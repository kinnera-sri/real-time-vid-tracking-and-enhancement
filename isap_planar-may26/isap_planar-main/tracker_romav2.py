import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sys

sys.path.append("models/LightGlueGyrus/")
from lightglue_gy import LightGlueGy
from lightglue_gy.utils import rbd
from robustpoint.robustpoint_top import RobustPointTop

sys.path.append("models/third_party/RoMaV2/src")
from romav2 import RoMaV2

from image_util import draw_matches, draw_keypoints
from lightglue import SuperPoint, LightGlue, viz2d

torch.set_grad_enabled(False)


class Tracker_Config:
    """Hyperparameters """
    KPTS_MATCH_USE_BB = False
    SPEED_UP = False
    MAX_NUM_KEYPOINTS = 2048
    DEBUG_REFINE_HG = False # To debug the refinement enable this


class PlaneTracker:
    def __init__(self, device=None, use_gyrus_lg=False):
        # setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: 
            device=device

        self.use_gyrus_lg = use_gyrus_lg

        # keypoint extractor and matcher models
        if self.use_gyrus_lg:
            extractor = RobustPointTop(conf={"max_num_keypoints":Tracker_Config.MAX_NUM_KEYPOINTS,
                                         "force_num_keypoints":False,
                                         "weights": "assets/checkpoints/tracker/robustpoint_ckpt.pth"})
            self.extractor = extractor.eval().to(self.device)

            matcher = LightGlueGy(weights="assets/checkpoints/tracker/checkpoint_lgrp.tar", features=None)
            self.matcher = matcher.eval().to(self.device)
        else:
            extractor = SuperPoint(max_num_keypoints=1024, detection_threshold=0.001)
            self.extractor = extractor.eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)


    def set_img0(self, img0, plane_mask0, empty_mask0):
        self.img0 = img0
        self.plane_mask0 = plane_mask0.astype(np.uint8)
        self.empty_mask0 = empty_mask0.astype(np.uint8)
        ########################################################################
        # Extract img0  features
        ########################################################################
        img0_t = torch.tensor((img0 / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        if self.use_gyrus_lg:
            self.feats0 = self.extractor(data={"image":img0_t})
        else:
            self.feats0 = self.extractor.extract(img0_t)

        #########################################################################
        ## Filter out keypoints which don't belong to this plane
        #########################################################################
        kpts0 = self.feats0["keypoints"][0]
        kpts0_np = kpts0.detach().cpu().numpy().round().astype(np.int32)
        kpts0_mask = plane_mask0[kpts0_np[:, 1], kpts0_np[:, 0]] == 0
        #Giving low scores to keypints not in this plane
        #self.feats0["keypoint_scores"][0][kpts0_mask] = torch.tensor(0.0).to(self.device)

        # we will set the coordinates for keypoints which don't belong to this
        # plane to (0,0). We will use this information to remove these points
        # from matching while doing homography computation
        self.feats0["keypoints"][0][kpts0_mask] = torch.tensor([0.,0.0]).to(self.device)

    def run(self, img1, dbg_matching_file=None):
        img_H, img_W = img1.shape[:2]

        img1_t = torch.tensor((img1/255.0).transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        if self.use_gyrus_lg:
            feats1 = self.extractor(data={"image":img1_t})
        else:
            feats1 = self.extractor.extract(img1_t)
        
        feats0 = self.feats0

        ########################################################################
        # match keypoints
        ########################################################################
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        # remove batch dimension
        feats0, feats1, matches01 = [ rbd(x) for x in [feats0, feats1, matches01] ]  
        kpts0, kpts1, matches = (feats0["keypoints"], 
                                 feats1["keypoints"], 
                                 matches01["matches"])

        m_kpts0_t, m_kpts1_t = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        m_kpts0 = m_kpts0_t.detach().cpu().numpy().astype(int)
        m_kpts1 = m_kpts1_t.detach().cpu().numpy().astype(int)


        ########################################################################
        # keep only the keypoints which belong to this plane
        ########################################################################
        mask = m_kpts0.all(axis=1) != 0 

        m_kpts0 = m_kpts0[mask]
        m_kpts1 = m_kpts1[mask]
        matches = matches[mask]

        if dbg_matching_file is not None and len(matches):
            axes = viz2d.plot_images([self.img0, img1])
            viz2d.plot_matches(m_kpts0, m_kpts1, lw=1)
            plt.savefig(f"{dbg_matching_file}")
            plt.close()


        if len(m_kpts0) < 4:
            # insufficient keypoints to compute homography, so return None
            return None, self.plane_mask0.astype(bool), self.empty_mask0.astype(bool)

        ########################################################################
        # compute homography between I0 and I1 matched keypoints
        ########################################################################
        Hg, mask_ = cv2.findHomography(m_kpts0, m_kpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.95)

        if Hg is None:
            # Sometimes cv2.findHomography doesn't converge
            return None, None, None
        else:
            plane_mask1 = cv2.warpPerspective(self.plane_mask0, Hg, (img_W,img_H), borderMode=cv2.BORDER_REPLICATE)
            empty_mask1 = cv2.warpPerspective(self.empty_mask0, Hg, (img_W,img_H), borderMode=cv2.BORDER_REPLICATE)

            plane_mask1 = plane_mask1.astype(bool)
            empty_mask1 = empty_mask1.astype(bool)


        return Hg, plane_mask1, empty_mask1


class ROITracker:
    def __init__(self, device=None):
        # setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device

        self.matcher = RoMaV2()
        self.matcher.apply_setting("base")
        self.matcher.to(self.device)

    def set_img0(self, img0, plane_mask0, roi_pts0, plane_idx=0, dbg_dir=None):
        self.dbg_dir = dbg_dir
        self.plane_idx = plane_idx
        self.img0 = img0
        self.plane_mask0 = plane_mask0
        self.roi_pts0 = roi_pts0

        ########################################################################
        # Extract img0  features
        ########################################################################
        self.img0_rgb = cv2.cvtColor(self.img0, cv2.COLOR_BGR2RGB)
        self.im_A = Image.fromarray(self.img0_rgb)
        
        self.feats0 = {}
        self.feats0["keypoints"] = [torch.tensor([[0,0]])] # Mock to not break dbg visualization if requested

    def run(self, img1, Hg_coarse=None, idx=None):
        img_H, img_W = img1.shape[:2]

        # from PIL import Image
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        if Hg_coarse is not None:
            img1w = cv2.warpPerspective(img1_rgb, np.linalg.inv(Hg_coarse), (img_W,img_H))
            im_B = Image.fromarray(img1w)
        else:
            im_B = Image.fromarray(img1_rgb)
        
        ########################################################################
        # match keypoints
        ########################################################################
        preds = self.matcher.match(self.im_A, im_B)
        matches, overlaps, prec_AB, prec_BA = self.matcher.sample(preds, 5000)
        kptsA, kptsB = self.matcher.to_pixel_coordinates(matches, img_H, img_W, img_H, img_W)
        
        m_kpts0 = kptsA.cpu().numpy()
        m_kpts1 = kptsB.cpu().numpy()

        ######################################################################
        # keep only the keypoints which belong to this plane
        ######################################################################
        kpts0_np = np.round(m_kpts0).astype(np.int32)
        # ensure indices are within bounds
        kpts0_np[:, 0] = np.clip(kpts0_np[:, 0], 0, img_W - 1)
        kpts0_np[:, 1] = np.clip(kpts0_np[:, 1], 0, img_H - 1)
        
        mask = self.plane_mask0[kpts0_np[:, 1], kpts0_np[:, 0]] != 0

        m_kpts0 = m_kpts0[mask]
        m_kpts1 = m_kpts1[mask]
        matches = matches[mask]


        if len(m_kpts0) < 4:
            # insufficient keypoints to compute homography, so return None
            print("insuffcient keypoints to do keypoint based matching")
            return [], None


        if idx is not None and self.dbg_dir is not None:
            axes = viz2d.plot_images([self.img0, img1])
            viz2d.plot_matches(m_kpts0, m_kpts1, lw=1)
            plt.savefig(f"{self.dbg_dir}/tracking/keypoints_matching_{self.plane_idx}_{idx}.png")
            plt.close()
            #plt.show()


        ######################################################################
        # compute homography between I0 and I1 matched keypoints
        ######################################################################
        Hg, mask_ = cv2.findHomography(m_kpts0, m_kpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.95)


        if Hg is None:
            return [], None

        if Hg_coarse is not None:
            Hg =  Hg @ Hg_coarse
        roi_pts0 = np.array(self.roi_pts0).astype(np.float32)[np.newaxis, :, :]
        roi_pts1 = cv2.perspectiveTransform(roi_pts0, Hg)[0]
        roi_pts1 = np.round(roi_pts1).astype(int)
        self.prev_Hg = Hg
        return roi_pts1, Hg


def main_video():
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--vid", required=True, help="Path to video file")
    parser.add_argument("--coarse_json", default=None, help="coarse hg generated by coarse tracking")
    args = parser.parse_args()
    DBG_DIR = "staging_dir/tracking"
    os.makedirs(DBG_DIR, exist_ok=True)

    if not os.path.exists(args.vid):
        raise Exception(f"Error: video file not found: {args.vid}")

    if args.coarse_json and not os.path.exists(args.coarse_json):
        raise Exception(f"Error: Coarse Hg file not found: {args.coarse_json}")

    if args.coarse_json:
        fs = cv2.FileStorage(args.coarse_json, cv2.FILE_STORAGE_READ)
        Hg_coarse_list = fs.getNode("Hgs").mat()
        fs.release()
    else:
        Hg_coarse_list = []

    #print(Hg_coarse_list.shape)

    #########################################################################
    # initialization
    #########################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roi_tracker =  ROITracker(device)

    #########################################################################
    # Read the first frame
    #########################################################################
    cap = cv2.VideoCapture(args.vid)
    ret, img0 = cap.read()
    img_H, img_W = img0.shape[:2]

    #########################################################################
    # Get the ad and plane ROIs
    #########################################################################
    # Press ENTER or SPACE to confirm, ESC to cancel
    x,y,w,h = cv2.selectROI("Select ROI Region", img0, fromCenter=False, showCrosshair=True)
    ad_roi_0 = np.array( [[x,y], [x+w,y], [x+w, y+h], [x, y+h]], dtype=np.int32)

    img_vis = img0.copy()
    img_vis = cv2.rectangle(img_vis, (ad_roi_0[0]), (ad_roi_0[2]), (0,255,0), 2)

    x,y,w,h = cv2.selectROI("Select plane Region", img_vis, fromCenter=False, showCrosshair=True)
    plane_roi = np.array( [[x,y], [x+w,y], [x+w, y+h], [x, y+h]], dtype=np.int32)

    img_vis = cv2.fillPoly(img_vis, [plane_roi], (0, 0, 100))
    img_vis = cv2.rectangle(img_vis, (ad_roi_0[0]), (ad_roi_0[2]), (0,255,0), 2)
    cv2.imshow("img", img_vis)
    cv2.waitKey(500)

    #########################################################################
    # Setup tracker
    #########################################################################
    plane_mask0 = np.zeros([img_H, img_W], dtype=np.uint8)
    plane_mask0 = cv2.fillPoly(plane_mask0, [plane_roi], 1)
    roi_tracker.set_img0(img0, plane_mask0, ad_roi_0, dbg_dir=None)


    #########################################################################
    # Output video file
    #########################################################################
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, filename = os.path.split(args.vid)
    filename, ext = os.path.splitext(filename)
    if args.coarse_json is None:
        dbg_out_file = f"{DBG_DIR}/{filename}_coarse{ext}"
    else:
        dbg_out_file = f"{DBG_DIR}/{filename}_fine{ext}"
    vid_out = cv2.VideoWriter(dbg_out_file, fourcc, 30.0, (img_W, img_H))
    vid_out.write(img0)


    #########################################################################
    # Start tracking
    #########################################################################
    Hg_list = []
    Hg_list.append(np.eye(3).reshape(-1))
    idx = 0
    ad_roi_prev = ad_roi_0
    Hg_coarse = None
    while(True):
        idx = idx+1
        ret, img1 = cap.read()
        if not ret:
            print("stream end? Exiting ...")
            break

        if len(Hg_coarse_list):
            Hg_coarse = Hg_coarse_list[idx].reshape(3,3)

        res, Hg = roi_tracker.run(img1, Hg_coarse, idx=idx)

        if len(res) > 0:
            ad_roi_i = res
        else:
            # we will use the previous ad_roi
            Hg = roi_tracker.prev_Hg
            ad_roi_i = ad_roi_prev

        img1 = cv2.fillPoly(img1, [ad_roi_i], (0,0,255))
        cv2.imshow("tracking", img1)
        cv2.waitKey(10)
        vid_out.write(img1)
        Hg_list.append(Hg.reshape(-1))

    print(f"Done....{idx}")
    vid_out.release()
    cap.release()

    #########################################################################
    # Save results
    #########################################################################
    # Open FileStorage for writing (.xml, .yml, or .json)
    if args.coarse_json is None:
        fs = cv2.FileStorage(f"{DBG_DIR}/{filename}_hgs_coarse.json", cv2.FILE_STORAGE_WRITE)
    else:
        fs = cv2.FileStorage(f"{DBG_DIR}/{filename}_hgs_fine.json", cv2.FILE_STORAGE_WRITE)
    Hg_arr = np.array(Hg_list)
    cv2.imwrite(f"{DBG_DIR}/{filename}_mask0.png", plane_mask0)
    cv2.imwrite(f"{DBG_DIR}/{filename}_img0.png", img0)
    fs.write("Hgs", Hg_arr)
    # Release the file handler
    fs.release()


if __name__ == "__main__":
    main_video()
