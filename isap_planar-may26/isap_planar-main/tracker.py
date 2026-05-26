import os
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import sys

sys.path.append("models/LightGlueGyrus/")
from lightglue_gy import LightGlueGy
from lightglue_gy.utils import rbd
from robustpoint.robustpoint_top import RobustPointTop

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


class ROITrackerCoarse:
    def __init__(self, device=None):
        # setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device

        # keypoint extractor and matcher models
        extractor = SuperPoint(max_num_keypoints=Tracker_Config.MAX_NUM_KEYPOINTS, detection_threshold=0.001)
        self.extractor = extractor.eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)


    def set_img0(self, img0, plane_mask0, roi_pts0, plane_idx=0, dbg_dir=None):
        self.prev_Hg = np.eye(3)
        self.dbg_dir = dbg_dir
        self.plane_idx = plane_idx
        self.img0 = img0
        self.plane_mask0 = plane_mask0
        self.roi_pts0 = roi_pts0

        img_H, img_W = img0.shape[:2]
        x,y,w,h = cv2.boundingRect(plane_mask0.astype(np.uint8))
        tl = np.array([x,y])
        br = np.array([x+w, y+h])
        ########################################################################
        # Extract img0  features
        ########################################################################
        img0_t = torch.tensor((img0 / 255.0).transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        feats0 = self.extractor.extract(img0_t)
        self.feats0 = feats0

        if 0: #self.dbg_dir is not None:
            kpts = feats0["keypoints"][0]
            img_vis = img0.copy()
            img_vis = draw_keypoints(img_vis, kpts, radius=1, color=(0, 255, 0))
            img_vis = cv2.rectangle(img_vis, (tl), (br), (0,0,255), 1)
            plt.imshow(img_vis)
            plt.show()
            #img0 = cv2.rectangle(img0, (ad_roi[0]), (ad_roi[2]), (0,255,0), 2)
            #plt.savefig(f"{dbg_dir}/tracking/tracking_plane_{plane_idx}.png")
            #plt.close()

        #########################################################################
        ## Filter out keypoints which don't belong to this plane
        # KNR::TODO Need to refine this. 
        #########################################################################
        kpts0 = self.feats0["keypoints"][0]
        kpts0_np = kpts0.detach().cpu().numpy().round().astype(np.int32)
        kpts0_mask = plane_mask0[kpts0_np[:, 1], kpts0_np[:, 0]] == 0
        #Giving low scores to keypints not in this plane
        #self.feats0["keypoint_scores"][0][kpts0_mask] = torch.tensor(0.0).to(self.device)

        ##### CAUTION ###############
        # we will set the coordinates for keypoints which don't belong to this
        # plane to (0,0). We will use this information to remove these points
        # from matching while doing homography computation
        self.feats0["keypoints"][0][kpts0_mask] = torch.tensor([0.,0.0]).to(self.device)


    def run(self, img1, idx=None):
        img_H, img_W = img1.shape[:2]
        img1_t = torch.tensor((img1/255.0).transpose(2, 0, 1), dtype=torch.float32).to(self.device)
        feats1 = self.extractor.extract(img1_t)
            
        feats0 = self.feats0
        
        ########################################################################
        # match keypoints
        ########################################################################
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        # remove batch dimension
        feats0, feats1, matches01 = [ rbd(x) for x in [feats0, feats1, matches01] ]  
        kpts0, kpts1, matches = (feats0["keypoints"], feats1["keypoints"], matches01["matches"])

        m_kpts0_t, m_kpts1_t = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        m_kpts0 = m_kpts0_t.detach().cpu().numpy().astype(int)
        m_kpts1 = m_kpts1_t.detach().cpu().numpy().astype(int)
    
        ######################################################################
        # keep only the keypoints which belong to this plane
        ######################################################################
        mask = m_kpts0.all(axis=1) != 0 
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


        ######################################################################
        # compute homography between I0 and I1 matched keypoints
        ######################################################################
        Hg, mask_ = cv2.findHomography(m_kpts0, m_kpts1, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.95)


        if Hg is None:
            return [], None
        roi_pts0 = np.array(self.roi_pts0).astype(np.float32)[np.newaxis, :, :]
        roi_pts1 = cv2.perspectiveTransform(roi_pts0, Hg)[0]
        roi_pts1 = np.round(roi_pts1).astype(int)
        self.prev_Hg = Hg
        return roi_pts1, Hg



class ROITrackerFineCPU:
    def __init__(self):
        # setup device
        pass

    def set_img0(self, img0, plane_mask0, roi_pts0, plane_idx=0, dbg_dir=None):
        self.prev_Hg = np.eye(3)
        #print(f"in set_img0 {img0.shape}, {plane_mask0.shape}")
        self.dbg_dir = dbg_dir
        self.plane_idx = plane_idx
        self.img0 = img0
        self.plane_mask0 = plane_mask0
        self.roi_pts0 = roi_pts0

        img_H, img_W = img0.shape[:2]
        x,y,w,h = cv2.boundingRect(plane_mask0.astype(np.uint8))
        tl = np.array([x,y])
        br = np.array([x+w, y+h])

    def run(self, img1, Hg_coarse):
        img_H, img_W = img1.shape[:2]
        W0_, H0_ = img_W, img_H
        tl, br = self.plane_bb 

        img1w0 = cv2.warpPerspective(img1, np.linalg.inv(Hg_coarse), (img_W,img_H))
        tlx, tly = tl
        brx, bry = br
        W0_ = brx-tlx
        H0_ = bry-tly

        if 1:
            plane_mask = self.plane_mask0[tly:bry, tlx:brx].astype(np.uint8)
        else:
            knl = np.ones((9, 9), np.uint8)
            plane_mask = np.zeros_like(self.plane_mask0)
            plane_mask[m_kpts1[:,1], m_kpts1[:,0]] = 1
            plane_mask = plane_mask[tly:bry, tlx:brx]
            plane_mask = cv2.dilate(plane_mask.astype(np.uint8), knl, iterations=2)
            #plt.imshow(plane_mask)
            #plt.show()
            #exit()
        img1w0 = img1w0[tly:bry, tlx:brx]


        ######################################################
        # Refinement using ECC
        ######################################################
        imgL_bgr = self.img0[tly:bry, tlx:brx]
        imgL = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2GRAY)
        imgR_bgr = img1w0
        imgR = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2GRAY)
        if Tracker_Config.SPEED_UP:
            plane_mask = cv2.resize(plane_mask, [W0_//4,H0_//4], interpolation=cv2.INTER_NEAREST)
            imgL = cv2.resize(imgL, [W0_//4,H0_//4], interpolation=cv2.INTER_NEAREST)
            imgR = cv2.resize(imgR, [W0_//4,H0_//4], interpolation=cv2.INTER_NEAREST)


        if 0:
            fig, axes = plt.subplots(nrows=3)
            axes[0].imshow(imgR)
            axes[1].imshow(imgL)
            axes[2].imshow(plane_mask)
            plt.show()

        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)

        number_of_iterations = 2000;
        termination_eps = 1e-10;
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
        
        try:
            (cc, warp_matrix) = cv2.findTransformECC(imgR.astype(np.float32), 
                                             imgL.astype(np.float32),
                                             warp_matrix,
                                             warp_mode,
                                             criteria,
                                             plane_mask)
            warp_matrix = np.linalg.inv(warp_matrix)
        except:
            print("cv2.findTransformECC didn't converge")
            warp_matrix = np.eye(3, 3, dtype=np.float32)

        if  Tracker_Config.DEBUG_REFINE_HG: 
            if Tracker_Config.SPEED_UP:
                imgLwR = cv2.warpPerspective(imgL_bgr, warp_matrix, (W0_//4,H0_//4))
            else:
                imgLwR = cv2.warpPerspective(imgL_bgr, warp_matrix, (W0_,H0_))
            cv2.destroyAllWindows()
            for i in range(50):
                cv2.imshow("imgc orig/pred", imgR_bgr)
                cv2.waitKey(50)
                cv2.imshow("imgc orig/pred", imgLwR)
                cv2.waitKey(50)
            cv2.destroyAllWindows()

    
        tfm = np.eye(3)
        tfm[0,2] = -tlx
        tfm[1,2] = -tly
        if Tracker_Config.SPEED_UP:
            tfm[0,0] = 1./4
            tfm[1,1] = 1./4
            tfm[0,2] /= 4.
            tfm[1,2] /= 4.
        Hg_refined = np.linalg.inv(tfm) @warp_matrix @ tfm @ Hg_coarse


        if Tracker_Config.DEBUG_REFINE_HG: 
            img0w1 = cv2.warpPerspective(self.img0, Hg_refined, (img_W,img_H))
            cv2.destroyAllWindows()
            for i in range(50):
                cv2.imshow("img orig", img1)
                cv2.waitKey(50)
                cv2.imshow("img orig", img0w1)
                cv2.waitKey(50)
            cv2.destroyAllWindows()

        roi_pts0 = np.array(self.roi_pts0).astype(np.float32)[np.newaxis, :, :]
        roi_pts1 = cv2.perspectiveTransform(roi_pts0, Hg_coarse)[0]
        roi_pts1 = np.round(roi_pts1).astype(int)

        roi_pts1_refined = cv2.perspectiveTransform(roi_pts0, Hg_refined)[0]
        roi_pts1_refined = np.round(roi_pts1_refined).astype(int)
        dist = (roi_pts1_refined - roi_pts1)**2
        max_dist = np.max(np.sum(dist, axis=-1))

        if max_dist > 32**2: #32 pixels distance
            return roi_pts1, Hg_coarse

        return roi_pts1_refined, Hg_refined


def main_video_coarse():
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--vid", required=True, help="Path to video file")
    args = parser.parse_args()
    DBG_DIR = "staging_dir/tracking"

    if os.path.exists(DBG_DIR):
        error_str = f"\nERROR! {DBG_DIR} exists! Perhaps it contains results from previous run\n"
        error_str += "Please delete/move this folder and run again..."
        raise Exception(error_str)

    os.makedirs(f"{DBG_DIR}", exist_ok=True)

    if not os.path.exists(args.vid):
        raise Exception(f"Error: video file not found: {args.vid}")

    #########################################################################
    # Some initialization
    #########################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    roi_tracker =  ROITrackerCoarse(device)

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
    #roi_tracker.set_img0(img0, plane_mask0, ad_roi, dbg_dir=DBG_DIR)


    #########################################################################
    # Output video file
    #########################################################################
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _, filename = os.path.split(args.vid)
    dbg_out_file = f"{DBG_DIR}/{filename}"
    vid_out = cv2.VideoWriter(dbg_out_file, fourcc, 30.0, (img_W, img_H))
    vid_out.write(img0)


    Hg_list = []
    Hg_list.append(np.eye(3).reshape(-1))
    idx = 0
    while(True):
        ret, img1 = cap.read()
        if not ret:
            print("stream end? Exiting ...")
            break
        
        res, Hg = roi_tracker.run(img1, idx=idx)
        if len(res) > 0:
            ad_roi_i = res
        else:
            # we will use the previous ad_roi
            Hg = roi_tracker.prev_Hg
            pass
        img1 = cv2.fillPoly(img1, [ad_roi_i], (0,0,255))
        cv2.imshow("img", img1)
        cv2.waitKey(10)
        vid_out.write(img1)
        Hg_list.append(Hg.reshape(-1))
        idx = idx+1

    cap.release()
    print(f"Done....{idx}")
    vid_out.release()

    #########################################################################
    # Save results
    #########################################################################
    # Open FileStorage for writing (.xml, .yml, or .json)
    filename, ext = os.path.splitext(filename)
    fs = cv2.FileStorage(f"{DBG_DIR}/{filename}_hgs_coarse.json", cv2.FILE_STORAGE_WRITE)
    Hg_arr = np.array(Hg_list)
    cv2.imwrite(f"{DBG_DIR}/{filename}_mask0.png", plane_mask0)
    cv2.imwrite(f"{DBG_DIR}/{filename}_img0.png", img0)
    fs.write("Hgs", Hg_arr)
    print(Hg_arr.shape)

    # Release the file handler
    fs.release()

def show_tracked_plane():
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--vid", required=True, help="Path to video file")
    parser.add_argument("--plane_mask", required=True, help="mask file")
    parser.add_argument("--Hgs", required=True, help="Homographies")
    args = parser.parse_args()

    plane_mask = cv2.imread(args.plane_mask, 0)
    fs = cv2.FileStorage(args.Hgs, cv2.FILE_STORAGE_READ)
    Hgs = fs.getNode("Hgs").mat()
    #plt.imshow(plane_mask.astype(bool))
    #plt.show()
    cap = cv2.VideoCapture(args.vid)
    ret, img0 = cap.read()
    img_H, img_W = img0.shape[:2]

    dbg_out_file = "out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_out = cv2.VideoWriter(dbg_out_file, fourcc, 30.0, (img_W, img_H))
    vid_out.write(img0)

    idx = 0
    while(True):
        ret, img1 = cap.read()
        if not ret:
            print("stream end? Exiting ...")
            break
        Hg = Hgs[idx].reshape(3,3)
        Hg = np.linalg.inv(Hg)
        mask = cv2.warpPerspective(plane_mask, Hg, (img_W,img_H),
                                   borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_NEAREST)
        img1[mask.astype(bool)] = (0,0,255)
        cv2.imshow("img", img1)
        cv2.waitKey(100)
        vid_out.write(img1)
        idx = idx+1

    cap.release()
    vid_out.release()
    print(f"Done....{idx}")




def main_video_refine_gpu():
    import sys
    import docker
    import subprocess
    DBG_DIR = "staging_dir/tracking"

    if os.path.exists("opencv_gpu_docker/tracking"):
        error_str = f"\nERROR! opencv_gpu_docker/tracking exists! Perhaps it contains results from previous run\n"
        error_str += "Please delete/move tracking folder and run again..."
        raise Exception(error_str)

    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--vid", required=True, help="name of the video file")
    args = parser.parse_args()

    _, filename_w_ext = os.path.split(args.vid)
    filename, ext = os.path.splitext(filename_w_ext)

    client = docker.DockerClient(base_url='unix:///run/user/1000/docker.sock')

    gpu_request = docker.types.DeviceRequest(driver='nvidia',
                                         count=-1,  # This is equivalent to "all"
                                         capabilities=[['gpu']])

    image_name = "knr_cv_cuda:latest"
    try:
        client.images.get(image_name)
    except docker.errors.ImageNotFound:
        print(client.images.list())
        print(f"not found {image_name}...")
        exit()

    container = client.containers.run(image_name, command="sleep infinity",
                                  device_requests=[gpu_request],
                                  detach=True)
    # Start the container in detached mode (background)

    print(f"Started container with ID: {container.id}")

    command = ["cp", "-r", f"{DBG_DIR}", "./opencv_gpu_docker/"]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)
        container.stop()
        exit()

    command = ["tar", "-cvf", "opencv_gpu_docker.tar", "./opencv_gpu_docker/"]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)
        container.stop()
        exit()

    #docker cp app_src.tar container_id:/
    command = ["docker", "cp", "./opencv_gpu_docker.tar", f"{container.id}:/"]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)
        container.stop()
        exit()

    exit_code, output = container.exec_run("tar -xvf opencv_gpu_docker.tar")
    print(output.decode())

    exe_file = f"/opencv_gpu_docker/app_src/ecc_cuda/ecc_video_app"
    img0_file = f"/opencv_gpu_docker/tracking/{filename}_img0.png"
    mask0_file = f"/opencv_gpu_docker/tracking/{filename}_mask0.png"
    vid_file = f"/opencv_gpu_docker/tracking/{filename_w_ext}"
    hg_coarse_file = f"/opencv_gpu_docker/tracking/{filename}_hgs_coarse.json"
    exit_code, output = container.exec_run(f"{exe_file} {img0_file} {mask0_file} {vid_file} {hg_coarse_file}")
    print(output.decode())

    command = ["docker", "cp", f"{container.id}:/opencv_gpu_docker/tracking/{filename}_hgs_fine.json", DBG_DIR]
    print(f"Running command {command}")
    try:
        # Run the command and wait for it to complete
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Script failed with exit code {e.returncode}")
        print("Error output:", e.stderr)
        container.stop()
        exit()

    container.stop(timeout=1)


if __name__ == "__main__":
    #main_video_coarse()
    main_video_refine_gpu()
    #show_tracked_plane()
