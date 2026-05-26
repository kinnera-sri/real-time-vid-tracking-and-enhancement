import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import torch
import re
import tqdm
import torch
import torch.nn.functional as F

sys.path.append("models/LightGlueGyrus/")
#sys.path.append("./roi_detection")
#sys.path.append("./tracker")

from image_util import draw_matches, draw_keypoints
from robustpoint.robustpoint_top import RobustPointTop

def gaussian_kernel_2d(kernel_size, sigma):
    # Create 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    gaussian_1d = torch.exp(-0.5 * (coords / sigma)**2)
    #gaussian_1d /= gaussian_1d.sum() # Normalize

    # Create 2D Gaussian kernel from 1D kernels
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    #gaussian_2d /= gaussian_2d.sum() # Normalize

    return gaussian_2d

def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kp_conf = {"max_num_keypoints":4096,
               "force_num_keypoints":False,
               "weights": "assets/checkpoints/tracker/robustpoint_ckpt.pth"}
    kp_extractor = RobustPointTop(conf=kp_conf).eval().to(device)

    return kp_extractor


def show_proposals(img, proposal_mask, ax):
    img_ = np.ones((img.shape[0], img.shape[1], 4))
    img_[:,:,3] = 0
    color_mask = np.concatenate([np.random.random(3), [0.85]])
    proposal_mask = cv2.resize(proposal_mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    proposal_mask = proposal_mask.astype(bool)
    img_[proposal_mask] = color_mask
    ax.imshow(img_) 
    return


def convert_bb_2_rois(bbs):
    rois = []
    for bb in bbs:
        x0,y0,x1,y1 = bb
        pt0 = [x0,y0]
        pt2 = [x1,y1]
        pt1 = [x1,y0]
        pt3 = [x0,y1]
        rois.append([pt0, pt1, pt2, pt3])
    return rois

def expand_bb(roi_img, bb, iters=100, dxy=4):
    img_H, img_W = roi_img.shape[:2]
    #tlx, tly, brx, bry = bb
    bbp = np.array(bb)

    # we will expand the bb in 4 modes in that order
    # 0 -> expand in all directions
    # 1 -> expand tl border in both directions
    # 2 -> expand br border in both directions
    # 3 -> expand tlx
    # 4 -> expand tly
    # 5 -> expand brx
    # 6 -> expand bry
    expand_mode = 0

    #while(True):
    for i in range(iters):

        exp_within_boundary = True
        match expand_mode:
            case 0: #expand on all sides
                bbc = bbp + np.array([-dxy,-dxy,dxy,dxy])
            case 1: # expand tl 
                bbc = bbp + np.array([-dxy,-dxy,0,0])
            case 2: # expand br
                bbc = bbp + np.array([0,0,dxy,dxy])
            case 3: # expand tlx
                bbc = bbp + np.array([-dxy,0,0,0])
            case 4: # expand brx
                bbc = bbp + np.array([0,0,dxy,0])
            case 5: # expand tly
                bbc = bbp + np.array([0,-dxy,0,0])
            case 6: # expand bry
                bbc = bbp + np.array([0,0,0,dxy])

        if bbc[0]<0 or bbc[1] <0 or bbc[2]>img_W or bbc[3]>img_H:
            exp_within_boundary = False
            
        img_bb = roi_img[bbc[1]:bbc[3], bbc[0]:bbc[2]]
        n_holes = (img_bb == 0).sum()
        if n_holes > 0 or not exp_within_boundary:
            if expand_mode >= 6:
                break
            expand_mode = expand_mode +1
        else:
            bbp = bbc
    return bbp

def check_if_bb_surrounded_by_kpts(bb, kpts, plane_mask): 
    # Seperate the plane into 4 regions around the center of the bb
    #ctx, cty = ctr
    tlx, tly, brx, bry = bb.astype(np.int32).reshape(-1)
    w_ = brx-tlx
    h_ = bry-tly
    wb8 = w_//8
    hb8 = h_//8
    ctx, cty = (tlx+brx)//2, (tly+bry)//2

    # tl_kpts
    tl_mask = np.zeros_like(plane_mask, dtype=bool)
    tl_mask[:cty-hb8, :ctx-wb8] = True
    tl_mask = np.logical_and(plane_mask, tl_mask)
    kpts_mask = tl_mask[kpts[:,1], kpts[:,0]] ==1
    tl_kpts = kpts[kpts_mask]

    # tr_kpts
    tr_mask = np.zeros_like(plane_mask, dtype=bool)
    tr_mask[:cty-hb8, ctx+wb8:] = True
    tr_mask = np.logical_and(plane_mask, tr_mask)
    kpts_mask = tr_mask[kpts[:,1], kpts[:,0]] ==1
    tr_kpts = kpts[kpts_mask]

    # bl_kpts
    bl_mask = np.zeros_like(plane_mask, dtype=bool)
    bl_mask[cty+hb8:, :ctx-wb8] = True
    bl_mask = np.logical_and(plane_mask, bl_mask)
    kpts_mask = bl_mask[kpts[:,1], kpts[:,0]] ==1
    bl_kpts = kpts[kpts_mask]

    # br_kpts
    br_mask = np.zeros_like(plane_mask, dtype=bool)
    br_mask[cty+hb8:, ctx+wb8:] = True
    br_mask = np.logical_and(plane_mask, br_mask)
    kpts_mask = br_mask[kpts[:,1], kpts[:,0]] ==1
    br_kpts = kpts[kpts_mask]

    kpts_hist = np.array([len(tl_kpts), len(tr_kpts), len(bl_kpts), len(br_kpts)], dtype=int)
    # 1 keypoints on each side
    res = np.all(kpts_hist>=1)

    return res, kpts_hist




def get_distribution_of_bb_kpts(bb, kpts):
    x0, y0, x1, y1 = bb
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    npts_quad0 = np.sum(np.logical_and((kpts[:,0] <  cx), (kpts[:,1] <  cy)))
    npts_quad1 = np.sum(np.logical_and((kpts[:,0] >= cx), (kpts[:,1] <  cy)))
    npts_quad2 = np.sum(np.logical_and((kpts[:,0] >= cx), (kpts[:,1] >= cy)))
    npts_quad3 = np.sum(np.logical_and((kpts[:,0] <  cx), (kpts[:,1] >= cy)))

    return np.array([npts_quad0, npts_quad1, npts_quad2, npts_quad3])





def get_rois(roi_img, min_wh=[200,150], itrs=300):
    roi_img = roi_img.astype(bool)
    img_H, img_W = roi_img.shape[:2]
    bb_list = []

    # we will start with min_xy
    min_w, min_h = min_wh
    bb_x,bb_y,bb_w,bb_h = cv2.boundingRect(roi_img.astype(np.uint8))

    # the bouning box is not big enough
    if bb_w <= min_w or bb_h <= min_h:
        return bb_list 
    
    centers = np.zeros([itrs,2], dtype=int)
    for i in tqdm.tqdm(range(itrs)):
        # pick a random point in the image

        ctx = np.random.randint(bb_x+min_w//2, bb_x+bb_w-min_w//2, 1)[0]
        cty = np.random.randint(bb_y+min_h//2, bb_y+bb_h-min_h//2, 1)[0]

        dist = np.sum((centers-np.array([ctx, cty]))**2, axis=1)
        if np.min(dist) < 50: # this point is near to another point
            continue

        tlx = ctx-min_w//2
        tly = cty-min_h//2
        brx = ctx+min_w//2
        bry = cty+min_h//2

        if tlx<0 or tly <0 or brx>img_W or bry>img_H:
            continue
        # check for any holes in the interest region 
        img_bb = roi_img[tly:bry, tlx:brx]
        n_holes = (img_bb == 0).sum()
        if n_holes>0:
            continue

        centers[i] = ctx, cty
        bb = [tlx, tly, brx, bry]
        bb =  expand_bb(roi_img, bb)
        if 0:
            # remove this region
            roi_img[bb[1]:bb[3], bb[0]:bb[2]] = 0
        bb_list.append(bb)
    rois_list = convert_bb_2_rois(bb_list)
    return rois_list
        



def get_trackable_roi(img, plane_mask, kpts, empty_region, device, ksize=7,
                      min_kpts_per_quandrant=2, min_score=20):

    is_trackable = False
    img_H, img_W = img.shape[:2]

    ##########################################################################
    # 1. Get kpts image for this plane
    ##########################################################################
    kpts_img = np.zeros([img_H, img_W], dtype=np.uint8)
    kpts_img[kpts[:,1], kpts[:,0]] = 1
    kpts_img = np.logical_and(kpts_img, plane_mask)
    kpts_img = np.logical_and(kpts_img, 1-empty_region)

    ##########################################################################
    # 2. Gridize this to 32x32 size grid and compute the histogram of
    # keypoints
    ##########################################################################
    # grid_size = 32, max histogram = (32/8)**2 = 16 < 255
    # grid_size = 64, max histogram = (64//8)**2 = 64 < 255
    # grid_size = 128, max histogram = (128/8)**2 = 256 > 255, but very unlikely
    grid_sz = 32 # pixels
    kpts_img_ds = cv2.resize(kpts_img.astype(np.float32), (img_W//grid_sz, img_H//grid_sz), interpolation=cv2.INTER_AREA)
    kpts_hist_img = (kpts_img_ds*(grid_sz**2))

    ##########################################################################
    # 3. Convert to binary image, Any grid with one or more kpts will be 1.
    ##########################################################################
    _, img_b = cv2.threshold(kpts_hist_img, 0.5, maxval=1, type=cv2.THRESH_BINARY)

    ##########################################################################
    # 4. Get a inverse Gaussian kernel. We will have weights with close to
    # zeros in the middle and close to 1 near edges
    ##########################################################################
    knl = 1-gaussian_kernel_2d(ksize, sigma=ksize/4)
    # Give some more weightage to corner
    knl[0,0] = 2 
    knl[0,ksize-1] = 2 
    knl[ksize-1,0] = 2 
    knl[ksize-1,ksize-1] = 2 
    knl = knl.reshape(1,1,ksize,ksize)
    knl = knl.to(device)

    ##########################################################################
    # Convert image to tensor and convolve with this kernel
    ##########################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_t = torch.tensor(img_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img_t = img_t.to(device)

    out_img_t = F.conv2d(img_t, knl, stride=1, padding=ksize//2)

    ##########################################################################
    # Find the peak location
    ##########################################################################
    out_img = out_img_t[0,0].detach().cpu().numpy()
    max_idx = np.argmax(out_img)
    y_idx = max_idx//out_img.shape[1]
    x_idx = max_idx - y_idx*out_img.shape[1]


    ##########################################################################
    # Get the bounding box in the orginal image
    ##########################################################################
    x_idx *= grid_sz
    y_idx *= grid_sz
    tlx = x_idx - (ksize//2)*grid_sz
    tly = y_idx - (ksize//2)*grid_sz


    brx = x_idx + (ksize//2+1)*grid_sz
    bry = y_idx + (ksize//2+1)*grid_sz

    # clip the bb to be within the image boundary
    tlx = max(0, tlx)
    tly = max(0, tly)
    brx = min(img_W, brx)
    bry = min(img_H, bry)

    roi_img = kpts_img[tly:bry, tlx:brx]
    kpts_score = np.sum(roi_img)


    if 0:
        kpts_img = cv2.rectangle(kpts_img, (tlx, tly), (brx, bry), 255, 2)
        kpts_img = draw_keypoints(kpts_img, kpts, color=255, radius=3)

        out_vis = np.zeros([out_img.shape[0], out_img.shape[1], 3])
        out_vis[:,:,0] = img_b
        out_vis[:,:,1] = img_b

        out_vis[y_idx-ksize//2:y_idx+ksize//2+1, x_idx-ksize//2:x_idx+ksize//2+1,2] = 1

        fig,ax = plt.subplots(nrows=2, ncols=2, figsize=(9,10))
        ax[0,0].imshow(kpts_img)
        ax[0,1].imshow(img_b)
        ax[1,0].imshow(out_img)
        ax[1,1].imshow(out_vis)
        plt.show()
        plt.close()

    trackable_bb = np.array([tlx,tly,brx,bry], dtype=np.int32)
    mask_x = np.logical_and((kpts[:,0] >= tlx), (kpts[:,0] < brx))
    mask_y = np.logical_and((kpts[:,1] >= tly), (kpts[:,1] < bry))
    mask = np.logical_and(mask_x, mask_y)
    kpts = kpts[mask]

    kpts_distribution = get_distribution_of_bb_kpts(trackable_bb, kpts)
    print(f"Keypoint distribution:{kpts_distribution} score:{kpts_score}")
    if (np.min(kpts_distribution) >= min_kpts_per_quandrant) and (kpts_score > min_score):
        is_trackable = True
    trackable_roi  = convert_bb_2_rois([trackable_bb])[0]
    return trackable_roi, is_trackable, kpts_score
  



def simplify_mask(mask, max_pts=10):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) 
    epsilon = perimeter/(5*max_pts) # Limit to 10 points
    contour_simple = cv2.approxPolyDP(contours[0], epsilon, True) 
    mask_simple = np.zeros_like(mask).astype(np.uint8)
    cv2.fillConvexPoly(mask_simple, contour_simple, 1)
    return mask_simple.astype(bool)






def main_img():
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument("--img", required=True, help="Path to input image file")
    args = parser.parse_args()

    if not os.path.exists(args.img):
        raise Exception(f"Error: Image file not found: {args.img}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kp_extractor = load_models()

    img = cv2.imread(args.img)
    if img is None:
        raise Exception("Empty img file")
    img_H, img_W = img.shape[:2]

    img_t = torch.tensor((img/255.0).transpose(2, 0, 1), dtype=torch.float32).to(device)
    feats = kp_extractor(data={"image":img_t})
    kpts = feats["keypoints"].detach().cpu().numpy().astype(int)[0]

    roi_img = np.zeros([img_H, img_W], dtype=np.uint8)
    roi_img = draw_keypoints(roi_img, kpts, color=1, radius=2)
    roi_img = 1 - roi_img

    rois = get_rois(roi_img)
    bb_img = roi_img.copy()
    if len(rois) ==0 :
        print("No rois found")

    for roi in rois:
        x0,y0 = roi[0]
        x1,y1 = roi[2]
        cv2.rectangle(bb_img, (x0, y0), (x1, y1), (0,0,255), 2)

    fig,ax = plt.subplots(nrows=2)
    ax[0].imshow(img)
    ax[1].imshow(bb_img)
    plt.show()



if __name__ == "__main__":
    np.set_printoptions(linewidth=np.inf)
    #for i in range(10):
    #    main_trackable_roi()
    main_img()
