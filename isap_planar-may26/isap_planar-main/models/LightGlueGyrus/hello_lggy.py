from lightglue_gy import LightGlue, LightGlueGy
from robustpoint.robustpoint_top import RobustPointTop
from lightglue_gy.utils import rbd
from lightglue_gy import viz2d
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

from pathlib import Path

def read_image(filename):
    image = cv2.imread(filename, 0)
    image = cv2.resize(image, (640, 480))
    image = image[None]  # add channel axis
    image = image[None]  # add batch axis
    image = torch.tensor(image / 255.0, dtype=torch.float)
    return image

torch.set_grad_enabled(False)
images = Path("LightGlue_infer/assets/")



def main():
    import argparse
    parser = argparse.ArgumentParser(description='main ')
    #parser.add_argument('--ckpt_path',  type=str, required=True)
    args =  parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'


    ###########################################################################
    # 1. Model loading, Keypoint and matchig models
    ###########################################################################
    extractor = RobustPointTop(conf={"max_num_keypoints":512,
                                     "force_num_keypoints":False,
                                     "weights": "assets/checkpoints/tracker/robustpoint_ckpt.pth"}
                               ).eval().to(device)

    matcher = LightGlueGy(weights="assets/checkpoints/tracker/checkpoint_lgrp.tar", features=None).eval().to(device)

    ###########################################################################
    # 2. Data loading
    ###########################################################################
    image0 = read_image(images / "DSC_0411.JPG")
    image1 = read_image(images / "DSC_0410.JPG")

    ###########################################################################
    # 3. keypoint extraction and describe
    ###########################################################################
    feats0 = extractor(data={"image":image0.to(device)})
    feats1 = extractor(data={"image":image1.to(device)})

    ###########################################################################
    # 4. keypoint matching
    ###########################################################################
    matches01 = matcher({"image0": feats0, "image1": feats1})
    
    ###########################################################################
    # 5. post processing
    ###########################################################################
    # remove batch dimension
    feats0, feats1, matches01 = [ rbd(x) for x in [feats0, feats1, matches01] ]  


    ###########################################################################
    # 5. Visualization
    ###########################################################################
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    axes = viz2d.plot_images([image0[0], image1[0]])
    viz2d.plot_matches(m_kpts0, m_kpts1, lw=1)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    
    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0[0], image1[0]])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    plt.show()


if __name__ == "__main__":
    main()
