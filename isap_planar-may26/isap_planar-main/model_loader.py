import sys
import torch
import os
import json
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation

sys.path.append("models/LightGlueGyrus/")
sys.path.append("models")
sys.path.append("models/third_party/TinySAM")
sys.path.append("models/third_party/sam3")
sys.path.append("models/third_party/UniDepth")
sys.path.append("models/third_party/RoMaV2/src")
sys.path.append("models/third_party/Depth-Anything-3/src")
sys.path.append("../../LightGlue/")
tinysam_wt_path = "assets/checkpoints/tinysam.pth"



from ultralytics import YOLO
import torchvision.models as tvmodels
from torchvision.models.feature_extraction import create_feature_extractor
from tinysam import sam_model_registry, SamPredictor, SamHierarchicalMaskGenerator
from robustpoint.robustpoint_top import RobustPointTop
from lcnn.model_inference import fclip_inference, lcnn_inference
from lcnn.infer_lcnn import load_lcnn_model
from depth_anything_3.api import DepthAnything3
#from models.third_party.TinySAMGyrus.custom_decoder import TinySAMGyrus



roi_detection_config = { 
    "method": "geometric",
    "min_area": 1000,
    "max_area": 100000,
    "aspect_ratio_range": [0.5, 2.0],
    "corner_detection_threshold": 0.01,
    "plane_angle_threshold": 30,
    "generate_sam_mask": False,
    "fclip_model_path": "assets/checkpoints/roi_detection/fclip/fclip_hr_512x512.onnx",
    #"lcnn_model_path": "assets/checkpoints/roi_detection/lcnn/190418-201834-f8934c6-lr4d10-312k.pth",
    "lcnn_model_path": "assets/checkpoints/roi_detection/lcnn/lcnn_custom.pth",
    "lcnn_config": "models/lcnn/config/wireframe.yaml",
    "fclip_target_resolution": [512, 512],
    "USE_FCLIP": True,
    "USE_LCNN": True,
    "bounding_box": {
      "initial_height": 10,
      "overlap_threshold_with_vp": 0.5,
      "min_overlap_threshold": 0.1,
      "max_expansion": 500,
      "max_attempts": 50,
      "expansion_step": 10,
      "shift_step": 15,
      "min_angle_threshold": 7.0,
      "empty_region_value": 255,
      "use_normal": True,
      "window_size": 10,
      "std_threshold": 5
    },
    "empty_region_detection": {
      "sam_base_ckpt": "assets/checkpoints/roi_detection/SamAnnotator/sam2.1_hiera_large.pt",
      "sam_custom_ckpt": "assets/checkpoints/roi_detection/SamAnnotator/best_model.pt",
      "distance_transform_mask_size": 5,
      "distance_threshold": 0.4,
      "empty_color": [127, 127, 127],
      "border_kernel_size": 3,
      "empty_region_value": 255,
      "border_size": 1
    },
    "save_debug_visualizations": True,
    "line_classification": {
      "vertical_angle_ranges": [
        [60, 120],
        [240, 300]
      ],
      "horizontal_angle_ranges": [
        [150, 210],
        [0, 30],
        [330, 360]
      ],
      "min_lines_required": 4
    },
    "vanishing_point": {
      "iterations": 3000,
      "threshold": 2,
      "min_inliers": 4
    },
    "bbox_filter": {
      "size_threshold": 0.005,
      "normal_threshold_angle": 10.0,
      "weights": [50, 30, 10, 10]
    }
}

def load_models(kp_extractor=False, sam_mask_generator=False,
                lcnn_model=False, yolo_model=False,
                fe_model=False, da=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if kp_extractor:
        kp_conf = {"max_num_keypoints":4096,
               "force_num_keypoints":False,
               "weights": "assets/checkpoints/tracker/robustpoint_ckpt.pth"}
        kp_extractor = RobustPointTop(conf=kp_conf).eval().to(device)
    else:
        kp_extractor = None

    if lcnn_model:
        lcnn_model = load_lcnn_model(roi_detection_config, device)
    else:
        lcnn_model = None

    if sam_mask_generator:
        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint=tinysam_wt_path)
        sam.to(device=device).eval()
        #sam_predictor = SamPredictor(sam)
        sam_mask_generator = SamHierarchicalMaskGenerator(sam,
                                                          points_per_side=16,
                                                      stability_score_thresh=0.6,
                                                      pred_iou_thresh=0.7)
    else:
        sam_mask_generator = None


    if yolo_model:
        yolo_model = YOLO("yolo26n-seg.pt")  # load an official model
        yolo_model = yolo_model.to(device)
    else:
        yolo_model = None

    if fe_model:
        vgg16 = tvmodels.vgg16(pretrained=True)
        vgg_fe = create_feature_extractor(vgg16, return_nodes={'features.8': 'feats'})
        #vgg_fe = create_feature_extractor(vgg16, return_nodes={'features.15': 'feats'})
        vgg_fe = vgg_fe.eval().to(device)
        fe_model = vgg_fe
    else:
        fe_model = None

    if da:
        if 0:
            #de_model = DepthAnything3.from_pretrained("depth-anything/da3nested-giant-large")
            da_model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
            #de_model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE-1.1")
            da_model = dav3_model.to(device).eval()
        else:
            # dav2_config = {'encoder': 'vitl', 
            #            'features': 256, 
            #            'out_channels': [256, 512, 1024, 1024],
            #            'max_depth':20}
            # dav2_model = DepthAnythingV2(**dav2_config)
            # dav2_model.load_state_dict(torch.load(dav2_ckpt, map_location='cpu'))
            # da_model = dav2_model.to(device).eval()
            
            from unidepth.models import UniDepthV2
            unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
            da_model = unidepth_model.to(device).eval()

    else:
        da_model = None

    return kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, da_model




class OneFormerSemanticSegmentation:
    """
    Semantic segmentation using OneFormer (ADE20K, 150 classes).

    Extracts a binary wall mask by matching class labels at runtime,
    ensuring robustness to different id↔label orderings.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large"
        ).to(self.device)
        self.model.eval()

        # # Build id → label mapping from model config
        self.id2label = {int(k)+1: v for k, v in self.model.config.id2label.items()}

        of_label_path = "assets/config/oneformer_id2label.json"

        if not os.path.exists(of_label_path):
            with open(of_label_path, "w") as f:
                json.dump(self.id2label, f, indent=4)

    # -----------------------------------------------------------------
    def predict(self, image: Image.Image):
        """
        Parameters
        ----------
        image : PIL.Image (RGB)

        Returns
        -------
        wall_mask : np.ndarray  bool  [H, W]
        seg_map   : np.ndarray  int   [H, W]   full semantic map
        """
        inputs = self.processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        seg_map = (
            self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(image.height, image.width)],
            )[0]
            .cpu()
            .numpy()
        )
        seg_map += 1

        return seg_map, self.id2label

    # -----------------------------------------------------------------
    def unload(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':

    kp_extractor, sam_mask_generator, lcnn_model, yolo_model, fe_model, dav3_model = load_models(load_kp_extractor=True)

    print(kp_extractor)
    print(sam_mask_generator)
