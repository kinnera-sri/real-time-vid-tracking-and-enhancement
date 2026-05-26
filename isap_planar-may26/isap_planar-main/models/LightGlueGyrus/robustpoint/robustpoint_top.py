import torch
from torch import nn

from .base_model import BaseModel
from .robustpoint import RobustPoint

#KNR:TODO This must come from config 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobustPointTop(BaseModel):
    default_conf = {
        "detection_threshold": 0.6,
        "max_num_keypoints": -1,
        "force_num_keypoints": False,
        "weights" : "../ckpts/robustpoint_ckpt.pth"
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        self.model = RobustPoint(prd_descr=True)
        print(f"Loading checkpoint from {conf.weights}")
        ckpt = torch.load(conf.weights, map_location=device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.max_num_keypoints = conf.max_num_keypoints
        self.model.force_num_keypoints = conf.force_num_keypoints
        self.model.det_thresh = conf.detection_threshold
        self.set_initialized()


    def _forward(self, data):

        op = self.model.infer(data)

        pred = {
                "keypoints": op["keypoints"],
                "keypoint_scores": op["scores"],
                "descriptors": op["descriptors"],
        }

        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
