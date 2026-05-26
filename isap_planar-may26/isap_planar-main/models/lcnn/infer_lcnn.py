#!/usr/bin/env python3
"""Lightweight wrapper around the LCNN demo to allow calling from Python.

Provides run_lcnn(checkpoint, image_path, devices='0') which returns a list
of generated output file paths (SVGs).

Defaults:
 - checkpoint: '190418-201834-f8934c6-lr4d10-312k.pth'
 - devices: '0'
"""

import os
import os.path as osp
import pprint
import random
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.transform
import torch

import lcnn
from lcnn.lcnn.config import C, M
from lcnn.lcnn.models.line_vectorizer import LineVectorizer
from lcnn.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from lcnn.lcnn.postprocess import postprocess
from lcnn.lcnn.utils import recursive_to

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])


def c(x):
    return sm.to_rgba(x)


def _ensure_rgb(im: np.ndarray) -> np.ndarray:
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, 2)
    return im[:, :, :3]


def load_lcnn_model(
    config: dict,
    device: torch.device,
):
    """Load LCNN model and return (model, device).

    This encapsulates configuration parsing, device setup, checkpoint loading,
    model construction and state_dict loading. It does not run inference.
    """
    checkpoint = config.get("lcnn_model_path", "assets/checkpoints/roi_detection/lcnn/190418-201834-f8934c6-lr4d10-312k.pth")
    config_file = config.get("lcnn_config", "roi_detection/lcnn/config/wireframe.yaml")
    
    # Load config and update global config objects
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)

    # Deterministic seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load checkpoint
    checkpoint_obj = torch.load(checkpoint, map_location=device)

    # Build model
    model = lcnn.lcnn.models.hg(
        depth=M.depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
    )
    model = MultitaskLearner(model)

    model = LineVectorizer(model)

    model.load_state_dict(checkpoint_obj["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def infer_single_image(
    model,
    device: torch.device,
    im: str | os.PathLike,
    thresholds: Optional[List[float]] = None,
) -> List[List[List[float]]]:
    """Run LCNN inference on a single image using a preloaded model.

    Args:
        model: Loaded LCNN model (already moved to the correct device).
        device: torch.device where model/input should reside.
        image_path: Path to the image file to process.
        thresholds: (unused) kept for compatibility.

    Returns:
        List of line segments in the format [[ [y1,x1], [y2,x2] ], ...].
    """
    # if image_path is None:
    #     raise ValueError("image_path must be provided")

    # imname = str(image_path)
    # print(f"Processing {imname}")
    # im = skimage.io.imread(imname)
    im = _ensure_rgb(im)
    im_resized = skimage.transform.resize(im, (512, 512)) * 255
    image = (im_resized - M.image.mean) / M.image.stddev
    image = torch.from_numpy(np.rollaxis(image, 2)[None].copy()).float()
    with torch.no_grad():
        input_dict = {
            "image": image.to(device),
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
            },
            "mode": "testing",
        }
        H = model(input_dict)["preds"]

    lines = H["lines"][0].cpu().numpy() / 128 * im.shape[:2]
    scores = H["score"][0].cpu().numpy()
    for i in range(1, len(lines)):
        if (lines[i] == lines[0]).all():
            lines = lines[:i]
            scores = scores[:i]
            break

    # postprocess lines to remove overlapped lines
    diag = (im.shape[0] ** 2 + im.shape[1] ** 2) ** 0.5
    nlines, nscores = postprocess(lines, scores, diag * 0.01, 0, False)

    # Convert to simple list format: [[[y1,x1],[y2,x2]], ...]
    lines_out: List[List[List[float]]] = []
    for (a, b), s in zip(nlines, nscores):
        if s < thresholds:
            continue
        # a and b are (y, x) coordinates in floating point
        lines_out.append([[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]])

    return lines_out


# Replace original monolithic run_lcnn with a small wrapper that uses the two functions above
def run_lcnn(
    checkpoint: str = "190418-201834-f8934c6-lr4d10-312k.pth",
    image_path: str | os.PathLike = None,
    devices: str = "0",
    config_file: str = "config/wireframe.yaml",
    thresholds: Optional[List[float]] = None,
) -> List[str]:
    """Convenience wrapper that loads the model and runs inference on one image."""
    model, device = load_lcnn_model(
        checkpoint=checkpoint, devices=devices, config_file=config_file
    )
    return infer_single_image(
        model, device, image_path=image_path, thresholds=thresholds
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LCNN inference on a single image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--checkpoint",
        default="190418-201834-f8934c6-lr4d10-312k.pth",
        help="Checkpoint filename (default the one requested)",
    )
    parser.add_argument(
        "--devices", default="0", help="CUDA devices string (default '0')"
    )
    parser.add_argument(
        "--config", default="config/wireframe.yaml", help="Config yaml file"
    )
    args = parser.parse_args()

    outputs = run_lcnn(
        args.checkpoint, args.image, devices=args.devices, config_file=args.config
    )
    print("Output line segments:")
    for line in outputs:
        print("  ", line)
