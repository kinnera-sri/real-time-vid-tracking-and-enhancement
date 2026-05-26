import os
import cv2
import json
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sam3_root = "models/third_party/sam3"
sys.path.append(sam3_root)

from models.third_party.sam3 import sam3
from models.third_party.sam3.sam3 import build_sam3_image_model
from models.third_party.sam3.sam3.model.sam3_image_processor import Sam3Processor

def get_watermark_bboxes(input_folder, op_bbox_dir=None):

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model_path = f"{sam3_root}/assets/models_sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
    model = build_sam3_image_model(checkpoint_path=model_path, bpe_path=bpe_path)

    image_files = sorted(Path(input_folder).glob("*.png"))
    processor = Sam3Processor(model, confidence_threshold=0.5)
    if op_bbox_dir:
        os.makedirs(op_bbox_dir, exist_ok=True)
        
    all_bboxes = {}
    knl = np.ones((9, 9), np.uint8)

    for image_path in image_files:
        image = Image.open(image_path).convert("RGB")
        
        # Run inference with autocast
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state, 
                prompt="logo, watermark, subtitle"
            )
        
        nb_objects = len(inference_state["scores"])
        image_bboxes = []
        
        if nb_objects > 0:
            combined_mask = None

            for i in range(nb_objects):
                mask = inference_state["masks"][i].squeeze(0).cpu().numpy().astype(np.uint8)
                
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            combined_mask = cv2.dilate(combined_mask, knl, iterations=1)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 20:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                bbox = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]
                image_bboxes.append(bbox)

        image_name = os.path.splitext(os.path.basename(image_path))[0][:-4]
        all_bboxes[image_name] = image_bboxes
        
        if op_bbox_dir and len(image_bboxes) > 0:
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for bbox in image_bboxes:
                pt1 = (bbox[0][0], bbox[0][1])
                pt2 = (bbox[2][0], bbox[2][1])
                cv2.rectangle(image_bgr, pt1, pt2, (0, 0, 255), 2)
            save_path = os.path.join(op_bbox_dir, image_path.name)
            cv2.imwrite(save_path, image_bgr)

    return all_bboxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type=str, required=True)
    parser.add_argument('--op_json', type=str, default="watermarks.json")
    parser.add_argument('--op_bbox_dir', type=str, default="watermark_bboxes")
    args = parser.parse_args()

    bboxes = get_watermark_bboxes(args.imgs, args.op_bbox_dir)
    
    with open(args.op_json, 'w') as f:
        json.dump(bboxes, f, indent=4)