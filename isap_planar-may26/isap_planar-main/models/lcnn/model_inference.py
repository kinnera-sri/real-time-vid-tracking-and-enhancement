import numpy as np
import cv2
import copy
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt
from lcnn.infer_lcnn import load_lcnn_model, infer_single_image


mean = [109.730, 103.832, 98.681] 
stddev = [22.275, 22.124, 23.229]


def fclip_inference(onnx_session, input_size, image, detected_lines):
    """
    Runs line detection using f-clip and apped the lines to detected lines and return them
    """
    # Pre process
    org_H, org_W = image.shape[0], image.shape[1]
    H, W = input_size
    input_image = copy.deepcopy(image)
    output_image = copy.deepcopy(image)

    # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, dsize=(W, H))
    input_image = (input_image - mean) / stddev
    input_image = input_image.transpose(2, 0, 1)
    input_image = input_image[np.newaxis, ...]
    input_image = input_image.astype(np.float32)

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    result = onnx_session.run(None, {input_name: input_image})
    # Post process
    lines = np.squeeze(result[0])
    score = np.squeeze(result[1])
    lines = lines[score > 0.4]
    lines = lines * 4
    lines[:, :, 0] = lines[:, :, 0] * org_H / H
    lines[:, :, 1] = lines[:, :, 1] * org_W / W

    for i in range(lines.shape[0]):
        start_coor = (int(lines[i][0][1]), int(lines[i][0][0]))
        end_coor = (int(lines[i][1][1]), int(lines[i][1][0]))
        detected_lines.append([int(lines[i][0][1]), int(lines[i][0][0]), int(lines[i][1][1]), int(lines[i][1][0])])
        cv2.line(output_image, start_coor, end_coor, (0, 0, 255), 2, lineType=16)  # red
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_image, detected_lines

def lcnn_inference(lcnn_model, image, detected_lines, device):
    lines = infer_single_image(lcnn_model, device, image, thresholds=[0.99])
    output_image = copy.deepcopy(image)

    # No scaling needed since we didn't resize the image
    for i in range(len(lines)):
        # LCNN outputs coordinates in (y, x) format, so we need to reverse them
        
        pt1 = lines[i][0][::-1]  # [y1, x1] -> [x1, y1]
        pt2 = lines[i][1][::-1]  # [y2, x2] -> [x2, y2]

        x1, y1 = pt1
        x2, y2 = pt2

        # Convert to integer coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add to detected lines in format [x1, y1, x2, y2]
        detected_lines.append([(x1, y1), (x2, y2)])
        
        # Draw on output image
        #cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2, lineType=16)  # red
        cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2, lineType=16)  # red
    
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image, detected_lines
