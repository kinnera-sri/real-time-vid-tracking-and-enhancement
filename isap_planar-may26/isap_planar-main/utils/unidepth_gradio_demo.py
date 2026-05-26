import gradio as gr
import numpy as np
import trimesh
import torch
from PIL import Image
import cv2
import os

import sys
sys.path.append("models/third_party/UniDepth")
from unidepth.models import UniDepthV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
da_model = unidepth_model.to(device).eval()
workspace_dir = "debug"

if not os.path.exists(workspace_dir):
    os.makedirs(workspace_dir)



def model_inference_depth(image):
    img_t = torch.from_numpy(image).permute(2, 0, 1).to(device)
    with torch.no_grad():
        depth_predictions = da_model.infer(img_t)
    depth_img = depth_predictions["depth"].squeeze().cpu().numpy()
    depth_img_vis  = 2*(depth_img - np.min(depth_img))/(np.max(depth_img) - np.min(depth_img))
    #inv_depth_img_vis = 1 - depth_img_vis
    depth_img_vis  = depth_img_vis -1
    #np.savez_compressed(f"{workspace_dir}/results.npz", depth_predictions)
    return depth_img_vis

def model_inference_pt3d(image):
    img_t = torch.from_numpy(image).permute(2, 0, 1).to(device)
    with torch.no_grad():
        depth_predictions = da_model.infer(img_t)
    pts_3d = depth_predictions["points"].squeeze().permute(1, 2, 0).cpu().numpy()
    pts_3d = pts_3d.reshape(-1, 3)
    
    colors = image.reshape(-1, 3)
    pcl = trimesh.PointCloud(vertices=pts_3d, colors=colors)
    pcl.vertices[:, 1] *= -1 
    pcl.vertices[:, 2] *= -1 

    if 0:
        output_path = f"{workspace_dir}/pts3d.ply"
        pcl.export(output_path)
    else:
        # convert to glb
        scene = trimesh.Scene()
        #if scene.metadata is None:
        #    scene.metadata = {}
        scene.add_geometry(pcl)
        output_path = f"{workspace_dir}/scene.glb"
        scene.export(output_path)
    return output_path

demo_depth = gr.Interface(
    fn=model_inference_depth,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Image(label="Displayed Image"),
    title="Image Loader & Viewer",
    description="Upload an image to see it displayed."
)

demo_pcl = gr.Interface(
    fn=model_inference_pt3d,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=gr.Model3D(label="Generated Point Cloud", display_mode="point_cloud"),
    title="Image Loader & Viewer",
    description="Upload an image to see it displayed."
)



if __name__ == "__main__":
    #demo_depth.launch()
    demo_pcl.launch()
