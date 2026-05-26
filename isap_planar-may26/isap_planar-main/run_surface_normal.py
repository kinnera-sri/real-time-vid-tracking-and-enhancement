import os
import sys
import glob
import numpy as np
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from types import SimpleNamespace
import sys

sys.path.append("models/third_party/DSINE")

import logging
logger = logging.getLogger('root')



def intrins_from_fov(new_fov, H, W, dtype=torch.float32, device='cpu'):
    """ define intrins based on field-of-view
        principal point is assumed to be at the center

        NOTE: new_fov should be in degrees
        NOTE: top-left is (0,0)
    """
    new_fx = new_fy = (max(H, W) / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    new_cx = (W / 2.0) - 0.5
    new_cy = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [new_fx,    0,          new_cx  ],
        [0,         new_fy,     new_cy  ],
        [0,         0,          1       ]
    ], dtype=dtype, device=device)

    return new_intrins


def load_checkpoint(fpath, model):
    assert os.path.exists(fpath)
    logger.info('loading checkpoint... %s' % fpath)
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    logger.info('loading checkpoint... / done')
    return model

def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b


config_dict = {
        'NNET_architecture':'v02',
        'NNET_output_dim': 3,
        'NNET_output_type': 'R',
        'NNET_feature_dim': 64,
        'NNET_hidden_dim': 64,
        'NNET_encoder_B': 5,
        'NNET_decoder_NF': 2048,
        'NNET_decoder_BN': False,
        'NNET_decoder_down': 8,
        'NNET_learned_upsampling': True,
        'NRN_prop_ps': 5,
        'NRN_num_iter_train': 5,
        'NRN_num_iter_test': 5,
        'NRN_ray_relu': True,
        'exp_name': "exp001_cvpr2024",
        'exp_id': "dsine",
        }
args0 = argparse.Namespace(**config_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tracker test")
    parser.add_argument('--imgs', type=str, required=True)
    parser.add_argument('--op_dir', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default="assets/checkpoints/dsine.pt")
    args = parser.parse_args()

    args = SimpleNamespace(**args0.__dict__, **args.__dict__)
    

    device = torch.device('cuda')
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    model = DSINE(args).to(device)
    model = load_checkpoint(args.ckpt_path, model)
    model.eval()

    img_paths = glob.glob(f'{args.imgs}/*.png') + glob.glob(f'{args.imgs}/*.jpg')
    img_paths.sort()
    os.makedirs(args.op_dir, exist_ok=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        for img_path in img_paths:
            print(img_path)
            _, filename = os.path.split(img_path)
            ext = os.path.splitext(img_path)[1]
            img = Image.open(img_path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            # pad input
            _, _, orig_H, orig_W = img.shape
            lrtb = get_padding(orig_H, orig_W)
            img = F.pad(img, lrtb, mode="constant", value=0.0)
            img = normalize(img)

            # get intrinsics
            intrins_path = img_path.replace(ext, '.txt')
            #if os.path.exists(intrins_path):
            #    # NOTE: camera intrinsics should be given as a txt file
            #    # it should contain the values of fx, fy, cx, cy
            #    intrins = intrins_from_txt(intrins_path, device=device).unsqueeze(0)
            #else:
            #    # NOTE: if intrins is not given, we just assume that the principal point is at the center
            #    # and that the field-of-view is 60 degrees (feel free to modify this assumption)
            #    intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
            intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
            intrins[:, 0, 2] += lrtb[0]
            intrins[:, 1, 2] += lrtb[2]

            pred_norm = model(img, intrins=intrins)[-1]
            pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]

            # save to output folder
            # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
            # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
            #target_path = filename.replace(ext, '_sn.png')
            target_path = filename.replace(ext, '.png')
            target_path = os.path.join(args.op_dir, target_path)

            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
            pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
            im = Image.fromarray(pred_norm[0,...])
            im.save(target_path)
