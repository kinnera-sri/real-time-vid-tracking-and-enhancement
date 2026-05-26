import torch
import torch.nn as nn
from torchsummary import summary
from .misc import pad_and_stack
import numpy as np
import matplotlib.pyplot as plt
import kornia
import torch.nn.functional as F

DXY_SCALE = 4.0


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x



class RobustPoint(torch.nn.Module):
    def __init__(self, prd_descr=True):
        super(RobustPoint, self).__init__()

        self.csz =  8         # Cell size 

        self.det_thresh = 0.65
        self.nms_radius = 4
        self.prd_descr = prd_descr
        self.remove_borders = 1
        self.max_num_keypoints = None
        self.force_num_keypoints = False

        csz = self.csz
        

        inc, c1, c2, c3, c4, desc_dim, det_dim = 1, 32, 64, 128, 256, 256, 1

        #self.block1 = ResBlock(inc,c1,1,nn.Conv2d(inc, c1, 1))
        #self.block2 = ResBlock(c1,c2,1,nn.Conv2d(c1, c2, 1))
        #self.block3 = ResBlock(c2,c3,1,nn.Conv2d(c2, c3, 1))
        #self.block4 = ResBlock(c3,c4,1,nn.Conv2d(c3, c4, 1))
        self.block1 = DoubleConv(inc,c1)
        self.block2 = DoubleConv(c1,c2)
        self.block3 = DoubleConv(c2,c3)
        self.block4 = DoubleConv(c3,c4)

        inch_ = c4
        self.det_head = nn.Sequential(nn.Conv2d(inch_, inch_//4, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(inch_//4),
                                      nn.ReLU(),
                                      nn.Conv2d(inch_//4, det_dim*8, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(det_dim*8),
                                      nn.ReLU(),
                                      nn.Conv2d(det_dim*8, det_dim, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(det_dim),
                                      )

        inch_ = c1 * csz**2 
        self.res_pos_head1 = nn.Sequential(nn.Conv2d(inch_, 2*csz, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(2*csz),
                                      nn.ReLU())

        inch_ = self.csz*2 + c4

        # predicts dx, dy values in the range +/-(csz/(2*DXY_SCALE))
        # the fnal dx, dy we will get by =>   dxy = csz/2 + pdx*DXY_SCALE
        self.res_pos_head2 = nn.Sequential(nn.Conv2d(inch_, inch_//4, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(inch_//4),
                                      nn.ReLU(),
                                      nn.Conv2d(inch_//4, 16, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.Conv2d(16, 2, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(2))

        inch_ = c4
        self.desc_head = nn.Sequential(nn.Conv2d(inch_, inch_, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(inch_),
                                       nn.ReLU(),
                                       nn.Conv2d(inch_, desc_dim, kernel_size=1, stride=1))


        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=3, stride=3)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):
        ######################################################################
        # feature extraction
        ######################################################################
        x1 = self.block1(x)              # [B, c1, H, W,]
        x2 = self.avg_pool2(x1)      # [B, c1, H/2, W/2]
        x2 = self.block2(x2)             # [B, c2, H/2, W/2]
        x3 = self.avg_pool2(x2)          # [B, c2, H/4, W/4] or [B, c2, H/6, W/6]
        x3 = self.block3(x3)             # [B, c3, H/4, W/4] or [B, c3, H/6, W/6]
        x4 = self.avg_pool2(x3)          # [B, c3, H/8, W/8] or [B, c3, H/12, W/12]
        x4 = self.block4(x4)             # [B, c4, H/8, W/8] or [B, c4, H/12, W/12]
        #print(f"x1:{x1.shape}, x2:{x2.shape}, x3:{x3.shape}, x4:{x4.shape}")

        ######################################################################
        # Keypoint detection
        # We will use x1 -> 1st level of features extracted at full scale
        #             x4 -> last level of features extracted at 1/8 scale
        # Instead of upsampling x4, we push the features from 8x8 grids of x1
        # to channel dimension to bring the x1 w,h on par with x4 and then
        # concat
        ######################################################################
        csz = self.csz
        n_,c_,h_,w_ = x1.shape

        #####################################################################
        # Detection of keypoints heatmap H/8, W/8 resolution
        #####################################################################
        # Concat x1 and x4 features
        det_feats = x4

        # detect keypoints 
        kpts_hm = self.det_head(det_feats)                         # [B, 1, h, w]

        #####################################################################
        # regress dx, xy positions
        #####################################################################
        # Concat x1 and x4 features
        n_,c_,h_,w_ = x1.shape
        x1_ = x1.reshape(n_, c_, h_//csz, csz, w_//csz, csz)
        x1_ = x1_.permute(0, 1, 3, 5, 2, 4)
        x1_ = x1_.reshape(n_, c_ * csz *csz, h_//csz, w_//csz)
        x1_pos_feats = self.res_pos_head1(x1_)                       # [B, 2*csz, h, w]
        pos_feats = torch.cat([x1_pos_feats, det_feats], dim=1)
        kpts_dxdy = self.res_pos_head2(pos_feats)                    # [B, 2, h, w]


        op = { "kpts_hm"   : kpts_hm,   # [B, 1, h, w]
               "kpts_dxdy" : kpts_dxdy  # [B, 2, h, w]
              }
        ######################################################################
        # Describe the keypoints 
        ######################################################################
        if self.prd_descr:
            desc_feats = x4
            desc = self.desc_head(desc_feats)          # [B, C, h, w]
            #desc = F.interpolate(desc, size=(h_//8, w_//8), mode='bilinear', align_corners=False)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
            op.update({"desc": desc})                   # [B, C, H/8, W/8]
        return op

    def infer(self, data):
        image = data["image"]
        if image.ndim ==3:
            image = image.unsqueeze(0)
        if image.shape[1] == 3:  # RGB
            image = kornia.color.rgb_to_grayscale(image)
        b, _, h, w = image.shape

        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        padding_tuple = (0, pad_w, 0, pad_h)
        image = F.pad(image, padding_tuple, mode='constant', value=0)

        #print(f"knr... Robustpoint:infer image:{image.shape}")
        csz = self.csz
        #with torch.no_grad():
        if True:
            op = self.forward(image)
            kpts_hm = op["kpts_hm"]                         # [B, 1, h, w]
            kpts_dxdy = op["kpts_dxdy"]                     # [B, 2, h, w]
            kpts_dx = kpts_dxdy[:,0,:,:]                    # [B, 1, h, w]
            kpts_dy = kpts_dxdy[:,1,:,:]                    # [B, 1, h, w]

            scores = torch.nn.functional.sigmoid(kpts_hm)
            #scores_ = scores[0][0].detach().cpu().numpy()
            #x_ = x[0][0].detach().cpu().numpy()
            #fig, axes = plt.subplots(nrows=2)
            #axes[0].imshow(scores_)
            #axes[1].imshow(x_)
            #plt.show()
            #scores = simple_nms(scores, self.nms_radius)
            b, _, h, w = scores.shape
            scores = scores.reshape(b, h, w)

            # Discard keypoints near the image borders
            if self.remove_borders:
                pad = self.remove_borders
                scores[:, :pad] = -1
                scores[:, :, :pad] = -1
                scores[:, -pad:] = -1
                scores[:, :, -pad:] = -1

            # Extract keypoints
            best_kp = torch.where(scores > self.det_thresh)

            clamp_val = csz/(2*DXY_SCALE)
            scores = scores[best_kp]
            dx = kpts_dx[best_kp]
            dx = torch.clamp(dx, min=-clamp_val, max=clamp_val)

            dy = kpts_dy[best_kp]
            dy = torch.clamp(dy, min=-clamp_val, max=clamp_val)
            #print(scores.shape, dx.shape, dy.shape)

            # Separate into batches
            keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i] for i in range(b)]
            scores = [scores[best_kp[0] == i] for i in range(b)]
            dx = [dx[best_kp[0] == i] for i in range(b)]
            dy = [dy[best_kp[0] == i] for i in range(b)]
            dyx = [torch.stack([dy[i], dx[i]], dim=-1) for i in range(b)]

            keypoints = [keypoints[i]*csz + csz//2 + DXY_SCALE*dyx[i] for i in range(b)]
            #print(keypoints[0].shape)


            # Keep the k keypoints with highest score
            if self.max_num_keypoints is not None: # and len(keypoints) > self.max_num_keypoints:
                keypoints, scores = list( zip(*[top_k_keypoints(k, s, self.max_num_keypoints)
                                                for k, s in zip(keypoints, scores)]
                                          ))

            # Convert (h, w) to (x, y)
            keypoints = [torch.flip(k, dims=[1]).float() for k in keypoints]

            if self.force_num_keypoints:
                max_kps = self.max_num_keypoints
                kpt_ubound = data.get("image_size", torch.tensor(image.shape[-2:])).min().item()
                if 0:
                    keypoints = pad_and_stack(keypoints, max_kps, -2, mode="random_c", bounds=(0, kpt_ubound))
                    scores = pad_and_stack(scores, max_kps, -1, mode="random_c", bounds = (-100, -1))
                else:
                    keypoints = pad_and_stack(keypoints, max_kps, -2, mode="random", bounds=(0, kpt_ubound))
                    scores = pad_and_stack(scores, max_kps, -1, mode="random", bounds = (0.1, 0.4))
            else:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)


            res = {"keypoints"  : keypoints,
                   "kpts_hm"    : kpts_hm,
                   "scores"     : scores}
            if self.prd_descr:
                # Extract descriptors
                desc = [ self.sample_descriptors(k[None], d[None], csz)[0]
                        for k, d in zip(keypoints, op["desc"])
                        ]
                if self.force_num_keypoints or b==1:
                    # We can stack descriptors only if all are of same size
                    desc = torch.stack(desc, 0).transpose(-1, -2)
                res.update({"descriptors": desc, "desc_img": op["desc"]})
        return res

    def sample_descriptors(self, keypoints, desc, s=8):
        b, c, h, w = desc.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],).to( keypoints)[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
        desc = torch.nn.functional.grid_sample(
                desc, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
        )
        desc = torch.nn.functional.normalize(desc.reshape(b, c, -1), p=2, dim=1)
        return desc




def test_simple():
    model = RobustPoint(prd_descr=True)
    model.max_num_keypoints = 128
    model.force_num_keypoints = True

    x = torch.rand(1, 1, 640, 480)

    
    data = {"image":x, "image_size": torch.tensor(x.shape)}

    op = model.infer(data)

    print(op.keys())
    print(op["keypoints"].shape)
    print(op["scores"].shape)
    print(op["descriptors"].shape)
    

    exit()
    
    summary(model, input_size=[(1, 640, 480)], device="cpu")
    #print(op["pts_map"].shape, op["pts_img"].shape, op["desc"].shape)


def test():
    import cv2
    import matplotlib.pyplot as plt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread("assets/01.jpg", 0)
    img = cv2.resize(img, (640,480))
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    img_t = img_t.float()/255.
    


    model = RobustPoint(prd_descr=True).to(device)
    #model.load_state_dict(torch.load("ckpts/robustpoint.pth", map_location=device))
    op = model(img_t)
    print(op.keys())
    print(op["desc"].shape, op["kpts_hm"].shape)
    exit()

    prd = model.infer(img_t)
    print(prd.keys())
    pnts = prd["points"][0].transpose()
    print(pnts.shape)

    plt.imshow(img)
    plt.scatter(pnts[:,0], pnts[:,1], c='r', s=1)
    plt.show()

    #print(prd['points'][0].shape, prd['desc'][0].shape, prd['conf'][0].shape)
    print(prd['points'][0].shape, prd['desc'][0].shape)
    


    

if __name__ == "__main__":
    #test()
    test_simple()
