# python scene_cluster_simple.py --input videos/Friends.S10E17-18.The.Last.One.720p.BluRay.2CH.x265.HEVC-PSA.mkv -o outputs/Friendss10e17_4_3

import os
import cv2
import glob
import shutil
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import re

import torch
from ultralytics import YOLO
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import rbd
from scene_change_cutter import detect_and_cut_scenes
from image_util import draw_keypoints


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Hyperparameters (simple variant - no block-based matching)."""
    IMG_SIZE = 512
    MAX_NUM_KEYPOINTS = 4096 * 2
    PEOPLE_MASK_THRESH = 0.6
    MATCH_THRESHOLD = 100  # min matches to group frames
    
    HOMOGRAPHY_MIN_INLIERS = 80
    HOMOGRAPHY_INLIER_RATIO = 0.90
    HOMOGRAPHY_MAX_DET_RATIO = 5.0
    HOMOGRAPHY_REPROJ_THRESH = 4.0
    HOMOGRAPHY_MAX_XY_TRANSLATIONS = 0.2
    HOMOGRAPHY_MAX_SCALE_DIFF = 0.2
    
    YOLO_WEIGHTS = "yolo26n-seg.pt"
    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
    MIN_CLIP_DURATION_SEC = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """SuperPoint extraction + people mask filtering."""
    
    def __init__(self, device: torch.device, max_kpts: int = Config.MAX_NUM_KEYPOINTS):
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(device)
    
    def extract(self, gray_tensor: torch.Tensor) -> dict:
        """Extract keypoints and descriptors."""
        with torch.no_grad():
            return self.extractor.extract(gray_tensor)
    
    def filter_by_mask(self, feats: dict, mask: np.ndarray) -> dict:
        """Remove keypoints on people regions."""
        kpts = feats["keypoints"][0]
        kpts_np = kpts.cpu().numpy()
        h, w = mask.shape[:2]
        
        xs = np.clip(np.round(kpts_np[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(kpts_np[:, 1]).astype(int), 0, h - 1)
        keep = ~mask[ys, xs]
        keep_t = torch.from_numpy(keep).to(self.device)
        
        filtered = {}
        for key, val in feats.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 2 \
                    and val.shape[1] == kpts.shape[0]:
                filtered[key] = val[:, keep_t]
            else:
                filtered[key] = val
        return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Matching
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureMatcher:
    """LightGlue matcher."""
    
    def __init__(self, device: torch.device):
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        # self.matcher.compile(mode="reduce-overhead")
    
    def match(self, feats0: dict, feats1: dict) -> np.ndarray:
        """Match two feature sets. Return Nx2 array of match indices."""
        with torch.no_grad():
            raw = self.matcher({"image0": feats0, "image1": feats1})
        _, _, mn = [rbd(x) for x in [feats0, feats1, raw]]
        return mn["matches"].cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Homography Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_homography(kpts0: np.ndarray, kpts1: np.ndarray, matches: np.ndarray, img_size: int,
                       min_inliers: int = Config.HOMOGRAPHY_MIN_INLIERS,
                       min_ratio: float = Config.HOMOGRAPHY_INLIER_RATIO,
                       max_det: float = Config.HOMOGRAPHY_MAX_DET_RATIO,
                       reproj_thr: float = Config.HOMOGRAPHY_REPROJ_THRESH,
                       max_xy_translations: float = Config.HOMOGRAPHY_MAX_XY_TRANSLATIONS,
                       max_scale_diff: float = Config.HOMOGRAPHY_MAX_SCALE_DIFF) -> bool:
    """Validate geometric consistency of matched points using homography."""
    n_in = 0
    det = 0
    in_ratio = 0
    xy_translations = 1
    scale_diff = 0
    res = False
    if len(matches) < 4:
        return res, det, in_ratio, xy_translations, scale_diff

    
    src = kpts0[matches[:, 0]].astype(np.float64)
    dst = kpts1[matches[:, 1]].astype(np.float64)
    
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC,
                                  reproj_thr, maxIters=2000, confidence=0.999)
    
    if H is None or mask is None:
        return res, det, in_ratio, xy_translations, scale_diff
    
    n_in = int(mask.flatten().astype(bool).sum())
    in_ratio = n_in/len(matches)
    if n_in < min_inliers or in_ratio < min_ratio:
        return res, det, in_ratio, xy_translations, scale_diff
    
    det = np.linalg.det(H)
    if det == 0 :#or abs(det) > max_det or abs(det) < (1.0 / max_det):
        return res, det, in_ratio, xy_translations, scale_diff
    
    # normalize translations
    H = H/H[2,2]
    dx = H[0,2]/img_size
    dy = H[1,2]/img_size
    xy_translations = max(abs(dx), abs(dy))

    scale_x = H[0,0]
    scale_y = H[1,1]

    scale_diff_x = (scale_x-1) if (scale_x > 1) else (1-scale_x)
    scale_diff_y = (scale_y-1) if (scale_y > 1) else (1-scale_y)
    scale_diff = max(scale_diff_x, scale_diff_y)

    if xy_translations <= max_xy_translations and scale_diff < max_scale_diff:
        res= True
    
    return res, det, in_ratio, xy_translations, scale_diff


# ═══════════════════════════════════════════════════════════════════════════════
# Union-Find for Grouping
# ═══════════════════════════════════════════════════════════════════════════════

class UnionFind:
    """Efficiently group similar frames."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, a: int, b: int) -> bool:
        """Merge sets. Return True if merged, False if already connected."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True
    
    def get_groups(self, n: int) -> dict:
        """Return {root: [indices]} grouping."""
        groups = {}
        for idx in range(n):
            groups.setdefault(self.find(idx), []).append(idx)
        return groups


# ═══════════════════════════════════════════════════════════════════════════════
# Simple Frame Grouper 
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleFrameGrouper:
    """Group frames by simple match count + optional homography validation."""
    
    def __init__(self,
                 img_size: int = Config.IMG_SIZE,
                 match_thresh: int = Config.MATCH_THRESHOLD,
                 use_homography: bool = True,
                 filter_mask: bool = True,
                 dbg_dir: str = "staging_dir/debug/scene_cluster"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.match_thresh = match_thresh
        self.use_homography = use_homography
        self.filter_mask = filter_mask
        self.dbg_dir = dbg_dir
        os.makedirs(f"{dbg_dir}", exist_ok=True)

        
        self.yolo_model = YOLO(Config.YOLO_WEIGHTS)
        self.feature_extractor = FeatureExtractor(self.device)
        self.feature_matcher = FeatureMatcher(self.device)
    
    def extract_features(self, image_paths: list) -> tuple:
        """Extract features from all images. Return (features_list, names)."""
        all_features = []
        all_names = [os.path.basename(p) for p in image_paths]
        
        for path in tqdm(image_paths, desc="Extracting features"):
            # Load image as BGR
            bgr = cv2.imread(path)
            
            # Convert BGR → grayscale → normalized tensor
            resized = cv2.resize(bgr, (self.img_size, self.img_size))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = torch.tensor(gray[None, None] / 255.0, dtype=torch.float32).to(self.device)
            
            # Extract people mask using YOLO
            results = self.yolo_model.predict(source=bgr, classes=[0], verbose=False)
            people_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            
            for result in results:
                if result.masks is not None:
                    for seg_mask in result.masks.xy:
                        contour = seg_mask.astype(np.int32).reshape(-1, 1, 2)
                        cv2.drawContours(people_mask, [contour], -1, 1, cv2.FILLED)
            
            people_mask = people_mask.astype(bool)
            if people_mask.shape[:2] != (self.img_size, self.img_size):
                people_mask = cv2.resize(people_mask.astype(np.uint8), (self.img_size, self.img_size),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            
            feats = self.feature_extractor.extract(gray)
            if self.filter_mask:
                feats = self.feature_extractor.filter_by_mask(feats, people_mask)
            
            all_features.append(feats)
        
        return all_features, all_names
    
    def match_and_group(self, all_features: list, image_paths: list) -> UnionFind:
        """Match all pairs and build groups. Return UnionFind object."""
        n = len(all_features)
        uf = UnionFind(n)
        
        for i, j in tqdm(list(combinations(range(n), 2)), desc="Matching pairs"):
            if uf.find(i) == uf.find(j):
                continue
            
            # Skip if either feature set has no keypoints
            kpts_i = all_features[i]["keypoints"][0]
            kpts_j = all_features[j]["keypoints"][0]
            if kpts_i.shape[0] == 0 or kpts_j.shape[0] == 0:
                continue

            # Match
            matches = self.feature_matcher.match(all_features[i], all_features[j])
            if len(matches) < self.match_thresh:
                continue
            
            # Validate homography if enabled
            if self.use_homography:
                kpts_i = kpts_i.cpu().numpy()
                kpts_j = kpts_j.cpu().numpy()
                
                res, det, in_ratio, xy_translations, scale_diff =  validate_homography(kpts_i, kpts_j, matches, self.img_size)
                if True: #debug
                    img0 = cv2.imread(image_paths[i])
                    img0 = cv2.resize(img0, (self.img_size, self.img_size))
                    img1 = cv2.imread(image_paths[j])
                    img1 = cv2.resize(img1, (self.img_size, self.img_size))


                    feats0 = all_features[i]
                    feats1 = all_features[j]
                    feats0, feats1, = [ rbd(x) for x in [feats0, feats1] ]  
                    kpts0, kpts1 = (feats0["keypoints"], feats1["keypoints"])
                    img0[20:70] = (255,255,255)
                    img0 = cv2.putText(img0, f'{res}, {xy_translations:.2f}, {scale_diff:.2f}, {det:.2f}, {in_ratio:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,255), 2)
                    img0 = draw_keypoints(img0, kpts0, radius=1, color=(0, 255, 0))
                    img1 = draw_keypoints(img1, kpts1, radius=1, color=(0, 255, 0))
    
                    m_kpts0_t, m_kpts1_t = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
                    m_kpts0 = m_kpts0_t.detach().cpu().numpy().astype(int)
                    m_kpts1 = m_kpts1_t.detach().cpu().numpy().astype(int)
    
                    axes = viz2d.plot_images([img0, img1], dpi=200)
                    viz2d.plot_matches(m_kpts0, m_kpts1, lw=1, a=0.5)
                    image_name_i = os.path.split(image_paths[i])[1]
                    image_name_j = os.path.split(image_paths[j])[1]
                    image_name_i  = os.path.splitext(image_name_i)[0]
                    image_name_j  = os.path.splitext(image_name_j)[0]
                    plt.savefig(f"{self.dbg_dir}/kpts_match_{image_name_i}__to__{image_name_j}.png")
                    plt.close()

                if not res:
                    continue
            
            # Group them
            uf.union(i, j)
        
        return uf
    
    def write_groups(self, image_paths: list, all_names: list, groups: dict, output_folder: str, group_id: int):
        """Write grouped images to disk."""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        #group_id = 0
        for root, members in sorted(groups.items(), key=lambda x: -len(x[1])):
            gdir = Path(output_folder) / f"group_{group_id:03d}"
            gdir.mkdir(parents=True, exist_ok=True)
            
            for idx in members:
                shutil.copy2(image_paths[idx], gdir / all_names[idx])
            
            group_id += 1
        
        print(f"[DONE] {group_id} group(s) written to '{output_folder}'.")
        return group_id
    
    def run(self, frames_folder: str, output_folder: str):
        image_paths = sorted(p for ext in Config.IMAGE_EXTS
                            for p in glob.glob(os.path.join(frames_folder, f"*{ext}")))
        all_names = [os.path.basename(p) for p in image_paths]

        total_clips = len(image_paths)

        BATCH_SZ = 100
        group_id = 0
        for i in range(0, total_clips+BATCH_SZ-1, BATCH_SZ):
            image_paths_samples = image_paths[i:i+BATCH_SZ]
            all_names_samples = all_names[i:i+BATCH_SZ]
            all_features, _ = self.extract_features(image_paths_samples)
            uf = self.match_and_group(all_features, image_paths_samples)
            #groups = uf.get_groups(len(image_paths))
            groups = uf.get_groups(len(image_paths_samples))

            group_id = self.write_groups(image_paths_samples, all_names_samples, groups, output_folder, group_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Frame Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class FrameExtractor:
    """Extract middle frame from video clips."""
    
    @staticmethod
    def extract(video_folder: str, output_folder: str, file_ext: str) -> None:
        """Extract middle frame from all .mkv files."""
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        mkv_files = [f for f in os.listdir(video_folder)
                     if f.lower().endswith(file_ext)]
        
        for file in tqdm(sorted(mkv_files), desc="Extracting frames"):
            video_path = os.path.join(video_folder, file)
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if duration < Config.MIN_CLIP_DURATION_SEC:
                cap.release()
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                stem = os.path.splitext(file)[0]
                out_path = os.path.join(output_folder, f"{stem}_middle.jpg")
                cv2.imwrite(out_path, frame)


# ═══════════════════════════════════════════════════════════════════════════════
# Video Merging
# ═══════════════════════════════════════════════════════════════════════════════

class GroupVideoMerger:
    """Merge clip videos in each group."""
    
    @staticmethod
    def merge(groups_root: str, video_folder: str, file_ext: str, overwrite: bool = False, dbg: str = False):
        """Merge videos for each group folder."""
        if not shutil.which("ffmpeg"):
            return
        
        groups_path = Path(groups_root)
        video_path = Path(video_folder)
        
        if not groups_path.is_dir() or not video_path.is_dir():
            return
        
        group_dirs = sorted(d for d in groups_path.iterdir()
                           if d.is_dir() and d.name.startswith("group_"))
        groups_dict = {}
        
        for i, gdir in enumerate(group_dirs):
            images = sorted(f for f in gdir.iterdir()
                           if f.is_file() and f.suffix.lower() in Config.IMAGE_EXTS)
            
            video_paths = []
            for img in images:
                stem = img.stem[:-len("_middle")] if img.stem.endswith("_middle") else img.stem
                vid = video_path / f"{stem}{file_ext}"
                if vid.exists():
                    video_paths.append(vid)
            
            
            if dbg:
                if len(video_paths) >= 2:
                    out_path = gdir / f"{gdir.name}{file_ext}"
                    if out_path.exists() and not overwrite:
                        continue
            
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                        for vp in video_paths:
                            f.write(f"file '{vp.resolve().as_posix()}'\n")
                        list_path = f.name
            
                    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", str(out_path)]
                    print(f"Running command {command}")
                    subprocess.run(command, capture_output=True)
                    os.unlink(list_path)

            groups_dict[i] = video_paths
        return groups_dict



# ═══════════════════════════════════════════════════════════════════════════════
# Video Splitting
# ═══════════════════════════════════════════════════════════════════════════════

def split_video(input_video: str, output_folder: str,
                threshold: float = 0.2, clip_prefix: str = "clip") -> list:
    """Split video by scene changes."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    clip_format = Path(input_video).suffix.lstrip(".") or "mkv"
    
    return detect_and_cut_scenes(
        video_path=input_video,
        output_dir=output_folder,
        threshold=threshold,
        clip_prefix=clip_prefix,
        clip_format=clip_format,
        reencode=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(input_video: str, output_folder: str, run_split_video: bool,
                      threshold: float = 0.2,
                      clip_prefix: str = "clip",
                      img_size: int = Config.IMG_SIZE,
                      match_thresh: int = Config.MATCH_THRESHOLD,
                      use_homography: bool = True,
                      filter_mask: bool = True):
    """End-to-end pipeline: split → extract → group → merge."""
    out = Path(output_folder)
    frames = str(out / "scene_cluster/middle_frames")
    groups = str(out / "scene_cluster/grouped_frames")
    
    print("\n" + "═" * 64)
    print("STEP 1 / 4  —  Split video into clips")
    print("═" * 64)
    filename, file_ext = os.path.splitext(input_video)
    if run_split_video:
        split_video(input_video, output_folder, threshold=threshold, clip_prefix=clip_prefix)
    
    print("\n" + "═" * 64)
    print("STEP 2 / 4  —  Extract middle frames")
    print("═" * 64)
    FrameExtractor.extract(output_folder, frames, file_ext)
    
    print("\n" + "═" * 64)
    print("STEP 3 / 4  —  Group similar frames")
    print("═" * 64)
    grouper = SimpleFrameGrouper(
        img_size=img_size,
        match_thresh=match_thresh,
        use_homography=use_homography,
        filter_mask=filter_mask,
    )
    grouper.run(frames, groups)
    
    print("\n" + "═" * 64)
    print("STEP 4 / 4  —  Merge group videos")
    print("═" * 64)
    group_dict = GroupVideoMerger.merge(groups, output_folder, file_ext)

    # remove the path and retain only the clip name
    group_dict_new = {}
    for group_id, grouped_clip_paths in group_dict.items():
        clip_names = []
        for clip_path in grouped_clip_paths:
            clip_path = str(clip_path)
            clip_name = re.sub(r"staging_dir\/", f"", clip_path)
            clip_names.append(clip_name)
        group_dict_new[group_id] = clip_names


    return group_dict_new



def main():
    parser = argparse.ArgumentParser(
        prog="scene_cluster_simple.py",
        description="Simple grouping (no block-based): split → extract → group → merge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, help="Source video file")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Scene-change threshold (default 0.2)")
    parser.add_argument("--prefix", default="clip", help="Clip filename prefix")
    parser.add_argument("--img-size", type=int, default=Config.IMG_SIZE)
    parser.add_argument("--match-threshold", type=int, default=Config.MATCH_THRESHOLD,
                        help="Minimum matches to group frames (default: 50)")
    parser.add_argument("--no-homography", action="store_true", help="Skip homography validation")
    parser.add_argument("--no-mask-filter", action="store_true", help="Skip people mask filtering")
    
    args = parser.parse_args()
    
    group_dict = run_full_pipeline(input_video=args.input,
                                   output_folder=args.output,
                                   run_split_video=True,
                                   threshold=args.threshold,
                                   clip_prefix=args.prefix,
                                   img_size=args.img_size,
                                   match_thresh=args.match_threshold,
                                   use_homography=not args.no_homography,
                                   filter_mask=not args.no_mask_filter,)
    print(group_dict)


if __name__ == "__main__":
    main()
