import os
import sys
from PIL import Image
import cv2
import torch
from pathlib import Path
import re
from tqdm import tqdm
import pickle
import numpy as np
from scipy import interpolate, signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import center_of_mass
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.signal import stft
from scipy.signal import welch
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean
from skimage import morphology, graph
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
sys.path.append("/home/maxime/prg/phd/segment-anything-2")
sys.path.append("/home/maxime/prg/phd/crawlingtracking")
from sam2.build_sam import build_sam2_video_predictor
import trackingfunctions as tfx


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000"

# scan all the png frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))