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


###Make prediction on fframe video###

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000"

# scan all the png frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
image = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.savefig("firstframe.png")
plt.close()

inference_state = predictor.init_state(video_path=video_dir)

###Add click on the first frame
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[1317, 1481]], dtype=np.float32) #full frame
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
tfx.show_points(points, labels, plt.gca())
tfx.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig("tstclick.png")
plt.close()

###Propagate to video
# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

#Save propagation results
with open('propagation_fullframe.pkl', 'wb') as file:
    pickle.dump(video_segments, file)


###Get detailed segmentation on fframe coordinates###
with open('propagation_fullframe.pkl', 'rb') as file:
    ffvideo_segments = pickle.load(file)

def get_hdsegmentation(ffvideo_segments, crop_size=94):
    hd_video_segments = {}
    for frame_num in sorted(ffvideo_segments.keys()):
        print(frame_num)
        #predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        or_mask = next(iter(ffvideo_segments[frame_num].values()))
        # Find the bounding box of the segment
        rows, cols = np.where(or_mask[0])
        center_y, center_x = rows.mean(), cols.mean()
        # Calculate the crop boundaries
        top = max(0, int(center_y - crop_size // 2))
        bottom = min(or_mask.shape[1], top + crop_size)
        left = max(0, int(center_x - crop_size // 2))
        right = min(or_mask.shape[2], left + crop_size)
        # Crop the original frame
        or_frame = cv2.imread(os.path.join("/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000", f"{frame_num:06}.jpg"))
        cropped_arr = or_frame[top:bottom, left:right]
        # Save the cropped frame to prediction folder
        cv2.imwrite("/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/cropprompt/000001.jpg", cropped_arr)

        # Make prediction on the cropped frame
        cropped_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/cropprompt"
        frame_names = [
            p for p in os.listdir(cropped_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        if frame_num == 0:
            inference_state = predictor.init_state(video_path=cropped_dir)
        else:
            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(video_path=cropped_dir)
        #Add click on the first frame
        ann_frame_idx = 0 #frame
        ann_obj_id = 1 #object
        points = np.array([[64, 45]], dtype=np.float32)
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        #Propagate to 'video' and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        cropped_hd_segment = video_segments[1][1]

        # Resize the cropped segment to the original size
        or_shape = or_frame.shape[:2]
        full_hd_segment = np.zeros(or_shape, dtype=bool)
        full_hd_segment[top:bottom, left:right] = cropped_hd_segment

        reshaped_hdseg = np.expand_dims(full_hd_segment, axis=0)
        hd_video_segments[frame_num] = {1: reshaped_hdseg}


    return hd_video_segments

#Check stuff
predimg = hd_video_segments[500][1][0].astype(int)
cv2.imwrite("tst.jpg", predimg*200)

#Use
hd_video_segments = get_hdsegmentation(ffvideo_segments, crop_size=94)

with open('hd_video_segments.pkl', 'wb') as file:
    pickle.dump(hd_video_segments, file)