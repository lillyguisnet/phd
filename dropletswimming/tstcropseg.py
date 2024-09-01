import os
import sys
sys.path.append("/home/maxime/prg/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
import re
from tqdm import tqdm
import pickle
from sam2.build_sam import build_sam2_video_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

###Get detailed segmentation on fframe coordinates###
with open('fframe_swimtst.pkl', 'rb') as file:
    ffvideo_segments = pickle.load(file)


def calculate_fixed_crop_window(video_segments, original_size, padding=10):
    orig_height, orig_width = original_size
    max_width, max_height = 0, 0
    centers = []

    for frame_num in sorted(video_segments.keys()):
        mask = next(iter(video_segments[frame_num].values()))
        y_coords, x_coords = np.where(mask[0])
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            left = x_coords.min()
            top = y_coords.min()
            right = x_coords.max()
            bottom = y_coords.max()
            
            width = right - left + 1 + 2 * padding
            height = bottom - top + 1 + 2 * padding
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            centers.append((center_x, center_y))
        else:
            # Fallback if no mask is found
            centers.append((orig_width // 2, orig_height // 2))

    # Ensure max_width and max_height are even (for potential video encoding)
    max_width = (max_width + 1) // 2 * 2
    max_height = (max_height + 1) // 2 * 2

    crop_windows = []
    for center_x, center_y in centers:
        left = max(0, center_x - max_width // 2)
        top = max(0, center_y - max_height // 2)
        right = min(orig_width, left + max_width)
        bottom = min(orig_height, top + max_height)
        
        # Adjust if crop window is out of bounds
        if right == orig_width:
            left = right - max_width
        if bottom == orig_height:
            top = bottom - max_height
        
        crop_windows.append((left, top, right, bottom))

    return crop_windows, (max_height, max_width)

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, padding=10):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate fixed crop windows
    crop_windows, (crop_height, crop_width) = calculate_fixed_crop_window(video_segments, original_size, padding)
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get crop window for this frame
        left, top, right, bottom = crop_windows[idx]
        
        # Crop the frame
        cropped_frame = frame[top:bottom, left:right]
        
        # Save the cropped frame
        cv2.imwrite(os.path.join(output_folder, frame_file), cropped_frame)
    
    return len(frame_files), (crop_height, crop_width)


# Usage
input_folder = '/home/maxime/prg/phd/dropletswimming/data_foranalysis/ngm/ngm-a-02182022162408-0000'
output_folder = '/home/maxime/prg/phd/dropletswimming/tstcrop'

# Read one frame to get original dimensions
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
original_size = first_frame.shape[:2]

num_frames, crop_size = process_frames_fixed_crop(input_folder, output_folder, ffvideo_segments, original_size, padding=5)

print(f"Processed {num_frames} frames.")
print(f"Fixed crop size: {crop_size[1]}x{crop_size[0]}")
print(f"Cropped frames saved in {output_folder}")

#Predict
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


video_dir = "/home/maxime/prg/phd/dropletswimming/tstcrop"

# scan all the png frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# take a look at frame to prompt
frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
image = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.savefig("tst.png")
plt.close()

#predictor.reset_state(inference_state) #if made previous inference
inference_state = predictor.init_state(video_path=video_dir)

#Add click on the first frame
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

points = np.array([[42, 50]], dtype=np.float32) #tight crop

# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


#Remove error
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

points = np.array([[41, 50], [50, 41]], dtype=np.float32)
labels = np.array([1, 0], np.int32)

#points = np.array([[50, 41]], dtype=np.float32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)


plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
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



f572 = video_segments[572][1][0]
image_array = np.uint8(f572 * 255)
image = Image.fromarray(image_array)
image.save('f572.png')




with open('tstswimcrop.pkl', 'wb') as file:
    pickle.dump(video_segments, file)

