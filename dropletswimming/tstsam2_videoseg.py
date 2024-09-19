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

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

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


video_dir = "/home/maxime/prg/phd/dropletswimming/data_foranalysis/ngm/ngm-a-02212022161253-0000"

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
plt.savefig("tst.png")
plt.close()

#predictor.reset_state(inference_state) #if made previous inference
inference_state = predictor.init_state(video_path=video_dir)

###Add click on the first frame
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

points = np.array([[1462, 1449]], dtype=np.float32) #full frame

labels = np.array([1], np.int32)
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

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


""" video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    for i, out_obj_id in enumerate(out_obj_ids)
} """
 

empty_frames = []
for frame, obj_dict in video_segments.items():
    if all(not mask.any() for mask in obj_dict.values()):
        empty_frames.append(frame)
if empty_frames:
    print(f"!!! Empty frames: {empty_frames}")



with open('fframe_swimtst.pkl', 'wb') as file:
    pickle.dump(video_segments, file)



###Make a video with results
def overlay_mask_on_image(image_path, mask, color=(0, 255, 0), alpha=0.5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #print(f"Image shape: {image.shape}")
    #print(f"Mask type: {type(mask)}")
    #print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'N/A'}")
    
    # Convert mask to binary numpy array if it's not already
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask.dtype != bool:
        mask = mask > 0.5
    
    # Ensure mask is 2D
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # Resize the mask to match the image dimensions
    mask_resized = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask_resized == 1] = color
    
    # Overlay the mask on the image
    overlaid_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlaid_image


# Prepare the video writer
output_video_path = "cropreversetst.mp4"
frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
#frame = cv2.imread(os.path.join(video_dir, "000000.jpg"))
if frame is None:
    raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

# Process each frame
for frame_idx in range(len(frame_names)):
    print(frame_idx)
    image_path = os.path.join(video_dir, frame_names[frame_idx])
    
    try:
        if frame_idx in video_segments:
            # Assuming we're only tracking one object (object ID 1)
            mask = video_segments[frame_idx][1]
            overlaid_frame = overlay_mask_on_image(image_path, mask)
        else:
            # If no segmentation for this frame, use the original image
            overlaid_frame = cv2.imread(image_path)
            if overlaid_frame is None:
                raise ValueError(f"Could not read image from {image_path}")
            overlaid_frame = cv2.cvtColor(overlaid_frame, cv2.COLOR_BGR2RGB)
        
        # Write the frame to the video
        out.write(cv2.cvtColor(overlaid_frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {str(e)}")
        # If there's an error, write the original frame
        original_frame = cv2.imread(image_path)
        if original_frame is not None:
            out.write(original_frame)
        else:
            print(f"Could not read original frame {frame_idx}")

# Release the video writer
out.release()

print(f"Video saved to {output_video_path}")