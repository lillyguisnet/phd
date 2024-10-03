import os
import sys
sys.path.append("/home/maxime/prg/phd/segment-anything-2")

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import shutil
from pathlib import Path

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


def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


#video_dir = '/home/maxime/prg/phd/ria/tstvideo/ria-MMH99_10s_20190813_03'
video_dir = '/home/maxime/prg/phd/ria/tstvideo/vidcrop'

#output_dir = Path("/home/maxime/prg/phd/dropletswimming/data_foranalysis/visc05")

# scan all the jpg frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=video_dir)


prompts = {}

ann_frame_idx = 380  #frame index
ann_obj_id = 2  #object id
#points = np.array([[277, 307]], dtype=np.float32) #full frame
points = np.array([[67, 39]], dtype=np.float32) #cropped frame nrd only
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
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
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()




ann_frame_idx = 380  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)
# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[47, 58]], dtype=np.float32) #cropped frame nrv only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()



ann_frame_idx = 380  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)
points = np.array([[43, 72], [41, 73], [43, 70], [46, 63], [55, 48], [63, 41]], dtype=np.float32) #cropped frame loop only
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
prompts[ann_obj_id] = points, labels
# `add_new_points` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()






#Propagate to 'video' in reverse and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    for i, out_obj_id in enumerate(out_obj_ids)
}
    

empty_frames = []
low_detection_frames = []
high_detection_frames = []
for frame, obj_dict in video_segments.items():
    if all(not mask.any() for mask in obj_dict.values()):
        empty_frames.append(frame)
    elif sum(mask.sum() for mask in obj_dict.values()) <= 200:
        low_detection_frames.append(frame)    
    elif sum(mask.sum() for mask in obj_dict.values()) >= 5000:
        high_detection_frames.append(frame)   
if empty_frames:
    print(f"!!! Empty frames: {empty_frames}")
else:
    print("Yay! No empty frames found, yay!")
if low_detection_frames:
    print(f"!!! Frames with 200 or fewer true elements: {low_detection_frames}")
else:
    print("Yay! No frames with 200 or fewer true elements found, yay!")
if high_detection_frames:
    print(f"!!! Frames with 5000 or more true elements: {high_detection_frames}")
else:
    print("Yay! No frames with 5000 or more true elements found, yay!")





video_segments[0][1][0]
video_segments[1][1][0][:] = False

#open pickle file
with open('/home/maxime/prg/phd/dropletswimming/data_analyzed/fframe_segmentations/ngm-b-02212022181139-0000_fframe_segmentation.pkl', 'rb') as file:
    video_segments = pickle.load(file)

from PIL import Image

mask = video_segments[4][1][0]

image_array = np.uint8(mask	 * 255)
image = Image.fromarray(image_array)
image.save('tst.png')



###Multimask video overlay###
import cv2
import numpy as np
import os
import torch

def overlay_masks_on_image(image_path, masks, colors=None, alpha=0.5):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a blank overlay
    overlay = np.zeros_like(image)
    
    # If colors are not provided, generate random colors
    if colors is None:
        colors = {mask_id: tuple(np.random.randint(0, 255, 3).tolist()) for mask_id in masks.keys()}
    
    # Process each mask
    for mask_id, mask in masks.items():
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
        color = colors[mask_id]
        colored_mask = np.zeros_like(image)
        colored_mask[mask_resized == 1] = color
        
        # Add the colored mask to the overlay
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
    
    # Overlay the masks on the image
    overlaid_image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    
    return overlaid_image

# Prepare the video writer
output_video_path = "crop_multi.mp4"
frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
if frame is None:
    raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

# Generate random colors for each unique mask ID
all_mask_ids = set()
for masks in video_segments.values():
    all_mask_ids.update(masks.keys())
colors = {mask_id: tuple(np.random.randint(0, 255, 3).tolist()) for mask_id in all_mask_ids}

# Process each frame
for frame_idx in range(len(frame_names)):
    print(frame_idx)
    image_path = os.path.join(video_dir, frame_names[frame_idx])
    
    try:
        if frame_idx in video_segments:
            masks = video_segments[frame_idx]
            overlaid_frame = overlay_masks_on_image(image_path, masks, colors)
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