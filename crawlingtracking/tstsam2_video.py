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

""" from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator """
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

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/tstcropped"
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000"
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/fixed_crop"
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/cropprompt"
video_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/cowz/cowz"

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

""" #Get random worm pixel from segmentation mask
worm0 = np.load("/home/maxime/prg/phd/tstvideo_1worm.npy") #Mask for first worm full frame
#Save array as image
plt.imshow(worm0)
plt.savefig("tstarr.png")
# Find the indices of all elements with a value of 1
indices = np.argwhere(worm0 == 1)
# Check if there are any elements with a value of 1
if indices.size > 0:
    # Randomly select one of these indices
    random_index = indices[np.random.choice(indices.shape[0])]
    print("Random coordinate with a value of 1:", random_index)
else:
    print("No elements with a value of 1 found in the array.") """


# Let's add a positive click at (x, y) = (210, 350) to get started
#SWITCH INDEX X Y
points = np.array([[1317, 1481]], dtype=np.float32) #full frame
points = np.array([[227, 250]], dtype=np.float32) #crop
points = np.array([[17, 40]], dtype=np.float32) #tight crop
points = np.array([[64, 45]], dtype=np.float32) #fixed crop
points = np.array([[1400, 500]], dtype=np.float32) #cowz head
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

###Two clicks
points = np.array([[500, 1300], [400, 1000]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
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

#Save propagation results
with open('propagation_cowz.pkl', 'wb') as file:
    pickle.dump(video_segments, file)

with open('propagation_fixedcrop.pkl', 'wb') as file:
    pickle.dump(video_segments, file)

""" # render the segmentation results every few frames
vis_frame_stride = 15
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id) """

#Make a video with results
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
output_video_path = "cowzhead_segmentation_result.mp4"
frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
#frame = cv2.imread(os.path.join(video_dir, "000000.jpg"))
if frame is None:
    raise ValueError(f"Could not read first frame from {os.path.join(video_dir, frame_names[0])}")
height, width, _ = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 29.90, (width, height))

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


###Crop video to make worm bigger
def find_crop_dimensions(video_segments):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = 0, 0

    for frame_num, frame_data in video_segments.items():
        # Assuming frame_data is a dictionary with a single key (1 in your example)
        mask = next(iter(frame_data.values()))
        
        # Assuming mask is a 3D array with shape (1, height, width)
        mask = mask[0]  # Remove the first dimension
        y_coords, x_coords = np.where(mask)
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            min_x = min(min_x, x_coords.min())
            max_x = max(max_x, x_coords.max())
            min_y = min(min_y, y_coords.min())
            max_y = max(max_y, y_coords.max())

    return min_x, min_y, max_x, max_y

# Usage
crop_dims = find_crop_dimensions(video_segments)
print(f"Crop dimensions: {crop_dims}")

# Try to find crop dimensions
crop_dims = find_crop_dimensions(video_segments)
print(f"\nCrop dimensions: {crop_dims}")

def process_frames(input_folder, output_folder, crop_dims, padding=10, target_size=None):
    min_x, min_y, max_x, max_y = crop_dims
    
    # Read one frame to get original dimensions
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    orig_height, orig_width = first_frame.shape[:2]

    # Adjust crop dimensions with padding
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(orig_width, max_x + padding)
    max_y = min(orig_height, max_y + padding)

    crop_width, crop_height = max_x - min_x, max_y - min_y

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Crop the frame with padding
        cropped_frame = frame[min_y:max_y, min_x:max_x]
        
        # Resize if necessary
        if target_size:
            cropped_frame = cv2.resize(cropped_frame, target_size)
        
        # Save the cropped frame
        cv2.imwrite(os.path.join(output_folder, frame_file), cropped_frame)

    return len(frame_files), (crop_width, crop_height)


def create_video_from_frames(input_folder, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in tqdm(frame_files, desc="Creating video"):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        out.write(frame)

    out.release()


# Usage
input_folder = '/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/a-02252022132222-0000'
output_folder = '/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/tstcropped'
output_video_path = '/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/tstcropped.avi'
crop_dims = (1100, 1256, 1335, 1533)  # Use the dimensions we found earlier
target_size = None  # Set this to a tuple like (640, 480) if you want to resize
padding = 10  # 10 pixels padding

num_frames, final_size = process_frames(input_folder, output_folder, crop_dims, padding=padding)

print(f"Processed {num_frames} frames.")
print(f"Final frame size after padding: {final_size[0]}x{final_size[1]}")

# Create video from processed frames
create_video_from_frames(output_folder, output_video_path, fps=10)

print(f"Processed {num_frames} frames and created video at {output_video_path}")



###Tight crop
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
input_folder = '/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/tstcropped'
output_folder = '/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/fixed_crop'

# Read one frame to get original dimensions
frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
original_size = first_frame.shape[:2]

num_frames, crop_size = process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, padding=5)

print(f"Processed {num_frames} frames.")
print(f"Fixed crop size: {crop_size[1]}x{crop_size[0]}")
print(f"Cropped frames saved in {output_folder}")