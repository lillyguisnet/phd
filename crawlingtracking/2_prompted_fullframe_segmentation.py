import os
import sys
sys.path.append("/home/lilly/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from pathlib import Path
import shutil
import re
from tqdm import tqdm
import pickle
import h5py
from sam2.build_sam import build_sam2_video_predictor

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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


video_dir = "/home/lilly/phd/crawlingtracking/data_foranalysis/food/food-a-02252022134507-0000"

sam2_checkpoint = "./segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
prompt_path = "/home/lilly/phd/crawlingtracking/000600.jpg"
# Extract the filename from the source path
filename = os.path.basename(prompt_path)

# Create the full destination path
destination_path = os.path.join(video_dir, filename)

# Copy the image to the destination folder
shutil.copy2(prompt_path, destination_path)


print(f"Processing video_dir: {video_dir}")
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
inference_state = predictor.init_state(video_path=video_dir)

ann_frame_idx = 600
ann_obj_id = 1
points = np.array([[1317, 1481]], dtype=np.float32) #full frame common prompt
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
plt.savefig("promptclick.png")
plt.close()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }



def analyze_masks(video_segments):
    results = {'empty': {}, 'large': {}, 'overlapping': {}}
    max_counts = {'empty': 0, 'large': 0, 'overlapping': 0}
    max_frames = {'empty': None, 'large': None, 'overlapping': None}

    for frame, mask_dict in video_segments.items():
        mask_ids = list(mask_dict.keys())
        
        # Track empty masks
        empty_masks = []
        large_masks = []
        overlapping_masks = []
        
        for mask_id in mask_ids:
            mask = mask_dict[mask_id]
            mask_sum = mask.sum()
            
            # Check for empty masks
            if mask_sum == 0:
                empty_masks.append(mask_id)
            
            # Check for large masks (more than 800 pixels)
            if mask_sum >= 800:
                large_masks.append(mask_id)
            
            # Check for overlaps with other masks
            for other_id in mask_ids:
                if other_id > mask_id:  # Only check each pair once
                    other_mask = mask_dict[other_id]
                    intersection = np.logical_and(mask, other_mask)
                    union = np.logical_or(mask, other_mask)
                    overlap_pixels = np.sum(intersection)
                    iou = overlap_pixels / np.sum(union) if np.sum(union) > 0 else 0
                    
                    if overlap_pixels > 0:
                        overlapping_masks.append((mask_id, other_id, iou, overlap_pixels))
        
        # Store results for this frame
        if empty_masks:
            results['empty'][frame] = empty_masks
        if large_masks:
            results['large'][frame] = large_masks
        if overlapping_masks:
            results['overlapping'][frame] = overlapping_masks
        
        # Update max counts
        for category, masks in [('empty', empty_masks), ('large', large_masks), ('overlapping', overlapping_masks)]:
            if len(masks) > max_counts[category]:
                max_counts[category] = len(masks)
                max_frames[category] = frame

    return results, max_counts, max_frames

def print_analysis_results(results, max_counts, max_frames):
    print("\nAnalysis Results:")
    print("-" * 50)
    
    # Print empty mask results
    if results['empty']:
        print("\nFrames with empty masks:")
        for frame, mask_ids in results['empty'].items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
        print(f"Maximum empty masks in a single frame: {max_counts['empty']} (Frame {max_frames['empty']})")
    else:
        print("\nNo frames with empty masks found!")
    
    # Print large mask results
    if results['large']:
        print("\nFrames with large masks (≥800 pixels):")
        for frame, mask_ids in results['large'].items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
        print(f"Maximum large masks in a single frame: {max_counts['large']} (Frame {max_frames['large']})")
    else:
        print("\nNo frames with large masks found!")
    
    # Print overlapping mask results
    if results['overlapping']:
        print("\nFrames with overlapping masks:")
        for frame, overlaps in results['overlapping'].items():
            overlap_info = [f"{a}-{b} ({iou:.2%}, {pixels} pixels)" for a, b, iou, pixels in overlaps]
            print(f"  Frame {frame}: Overlapping pairs {', '.join(overlap_info)}")
        print(f"Maximum overlapping pairs in a single frame: {max_counts['overlapping']} (Frame {max_frames['overlapping']})")
    else:
        print("\nNo frames with overlapping masks found!")
    
    # Print summary statistics
    total_frames = len(video_segments)
    frames_with_issues = len(set().union(*[set(frames.keys()) for frames in results.values()]))
    print(f"\nSummary:")
    print(f"Total frames analyzed: {total_frames}")
    print(f"Frames with any issues: {frames_with_issues} ({frames_with_issues/total_frames*100:.1f}%)")

# Analyze the video segments
results, max_counts, max_frames = analyze_masks(video_segments)
print_analysis_results(results, max_counts, max_frames)



def create_overlay_video(video_segments, video_dir, frame_names, output_path, mask_color=(0, 255, 0), alpha=0.5, fps=10):
    """
    Create an overlay video with segmentation masks overlaid on the original frames.
    
    Args:
        video_segments: Dictionary containing segmentation masks for each frame
        video_dir: Directory containing the original video frames
        frame_names: List of frame filenames sorted by frame number
        output_path: Path where the output video will be saved
        mask_color: RGB color for the mask overlay (default: green)
        alpha: Transparency of the mask overlay (0.0 to 1.0)
        fps: Frames per second for the output video
    """
    # Get sorted frame indices that have masks
    sorted_frames = sorted(video_segments.keys())
    
    if not sorted_frames:
        print("No frames with masks found in video_segments")
        return
    
    # Read the first frame to get video dimensions
    first_frame_path = os.path.join(video_dir, frame_names[sorted_frames[0]])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Could not read first frame: {first_frame_path}")
        return
    
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating overlay video with {len(sorted_frames)} frames...")
    
    for frame_idx in tqdm(sorted_frames, desc="Creating overlay video"):
        # Read the original frame
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
        
        # Get the mask for this frame (assuming single object with obj_id=1)
        if ann_obj_id in video_segments[frame_idx]:
            mask = video_segments[frame_idx][ann_obj_id]
            
            # Ensure mask is 2D and matches frame dimensions
            if len(mask.shape) > 2:
                mask = mask.squeeze()
            
            # Resize mask if it doesn't match frame dimensions
            if mask.shape != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)
            
            # Create colored overlay
            overlay = frame.copy()
            overlay[mask] = mask_color
            
            # Blend the overlay with the original frame
            frame_with_overlay = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
        else:
            # No mask for this frame, use original frame
            frame_with_overlay = frame
        
        # Write frame to video
        out.write(frame_with_overlay)
    
    # Release video writer
    out.release()
    print(f"Overlay video saved to: {output_path}")

# Create overlay video
overlay_video_path = os.path.join(os.path.dirname(video_dir), f"{os.path.basename(video_dir)}_overlay.mp4")
create_overlay_video(video_segments, video_dir, frame_names, overlay_video_path)





#Save segmentation dict
save_dir = "/home/maxime/prg/phd/crawlingtracking/final_data/fullframe_segmentations/"
save_name = save_dir + os.path.basename(os.path.normpath(video_dir)) + ".h5"
# Convert the dictionary to a single large numpy array
frames = sorted(video_segments.keys())
all_masks = np.array([list(video_segments[frame].values())[0].astype(np.uint8) for frame in frames])
with h5py.File(save_name, 'w') as f:
    # Create a single large dataset
    f.create_dataset('masks', data=all_masks, compression="gzip", compression_opts=9)
    # Save frame numbers as a separate dataset for reference
    f.create_dataset('frame_numbers', data=np.array(frames))

if not os.path.exists(save_name):
    raise IOError(f"File {save_name} was not created successfully.")
else:
    print(f"Successfully saved {save_name}")
# Delete the copied image after processing
os.remove(destination_path)



