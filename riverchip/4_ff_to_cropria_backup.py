import os
import sys
sys.path.append("/home/lilly/phd/segment-anything-2")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import random
from scipy.ndimage import maximum_filter, label

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


def get_random_unprocessed_video(parent_dir, crop_dir):
    all_videos = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(crop_dir, video + "_crop"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(parent_dir, random.choice(unprocessed_videos))



parent_video_dir = '/home/lilly/phd/riverchip/data_foranalysis/videotojpg/'
crop_dir = '/home/lilly/phd/riverchip/data_foranalysis/riacrop/'


# Get a random unprocessed video
random_video_dir = get_random_unprocessed_video(parent_video_dir, crop_dir)
print(f"Processing video: {random_video_dir}")


# Scan all the jpg frame names in the directory
frame_names = [
    p for p in os.listdir(random_video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

inference_state = predictor.init_state(video_path=random_video_dir)


points=np.array([[560, 700]], dtype=np.float32) #RIA region
labels=np.array([1], np.int32)

prompts = {}
ann_frame_idx = len(frame_names) - 1 # Use the last frame
ann_obj_id = 2
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Show the results on the prompt frame
plt.figure(figsize=(12, 8))
plt.title(f"Prompt frame")
plt.imshow(Image.open(os.path.join(random_video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.savefig("tstclick.png")
plt.close()



video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=ann_frame_idx, reverse=True):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


empty_masks = {}
low_detection_masks = {}
high_detection_masks = {}
for frame, mask_dict in video_segments.items():
    for mask_id, mask in mask_dict.items():
        mask_sum = mask.sum()        
        if mask_sum == 0:
            if frame not in empty_masks:
                empty_masks[frame] = []
            empty_masks[frame].append(mask_id)
        elif mask_sum <= 200:
            if frame not in low_detection_masks:
                low_detection_masks[frame] = []
            low_detection_masks[frame].append(mask_id)
        elif mask_sum >= 5000:
            if frame not in high_detection_masks:
                high_detection_masks[frame] = []
            high_detection_masks[frame].append(mask_id)
def print_results(result_dict, condition):
    if result_dict:
        print(f"!!! Frames with masks {condition}:")
        for frame, mask_ids in result_dict.items():
            print(f"  Frame {frame}: Mask IDs {mask_ids}")
    else:
        print(f"Yay! No masks {condition} found, yay!")
print_results(empty_masks, "that are empty")
#print_results(low_detection_masks, "having 200 or fewer true elements")
print_results(high_detection_masks, "having 5000 or more true elements")



def calculate_fixed_crop_window(video_segments, original_size, crop_size):
    orig_height, orig_width = original_size
    centers = []
    empty_masks = 0
    total_masks = 0

    for frame_num in sorted(video_segments.keys()):
        mask = next(iter(video_segments[frame_num].values()))
        total_masks += 1
        y_coords, x_coords = np.where(mask[0])
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            center_x = (x_coords.min() + x_coords.max()) // 2
            center_y = (y_coords.min() + y_coords.max()) // 2
            centers.append((center_x, center_y))
        else:
            empty_masks += 1
            centers.append((orig_width // 2, orig_height // 2))

    if empty_masks > 0:
        avg_center_x = sum(center[0] for center in centers) // len(centers)
        avg_center_y = sum(center[1] for center in centers) // len(centers)
        centers = [(avg_center_x, avg_center_y)] * len(centers)

    crop_windows = []
    for center_x, center_y in centers:
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(orig_width, left + crop_size)
        bottom = min(orig_height, top + crop_size)
        
        # Adjust if crop window is out of bounds
        if right == orig_width:
            left = right - crop_size
        if bottom == orig_height:
            top = bottom - crop_size
        
        crop_windows.append((left, top, right, bottom))

    return crop_windows, (crop_size, crop_size), empty_masks, total_masks

def process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, crop_size):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate fixed crop windows
    crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, crop_size)
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get crop window for this frame
        left, top, right, bottom = crop_windows[idx]
        
        # Crop the frame
        cropped_frame = frame[top:bottom, left:right]
        
        # Ensure the cropped frame is exactly crop_size x crop_size
        if cropped_frame.shape[:2] != (crop_height, crop_width):
            cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
        
        # Save the cropped frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, cropped_frame)
    
    print(f"Cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_height, crop_width)


def calculate_weighted_centroid(image, intensity_threshold_percentile=85):
    """
    Calculate weighted centroid of bright regions in an image.
    
    Args:
        image: Input image (grayscale or color)
        intensity_threshold_percentile: Percentile threshold for considering pixels as "bright"
    
    Returns:
        (center_x, center_y): Weighted centroid coordinates
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate intensity threshold
    threshold_value = np.percentile(gray, intensity_threshold_percentile)
    
    # Create mask for bright regions
    bright_mask = gray >= threshold_value
    
    # Get coordinates of bright pixels
    y_coords, x_coords = np.where(bright_mask)
    
    if len(x_coords) == 0:
        # Fallback to image center if no bright regions found
        return gray.shape[1] // 2, gray.shape[0] // 2
    
    # Get intensities of bright pixels for weighting
    intensities = gray[bright_mask]
    
    # Calculate weighted centroid
    total_weight = np.sum(intensities)
    weighted_x = np.sum(x_coords * intensities) / total_weight
    weighted_y = np.sum(y_coords * intensities) / total_weight
    
    return int(weighted_x), int(weighted_y)


def calculate_alignment_offsets(input_folder, video_segments, crop_size, intensity_threshold_percentile=85):
    """
    Calculate alignment offsets for each frame based on weighted centroids.
    
    Args:
        input_folder: Path to folder containing frame images
        video_segments: Segmentation results from SAM2
        crop_size: Size of the crop window
        intensity_threshold_percentile: Percentile threshold for bright regions
    
    Returns:
        List of (offset_x, offset_y) tuples for each frame
    """
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    # First, get initial crop windows based on segmentation (existing method)
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    original_size = first_frame.shape[:2]
    initial_crop_windows, _, _, _ = calculate_fixed_crop_window(video_segments, original_size, crop_size)
    
    centroids = []
    cropped_frames = []
    
    # Calculate weighted centroids for each initially cropped frame
    print("Calculating weighted centroids for alignment...")
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing centroids")):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get initial crop
        left, top, right, bottom = initial_crop_windows[idx]
        cropped_frame = frame[top:bottom, left:right]
        
        # Ensure consistent size
        if cropped_frame.shape[:2] != (crop_size, crop_size):
            cropped_frame = cv2.resize(cropped_frame, (crop_size, crop_size))
        
        cropped_frames.append(cropped_frame)
        
        # Calculate weighted centroid
        centroid_x, centroid_y = calculate_weighted_centroid(cropped_frame, intensity_threshold_percentile)
        centroids.append((centroid_x, centroid_y))
    
    # Calculate reference centroid (median position to be robust to outliers)
    ref_x = np.median([c[0] for c in centroids])
    ref_y = np.median([c[1] for c in centroids])
    
    print(f"Reference centroid: ({ref_x:.1f}, {ref_y:.1f})")
    
    # Calculate offsets needed to align each frame's centroid to the reference
    alignment_offsets = []
    for centroid_x, centroid_y in centroids:
        offset_x = ref_x - centroid_x
        offset_y = ref_y - centroid_y
        alignment_offsets.append((offset_x, offset_y))
    
    return alignment_offsets, (ref_x, ref_y), cropped_frames


def apply_alignment_crop(image, crop_window, alignment_offset, crop_size):
    """
    Apply alignment offset to crop window and extract aligned crop.
    
    Args:
        image: Input image
        crop_window: (left, top, right, bottom) initial crop coordinates
        alignment_offset: (offset_x, offset_y) alignment adjustment
        crop_size: Final crop size
    
    Returns:
        Aligned and cropped image
    """
    left, top, right, bottom = crop_window
    offset_x, offset_y = alignment_offset
    
    # Apply alignment offset to crop window
    aligned_left = int(left - offset_x)
    aligned_top = int(top - offset_y)
    aligned_right = aligned_left + crop_size
    aligned_bottom = aligned_top + crop_size
    
    # Ensure crop window stays within image bounds
    img_height, img_width = image.shape[:2]
    
    # Adjust if out of bounds
    if aligned_left < 0:
        aligned_right -= aligned_left
        aligned_left = 0
    if aligned_top < 0:
        aligned_bottom -= aligned_top
        aligned_top = 0
    if aligned_right > img_width:
        aligned_left -= (aligned_right - img_width)
        aligned_right = img_width
    if aligned_bottom > img_height:
        aligned_top -= (aligned_bottom - img_height)
        aligned_bottom = img_height
    
    # Final bounds check
    aligned_left = max(0, aligned_left)
    aligned_top = max(0, aligned_top)
    aligned_right = min(img_width, aligned_left + crop_size)
    aligned_bottom = min(img_height, aligned_top + crop_size)
    
    # Crop the image
    cropped = image[aligned_top:aligned_bottom, aligned_left:aligned_right]
    
    # Ensure exact crop size
    if cropped.shape[:2] != (crop_size, crop_size):
        cropped = cv2.resize(cropped, (crop_size, crop_size))
    
    return cropped


def process_frames_aligned_crop(input_folder, output_folder, video_segments, original_size, crop_size, intensity_threshold_percentile=85):
    """
    Process frames with centroid-based alignment.
    
    Args:
        input_folder: Input folder containing frame images
        output_folder: Output folder for aligned crops
        video_segments: Segmentation results from SAM2
        original_size: Original image dimensions
        crop_size: Size of crop window
        intensity_threshold_percentile: Percentile threshold for bright regions
    
    Returns:
        Number of processed frames and crop dimensions
    """
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate initial crop windows (based on segmentation)
    initial_crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, crop_size)
    
    # Calculate alignment offsets
    alignment_offsets, ref_centroid, _ = calculate_alignment_offsets(input_folder, video_segments, crop_size, intensity_threshold_percentile)
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    print(f"Reference centroid: ({ref_centroid[0]:.1f}, {ref_centroid[1]:.1f})")
    
    # Process each frame with alignment
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing aligned frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get initial crop window and alignment offset
        crop_window = initial_crop_windows[idx]
        alignment_offset = alignment_offsets[idx]
        
        # Apply aligned crop
        aligned_cropped_frame = apply_alignment_crop(frame, crop_window, alignment_offset, crop_size)
        
        # Save the aligned cropped frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, aligned_cropped_frame)
    
    print(f"Aligned cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_height, crop_width)


def visualize_alignment_results(output_folder, num_frames_to_show=5):
    """
    Create a visualization showing the alignment results.
    
    Args:
        output_folder: Folder containing aligned cropped frames
        num_frames_to_show: Number of frames to include in visualization
    """
    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print("No frames found for visualization")
        return
    
    # Select frames evenly spaced throughout the video
    indices = np.linspace(0, len(frame_files)-1, min(num_frames_to_show, len(frame_files)), dtype=int)
    
    fig, axes = plt.subplots(2, len(indices), figsize=(3*len(indices), 6))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(indices):
        frame_file = frame_files[idx]
        frame = cv2.imread(os.path.join(output_folder, frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Original frame
        axes[0, i].imshow(frame_rgb)
        axes[0, i].set_title(f'Frame {idx}')
        axes[0, i].axis('off')
        
        # Frame with centroid marked
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centroid_x, centroid_y = calculate_weighted_centroid(frame)
        
        axes[1, i].imshow(frame_rgb)
        axes[1, i].scatter(centroid_x, centroid_y, c='red', s=50, marker='x', linewidths=2)
        axes[1, i].set_title(f'Centroid: ({centroid_x}, {centroid_y})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'alignment_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Alignment visualization saved to: {os.path.join(output_folder, 'alignment_visualization.png')}")


###Crop around RIA region with feature-based alignment
output_folder = os.path.join(os.path.dirname(crop_dir), os.path.basename(random_video_dir) + "_crop_feature")
first_frame = cv2.imread(os.path.join(random_video_dir, frame_names[0]))
original_size = first_frame.shape[:2]

# Use feature-based alignment instead of centroid-based alignment
num_frames, crop_dims, trajectories = process_frames_feature_aligned_crop(random_video_dir, output_folder, video_segments, original_size, 110, num_spots=3)

# Add visualization after processing
visualize_feature_alignment_results(output_folder, trajectories, num_frames_to_show=5)


def detect_bright_spots(image, num_spots=3, min_distance=10):
    """
    Detect bright spots in an image using local maxima detection.
    
    Args:
        image: Input image (grayscale or color)
        num_spots: Expected number of bright spots
        min_distance: Minimum distance between spots
    
    Returns:
        List of (x, y) coordinates of detected spots
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find local maxima
    # Create a mask for local maxima
    local_maxima = maximum_filter(blurred, size=min_distance) == blurred
    
    # Apply intensity threshold (top 15% of pixels)
    threshold = np.percentile(blurred, 85)
    bright_mask = blurred >= threshold
    
    # Combine masks
    spots_mask = local_maxima & bright_mask
    
    # Get coordinates of potential spots
    y_coords, x_coords = np.where(spots_mask)
    intensities = blurred[spots_mask]
    
    if len(x_coords) == 0:
        return []
    
    # Sort by intensity (brightest first)
    sorted_indices = np.argsort(intensities)[::-1]
    
    # Select top spots with minimum distance constraint
    selected_spots = []
    for idx in sorted_indices:
        x, y = x_coords[idx], y_coords[idx]
        
        # Check minimum distance from already selected spots
        too_close = False
        for sx, sy in selected_spots:
            if np.sqrt((x - sx)**2 + (y - sy)**2) < min_distance:
                too_close = True
                break
        
        if not too_close:
            selected_spots.append((x, y))
            
        if len(selected_spots) >= num_spots:
            break
    
    return selected_spots


def match_spots_across_frames(spots_list, max_distance=20):
    """
    Match spots across frames to create consistent tracking.
    
    Args:
        spots_list: List of spot lists for each frame
        max_distance: Maximum distance for matching spots
    
    Returns:
        List of matched spot trajectories
    """
    if not spots_list or not spots_list[0]:
        return []
    
    # Initialize trajectories with first frame
    trajectories = [[spot] for spot in spots_list[0]]
    
    for frame_spots in spots_list[1:]:
        if not frame_spots:
            # Add None for missing spots in this frame
            for traj in trajectories:
                traj.append(None)
            continue
        
        # Match spots to existing trajectories
        used_spots = set()
        
        for traj_idx, trajectory in enumerate(trajectories):
            # Get last known position
            last_pos = None
            for pos in reversed(trajectory):
                if pos is not None:
                    last_pos = pos
                    break
            
            if last_pos is None:
                trajectory.append(None)
                continue
            
            # Find closest unused spot
            best_spot = None
            best_distance = float('inf')
            
            for spot_idx, spot in enumerate(frame_spots):
                if spot_idx in used_spots:
                    continue
                
                distance = np.sqrt((spot[0] - last_pos[0])**2 + (spot[1] - last_pos[1])**2)
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_spot = (spot_idx, spot)
            
            if best_spot is not None:
                trajectory.append(best_spot[1])
                used_spots.add(best_spot[0])
            else:
                trajectory.append(None)
        
        # Add new trajectories for unmatched spots
        for spot_idx, spot in enumerate(frame_spots):
            if spot_idx not in used_spots:
                new_traj = [None] * len(trajectories[0])
                new_traj[-1] = spot
                trajectories.append(new_traj)
    
    return trajectories


def calculate_feature_alignment_offsets(input_folder, video_segments, crop_size, num_spots=3):
    """
    Calculate alignment offsets based on feature tracking of bright spots.
    
    Args:
        input_folder: Path to folder containing frame images
        video_segments: Segmentation results from SAM2
        crop_size: Size of the crop window
        num_spots: Number of bright spots to track
    
    Returns:
        List of (offset_x, offset_y) tuples for each frame
    """
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    # First, get initial crop windows based on segmentation
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    original_size = first_frame.shape[:2]
    initial_crop_windows, _, _, _ = calculate_fixed_crop_window(video_segments, original_size, crop_size)
    
    # Detect spots in each initially cropped frame
    print("Detecting bright spots for feature-based alignment...")
    all_spots = []
    cropped_frames = []
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Detecting spots")):
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get initial crop
        left, top, right, bottom = initial_crop_windows[idx]
        cropped_frame = frame[top:bottom, left:right]
        
        # Ensure consistent size
        if cropped_frame.shape[:2] != (crop_size, crop_size):
            cropped_frame = cv2.resize(cropped_frame, (crop_size, crop_size))
        
        cropped_frames.append(cropped_frame)
        
        # Detect spots in this frame
        spots = detect_bright_spots(cropped_frame, num_spots=num_spots)
        all_spots.append(spots)
    
    # Match spots across frames
    print("Matching spots across frames...")
    trajectories = match_spots_across_frames(all_spots)
    
    if not trajectories:
        print("Warning: No spots detected, falling back to centroid alignment")
        return calculate_alignment_offsets(input_folder, video_segments, crop_size)
    
    # Calculate reference positions (median of each trajectory)
    reference_spots = []
    for traj in trajectories:
        valid_positions = [pos for pos in traj if pos is not None]
        if valid_positions:
            ref_x = np.median([pos[0] for pos in valid_positions])
            ref_y = np.median([pos[1] for pos in valid_positions])
            reference_spots.append((ref_x, ref_y))
    
    print(f"Detected {len(reference_spots)} spot trajectories")
    print(f"Reference spots: {reference_spots}")
    
    # Calculate alignment offsets for each frame
    alignment_offsets = []
    
    for frame_idx in range(len(frame_files)):
        # Get current spots for this frame
        current_spots = []
        for traj in trajectories:
            if frame_idx < len(traj) and traj[frame_idx] is not None:
                current_spots.append(traj[frame_idx])
        
        if len(current_spots) == 0 or len(reference_spots) == 0:
            # No spots detected, no offset
            alignment_offsets.append((0, 0))
            continue
        
        # Calculate transformation to align current spots to reference
        # Use the centroid of available spots for simplicity
        current_centroid_x = np.mean([spot[0] for spot in current_spots])
        current_centroid_y = np.mean([spot[1] for spot in current_spots])
        
        ref_centroid_x = np.mean([spot[0] for spot in reference_spots[:len(current_spots)]])
        ref_centroid_y = np.mean([spot[1] for spot in reference_spots[:len(current_spots)]])
        
        offset_x = ref_centroid_x - current_centroid_x
        offset_y = ref_centroid_y - current_centroid_y
        
        alignment_offsets.append((offset_x, offset_y))
    
    return alignment_offsets, reference_spots, trajectories


def process_frames_feature_aligned_crop(input_folder, output_folder, video_segments, original_size, crop_size, num_spots=3):
    """
    Process frames with feature-based alignment using bright spot detection.
    
    Args:
        input_folder: Input folder containing frame images
        output_folder: Output folder for aligned crops
        video_segments: Segmentation results from SAM2
        original_size: Original image dimensions
        crop_size: Size of crop window
        num_spots: Number of bright spots to track
    
    Returns:
        Number of processed frames and crop dimensions
    """
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Calculate initial crop windows (based on segmentation)
    initial_crop_windows, (crop_height, crop_width), empty_masks, total_masks = calculate_fixed_crop_window(video_segments, original_size, crop_size)
    
    # Calculate feature-based alignment offsets
    alignment_offsets, reference_spots, trajectories = calculate_feature_alignment_offsets(input_folder, video_segments, crop_size, num_spots)
    
    print(f"Empty masks: {empty_masks}/{total_masks}")
    print(f"Crop size: {crop_height}x{crop_width}")
    print(f"Reference spots: {reference_spots}")
    
    # Process each frame with alignment
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing feature-aligned frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get initial crop window and alignment offset
        crop_window = initial_crop_windows[idx]
        alignment_offset = alignment_offsets[idx]
        
        # Apply aligned crop
        aligned_cropped_frame = apply_alignment_crop(frame, crop_window, alignment_offset, crop_size)
        
        # Save the aligned cropped frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, aligned_cropped_frame)
    
    print(f"Feature-aligned cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_height, crop_width), trajectories


def visualize_feature_alignment_results(output_folder, trajectories, num_frames_to_show=5):
    """
    Create a visualization showing the feature-based alignment results.
    
    Args:
        output_folder: Folder containing aligned cropped frames
        trajectories: Spot trajectories from feature tracking
        num_frames_to_show: Number of frames to include in visualization
    """
    frame_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print("No frames found for visualization")
        return
    
    # Select frames evenly spaced throughout the video
    indices = np.linspace(0, len(frame_files)-1, min(num_frames_to_show, len(frame_files)), dtype=int)
    
    fig, axes = plt.subplots(2, len(indices), figsize=(3*len(indices), 6))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    
    for i, idx in enumerate(indices):
        frame_file = frame_files[idx]
        frame = cv2.imread(os.path.join(output_folder, frame_file))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Original frame
        axes[0, i].imshow(frame_rgb)
        axes[0, i].set_title(f'Frame {idx}')
        axes[0, i].axis('off')
        
        # Frame with detected spots marked
        axes[1, i].imshow(frame_rgb)
        
        # Plot spots from trajectories
        for traj_idx, trajectory in enumerate(trajectories):
            if idx < len(trajectory) and trajectory[idx] is not None:
                x, y = trajectory[idx]
                color = colors[traj_idx % len(colors)]
                axes[1, i].scatter(x, y, c=color, s=50, marker='o', linewidths=2, edgecolors='white')
        
        axes[1, i].set_title(f'Detected Spots')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_alignment_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature alignment visualization saved to: {os.path.join(output_folder, 'feature_alignment_visualization.png')}")


