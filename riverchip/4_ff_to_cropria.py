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
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from skimage import measure

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "/home/lilly/phd/segment-anything-2/checkpoints/sam2_hiera_large.pt"
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

def compute_mask_centroid_and_orientation(mask):
    """Compute centroid and principal orientation of a binary mask."""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Compute moments
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None, None, None
    
    # Centroid
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    
    # Principal orientation using second moments
    mu20 = M['m20'] / M['m00'] - cx * cx
    mu02 = M['m02'] / M['m00'] - cy * cy
    mu11 = M['m11'] / M['m00'] - cx * cy
    
    # Angle of principal axis
    if mu20 != mu02:
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    else:
        theta = 0.0
    
    return (cx, cy), theta, largest_contour

def align_mask_to_reference_simple(ref_mask, target_mask):
    """Simple centroid-based alignment - much more robust for similar masks."""
    
    # Get centroids of both masks
    ref_centroid, ref_theta, _ = compute_mask_centroid_and_orientation(ref_mask)
    target_centroid, target_theta, _ = compute_mask_centroid_and_orientation(target_mask)
    
    if ref_centroid is None or target_centroid is None:
        return 0, 0, 0, target_centroid if target_centroid else (0, 0)
    
    # Simple translation to align centroids
    dx = ref_centroid[0] - target_centroid[0] 
    dy = ref_centroid[1] - target_centroid[1]
    
    # Simple rotation difference
    dtheta = ref_theta - target_theta
    # Normalize angle to [-pi, pi]
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    
    # Limit rotation to reasonable values
    max_rotation = np.pi/6  # 30 degrees max
    dtheta = np.clip(dtheta, -max_rotation, max_rotation)
    
    print(f"  Centroid alignment: dx={dx:.1f}, dy={dy:.1f}, rot={np.degrees(dtheta):.1f}°")
    
    return dx, dy, dtheta, target_centroid

def apply_transformation_to_image_simple(image, dx, dy, dtheta, mask_centroid):
    """Apply INVERSE transformation to image to align the object - simplified version."""
    h, w = image.shape[:2]
    
    # We need to apply the INVERSE transformation to the image
    # If we need to move the mask by (dx, dy, dtheta), we move the image by (-dx, -dy, -dtheta)
    inv_dx, inv_dy, inv_dtheta = -dx, -dy, -dtheta
    
    # For small rotations, we can simplify by rotating around the mask centroid
    if mask_centroid is None:
        mask_centroid = (w//2, h//2)
    
    # Create transformation matrix
    # First translate so rotation center is at origin, then rotate, then translate back and apply translation
    cx, cy = mask_centroid[0], mask_centroid[1]
    
    cos_theta = np.cos(inv_dtheta)
    sin_theta = np.sin(inv_dtheta)
    
    # Combined transformation matrix
    M = np.array([
        [cos_theta, -sin_theta, inv_dx + cx - cos_theta * cx + sin_theta * cy],
        [sin_theta, cos_theta, inv_dy + cy - sin_theta * cx - cos_theta * cy]
    ], dtype=np.float32)
    
    # Apply transformation
    transformed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return transformed_image

def process_frames_with_mask_alignment_correct(input_folder, output_folder, video_segments, original_size, crop_size):
    """Process frames with mask-based alignment - CORRECTLY handling mask cropping."""
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get reference mask
    reference_mask = None
    reference_frame_idx = None
    
    for frame_idx in sorted(video_segments.keys()):
        if frame_idx in video_segments and video_segments[frame_idx]:
            mask = next(iter(video_segments[frame_idx].values()))[0]
            if mask.sum() > 100:
                reference_mask = mask
                reference_frame_idx = frame_idx
                break
    
    if reference_mask is None:
        print("Warning: No valid reference mask found. Falling back to fixed crop.")
        return process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, crop_size)
    
    print(f"Using frame {reference_frame_idx} as reference for alignment")
    
    # Calculate final crop window based on reference mask
    ref_y_coords, ref_x_coords = np.where(reference_mask)
    ref_center_x = (ref_x_coords.min() + ref_x_coords.max()) // 2
    ref_center_y = (ref_y_coords.min() + ref_y_coords.max()) // 2
    
    # Define FINAL crop boundaries (where we want the aligned object to be)
    final_left = max(0, ref_center_x - crop_size // 2)
    final_top = max(0, ref_center_y - crop_size // 2)
    final_right = min(original_size[1], final_left + crop_size)
    final_bottom = min(original_size[0], final_top + crop_size)
    
    if final_right == original_size[1]:
        final_left = final_right - crop_size
    if final_bottom == original_size[0]:
        final_top = final_bottom - crop_size
    
    print(f"Final crop region: ({final_left}, {final_top}) to ({final_right}, {final_bottom})")
    
    # Use FULL masks for alignment calculations (no cropping!)
    ref_centroid, ref_theta, _ = compute_mask_centroid_and_orientation(reference_mask)
    
    if ref_centroid is None:
        print("Warning: Invalid reference mask.")
        return process_frames_fixed_crop(input_folder, output_folder, video_segments, original_size, crop_size)
    
    print(f"Reference centroid (full image): {ref_centroid}")
    
    # Process each frame
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Processing frames")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Get FULL mask for this frame (no cropping for alignment calculation!)
        frame_idx = idx
        if frame_idx in video_segments and video_segments[frame_idx]:
            current_mask_full = next(iter(video_segments[frame_idx].values()))[0]
            
            # Calculate alignment using FULL masks
            if current_mask_full.sum() > 100:  # Valid mask
                current_centroid, current_theta, _ = compute_mask_centroid_and_orientation(current_mask_full)
                
                if current_centroid is not None:
                    # Calculate transformation needed (in full image coordinates)
                    dx = ref_centroid[0] - current_centroid[0]
                    dy = ref_centroid[1] - current_centroid[1]
                    dtheta = ref_theta - current_theta
                    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                    
                    # Limit transformations to reasonable values
                    dx = np.clip(dx, -50, 50)  # Allow larger movements
                    dy = np.clip(dy, -50, 50)
                    dtheta = np.clip(dtheta, -np.pi/6, np.pi/6)
                    
                    # Apply inverse transformation to the full image
                    aligned_frame = apply_transformation_to_image_simple(frame, -dx, -dy, -dtheta, current_centroid)
                    
                    print(f"Frame {idx}: dx={dx:.1f}, dy={dy:.1f}, rot={np.degrees(dtheta):.1f}°")
                else:
                    aligned_frame = frame
            else:
                aligned_frame = frame
        else:
            aligned_frame = frame
        
        # NOW crop the aligned frame to the final region
        final_crop = aligned_frame[final_top:final_bottom, final_left:final_right]
        
        # Ensure exact crop size
        if final_crop.shape[:2] != (crop_size, crop_size):
            final_crop = cv2.resize(final_crop, (crop_size, crop_size))
        
        # Save the aligned crop
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, final_crop)
    
    print(f"Aligned and cropped frames saved to: {output_folder}")
    return len(frame_files), (crop_size, crop_size)

def create_mask_overlay_video(input_folder, video_segments, output_video_path, fps=10):
    """Create a video showing the original frames with mask overlays."""
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Read first frame to get video dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating overlay video with {len(frame_files)} frames...")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Creating overlay video")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Create overlay
        overlay = frame.copy()
        
        # Add mask if available
        if idx in video_segments and video_segments[idx]:
            for obj_id, mask in video_segments[idx].items():
                # Convert mask to uint8
                mask_uint8 = (mask[0] * 255).astype(np.uint8)
                
                # Create colored mask (green with transparency)
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = mask_uint8  # Green channel
                
                # Blend with original frame
                alpha = 0.4  # Transparency
                overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
                
                # Add contour outline
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contour
        
        # Add frame number text
        cv2.putText(overlay, f"Frame {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame
        out.write(overlay)
    
    # Release everything
    out.release()
    print(f"Overlay video saved to: {output_video_path}")

def create_alignment_comparison_video(input_folder, video_segments, original_size, crop_size, output_video_path, fps=10):
    """Create a side-by-side comparison video showing original crops vs aligned crops."""
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    
    if not frame_files:
        raise ValueError("No jpg files found in the input folder")
    
    # Get reference mask (first frame with valid mask)
    reference_mask = None
    reference_frame_idx = None
    
    for frame_idx in sorted(video_segments.keys()):
        if frame_idx in video_segments and video_segments[frame_idx]:
            mask = next(iter(video_segments[frame_idx].values()))[0]
            if mask.sum() > 100:
                reference_mask = mask
                reference_frame_idx = frame_idx
                break
    
    if reference_mask is None:
        print("No valid reference mask found for comparison video.")
        return
    
    # Calculate crop window based on reference mask
    ref_y_coords, ref_x_coords = np.where(reference_mask)
    ref_center_x = (ref_x_coords.min() + ref_x_coords.max()) // 2
    ref_center_y = (ref_y_coords.min() + ref_y_coords.max()) // 2
    
    # Calculate transformations for all frames
    transformations = {}
    for frame_idx in sorted(video_segments.keys()):
        if frame_idx in video_segments and video_segments[frame_idx]:
            current_mask = next(iter(video_segments[frame_idx].values()))[0]
            if current_mask.sum() > 100:
                dx, dy, dtheta, mask_centroid = align_mask_to_reference_simple(reference_mask, current_mask)
                transformations[frame_idx] = (dx, dy, dtheta, mask_centroid)
            else:
                ref_centroid, _, _ = compute_mask_centroid_and_orientation(reference_mask)
                transformations[frame_idx] = (0, 0, 0, ref_centroid)
        else:
            ref_centroid, _, _ = compute_mask_centroid_and_orientation(reference_mask)
            transformations[frame_idx] = (0, 0, 0, ref_centroid)
    
    # Video dimensions (side by side)
    video_width = crop_size * 2 + 10  # 10 pixels separator
    video_height = crop_size
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))
    
    print(f"Creating alignment comparison video...")
    
    for idx, frame_file in enumerate(tqdm(frame_files, desc="Creating comparison video")):
        # Read the frame
        frame = cv2.imread(os.path.join(input_folder, frame_file))
        
        # Crop original (fixed crop)
        left = max(0, ref_center_x - crop_size // 2)
        top = max(0, ref_center_y - crop_size // 2)
        right = min(original_size[1], left + crop_size)
        bottom = min(original_size[0], top + crop_size)
        
        if right == original_size[1]:
            left = right - crop_size
        if bottom == original_size[0]:
            top = bottom - crop_size
            
        original_crop = frame[top:bottom, left:right]
        if original_crop.shape[:2] != (crop_size, crop_size):
            original_crop = cv2.resize(original_crop, (crop_size, crop_size))
        
        # Crop aligned
        frame_idx = idx
        if frame_idx in transformations:
            dx, dy, dtheta, mask_centroid = transformations[frame_idx]
            aligned_frame = apply_transformation_to_image_simple(frame, dx, dy, dtheta, mask_centroid)
        else:
            aligned_frame = frame
            
        aligned_crop = aligned_frame[top:bottom, left:right]
        if aligned_crop.shape[:2] != (crop_size, crop_size):
            aligned_crop = cv2.resize(aligned_crop, (crop_size, crop_size))
        
        # Create side-by-side comparison
        comparison = np.zeros((crop_size, video_width, 3), dtype=np.uint8)
        comparison[:, :crop_size] = original_crop
        comparison[:, crop_size+10:] = aligned_crop
        
        # Add separator line
        comparison[:, crop_size:crop_size+10] = [100, 100, 100]
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(comparison, "Aligned", (crop_size + 20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(comparison, f"Frame {idx}", (10, crop_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add transformation info
        if frame_idx in transformations:
            dx, dy, dtheta = transformations[frame_idx][:3]
            transform_text = f"dx:{dx:.1f} dy:{dy:.1f} rot:{np.degrees(dtheta):.1f}°"
            cv2.putText(comparison, transform_text, (crop_size + 20, crop_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        out.write(comparison)
    
    out.release()
    print(f"Alignment comparison video saved to: {output_video_path}")

###Crop around RIA region
output_folder = os.path.join(os.path.dirname(crop_dir), os.path.basename(random_video_dir) + "_crop")
first_frame = cv2.imread(os.path.join(random_video_dir, frame_names[0]))
original_size = first_frame.shape[:2]

# Create visualization videos for debugging
print("Creating mask overlay video...")
overlay_video_path = os.path.join(os.path.dirname(crop_dir), f"{os.path.basename(random_video_dir)}_mask_overlay.mp4")
create_mask_overlay_video(random_video_dir, video_segments, overlay_video_path, fps=15)

print("Creating alignment comparison video...")
comparison_video_path = os.path.join(os.path.dirname(crop_dir), f"{os.path.basename(random_video_dir)}_alignment_comparison.mp4")
create_alignment_comparison_video(random_video_dir, video_segments, original_size, 110, comparison_video_path, fps=15)

# Also run the mask-based alignment
print("Processing frames with mask alignment...")
process_frames_with_mask_alignment_correct(random_video_dir, output_folder, video_segments, original_size, 110)


