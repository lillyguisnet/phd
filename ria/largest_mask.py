import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
from typing import Dict, Set, List, Tuple
import itertools


def get_mask_orientation(mask_3d):
    """
    Get the orientation of a binary mask using PCA.
    Returns angle in degrees.
    Works with both (h,w) and (1,h,w) masks.
    """
    # If mask is 3D, take the first channel
    mask = mask_3d[0] if len(mask_3d.shape) == 3 else mask_3d
    y_coords, x_coords = np.nonzero(mask)
    if len(y_coords) == 0:
        return 0.0
    
    coords = np.column_stack((x_coords, y_coords))
    pca = PCA(n_components=2)
    pca.fit(coords)
    
    angle = np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0]))
    return angle

def get_exact_mask_points(mask):
    """
    Get the exact set of points in the mask.
    Returns a numpy array of coordinates.
    """
    y_coords, x_coords = np.nonzero(mask)
    return np.column_stack((y_coords, x_coords))

def create_mask_at_position(points, shape, shift):
    """
    Create a mask by shifting points by a given amount.
    Ensures all points stay within bounds.
    """
    shifted_points = points + shift
    
    # Check if all points are within bounds
    if (np.all(shifted_points[:, 0] >= 0) and 
        np.all(shifted_points[:, 0] < shape[0]) and
        np.all(shifted_points[:, 1] >= 0) and
        np.all(shifted_points[:, 1] < shape[1])):
        
        new_mask = np.zeros(shape, dtype=bool)
        new_mask[shifted_points[:, 0], shifted_points[:, 1]] = True
        return new_mask, True
    
    return None, False

def find_best_mask_position(template_points, target_mask):
    """
    Find the best position for the template mask that aligns with the target mask.
    Uses the center of mass for initial positioning and tries small adjustments.
    """
    # Get centers
    template_center = np.mean(template_points, axis=0)
    target_y, target_x = np.nonzero(target_mask)
    target_center = np.array([np.mean(target_y), np.mean(target_x)])
    
    # Calculate initial shift
    base_shift = (target_center - template_center).astype(int)
    
    # Try various small adjustments around the base position
    best_mask = None
    min_diff = float('inf')
    best_shift = None
    
    # Try shifts in a small window around the base position
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shift = base_shift + np.array([dy, dx])
            mask, valid = create_mask_at_position(template_points, target_mask.shape, shift)
            
            if valid:
                # Calculate difference in center of mass
                mask_com = np.array(center_of_mass(mask))
                target_com = np.array(center_of_mass(target_mask))
                diff = np.sum((mask_com - target_com) ** 2)
                
                if diff < min_diff:
                    min_diff = diff
                    best_mask = mask
                    best_shift = shift
    
    return best_mask, best_shift

def get_unique_object_ids(frame_masks: Dict) -> Set[int]:
    """
    Extract all unique object IDs from the frame masks dictionary.
    """
    object_ids = set()
    for objects in frame_masks.values():
        object_ids.update(objects.keys())
    return object_ids

def find_largest_mask_for_object(frame_masks: Dict, object_id: int) -> Tuple[np.ndarray, int]:
    """
    Find the largest mask instance for a given object ID.
    Returns the mask and its frame number.
    """
    max_area = 0
    largest_mask = None
    frame_num_largest = None
    
    for frame_num, objects in frame_masks.items():
        if object_id in objects:
            mask = objects[object_id]
            # Handle both (1,h,w) and (h,w) shapes
            mask = mask[0] if len(mask.shape) == 3 else mask
            area = np.sum(mask)
            if area > max_area:
                max_area = area
                largest_mask = mask.copy()
                frame_num_largest = frame_num
    
    if largest_mask is None:
        raise ValueError(f"No masks found for object {object_id}")
        
    return largest_mask, frame_num_largest

def process_all_masks(frame_masks: Dict) -> Dict:
    """
    Process all masks in the dictionary while preserving exact shape.
    Returns a dictionary with the same structure as input, with all masks having shape (1,h,w).
    
    Args:
        frame_masks: Dictionary of frame_number -> {object_id -> mask}
        
    Returns:
        Dictionary with processed masks maintaining the same structure
    """
    # Get all unique object IDs
    object_ids = get_unique_object_ids(frame_masks)
    
    # Initialize output dictionary
    processed_masks = {}
    
    # Process each object separately
    for object_id in object_ids:
        # Find the largest mask for this object
        largest_mask, frame_num_largest = find_largest_mask_for_object(frame_masks, object_id)
        
        # Get the exact points of the largest mask
        template_points = get_exact_mask_points(largest_mask)
        total_pixels = len(template_points)
        
        # Process each frame for this object
        for frame_num, objects in frame_masks.items():
            if object_id in objects:
                current_mask = objects[object_id]
                # Handle both (1,h,w) and (h,w) shapes
                current_mask = current_mask[0] if len(current_mask.shape) == 3 else current_mask
                
                # Find best position for template mask
                new_mask, shift = find_best_mask_position(template_points, current_mask)
                
                if new_mask is None:
                    print(f"Warning: Could not place mask for object {object_id} in frame {frame_num}")
                    continue
                
                # Ensure mask is 3D with shape (1,h,w)
                new_mask = new_mask[np.newaxis, ...] if len(new_mask.shape) == 2 else new_mask
                
                # Initialize frame dictionary if needed
                if frame_num not in processed_masks:
                    processed_masks[frame_num] = {}
                
                # Store result
                processed_masks[frame_num][object_id] = new_mask
                
                # Verify pixel count
                current_pixels = np.sum(new_mask)
                if current_pixels != total_pixels:
                    print(f"Warning: Object {object_id} Frame {frame_num} has different number of pixels "
                          f"({current_pixels} vs {total_pixels})")
    
    return processed_masks


frame_masks = cleaned_distance_segments


processed_masks = process_all_masks(frame_masks)




def ensure_3d(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is 3D with shape (1, h, w)."""
    if len(mask.shape) == 2:
        return mask[np.newaxis, ...]
    return mask

def get_mask_indices(mask: np.ndarray) -> List[int]:
    """
    Get indices of active pixels within the mask's own coordinate system.
    Returns list of indices where mask is True.
    """
    mask = ensure_3d(mask)
    indices = np.where(mask[0].ravel())[0]
    return indices.tolist()

def get_overlapping_relative_indices(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Find overlapping pixels between two masks and return their indices
    within each mask's own coordinate system.
    """
    mask1 = ensure_3d(mask1)
    mask2 = ensure_3d(mask2)
    overlap = mask1 & mask2
    
    if not np.any(overlap):
        return [], []
    
    # Get indices of active pixels in each mask
    mask1_indices = get_mask_indices(mask1)
    mask2_indices = get_mask_indices(mask2)
    overlap_indices = get_mask_indices(overlap)
    
    # Find which indices in each mask correspond to overlapping pixels
    mask1_overlap = []
    mask2_overlap = []
    
    # For each overlapping pixel, find its index in each original mask
    y_coords, x_coords = np.where(overlap[0])
    for y, x in zip(y_coords, x_coords):
        # Find position in mask1's active pixels
        flat_idx = y * mask1.shape[2] + x
        mask1_pos = mask1_indices.index(flat_idx)
        mask1_overlap.append(mask1_pos)
        
        # Find position in mask2's active pixels
        mask2_pos = mask2_indices.index(flat_idx)
        mask2_overlap.append(mask2_pos)
    
    return mask1_overlap, mask2_overlap

def find_all_overlapping_pairs(frame_masks: Dict) -> Set[Tuple[int, int]]:
    """Find all pairs of object IDs that overlap in any frame."""
    overlapping_pairs = set()
    
    for frame_num, objects in frame_masks.items():
        object_ids = list(objects.keys())
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                obj1_id = object_ids[i]
                obj2_id = object_ids[j]
                
                mask1 = objects[obj1_id]
                mask2 = objects[obj2_id]
                
                mask1_overlap, mask2_overlap = get_overlapping_relative_indices(mask1, mask2)
                if mask1_overlap:
                    pair = tuple(sorted([obj1_id, obj2_id]))
                    overlapping_pairs.add(pair)
    
    return overlapping_pairs

def find_all_overlapping_indices(masks1: Dict[int, np.ndarray],
                              masks2: Dict[int, np.ndarray]) -> Tuple[Set[int], Set[int]]:
    """
    Find all relative indices that overlap in any frame.
    Returns two sets of indices - one for each mask.
    """
    mask1_all_overlaps = set()
    mask2_all_overlaps = set()
    
    for frame in set(masks1.keys()) & set(masks2.keys()):
        mask1_overlap, mask2_overlap = get_overlapping_relative_indices(
            masks1[frame], masks2[frame])
        mask1_all_overlaps.update(mask1_overlap)
        mask2_all_overlaps.update(mask2_overlap)
    
    return mask1_all_overlaps, mask2_all_overlaps

def remove_indices_from_mask(mask: np.ndarray, indices_to_remove: Set[int]) -> np.ndarray:
    """
    Remove pixels at specified relative indices from mask.
    """
    mask = ensure_3d(mask)
    active_indices = get_mask_indices(mask)
    
    # Create new mask
    new_mask = mask.copy()
    for idx in indices_to_remove:
        if idx < len(active_indices):
            flat_idx = active_indices[idx]
            y = flat_idx // mask.shape[2]
            x = flat_idx % mask.shape[2]
            new_mask[0, y, x] = False
    
    return new_mask

def remove_overlaps_from_masks(processed_masks: Dict) -> Dict:
    """
    Remove overlapping pixels from all masks while maintaining consistency across frames.
    """
    print("Starting overlap removal process...")
    
    # Print initial sizes
    print("\nInitial mask sizes:")
    for obj_id in get_unique_object_ids(processed_masks):
        masks = get_all_object_masks(processed_masks, obj_id)
        sizes = [np.sum(mask) for mask in masks.values()]
        print(f"Object {obj_id}: {len(sizes)} frames, all size {sizes[0]} pixels")
    
    # Find all pairs that overlap
    overlapping_pairs = find_all_overlapping_pairs(processed_masks)
    print(f"\nFound {len(overlapping_pairs)} overlapping pairs")
    
    if not overlapping_pairs:
        return processed_masks
    
    # Create output dictionary
    final_masks = {frame: {obj_id: mask.copy() 
                          for obj_id, mask in objects.items()}
                  for frame, objects in processed_masks.items()}
    
    # Process each overlapping pair
    for obj1_id, obj2_id in overlapping_pairs:
        masks1 = get_all_object_masks(final_masks, obj1_id)
        masks2 = get_all_object_masks(final_masks, obj2_id)
        
        # Find all overlapping indices
        mask1_indices, mask2_indices = find_all_overlapping_indices(masks1, masks2)
        
        if mask1_indices:
            print(f"\nObjects {obj1_id} and {obj2_id}:")
            print(f"Removing {len(mask1_indices)} pixels from object {obj1_id}")
            print(f"Removing {len(mask2_indices)} pixels from object {obj2_id}")
            
            # Remove these indices from all frames
            for frame_num in final_masks:
                if obj1_id in final_masks[frame_num]:
                    final_masks[frame_num][obj1_id] = remove_indices_from_mask(
                        final_masks[frame_num][obj1_id], mask1_indices)
                if obj2_id in final_masks[frame_num]:
                    final_masks[frame_num][obj2_id] = remove_indices_from_mask(
                        final_masks[frame_num][obj2_id], mask2_indices)
    
    # Verify final sizes
    print("\nFinal sizes:")
    for obj_id in get_unique_object_ids(final_masks):
        sizes = [np.sum(mask) for frame_num, objects in final_masks.items() 
                if obj_id in objects
                for mask in [objects[obj_id]]]
        
        if len(set(sizes)) > 1:
            print(f"\nObject {obj_id} has inconsistent sizes:")
            size_counts = {}
            for size in sizes:
                if size not in size_counts:
                    size_counts[size] = 0
                size_counts[size] += 1
            
            for size, count in sorted(size_counts.items()):
                print(f"  {size} pixels: {count} frames")
            
            raise AssertionError(
                f"Inconsistent mask sizes for object {obj_id}: "
                f"range {min(sizes)}-{max(sizes)} pixels"
            )
        else:
            print(f"Object {obj_id}: all {len(sizes)} frames have {sizes[0]} pixels")
    
    print("\nOverlap removal complete")
    return final_masks


final_masks = remove_overlaps_from_masks(processed_masks)





def create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=0.99):
    """
    Create a video with mask overlays from a directory of images and a dictionary of masks.
    
    Args:
        image_dir (str): Directory containing the input images
        masks_dict (dict): Dictionary where keys are frame indices and values are
                          dictionaries of mask_id: mask pairs for that frame
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        alpha (float): Transparency of the mask overlay (0-1)
    """
    # Predefined colors for different masks
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow
        (0, 255, 128)   # Spring Green
    ]

    def overlay_masks(image, frame_masks, mask_colors, alpha):
        """Helper function to overlay masks on an image"""
        overlay = np.zeros_like(image)
        
        for mask_id, mask in frame_masks.items():
            # Convert to binary mask if needed
            if mask.dtype != bool:
                mask = mask > 0.5
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = mask_colors[mask_id]
            
            # Add to overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        # Combine with original image
        return cv2.addWeighted(image, 1, overlay, alpha, 0)

    # Get sorted list of image files
    image_files = sorted([f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {image_files[0]}")
    
    height, width, _ = first_image.shape

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create color mapping for mask IDs
    all_mask_ids = set()
    for masks in masks_dict.values():
        all_mask_ids.update(masks.keys())
    mask_colors = {mask_id: COLORS[i % len(COLORS)] 
                  for i, mask_id in enumerate(all_mask_ids)}

    # Process each frame
    for frame_idx, image_file in enumerate(image_files):
        try:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply masks if available for this frame
            if frame_idx in masks_dict:
                frame = overlay_masks(frame, masks_dict[frame_idx], 
                                   mask_colors, alpha)

            # Write frame
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")

# Example usage:
"""
image_dir = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"
masks_dict = final_masks
output_path = "largest_segments_video_nonoverlapping.mp4"

create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=1)
"""




frame = 9
mask = 4
mask = final_masks[frame][mask][0]

image_array = np.uint8(mask	 * 255)
image = Image.fromarray(image_array)
image.save('tst.png')







def save_cleaned_segments_to_h5(cleaned_segments, filename):
    # Create the output filename
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"/home/lilly/phd/ria/data_analyzed/aligned_segments/{name_without_ext}_alignedsegments.h5"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    with h5py.File(output_filename, 'w') as f:
        num_frames = len(cleaned_segments)
        f.attrs['num_frames'] = num_frames
        f.attrs['object_ids'] = list(cleaned_segments[0].keys())

        masks_group = f.create_group('masks')

        first_frame = list(cleaned_segments.keys())[0]
        first_obj = list(cleaned_segments[first_frame].keys())[0]
        mask_shape = cleaned_segments[first_frame][first_obj].shape

        for obj_id in cleaned_segments[first_frame].keys():
            masks_group.create_dataset(str(obj_id), (num_frames, *mask_shape), dtype=np.uint8)

        # Sort frame indices to ensure consistent ordering
        sorted_frames = sorted(cleaned_segments.keys())
        
        for idx, frame in enumerate(sorted_frames):
            frame_data = cleaned_segments[frame]
            for obj_id, mask in frame_data.items():
                masks_group[str(obj_id)][idx] = mask.astype(np.uint8) * 255
            
            # Debug print
            print(f"Saving frame {frame} at index {idx}")

    print(f"Cleaned segments saved to {output_filename}")
    return output_filename

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            
            # Debug print
            print(f"Loading frame {frame_idx}")
    
    print(f"Cleaned segments loaded from {filename}")
    return cleaned_segments

def compare_cleaned_segments(original, loaded):
    assert len(original) == len(loaded), "Number of frames doesn't match"
    
    # Sort frame indices for both original and loaded data
    original_frames = sorted(original.keys())
    loaded_frames = sorted(loaded.keys())
    
    for orig_frame, loaded_frame in zip(original_frames, loaded_frames):
        assert original[orig_frame].keys() == loaded[loaded_frame].keys(), f"Object IDs don't match in frame {orig_frame}"
        
        for obj_id in original[orig_frame]:
            original_mask = original[orig_frame][obj_id]
            loaded_mask = loaded[loaded_frame][obj_id]
            
            if not np.array_equal(original_mask, loaded_mask):
                print(f"Mismatch found in original frame {orig_frame}, loaded frame {loaded_frame}, object {obj_id}")
                print(f"Original mask shape: {original_mask.shape}")
                print(f"Loaded mask shape: {loaded_mask.shape}")
                print(f"Original mask dtype: {original_mask.dtype}")
                print(f"Loaded mask dtype: {loaded_mask.dtype}")
                print(f"Number of True values in original: {np.sum(original_mask)}")
                print(f"Number of True values in loaded: {np.sum(loaded_mask)}")
                
                diff_positions = np.where(original_mask != loaded_mask)
                print(f"Number of differing positions: {len(diff_positions[0])}")
                
                if len(diff_positions[0]) > 0:
                    print("First 5 differing positions:")
                    for i in range(min(5, len(diff_positions[0]))):
                        pos = tuple(dim[i] for dim in diff_positions)
                        print(f"  Position {pos}: Original = {original_mask[pos]}, Loaded = {loaded_mask[pos]}")
                
                return False
    
    print("All masks match exactly!")
    return True

# Example usage:
filename = '/home/lilly/phd/ria/data_analyzed/ria_segmentation/AG-MMH99_10s_20190306_02_crop_riasegmentation.h5'

# Save the cleaned segments
output_filename = save_cleaned_segments_to_h5(final_masks, filename)

# Load the cleaned segments
output_filename = "/home/lilly/phd/ria/data_analyzed/aligned_segments/AG-MMH99_10s_20190306_02_crop_riasegmentation_alignedsegments.h5"
loaded_segments = load_cleaned_segments_from_h5(output_filename)

# Perform detailed comparison
compare_cleaned_segments(final_masks, loaded_segments)




####Compare segmentations
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

#Get mean and std of brightness values from masked regions using average mask
def extract_brightness_values(aligned_images: List[np.ndarray],
                            masks_dict: Dict[int, Dict[int, np.ndarray]], 
                            object_id: int) -> pd.DataFrame:
    """
    Extracts mean and standard deviation of brightness values from masked regions.
    
    Parameters:
        aligned_images: List of grayscale images (numpy arrays)
        masks_dict: Dictionary where keys are frame indices and values are
                   dictionaries of mask_id: mask pairs for that frame
        object_id: ID of the object whose mask should be used
    
    Returns:
        DataFrame containing frame number, mean and std brightness values for each frame
    """
    data = []
    
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            # Get mask for this frame and object
            mask = masks_dict[frame_idx][object_id][0]
            
            # Extract pixels within mask
            masked_pixels = image[mask]
            
            # Calculate statistics
            mean_val = np.mean(masked_pixels)
            std_val = np.std(masked_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_brightness': mean_val,
                'std_brightness': std_val
            })
    
    return pd.DataFrame(data)


def extract_top_percent_brightness(aligned_images: List[np.ndarray],
                                 masks_dict: Dict[int, Dict[int, np.ndarray]], 
                                 object_id: int,
                                 percent: float) -> pd.DataFrame:
    """
    Extracts mean brightness of the top X% brightest pixels from masked regions.
    
    Parameters:
        aligned_images: List of grayscale images (numpy arrays)
        masks_dict: Dictionary where keys are frame indices and values are
                   dictionaries of mask_id: mask pairs for that frame
        object_id: ID of the object whose mask should be used
        percent: Percentage of brightest pixels to consider (0-100)
        
    Returns:
        DataFrame containing frame number and mean brightness of top X% pixels for each frame
    """
    # Validate percentage input
    if not 0 < percent <= 100:
        raise ValueError("Percentage must be between 0 and 100")
        
    data = []
    
    for frame_idx, image in enumerate(aligned_images):
        if frame_idx in masks_dict and object_id in masks_dict[frame_idx]:
            # Get mask for this frame and object
            mask = masks_dict[frame_idx][object_id][0]
            
            # Extract pixels within mask
            masked_pixels = image[mask]
            
            # Calculate number of pixels to use based on percentage
            n_pixels = int(round(len(masked_pixels) * (percent / 100)))
            # Ensure at least 1 pixel is used
            n_pixels = max(1, n_pixels)
            
            # Get top percentage of brightest pixels
            top_n_pixels = np.sort(masked_pixels)[-n_pixels:]
            
            # Calculate mean of top pixels
            mean_top_percent = np.mean(top_n_pixels)
            
            data.append({
                'frame': frame_idx,
                'mean_top_percent_brightness': mean_top_percent,
                'n_pixels_used': n_pixels,
                'total_pixels': len(masked_pixels),
                'percent_used': percent
            })
    
    return pd.DataFrame(data)


img_dir = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"

# Get list of image files sorted by frame number
image_files = sorted(os.listdir(img_dir))

# Load and preprocess images
aligned_images = []
for img_file in image_files:
    if img_file.endswith('.jpg'):
        img_path = os.path.join(img_dir, img_file)
        # Read as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            aligned_images.append(img)

final_masks = load_cleaned_segments_from_h5(output_filename)
brightness_loop = extract_brightness_values(aligned_images, final_masks, 4)
brightness_nrd = extract_brightness_values(aligned_images, final_masks, 2)
brightness_nrv = extract_brightness_values(aligned_images, final_masks, 3)

loop_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=10)
nrd_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=10)
nrv_top_10percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=10)

loop_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=50)
nrd_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=50)
nrv_top_50percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=50)

loop_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 4, percent=25)
nrd_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 2, percent=25)
nrv_top_25percent = extract_top_percent_brightness(aligned_images, final_masks, 3, percent=25)

# Load all Fiji data files
fiji_dir = "/home/lilly/phd/ria/MMH99_10s_20190306_02"
fiji_files = [f for f in os.listdir(fiji_dir) if f.endswith('.xlsx')]

all_fiji_data = []
for file in fiji_files:
    df = pd.read_excel(os.path.join(fiji_dir, file))
    all_fiji_data.append(df)

# Combine all Fiji data
fiji_combined = pd.concat(all_fiji_data)

# Group by frame and calculate mean and std for each segment
fiji_stats = fiji_combined.groupby('Frame').agg({
    'loop': ['mean', 'std'],
    'nrD': ['mean', 'std'],
    'nrV': ['mean', 'std']
}).reset_index()

# Normalize means and calculate normalized std and 95% confidence intervals
# For each segment, calculate std relative to the range of the mean
for segment in ['loop', 'nrD', 'nrV']:
    mean_range = fiji_stats[(segment, 'mean')].max() - fiji_stats[(segment, 'mean')].min()
    fiji_stats[(segment, 'std_norm')] = fiji_stats[(segment, 'std')] / mean_range
    
    # Calculate 95% confidence interval (1.96 * std error)
    n = len(fiji_combined[fiji_combined['Frame'] == fiji_stats['Frame'][0]][segment])  # samples per frame
    fiji_stats[(segment, 'ci_norm')] = (1.96 * fiji_stats[(segment, 'std')] / np.sqrt(n)) / mean_range
    fiji_stats[(segment, 'mean_norm')] = normalize(fiji_stats[(segment, 'mean')])

# Create plot with ribbons
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(40, 32))

# Define colors
loop_color = 'blue'
nrv_color = 'red'
nrd_color = 'green'
fiji_color = 'purple'
alpha_ribbon = 0.3

# Top subplot - loop
ax1.plot(brightness_loop['frame'].to_numpy(), normalize(brightness_loop['mean_brightness']).to_numpy(),
         color=loop_color, linewidth=2, label='Aligned Loop')
ax1.plot(loop_top_10percent['frame'].to_numpy(), normalize(loop_top_10percent['mean_top_percent_brightness']).to_numpy(),
         color='cyan', linestyle='-', linewidth=1, label='Top 10%')
ax1.plot(loop_top_25percent['frame'].to_numpy(), normalize(loop_top_25percent['mean_top_percent_brightness']).to_numpy(),
         color='magenta', linestyle='-', linewidth=1, label='Top 25%')
ax1.plot(loop_top_50percent['frame'].to_numpy(), normalize(loop_top_50percent['mean_top_percent_brightness']).to_numpy(),
         color='yellow', linestyle='-', linewidth=1, label='Top 50%')

ax1.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('loop', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji Loop Mean')
ax1.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('loop', 'mean_norm')].to_numpy() - fiji_stats[('loop', 'std_norm')].to_numpy(),
                 fiji_stats[('loop', 'mean_norm')].to_numpy() + fiji_stats[('loop', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

ax1.set_xlabel('Frame Number', fontsize=14)
ax1.set_ylabel('Normalized Brightness', fontsize=14)
ax1.set_title('Normalized Brightness Over Frames (Loop)', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(np.arange(0, len(brightness_loop), 10))
ax1.tick_params(axis='both', labelsize=12)
ax1.legend(fontsize=12)
ax1.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Middle subplot - nrV
ax2.plot(brightness_nrv['frame'].to_numpy(), normalize(brightness_nrv['mean_brightness']).to_numpy(),
         color=nrv_color, linewidth=2, label='Aligned nrV')
ax2.plot(nrv_top_10percent['frame'].to_numpy(), normalize(nrv_top_10percent['mean_top_percent_brightness']).to_numpy(),
         color='cyan', linestyle='-', linewidth=1, label='Top 10%')
ax2.plot(nrv_top_25percent['frame'].to_numpy(), normalize(nrv_top_25percent['mean_top_percent_brightness']).to_numpy(),
         color='magenta', linestyle='-', linewidth=1, label='Top 25%')
ax2.plot(nrv_top_50percent['frame'].to_numpy(), normalize(nrv_top_50percent['mean_top_percent_brightness']).to_numpy(),
         color='yellow', linestyle='-', linewidth=1, label='Top 50%')

ax2.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('nrV', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji nrV Mean')
ax2.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('nrV', 'mean_norm')].to_numpy() - fiji_stats[('nrV', 'std_norm')].to_numpy(),
                 fiji_stats[('nrV', 'mean_norm')].to_numpy() + fiji_stats[('nrV', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

ax2.set_xlabel('Frame Number', fontsize=14)
ax2.set_ylabel('Normalized Brightness', fontsize=14)
ax2.set_title('Normalized Brightness Over Frames (nrV)', fontsize=18)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(np.arange(0, len(brightness_nrv), 10))
ax2.tick_params(axis='both', labelsize=12)
ax2.legend(fontsize=12)
ax2.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Bottom subplot - nrD
ax3.plot(brightness_nrd['frame'].to_numpy(), normalize(brightness_nrd['mean_brightness']).to_numpy(),
         color=nrd_color, linewidth=2, label='Aligned nrD')
ax3.plot(nrd_top_10percent['frame'].to_numpy(), normalize(nrd_top_10percent['mean_top_percent_brightness']).to_numpy(),
         color='cyan', linestyle='-', linewidth=1, label='Top 10%')
ax3.plot(nrd_top_25percent['frame'].to_numpy(), normalize(nrd_top_25percent['mean_top_percent_brightness']).to_numpy(),
         color='magenta', linestyle='-', linewidth=1, label='Top 25%')
ax3.plot(nrd_top_50percent['frame'].to_numpy(), normalize(nrd_top_50percent['mean_top_percent_brightness']).to_numpy(),
         color='yellow', linestyle='-', linewidth=1, label='Top 50%')

ax3.plot(fiji_stats['Frame'].to_numpy(), fiji_stats[('nrD', 'mean_norm')].to_numpy(),
         color=fiji_color, linewidth=2, label='Fiji nrD Mean')
ax3.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_stats[('nrD', 'mean_norm')].to_numpy() - fiji_stats[('nrD', 'std_norm')].to_numpy(),
                 fiji_stats[('nrD', 'mean_norm')].to_numpy() + fiji_stats[('nrD', 'std_norm')].to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

ax3.set_xlabel('Frame Number', fontsize=14)
ax3.set_ylabel('Normalized Brightness', fontsize=14)
ax3.set_title('Normalized Brightness Over Frames (nrD)', fontsize=18)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_xticks(np.arange(0, len(brightness_nrd), 10))
ax3.tick_params(axis='both', labelsize=12)
ax3.legend(fontsize=12)
ax3.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax3.axvline(x=500, color='k', linestyle='--', alpha=0.5)

# Fourth subplot - sum of all segments
sum_aligned = brightness_loop['mean_brightness'].to_numpy() + \
             brightness_nrv['mean_brightness'].to_numpy() + \
             brightness_nrd['mean_brightness'].to_numpy()
sum_aligned_normalized = normalize(sum_aligned)

# Sum of top percentages
sum_10percent = loop_top_10percent['mean_top_percent_brightness'].to_numpy() + \
                nrv_top_10percent['mean_top_percent_brightness'].to_numpy() + \
                nrd_top_10percent['mean_top_percent_brightness'].to_numpy()
sum_25percent = loop_top_25percent['mean_top_percent_brightness'].to_numpy() + \
                nrv_top_25percent['mean_top_percent_brightness'].to_numpy() + \
                nrd_top_25percent['mean_top_percent_brightness'].to_numpy()
sum_50percent = loop_top_50percent['mean_top_percent_brightness'].to_numpy() + \
                nrv_top_50percent['mean_top_percent_brightness'].to_numpy() + \
                nrd_top_50percent['mean_top_percent_brightness'].to_numpy()

# For Fiji data, sum the means first, then normalize
fiji_sum_mean = fiji_stats[('loop', 'mean')] + fiji_stats[('nrV', 'mean')] + fiji_stats[('nrD', 'mean')]
# Calculate normalized std for sum using error propagation
fiji_sum_std = np.sqrt(
    (fiji_stats[('loop', 'std_norm')])**2 + 
    (fiji_stats[('nrV', 'std_norm')])**2 + 
    (fiji_stats[('nrD', 'std_norm')])**2
)
fiji_sum_normalized = normalize(fiji_sum_mean)

ax4.plot(brightness_loop['frame'].to_numpy(), sum_aligned_normalized,
         color='black', linewidth=2, label='Sum of Aligned Segments')
ax4.plot(brightness_loop['frame'].to_numpy(), normalize(sum_10percent),
         color='cyan', linestyle='-', linewidth=1, label='Sum Top 10%')
ax4.plot(brightness_loop['frame'].to_numpy(), normalize(sum_25percent),
         color='magenta', linestyle='-', linewidth=1, label='Sum Top 25%')
ax4.plot(brightness_loop['frame'].to_numpy(), normalize(sum_50percent),
         color='yellow', linestyle='-', linewidth=1, label='Sum Top 50%')
ax4.plot(fiji_stats['Frame'].to_numpy(), fiji_sum_normalized.to_numpy(),
         color=fiji_color, linewidth=2, label='Sum of Fiji Segments Mean')
ax4.fill_between(fiji_stats['Frame'].to_numpy(),
                 fiji_sum_normalized.to_numpy() - fiji_sum_std.to_numpy(),
                 fiji_sum_normalized.to_numpy() + fiji_sum_std.to_numpy(),
                 color=fiji_color, alpha=alpha_ribbon)

ax4.set_xlabel('Frame Number', fontsize=14)
ax4.set_ylabel('Normalized Sum of Brightness', fontsize=14)
ax4.set_title('Normalized Sum of All Segments Over Frames', fontsize=18)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.set_xticks(np.arange(0, len(brightness_loop), 10))
ax4.tick_params(axis='both', labelsize=12)
ax4.legend(fontsize=12)
ax4.axvline(x=100, color='k', linestyle='--', alpha=0.5)
ax4.axvline(x=300, color='k', linestyle='--', alpha=0.5)
ax4.axvline(x=500, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('normalized_brightness_over_frames_ria9902_largest.png', dpi=300, bbox_inches='tight')
plt.close()