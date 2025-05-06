import h5py
import numpy as np
import os
import cv2
from scipy import ndimage
from PIL import Image
import random
from skimage.measure import label
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Set
from scipy.ndimage import center_of_mass
from sklearn.decomposition import PCA
import itertools

def load_video_segments_from_h5(filename):
    video_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        # Convert 'None' back to None and others to int
        object_ids = [None if obj_id == 'None' else int(obj_id) for obj_id in object_ids]
        
        for i in range(num_frames):
            frame_idx = num_frames - 1 - i  # Reverse the frame index
            video_segments[frame_idx] = {}
            for obj_id in object_ids:
                obj_id_str = str(obj_id) if obj_id is not None else 'None'
                video_segments[frame_idx][obj_id] = f[f'masks/{obj_id_str}'][i]
    
    return video_segments


def get_random_unprocessed_video(segmented_videos_dir, cleaned_aligned_segments_dir):
    all_videos = [os.path.splitext(d)[0] for d in os.listdir(segmented_videos_dir)]
    unprocessed_videos = [
        video for video in all_videos
        if not os.path.exists(os.path.join(cleaned_aligned_segments_dir, video + "_cleanedalignedsegments.h5"))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(segmented_videos_dir, random.choice(unprocessed_videos) + ".h5")


segmented_videos_dir = '/home/lilly/phd/ria/data_analyzed/AG_WT/ria_segmentation'
cleaned_aligned_segments_dir = '/home/lilly/phd/ria/data_analyzed/AG_WT/cleaned_aligned_segments'

ria_segments = get_random_unprocessed_video(segmented_videos_dir, cleaned_aligned_segments_dir)
print(f"Processing video: {ria_segments}")

#full_path = '/home/lilly/phd/ria/data_analyzed/ria_segmentation/AG-MMH99_10s_20190306_02_crop_riasegmentation.h5'
loaded_video_segments = load_video_segments_from_h5(ria_segments)


###Fill missing masks
def fill_missing_masks(video_segments: Dict[int, Dict[int, np.ndarray]], required_ids: List[int] = [2, 3, 4]) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Fill in missing masks by interpolating between the nearest available masks.
    
    Args:
        video_segments: Dictionary of frame_number -> {object_id -> mask}
        required_ids: List of object IDs that should be present in every frame
    
    Returns:
        Dictionary with the same structure but with missing masks filled in
    """
    print("Checking for missing or empty masks...")
    
    # Create a copy to avoid modifying the original
    filled_segments = {frame: masks.copy() for frame, masks in video_segments.items()}
    
    # Get sorted frame numbers
    frames = sorted(video_segments.keys())
    
    # Track if any changes were made
    any_changes_made = False
    
    # For each required object ID
    for obj_id in required_ids:
        # Find frames with missing masks (including empty masks)
        missing_frames = [
            frame for frame in frames 
            if (obj_id not in video_segments[frame] or 
                video_segments[frame][obj_id] is None or 
                np.sum(video_segments[frame][obj_id]) == 0)  # Check if mask is empty
        ]
        
        if not missing_frames:
            continue
            
        any_changes_made = True
        print(f"Found {len(missing_frames)} frames with missing masks for object {obj_id}")
        
        # Process each missing frame
        for missing_frame in missing_frames:
            # Find nearest previous frame with non-empty mask
            prev_frame = None
            prev_mask = None
            for frame in reversed(frames[:frames.index(missing_frame)]):
                if (obj_id in video_segments[frame] and 
                    video_segments[frame][obj_id] is not None and 
                    np.sum(video_segments[frame][obj_id]) > 0):
                    prev_frame = frame
                    prev_mask = video_segments[frame][obj_id]
                    break
            
            # Find nearest next frame with non-empty mask
            next_frame = None
            next_mask = None
            for frame in frames[frames.index(missing_frame) + 1:]:
                if (obj_id in video_segments[frame] and 
                    video_segments[frame][obj_id] is not None and 
                    np.sum(video_segments[frame][obj_id]) > 0):
                    next_frame = frame
                    next_mask = video_segments[frame][obj_id]
                    break
            
            # Interpolate mask based on available neighboring masks
            if prev_mask is not None and next_mask is not None:
                # Calculate weights based on distance
                total_dist = next_frame - prev_frame
                weight_next = (missing_frame - prev_frame) / total_dist
                weight_prev = (next_frame - missing_frame) / total_dist
                
                # Interpolate between masks
                interpolated_mask = (prev_mask * weight_prev + next_mask * weight_next) > 0.5
                filled_segments[missing_frame][obj_id] = interpolated_mask
                
                print(f"Frame {missing_frame}: Interpolated mask {obj_id} using frames {prev_frame} and {next_frame}")
                
            elif prev_mask is not None:
                # If only previous mask is available, use it
                filled_segments[missing_frame][obj_id] = prev_mask
                print(f"Frame {missing_frame}: Used previous mask {obj_id} from frame {prev_frame}")
                
            elif next_mask is not None:
                # If only next mask is available, use it
                filled_segments[missing_frame][obj_id] = next_mask
                print(f"Frame {missing_frame}: Used next mask {obj_id} from frame {next_frame}")
                
            else:
                print(f"Warning: Could not fill mask for object {obj_id} in frame {missing_frame} - no neighboring masks available")
    
    if not any_changes_made:
        print("\nNo missing or empty masks found. All masks are present and non-empty!")
        return filled_segments
    
    # Verify all required masks are present and non-empty
    missing_after_fill = []
    for frame in frames:
        for obj_id in required_ids:
            if (obj_id not in filled_segments[frame] or 
                filled_segments[frame][obj_id] is None or 
                np.sum(filled_segments[frame][obj_id]) == 0):  # Added empty mask check
                missing_after_fill.append((frame, obj_id))
    
    if missing_after_fill:
        print("\nWarning: Some masks could not be filled:")
        for frame, obj_id in missing_after_fill:
            print(f"Frame {frame}, Object {obj_id}")
    else:
        print("\nAll missing masks have been filled successfully!")
    
    return filled_segments

# Add this line after loading the video segments and before starting the processing
filled_video_segments = fill_missing_masks(loaded_video_segments)



###Clean size/distance. Remove pixels in mask4 that are within min_distance pixels from any pixel in mask3.
def find_connected_components(mask):
    # Remove the single-dimensional entries
    mask = np.squeeze(mask)
    
    # Use scipy's label function to find connected components
    labeled_array, num_features = ndimage.label(mask)
    
    return labeled_array, num_features

def filter_masks(segments):
    """
    Filter out masks that are not 2, 3, or 4.
    """
    
    filtered_segments = {}
    for frame, masks in segments.items():
        filtered_segments[frame] = {mask_id: mask for mask_id, mask in masks.items() if mask_id in [2, 3, 4]}
    return filtered_segments

def remove_small_components(mask, min_size=4):
    labeled_array, num_features = find_connected_components(mask)
    
    # First pass: identify components larger than min_size
    cleaned_mask = np.zeros_like(mask)
    valid_components = []
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        size = np.sum(component)
        if size > min_size:
            valid_components.append((size, component))
            cleaned_mask = np.logical_or(cleaned_mask, component)
    
    # Second pass: if there are multiple components, keep only the largest
    if len(valid_components) > 1:
        # Sort components by size (largest first)
        valid_components.sort(key=lambda x: x[0], reverse=True)
        # Keep only the largest component
        cleaned_mask = valid_components[0][1]
    
    return cleaned_mask

def filter_by_distance(mask3_orig: np.ndarray, mask4_orig: np.ndarray, min_distance: float = 5, min_pixels: int = 4) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Filter out pixels in mask3 and mask4 that are within min_distance pixels from each other.
    If all pixels in a mask would be filtered out or result in fewer than min_pixels,
    keep the min_pixels furthest pixels for that mask.

    Parameters:
    -----------
    mask3_orig : numpy.ndarray
        Binary mask for segment 3
    mask4_orig : numpy.ndarray
        Binary mask for segment 4
    min_distance : float
        Minimum Euclidean distance in pixels
    min_pixels : int
        Minimum number of pixels to preserve from each mask if they would become too small

    Returns:
    --------
    Tuple[numpy.ndarray, numpy.ndarray, int, int]
        - Filtered mask3
        - Filtered mask4
        - Number of pixels removed from mask3
        - Number of pixels removed from mask4
    """
    # Ensure masks are 2D and boolean copies
    m3 = np.squeeze(mask3_orig).astype(bool)
    m4 = np.squeeze(mask4_orig).astype(bool)

    if m3.ndim != 2 or m4.ndim != 2:
        raise ValueError(f"Masks must be 2D after squeezing. Got shapes: m3={m3.shape}, m4={m4.shape}")

    initial_pixels_m3 = np.sum(m3)
    initial_pixels_m4 = np.sum(m4)

    # Initialize filtered masks as copies of originals
    filtered_m3 = m3.copy()
    filtered_m4 = m4.copy()

    # --- Filter m4 based on proximity to original m3 ---
    if initial_pixels_m3 > 0 and initial_pixels_m4 > 0:
        m3_uint8 = np.ascontiguousarray(m3.astype(np.uint8))
        dist_transform_from_m3 = cv2.distanceTransform((1 - m3_uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # Initial filtering of m4: remove pixels in original m4 too close to original m3
        m4_candidate = np.logical_and(m4, dist_transform_from_m3 >= min_distance)
        
        # If m4_candidate is too small, try to keep min_pixels furthest pixels from original m4 (relative to m3)
        if np.sum(m4_candidate) < min_pixels:
            m4_distances_from_m3 = dist_transform_from_m3[m4] # Distances of original m4 pixels from m3
            if len(m4_distances_from_m3) > 0: # If original m4 had any pixels
                distances_sorted = np.sort(m4_distances_from_m3)[::-1]
                # Threshold to keep min_pixels or all if fewer than min_pixels exist in original m4
                threshold_idx = min(min_pixels - 1, len(distances_sorted) - 1)
                # threshold_idx will be >= 0 if len(distances_sorted) > 0
                distance_threshold_val = distances_sorted[threshold_idx]
                # Final m4 has pixels from original m4 that are >= this distance_threshold_val from m3
                filtered_m4 = np.logical_and(m4, dist_transform_from_m3 >= distance_threshold_val)
            else: # Original m4 was empty, so filtered_m4 remains empty
                filtered_m4 = np.zeros_like(m4, dtype=bool)
        else:
            filtered_m4 = m4_candidate
    elif initial_pixels_m4 == 0 : # m4 is already empty
        filtered_m4 = m4.copy() # no change, remains empty
    # else m3 is empty, so m4 is not filtered based on m3, filtered_m4 remains m4.copy()

    # --- Filter m3 based on proximity to original m4 ---
    if initial_pixels_m4 > 0 and initial_pixels_m3 > 0:
        m4_uint8 = np.ascontiguousarray(m4.astype(np.uint8)) # Original m4
        dist_transform_from_m4 = cv2.distanceTransform((1 - m4_uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Initial filtering of m3: remove pixels in original m3 too close to original m4
        m3_candidate = np.logical_and(m3, dist_transform_from_m4 >= min_distance)

        # If m3_candidate is too small, try to keep min_pixels furthest pixels from original m3 (relative to m4)
        if np.sum(m3_candidate) < min_pixels:
            m3_distances_from_m4 = dist_transform_from_m4[m3] # Distances of original m3 pixels from m4
            if len(m3_distances_from_m4) > 0: # If original m3 had any pixels
                distances_sorted = np.sort(m3_distances_from_m4)[::-1]
                threshold_idx = min(min_pixels - 1, len(distances_sorted) - 1)
                distance_threshold_val = distances_sorted[threshold_idx]
                filtered_m3 = np.logical_and(m3, dist_transform_from_m4 >= distance_threshold_val)
            else: # Original m3 was empty
                filtered_m3 = np.zeros_like(m3, dtype=bool)
        else:
            filtered_m3 = m3_candidate
    elif initial_pixels_m3 == 0: # m3 is already empty
        filtered_m3 = m3.copy() # no change, remains empty
    # else m4 is empty, so m3 is not filtered based on m4, filtered_m3 remains m3.copy()


    pixels_removed_m3_count = initial_pixels_m3 - np.sum(filtered_m3)
    pixels_removed_m4_count = initial_pixels_m4 - np.sum(filtered_m4)

    # Ensure output has same shape as input (2D squeezed versions)
    if filtered_m3.shape != m3.shape:
        raise ValueError(f"Output mask3 shape {filtered_m3.shape} doesn't match input shape {m3.shape}")
    if filtered_m4.shape != m4.shape:
        raise ValueError(f"Output mask4 shape {filtered_m4.shape} doesn't match input shape {m4.shape}")
        
    return filtered_m3, filtered_m4, pixels_removed_m3_count, pixels_removed_m4_count

def distance_clean_segments(segments, min_size=4, min_distance=5, min_pixels=4):
    """
    Master function to clean all segments with multiple cleaning operations.
    
    Parameters:
    -----------
    segments : dict
        Dictionary of frame masks
    min_size : int
        Minimum size for connected components
    min_distance : float
        Minimum distance between mask3 and mask4 pixels
    min_pixels : int
        Minimum number of pixels to preserve from mask3 and mask4 during distance filtering
        
    Returns:
    --------
    dict
        Cleaned segments
    """
    # Initialize tracking dictionaries
    modifications = {
        'small_components_modified_frames': {2: [], 3: [], 4: []},
        'small_components_pixels_removed': {2: {}, 3: {}, 4: {}}, # frame: count
        'overlap_removal_modified_frames': { (2,3): [], (2,4): [], (3,4): [] }, # (mask_id1, mask_id2): [frames]
        'overlap_pixels_removed': { (2,3): {}, (2,4): {}, (3,4): {} }, # (mask_id1, mask_id2): {frame: count}
        'distance_filter_modified_frames': [], 
        'distance_filter_pixels_removed': {3: {}, 4: {}}  # frame: count
    }
    
    # Step 1: Filter to keep only masks 2, 3, and 4
    cleaned_segments = filter_masks(segments)
    
    # Step 2: Remove small components
    for frame, masks in cleaned_segments.items():
        for mask_id, mask in masks.items():
            # original_pixels = np.sum(mask) # Not needed here with current modifications structure
            cleaned_mask = remove_small_components(mask, min_size)
            pixels_removed = np.sum(mask) - np.sum(cleaned_mask)
            
            if pixels_removed > 0:
                modifications['small_components_modified_frames'][mask_id].append(frame)
                modifications['small_components_pixels_removed'][mask_id][frame] = \
                    modifications['small_components_pixels_removed'][mask_id].get(frame, 0) + pixels_removed
                
            cleaned_segments[frame][mask_id] = cleaned_mask

    # Step 3: Remove overlapping pixels between mask pairs (2&3, 2&4, 3&4)
    for frame, masks in cleaned_segments.items():
        # Ensure masks are squeezed to 2D for overlap operations
        current_masks = {mid: np.squeeze(m) for mid, m in masks.items() if m is not None}

        mask_ids = sorted([mid for mid in current_masks.keys() if np.sum(current_masks[mid]) > 0]) # Process only non-empty masks

        for i in range(len(mask_ids)):
            for j in range(i + 1, len(mask_ids)):
                id1, id2 = mask_ids[i], mask_ids[j]
                
                # Ensure we are working with copies if they exist in cleaned_segments for this frame
                m1 = current_masks.get(id1)
                m2 = current_masks.get(id2)

                if m1 is None or m2 is None: # Should not happen due to mask_ids logic but defensive
                    continue

                overlap = np.logical_and(m1, m2)
                overlap_count = np.sum(overlap)

                if overlap_count > 0:
                    pair_key = tuple(sorted((id1, id2))) # Consistent key for modifications
                    
                    # Remove overlap: pixels in overlap are removed from both masks
                    m1_cleaned = np.logical_and(m1, np.logical_not(overlap))
                    m2_cleaned = np.logical_and(m2, np.logical_not(overlap))
                    
                    # Update the masks in cleaned_segments (ensure they go back to 3D if that's the convention)
                    # Assuming original masks in `segments` could be 3D (1,H,W)
                    # and filter_masks and remove_small_components preserve this or work with squeezed versions.
                    # Let's ensure they are 2D for now and handle 3D conversion at the end if necessary.
                    cleaned_segments[frame][id1] = m1_cleaned
                    cleaned_segments[frame][id2] = m2_cleaned
                    current_masks[id1] = m1_cleaned # Update local copy for subsequent overlaps in same frame
                    current_masks[id2] = m2_cleaned

                    if frame not in modifications['overlap_removal_modified_frames'].get(pair_key, []):
                        modifications['overlap_removal_modified_frames'].setdefault(pair_key, []).append(frame)
                    
                    modifications['overlap_pixels_removed'].setdefault(pair_key, {})[frame] = \
                        modifications['overlap_pixels_removed'][pair_key].get(frame, 0) + overlap_count
    
    # Step 4: Filter mask3 and mask4 based on distance from each other
    for frame, masks in cleaned_segments.items():
        if 3 in masks and 4 in masks:
            # Ensure masks are 2D for filter_by_distance
            original_mask3 = np.squeeze(masks[3]) if masks[3] is not None else np.zeros_like(next(iter(masks.values()))[0], dtype=bool)
            original_mask4 = np.squeeze(masks[4]) if masks[4] is not None else np.zeros_like(next(iter(masks.values()))[0], dtype=bool)
            
            if masks[3] is None: # If mask was entirely removed by overlap or was initially empty
                original_mask3_shape = np.squeeze(segments[frame].get(3, np.zeros((1,10,10)))).shape # Get a reference shape
                original_mask3 = np.zeros(original_mask3_shape, dtype=bool)
            if masks[4] is None:
                original_mask4_shape = np.squeeze(segments[frame].get(4, np.zeros((1,10,10)))).shape
                original_mask4 = np.zeros(original_mask4_shape, dtype=bool)


            filtered_m3, filtered_m4, removed_m3, removed_m4 = filter_by_distance(
                original_mask3,
                original_mask4,
                min_distance=min_distance,
                min_pixels=min_pixels
            )
            
            if removed_m3 > 0:
                if frame not in modifications['distance_filter_modified_frames']:
                    modifications['distance_filter_modified_frames'].append(frame)
                modifications['distance_filter_pixels_removed'][3][frame] = \
                    modifications['distance_filter_pixels_removed'][3].get(frame, 0) + removed_m3
            # Update mask 3 if it exists or if it was modified (it will be 2D from filter_by_distance)
            cleaned_segments[frame][3] = filtered_m3
            
            if removed_m4 > 0:
                if frame not in modifications['distance_filter_modified_frames']:
                    modifications['distance_filter_modified_frames'].append(frame)
                modifications['distance_filter_pixels_removed'][4][frame] = \
                    modifications['distance_filter_pixels_removed'][4].get(frame, 0) + removed_m4
            # Update mask 4 if it exists or if it was modified
            cleaned_segments[frame][4] = filtered_m4
    
    # Print summary of modifications
    print("\nSegment Cleaning Summary:")
    print("\nSmall Components Removed:")
    for mask_id in [2, 3, 4]:
        modified_frames_sc = sorted(list(set(modifications['small_components_modified_frames'][mask_id])))
        if modified_frames_sc:
            print(f"  Mask {mask_id}:")
            print(f"    Modified frames: {modified_frames_sc}")
            total_pixels_sc = sum(modifications['small_components_pixels_removed'][mask_id].get(f, 0) 
                                 for f in modified_frames_sc)
            avg_pixels_sc = total_pixels_sc / len(modified_frames_sc) if len(modified_frames_sc) > 0 else 0
            print(f"    Average pixels removed per modified frame: {avg_pixels_sc:.2f}")
        else:
            print(f"  Mask {mask_id}: No frames modified by small component removal")

    print("\nOverlap Removal:")
    any_overlap_removed = False
    for pair_key, modified_frames_overlap in modifications['overlap_removal_modified_frames'].items():
        if modified_frames_overlap:
            any_overlap_removed = True
            sorted_frames_overlap = sorted(list(set(modified_frames_overlap)))
            pixels_removed_map = modifications['overlap_pixels_removed'].get(pair_key, {})
            total_pixels_overlap = sum(pixels_removed_map.get(f, 0) for f in sorted_frames_overlap)
            avg_pixels_overlap = total_pixels_overlap / len(sorted_frames_overlap) if len(sorted_frames_overlap) > 0 else 0
            print(f"  Masks {pair_key[0]} & {pair_key[1]}:")
            print(f"    Frames with overlap removed: {sorted_frames_overlap}")
            print(f"    Average overlapping pixels removed per modified frame: {avg_pixels_overlap:.2f}")
    if not any_overlap_removed:
        print("  No overlaps removed between any mask pairs.")
            
    print("\nDistance Filtering (Masks 3 and 4):")
    distance_modified_overall_frames = sorted(list(set(modifications['distance_filter_modified_frames'])))
    if distance_modified_overall_frames:
        print(f"  Frames with any distance filtering: {distance_modified_overall_frames}")
        for mask_id_dist in [3, 4]:
            pixels_removed_by_dist_map = modifications['distance_filter_pixels_removed'][mask_id_dist]
            
            actual_modified_frames_for_mask_dist = sorted([
                f for f in distance_modified_overall_frames 
                if f in pixels_removed_by_dist_map and pixels_removed_by_dist_map[f] > 0
            ])
            
            if actual_modified_frames_for_mask_dist:
                total_pixels_dist = sum(pixels_removed_by_dist_map.get(f, 0) for f in actual_modified_frames_for_mask_dist)
                avg_pixels_dist = total_pixels_dist / len(actual_modified_frames_for_mask_dist) if len(actual_modified_frames_for_mask_dist) > 0 else 0
                print(f"  Mask {mask_id_dist}:")
                print(f"    Modified frames by distance: {actual_modified_frames_for_mask_dist}")
                print(f"    Average pixels removed per modified frame: {avg_pixels_dist:.2f}")
            else:
                 print(f"  Mask {mask_id_dist}: No pixels removed by distance filtering.")
    else:
        print("  No frames modified by distance filtering.")
    
    # Final check to ensure masks are 3D (1, H, W) if they were originally
    # This assumes input `segments` might have 3D masks.
    # And `filter_by_distance` returns 2D masks.
    # `remove_small_components` also likely operates on and returns 2D.
    # Overlap removal was done on 2D.
    ref_mask_shape_dim = None
    if segments:
        first_frame_key = next(iter(segments))
        if segments[first_frame_key]:
            first_mask_key = next(iter(segments[first_frame_key]))
            if segments[first_frame_key][first_mask_key] is not None:
                 ref_mask_shape_dim = len(segments[first_frame_key][first_mask_key].shape)

    if ref_mask_shape_dim == 3:
        for frame in cleaned_segments:
            for mask_id in cleaned_segments[frame]:
                if cleaned_segments[frame][mask_id] is not None and len(cleaned_segments[frame][mask_id].shape) == 2:
                    cleaned_segments[frame][mask_id] = cleaned_segments[frame][mask_id][np.newaxis, ...]
    
    return cleaned_segments


cleaned_distance_segments = distance_clean_segments(filled_video_segments, min_size=4, min_distance=5, min_pixels=4)
modified_segments = cleaned_distance_segments 


""" 
### Check for overlaps between the segments (Modify masks to remove overlapping pixels)
def check_mask_overlap(loaded_video_segments):
    results = {}
    total_frames = len(loaded_video_segments)
    frames_with_overlap = 0
    
    for frame, masks in loaded_video_segments.items():
        overlap = False
        mask_combination = []
        
        required_masks = [masks.get(i) for i in range(2, 5)]
        if all(mask is not None for mask in required_masks): 
            for i in range(len(required_masks)):
                for j in range(i+1, len(required_masks)):
                    overlap_mask = np.logical_and(required_masks[i], required_masks[j])
                    overlap_count = np.sum(overlap_mask)
                    if overlap_count > 0:
                        overlap = True
                        frames_with_overlap += 1
                        mask_combination.append((i+2, j+2, overlap_count))
        
        if overlap:
            results[frame] = {
                'overlap': overlap,
                'mask_combination': mask_combination
            }

    print("\nOverlap Analysis Results:")
    print(f"Total frames analyzed: {total_frames}")
    
    if results:
        print(f"Found overlaps in {frames_with_overlap} frames:")
        for frame, result in results.items():
            for combination in result['mask_combination']:
                print(f"Frame {frame}:  Masks {combination[0]} and {combination[1]} overlap: {combination[2]} pixels")
    else:
        print("No overlaps found between any masks in any frames!")

    return results

def remove_overlap(mask1, mask2):
    overlap = np.logical_and(mask1, mask2)
    return mask1 & ~overlap, mask2 & ~overlap, np.sum(overlap)

def remove_overlapping_pixels(loaded_video_segments, overlap_results):
    modified_segments = {}
    
    for frame, masks in loaded_video_segments.items():
        modified_masks = masks.copy()
        
        if frame in overlap_results:
            for combination in overlap_results[frame]['mask_combination']:
                mask1_id, mask2_id, _ = combination
                mask1, mask2, _ = remove_overlap(masks[mask1_id], masks[mask2_id])
                modified_masks[mask1_id] = mask1
                modified_masks[mask2_id] = mask2
        
        modified_segments[frame] = modified_masks
    
    return modified_segments

 """# overlap_results = check_mask_overlap(cleaned_distance_segments)

# modified_segments = remove_overlapping_pixels(cleaned_distance_segments, overlap_results)
# modified_overlap_results = check_mask_overlap(modified_segments)

# Instead, the `cleaned_distance_segments` will now be the `modified_segments`

# Check if any overlaps remain (should be none if logic is correct)
# We can repurpose check_mask_overlap for verification or rely on the internal summary.
# For now, let's remove the explicit check after the new distance_clean_segments.
# modified_overlap_results = check_mask_overlap(modified_segments) 




### Check for local movement of the segments (print only)
def check_local_movement(cleaned_segments, overlap_threshold=0.25, extended_overlap_threshold=0.10, mask2_threshold=0.10):
    movement_results = {}
    frames = sorted(cleaned_segments.keys())
    
    for i, current_frame in enumerate(frames):
        frame_result = {}
        
        for mask_id in cleaned_segments[current_frame].keys():
            current_mask = cleaned_segments[current_frame][mask_id]
            current_area = np.sum(current_mask)
            
            if current_area == 0:
                frame_result[mask_id] = {
                    'is_empty': True,
                    'meets_criteria': False,
                    'prev_overlaps': [],
                    'next_overlaps': []
                }
                continue
            
            # Check previous frames
            prev_overlaps = []
            for j in range(max(0, i-3), i):
                prev_frame = frames[j]
                prev_mask = cleaned_segments[prev_frame][mask_id]
                overlap = np.sum(np.logical_and(current_mask, prev_mask))
                prev_overlaps.append(overlap / current_area)
            
            # Check next frames
            next_overlaps = []
            for j in range(i+1, min(len(frames), i+4)):
                next_frame = frames[j]
                next_mask = cleaned_segments[next_frame][mask_id]
                overlap = np.sum(np.logical_and(current_mask, next_mask))
                next_overlaps.append(overlap / current_area)
            
            # Check if movement criteria are met
            if mask_id == 2:
                meets_criteria = all(overlap >= mask2_threshold for overlap in prev_overlaps + next_overlaps)
            else:
                meets_criteria = True
                if len(prev_overlaps) > 0:
                    meets_criteria = meets_criteria and prev_overlaps[-1] >= overlap_threshold
                    if len(prev_overlaps) >= 3:
                        meets_criteria = meets_criteria and all(overlap >= extended_overlap_threshold for overlap in prev_overlaps[:3])
                if len(next_overlaps) > 0:
                    meets_criteria = meets_criteria and next_overlaps[0] >= overlap_threshold
                    if len(next_overlaps) >= 3:
                        meets_criteria = meets_criteria and all(overlap >= extended_overlap_threshold for overlap in next_overlaps[:3])
            
            frame_result[mask_id] = {
                'is_empty': False,
                'meets_criteria': meets_criteria,
                'prev_overlaps': prev_overlaps,
                'next_overlaps': next_overlaps
            }
        
        movement_results[current_frame] = frame_result
    
    return movement_results

def analyze_movement_results(movement_results):
    total_frames = len(movement_results)
    masks_meeting_criteria = {2: 0, 3: 0, 4: 0}
    empty_masks = {2: 0, 3: 0, 4: 0}
    zero_overlap_masks = {2: [], 3: [], 4: []}
    not_meeting_criteria_masks = {2: [], 3: [], 4: []}
    problem_frames = {}
    
    for frame, frame_result in movement_results.items():
        frame_problems = {}
        for mask_id, mask_result in frame_result.items():
            if mask_result['is_empty']:
                empty_masks[mask_id] += 1
                frame_problems[mask_id] = "Empty mask"
            elif not mask_result['meets_criteria']:
                not_meeting_criteria_masks[mask_id].append(frame)
                all_overlaps = mask_result['prev_overlaps'] + mask_result['next_overlaps']
                has_zero_overlap = any(overlap == 0 for overlap in all_overlaps)
                if has_zero_overlap:
                    zero_overlap_masks[mask_id].append(frame)
                    frame_problems[mask_id] = {
                        "zero_overlap": True,
                        "prev_overlaps": mask_result['prev_overlaps'],
                        "next_overlaps": mask_result['next_overlaps']
                    }
                else:
                    frame_problems[mask_id] = {
                        "zero_overlap": False,
                        "prev_overlaps": mask_result['prev_overlaps'],
                        "next_overlaps": mask_result['next_overlaps']
                    }
            else:
                masks_meeting_criteria[mask_id] += 1
        
        if frame_problems:
            problem_frames[frame] = frame_problems
    
    return {
        "total_frames": total_frames,
        "masks_meeting_criteria": masks_meeting_criteria,
        "empty_masks": empty_masks,
        "zero_overlap_masks": zero_overlap_masks,
        "not_meeting_criteria_masks": not_meeting_criteria_masks,
        "problem_frames": problem_frames
    }

def print_movement_analysis_summary(analysis_results):
    print("Local Movement Analysis Summary:")
    total_frames = analysis_results["total_frames"]
    
    print("\nProblem Frames:")
    for frame, problems in analysis_results["problem_frames"].items():
        print(f"  Frame {frame}:")
        for mask_id, problem in problems.items():
            if problem == "Empty mask":
                print(f"    Mask {mask_id}: Empty mask")
            else:
                if problem["zero_overlap"]:
                    print(f"    Mask {mask_id}: ZERO OVERLAP")
                else:
                    print(f"    Mask {mask_id}:")
                print(f"      Previous overlaps: {[f'{overlap:.2f}' for overlap in problem['prev_overlaps']]}")
                print(f"      Next overlaps: {[f'{overlap:.2f}' for overlap in problem['next_overlaps']]}")
    
    print("\nSummary Statistics:")
    for mask_id in [2, 3, 4]:
        non_empty_frames = total_frames - analysis_results["empty_masks"][mask_id]
        zero_overlap_frames = analysis_results["zero_overlap_masks"][mask_id]
        not_meeting_criteria_frames = analysis_results["not_meeting_criteria_masks"][mask_id]
        if non_empty_frames > 0:
            percentage = (analysis_results["masks_meeting_criteria"][mask_id] / non_empty_frames) * 100
            print(f"\nMask {mask_id}:")
            if mask_id == 2:
                print("  10% threshold")
            else:
                print("  25% adjacent, 10% extended")
            print(f"  Empty: {analysis_results['empty_masks'][mask_id]}/{total_frames} frames")
            print(f"  Zero overlap: {len(zero_overlap_frames)}/{non_empty_frames} non-empty frames")
            if zero_overlap_frames:
                print(f"    Frames: {', '.join(map(str, zero_overlap_frames))}")
            print(f"  Bellow threshold: {len(not_meeting_criteria_frames)}/{non_empty_frames} non-empty frames")
            if not_meeting_criteria_frames:
                print(f"    Frames: {', '.join(map(str, not_meeting_criteria_frames))}")
            print(f"  Passed: {analysis_results['masks_meeting_criteria'][mask_id]}/{non_empty_frames} non-empty frames ({percentage:.2f}%)")
        else:
            print(f"Mask {mask_id}: Empty in all frames")

# Perform the local movement check
movement_results = check_local_movement(modified_segments)

# Analyze and print results
analysis_results = analyze_movement_results(movement_results)
print_movement_analysis_summary(analysis_results)




### Align largest segment
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
    for object_id in tqdm(object_ids, desc="Processing objects"):
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


processed_masks = process_all_masks(modified_segments)




###Clean overlaps
def get_all_object_masks(processed_masks: Dict[int, Dict[int, np.ndarray]], 
                        object_id: int) -> Dict[int, np.ndarray]:
    """
    Get all masks for a specific object across all frames.
    
    Args:
        processed_masks: Dictionary mapping frame numbers to dictionaries of object masks
                       {frame_num: {obj_id: mask}}
        object_id: ID of the object to collect masks for
    
    Returns:
        Dictionary mapping frame numbers to masks for the specified object
        {frame_num: mask}
    """
    return {
        frame_num: objects[object_id]
        for frame_num, objects in processed_masks.items()
        if object_id in objects
    }

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




#Save clean aligned segments to h5
def save_cleaned_segments_to_h5(cleaned_segments, filename):
    # Create the output filename
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"/home/lilly/phd/ria/data_analyzed/AG_WT/cleaned_aligned_segments/{name_without_ext}_cleanedalignedsegments.h5"
    
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


filename = ria_segments
# Save the cleaned segments
output_filename = save_cleaned_segments_to_h5(final_masks, filename)

# Load the cleaned segments
loaded_segments = load_cleaned_segments_from_h5(output_filename)

# Perform detailed comparison
compare_cleaned_segments(final_masks, loaded_segments)






#####Test crap
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
                (0, 255, 255),  # Cyan
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta

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


image_dir = "/home/lilly/phd/ria/data_foranalysis/AG_WT/riacrop/AG_WT-MMH99_10s_20190306_03_crop"
masks_dict = loaded_segments
output_path = "filled_segments_video.mp4"

create_mask_video(image_dir, masks_dict, output_path, fps=10, alpha=1)








def load_original_image(frame, image_folder):
    # Assuming the images are named as frame_000.png, frame_001.png, etc.
    image_filename = f"{frame:06d}.jpg"
    image_path = os.path.join(image_folder, image_filename)
    return cv2.imread(image_path)

def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    # Create a color overlay
    overlay = np.zeros(image.shape, dtype=np.uint8)
    overlay[mask == 1] = color

    # Combine the image with the overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

# Modify the last section
video_folder = '/home/lilly/phd/ria/data_foranalysis/riacrop'  # Replace with the actual path to your image folder
video_name = filename.split('_riasegmentation.h5')[0]
image_folder = os.path.join(video_folder, video_name)


frame = 537
mask_id = 4
image_folder = "/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH99_10s_20190306_02_crop"
# Load the original image
original_image = load_original_image(frame, image_folder)

# Get the mask
mask = cleaned_segments[frame][mask_id][0]

# Overlay the mask on the original image
overlayed_image = overlay_mask_on_image(original_image, mask)

# Save the overlayed image
cv2.imwrite("tst.png", overlayed_image)





frame = 21
mask = 4
mask = modified_segments[frame][mask][0]

image_array = np.uint8(mask	 * 255)
image = Image.fromarray(image_array)
image.save('tst.png')



