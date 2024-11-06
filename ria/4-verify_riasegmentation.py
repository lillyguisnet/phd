import h5py
import numpy as np
import os
from skimage.measure import label
import cv2
from scipy import ndimage
from PIL import Image


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

output_dir = '/home/lilly/phd/ria/data_analyzed/ria_segmentation'
filename = 'AG-MMH122_10s_20190830_04_crop_riasegmentation.h5'
full_path = os.path.join(output_dir, filename)
full_path = '/home/lilly/phd/ria/data_analyzed/ria_segmentation/AG-MMH99_10s_20190306_02_crop_riasegmentation.h5'
loaded_video_segments = load_video_segments_from_h5(full_path)



np.sum(loaded_video_segments[0][2])
np.sum(loaded_video_segments[0][3])
np.sum(loaded_video_segments[0][4])


# Check for overlaps between the segments (Modify masks to remove overlapping pixels)
def check_mask_overlap(loaded_video_segments):
    results = {}
    
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
                        mask_combination.append((i+2, j+2, overlap_count))
        
        if overlap:
            results[frame] = {
                'overlap': overlap,
                'mask_combination': mask_combination
            }

    print("Overlap Results:")
    for frame, result in results.items():
        for combination in result['mask_combination']:
            print(f"Frame {frame}:  Masks {combination[0]} and {combination[1]} overlap: {combination[2]} pixels")

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

overlap_results = check_mask_overlap(loaded_video_segments)

modified_segments = remove_overlapping_pixels(loaded_video_segments, overlap_results)
modified_overlap_results = check_mask_overlap(modified_segments)



#Check for discontinuous segments
def find_connected_components(mask):
    # Remove the single-dimensional entries
    mask = np.squeeze(mask)
    
    # Use scipy's label function to find connected components
    labeled_array, num_features = ndimage.label(mask)
    
    return labeled_array, num_features

def calculate_statistics(sizes):
    if not sizes:
        return {
            'mean': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'std': 0
        }
    return {
        'mean': np.mean(sizes),
        'median': np.median(sizes),
        'min': np.min(sizes),
        'max': np.max(sizes),
        'std': np.std(sizes)
    }

def check_discontinuous_segments(modified_segments):
    discontinuous_results = {}
    continuous_results = {}
    segment_statistics = {2: [], 3: [], 4: []}

    for frame, masks in modified_segments.items():
        frame_results = {}

        for mask_id, mask in masks.items():
            if mask_id in [2, 3, 4]:  # Only check masks 2, 3, and 4
                labeled_array, num_features = find_connected_components(mask)
                
                component_sizes = [np.sum(labeled_array == i) for i in range(1, num_features + 1)]
                segment_statistics[mask_id].extend(component_sizes)
                
                if num_features > 1:
                    frame_results[mask_id] = {
                        'discontinuous': True,
                        'num_components': num_features,
                        'component_sizes': component_sizes
                    }
                else:
                    frame_results[mask_id] = {
                        'discontinuous': False,
                        'num_components': 1,
                        'component_sizes': component_sizes
                    }

        if any(result['discontinuous'] for result in frame_results.values()):
            discontinuous_results[frame] = frame_results
        else:
            continuous_results[frame] = frame_results

    # Calculate statistics for each segment
    for mask_id in [2, 3, 4]:
        segment_statistics[mask_id] = calculate_statistics(segment_statistics[mask_id])

    return discontinuous_results, continuous_results, segment_statistics

def print_segment_analysis_results(discontinuous_results, continuous_results, segment_statistics, modified_segments):
    print("Discontinuous segments:")
    if not discontinuous_results:
        print("No discontinuous segments found.")
    else:
        print(f"Found discontinuous segments in {len(discontinuous_results)} frames.")
        print("\nDetails of discontinuous segments:")
        for frame, results in discontinuous_results.items():
            print(f"\nFrame {frame}:")
            for mask_id, result in results.items():
                if result['discontinuous']:
                    print(f"  Mask {mask_id}:")
                    print(f"    Number of components: {result['num_components']}")
                    print(f"    Component sizes: {result['component_sizes']}")

    print("\nContinuous segments:")
    if not continuous_results:
        print("No continuous segments found.")
    else:
        print(f"Found {len(continuous_results)} frames with all continuous segments.")

    print("\nSegment Statistics:")
    for mask_id in [2, 3, 4]:
        print(f"\nMask {mask_id}:")
        for stat, value in segment_statistics[mask_id].items():
            print(f"  {stat}: {value:.2f}")

    print(f"\nTotal frames processed: {len(modified_segments)}")
    print(f"Frames with discontinuous segments: {len(discontinuous_results)}")
    print(f"Frames with all continuous segments: {len(continuous_results)}")


discontinuous_results, continuous_results, segment_statistics = check_discontinuous_segments(modified_segments)

print_segment_analysis_results(discontinuous_results, continuous_results, segment_statistics, modified_segments)



# Clean the segments
    # Filter to keep only masks 2, 3, and 4
    # Remove small components (min size 3)
def remove_small_components(mask, min_size=3):
    labeled_array, num_features = find_connected_components(mask)
    
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        if np.sum(component) > min_size:
            cleaned_mask = np.logical_or(cleaned_mask, component)
    
    return cleaned_mask

def filter_masks(segments):
    filtered_segments = {}
    for frame, masks in segments.items():
        filtered_segments[frame] = {mask_id: mask for mask_id, mask in masks.items() if mask_id in [2, 3, 4]}
    return filtered_segments

def master_clean_segments(segments, min_size=3):
    # Step 1: Filter to keep only masks 2, 3, and 4
    cleaned_segments = filter_masks(segments)
    
    # Step 2: Remove small components
    for frame, masks in cleaned_segments.items():
        for mask_id, mask in masks.items():
            cleaned_segments[frame][mask_id] = remove_small_components(mask, min_size)
    
    # Add more cleaning operations here as needed
    # cleaned_segments = another_cleaning_function(cleaned_segments)
    
    return cleaned_segments


cleaned_segments = master_clean_segments(modified_segments)

discontinuous_results, continuous_results, segment_statistics = check_discontinuous_segments(cleaned_segments)
print_segment_analysis_results(discontinuous_results, continuous_results, segment_statistics, cleaned_segments)



# Check for local movement of the segments
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
movement_results = check_local_movement(cleaned_segments)

# Analyze the results
analysis_results = analyze_movement_results(movement_results)

# Print the summary
print_movement_analysis_summary(analysis_results)


#Remove pixels in mask4 that are within min_distance pixels from any pixel in mask3.
def filter_by_distance(mask3, mask4, min_distance=6, min_pixels=3):
    """
    Filter out pixels in mask4 that are within min_distance pixels from any pixel in mask3.
    If all pixels would be filtered out, keep the min_pixels furthest pixels.
    
    Parameters:
    -----------
    mask3 : numpy.ndarray
        Binary mask for segment 3
    mask4 : numpy.ndarray
        Binary mask for segment 4
    min_distance : float
        Minimum Euclidean distance in pixels (default: 6)
    min_pixels : int
        Minimum number of pixels to preserve from mask4 (default: 3)
        
    Returns:
    --------
    numpy.ndarray
        Filtered mask4 with pixels too close to mask3 removed
    """
    # Ensure masks are 2D
    mask3 = np.squeeze(mask3)
    mask4 = np.squeeze(mask4)
    
    if mask3.ndim != 2 or mask4.ndim != 2:
        raise ValueError(f"Masks must be 2D after squeezing. Got shapes: mask3={mask3.shape}, mask4={mask4.shape}")
    
    # Ensure masks are boolean
    mask3 = mask3.astype(bool)
    mask4 = mask4.astype(bool)
    
    # Convert to correct format for distanceTransform (8-bit unsigned integer)
    mask3_uint8 = np.ascontiguousarray(mask3.astype(np.uint8))
    
    # Calculate distance transform from mask3
    dist_transform = cv2.distanceTransform(
        (1 - mask3_uint8),  # Invert mask3
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )
    
    # Create initial distance mask
    distance_mask = dist_transform >= min_distance
    
    # Apply the distance mask to mask4
    filtered_mask4 = np.logical_and(mask4, distance_mask)
    
    # If filtered mask is empty or has fewer than min_pixels pixels,
    # keep the furthest min_pixels pixels from mask3
    if np.sum(filtered_mask4) < min_pixels:
        # Get distances for all mask4 pixels
        mask4_distances = dist_transform[mask4]
        if len(mask4_distances) > 0:
            # Sort distances in descending order
            distances_sorted = np.sort(mask4_distances)[::-1]
            # Get the distance threshold that keeps exactly min_pixels
            # (or all pixels if there are fewer than min_pixels)
            distance_threshold = distances_sorted[min(min_pixels - 1, len(distances_sorted) - 1)]
            # Create new mask keeping only the furthest pixels
            filtered_mask4 = np.logical_and(mask4, dist_transform >= distance_threshold)
    
    # Ensure output has same shape as input
    if filtered_mask4.shape != mask4.shape:
        raise ValueError(f"Output shape {filtered_mask4.shape} doesn't match input shape {mask4.shape}")
    
    return filtered_mask4


def master_clean_segments(segments, min_size=3, min_distance=6, min_pixels=3):
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
        Minimum number of pixels to preserve from mask4
        
    Returns:
    --------
    dict
        Cleaned segments
    """
    # Step 1: Filter to keep only masks 2, 3, and 4
    cleaned_segments = filter_masks(segments)
    
    # Step 2: Remove small components
    for frame, masks in cleaned_segments.items():
        for mask_id, mask in masks.items():
            cleaned_segments[frame][mask_id] = remove_small_components(mask, min_size)
    
    # Step 3: Filter mask4 based on distance from mask3
    for frame, masks in cleaned_segments.items():
        if 3 in masks and 4 in masks:
            cleaned_segments[frame][4] = filter_by_distance(
                masks[3],
                masks[4],
                min_distance=min_distance,
                min_pixels=min_pixels
            )
    
    return cleaned_segments


cleaned_distance_segments = master_clean_segments(cleaned_segments, min_size=3, min_distance=6, min_pixels=3)



def save_cleaned_segments_to_h5(cleaned_segments, filename):
    # Create the output filename
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"/home/lilly/phd/ria/data_analyzed/cleaned_segments/{name_without_ext}_cleanedsegments.h5"
    
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
filename = 'AG-MMH122_10s_20190830_04_crop_riasegmentation.h5'

# Save the cleaned segments
output_filename = save_cleaned_segments_to_h5(cleaned_segments, filename)

# Load the cleaned segments
loaded_segments = load_cleaned_segments_from_h5(output_filename)

# Perform detailed comparison
compare_cleaned_segments(cleaned_segments, loaded_segments)




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


