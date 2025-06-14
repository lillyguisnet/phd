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
from typing import Dict, Set, List, Tuple
from scipy.ndimage import distance_transform_edt, center_of_mass
from sklearn.decomposition import PCA
import itertools
import glob

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        # Filter out None objects and convert to integers
        valid_object_ids = []
        for obj_id in object_ids:
            if obj_id is not None and obj_id != 'None' and str(obj_id) != 'None':
                # Convert to integer if it's not already
                try:
                    valid_object_ids.append(int(obj_id))
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert object_id '{obj_id}' to integer, skipping")
        
        print(f"Valid object IDs found: {valid_object_ids}")
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in valid_object_ids:
                # Use string key to access the mask data (since H5 keys are strings)
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                # But store with integer key for consistency
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1

    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(raw_segments_dir, final_data_dir):
    all_videos = [os.path.splitext(d)[0] for d in os.listdir(raw_segments_dir)]
    
    # Get all files in final_data_dir (without extensions)
    final_data_files = [os.path.splitext(f)[0] for f in os.listdir(final_data_dir)]
    
    unprocessed_videos = []
    for video in all_videos:
        # Extract core name from segmentation file
        # Example: "data_original-hannah_crop_riasegmentation" -> "data_original-hannah"
        if '_crop_' in video:
            core_name = video[:video.find('_crop_')]
        else:
            # Fallback: use the whole name if no '_crop_' pattern found
            core_name = video
        
        # Find all matching files with this core name
        matching_files = [f for f in final_data_files if f.startswith(core_name)]
        
        # Check if this video has been processed by this brightness extraction script
        # It's unprocessed if:
        # 1. No matching files at all, OR
        # 2. Only matching files end with "_headsegmentation_head_angles" (from previous processing)
        is_processed_by_brightness_script = any(
            not f.endswith('_headsegmentation_head_angles') 
            for f in matching_files
        )
        
        if not is_processed_by_brightness_script:
            unprocessed_videos.append(video)
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(raw_segments_dir, random.choice(unprocessed_videos) + ".h5")

raw_segments_dir = '/home/lilly/phd/riverchip/data_analyzed/ria_segmentation'
final_data_dir = '/home/lilly/phd/riverchip/data_analyzed/final_data'

filename = get_random_unprocessed_video(raw_segments_dir, final_data_dir)
raw_segments = load_cleaned_segments_from_h5(filename)



###Fill missing masks
def fill_missing_masks(video_segments: Dict[int, Dict[int, np.ndarray]], required_ids: List[int] = [2, 3]) -> Dict[int, Dict[int, np.ndarray]]:
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
    
    # Check what object IDs are actually available in the data
    sample_frame_data = next(iter(video_segments.values()))
    available_obj_ids = list(sample_frame_data.keys())
    print(f"Available object IDs in data: {available_obj_ids}")
    print(f"Required object IDs: {required_ids}")
    
    # Only process object IDs that are actually supposed to be in the data
    valid_required_ids = [obj_id for obj_id in required_ids if obj_id in available_obj_ids]
    if len(valid_required_ids) != len(required_ids):
        missing_from_data = [obj_id for obj_id in required_ids if obj_id not in available_obj_ids]
        print(f"Warning: These required object IDs are not found in the data: {missing_from_data}")
        print(f"Will only process: {valid_required_ids}")
    
    # For each required object ID
    for obj_id in valid_required_ids:
        # Find frames with missing masks (including empty masks)
        missing_frames = [
            frame for frame in frames 
            if (obj_id not in video_segments[frame] or 
                video_segments[frame][obj_id] is None or 
                np.sum(video_segments[frame][obj_id]) == 0)  # Check if mask is empty
        ]
        
        if not missing_frames:
            print(f"Object {obj_id}: No missing or empty masks found")
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
        for obj_id in valid_required_ids:
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
filled_video_segments = fill_missing_masks(raw_segments)




###Extract brightness and background
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

def get_background_sample(frame_masks, image_shape, num_samples=100, min_distance=40):
    combined_mask = np.zeros(image_shape[1:], dtype=bool)
    for mask in frame_masks.values():
        combined_mask |= mask.squeeze()
    
    distance_map = distance_transform_edt(~combined_mask)
    valid_bg = (distance_map >= min_distance)
    valid_coords = np.column_stack(np.where(valid_bg))
    
    if len(valid_coords) < num_samples:
        print(f"Warning: Only {len(valid_coords)} valid background pixels found. Sampling all of them.")
        return valid_coords
    else:
        sampled_indices = random.sample(range(len(valid_coords)), num_samples)
        return valid_coords[sampled_indices]

def load_image(frame_idx):
    # Extract core name from the current filename being processed
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Extract core name from segmentation file
    # Example: "data_original-hannah_crop_riasegmentation" -> "data_original-hannah"
    if '_crop_' in base_filename:
        core_name = base_filename[:base_filename.find('_crop_')]
    else:
        # Fallback: use the whole name if no '_crop_' pattern found
        core_name = base_filename
    
    # Construct the path to the video folder
    video_folder_path = f"/home/lilly/phd/riverchip/data_foranalysis/riacrop/{core_name}_crop"
    image_path = f"{video_folder_path}/{frame_idx:06d}.jpg"
    
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def count_mask_pixels(masks):
    pixel_counts = {}
    for obj_id, mask in masks.items():
        # Properly squeeze the mask to ensure correct pixel counting
        mask_2d = mask.squeeze()
        pixel_counts[obj_id] = np.sum(mask_2d)
    return pixel_counts

def calculate_mean_values_and_pixel_counts(image, masks, background_coordinates):
    mean_values = {}
    pixel_counts = count_mask_pixels(masks)
    
    # Calculate mean background value
    bg_pixel_values = image[background_coordinates[:, 0], background_coordinates[:, 1]]
    mean_values['background'] = np.mean(bg_pixel_values)
    
    # Calculate mean value and top percentage values for each mask
    for obj_id, mask in masks.items():
        # Properly squeeze the mask to ensure 2D dimensions match the image
        mask_2d = mask.squeeze()
        mask_pixel_values = image[mask_2d]
        
        # Mean of all pixels
        mean_values[obj_id] = np.mean(mask_pixel_values)
        
        # Top percentage brightness values
        sorted_pixels = np.sort(mask_pixel_values)
        
        # Top 25% brightest pixels
        n_pixels_25 = max(1, int(round(len(mask_pixel_values) * 0.25)))
        top_25_pixels = sorted_pixels[-n_pixels_25:]
        mean_values[f"{obj_id}_top25"] = np.mean(top_25_pixels)
        
        # Top 10% brightest pixels
        n_pixels_10 = max(1, int(round(len(mask_pixel_values) * 0.10)))
        top_10_pixels = sorted_pixels[-n_pixels_10:]
        mean_values[f"{obj_id}_top10"] = np.mean(top_10_pixels)
        
        # Top 1% brightest pixels
        n_pixels_1 = max(1, int(round(len(mask_pixel_values) * 0.01)))
        top_1_pixels = sorted_pixels[-n_pixels_1:]
        mean_values[f"{obj_id}_top1"] = np.mean(top_1_pixels)
    
    return mean_values, pixel_counts

def create_wide_format_table_with_pixel_count(mean_values, pixel_counts):
    data = {'frame': []}
    
    all_objects = set()
    for frame_data in mean_values.values():
        # Get base object IDs (excluding background and top percentage variants)
        for key in frame_data.keys():
            key_str = str(key)
            if key_str != 'background' and not any(key_str.endswith(suffix) for suffix in ['_top25', '_top10', '_top1']):
                all_objects.add(key)
    
    # Create columns for each object
    for obj in all_objects:
        data[f"{obj}_mean"] = []
        data[f"{obj}_top25"] = []
        data[f"{obj}_top10"] = []
        data[f"{obj}_top1"] = []
        data[f"{obj}_pixel_count"] = []
    
    for frame_idx, frame_data in mean_values.items():
        data['frame'].append(frame_idx)
        frame_pixel_counts = pixel_counts[frame_idx]
        
        for obj in all_objects:
            # Mean brightness
            obj_mean = frame_data.get(obj, np.nan)
            data[f"{obj}_mean"].append(obj_mean)
            
            # Top percentage brightness values
            data[f"{obj}_top25"].append(frame_data.get(f"{obj}_top25", np.nan))
            data[f"{obj}_top10"].append(frame_data.get(f"{obj}_top10", np.nan))
            data[f"{obj}_top1"].append(frame_data.get(f"{obj}_top1", np.nan))
            
            # Pixel count
            data[f"{obj}_pixel_count"].append(frame_pixel_counts.get(obj, 0))
    
    df = pd.DataFrame(data)
    
    return df

def process_cleaned_segments(cleaned_segments):
    first_frame = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame.values()))
    image_shape = first_mask.shape

    mean_values = {}
    pixel_counts = {}

    for frame_idx, frame_masks in tqdm(cleaned_segments.items(), desc="Processing frames"):
        bg_coordinates = get_background_sample(frame_masks, image_shape)
        image = load_image(frame_idx)
        
        if image is None:
            print(f"Warning: Could not load image for frame {frame_idx}")
            continue
        
        frame_mean_values, frame_pixel_counts = calculate_mean_values_and_pixel_counts(image, frame_masks, bg_coordinates)
        mean_values[frame_idx] = frame_mean_values
        pixel_counts[frame_idx] = frame_pixel_counts

    df_wide = create_wide_format_table_with_pixel_count(mean_values, pixel_counts)
    df_wide.columns = df_wide.columns.astype(str)

    if 'background' not in df_wide.columns:
        background_values = [frame_data['background'] for frame_data in mean_values.values()]
        df_wide['background'] = background_values

    # Organize columns in a logical order
    background_column = ['background']
    mean_columns = [col for col in df_wide.columns if col.endswith('_mean')]
    top25_columns = [col for col in df_wide.columns if col.endswith('_top25')]
    top10_columns = [col for col in df_wide.columns if col.endswith('_top10')]
    top1_columns = [col for col in df_wide.columns if col.endswith('_top1')]
    pixel_count_columns = [col for col in df_wide.columns if col.endswith('_pixel_count')]

    all_columns = ['frame'] + background_column + mean_columns + top25_columns + top10_columns + top1_columns + pixel_count_columns

    print(df_wide[all_columns].describe())

    return df_wide


df_wide_brightness_and_background = process_cleaned_segments(filled_video_segments)
print("Columns in df_wide_brightness_and_background:")
print(list(df_wide_brightness_and_background.columns))




def save_brightness_data(df_wide_brightness, filename, final_data_dir):
    """
    Save the brightness dataframe as a CSV file.
    
    Parameters:
        df_wide_brightness: DataFrame containing brightness data
        filename: Original H5 filename being processed
        final_data_dir: Directory to save the CSV file
    
    Returns:
        str: Path to the saved CSV file
    """
    # Extract base filename from full path
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    
    # Extract core name from segmentation file
    # Example: "data_original-hannah_crop_riasegmentation" -> "data_original-hannah"
    if '_crop_' in base_filename:
        core_name = base_filename[:base_filename.find('_crop_')]
    else:
        # Fallback: use the whole name if no '_crop_' pattern found
        core_name = base_filename
    
    # Create output filename with the specified suffix
    output_filename = os.path.join(final_data_dir, core_name + "_riabrightnessextraction.csv")
    
    # Save dataframe as CSV
    df_wide_brightness.to_csv(output_filename, index=False)
    
    print(f"Brightness data saved to: {output_filename}")
    print(f"DataFrame shape: {df_wide_brightness.shape}")
    print("Columns saved:")
    for col in df_wide_brightness.columns:
        print(f"  - {col}")
    
    return output_filename

saved_file_path = save_brightness_data(df_wide_brightness_and_background, filename, final_data_dir)




""" 
###Extract side
def get_centroid(mask):
    # Get indices of True values - mask is already 2D
    y_indices, x_indices = np.where(mask[0])
    if len(x_indices) == 0:  # If no True values found
        return None
    
    # Calculate centroid
    centroid_x = np.mean(x_indices)
    centroid_y = np.mean(y_indices)
    return (centroid_x, centroid_y)

def get_relative_position(first_frame):
    # Get available object IDs
    available_objects = list(first_frame.keys())
    print(f"Available objects for side position calculation: {available_objects}")
    
    # Try to find two objects to compare
    if len(available_objects) < 2:
        return f"Not enough objects for comparison (only {len(available_objects)} found)"
    
    # Sort objects by ID and take the first two for consistency
    sorted_objects = sorted(available_objects)
    obj1, obj2 = sorted_objects[0], sorted_objects[1]
    
    # Get centroids for the two objects
    centroid1 = get_centroid(first_frame[obj1])
    centroid2 = get_centroid(first_frame[obj2])
    
    if centroid1 is None or centroid2 is None:
        return f"One or both objects ({obj1}, {obj2}) have no valid mask in frame"
    
    # Compare x-coordinates of centroids
    # Define which object is "reference" - using the lower ID as reference
    if centroid2[0] < centroid1[0]:
        position = "left"
    else:
        position = "right"
    
    print(f"Object {obj2} is {position} of object {obj1}")
    print(f"Centroid {obj1}: {centroid1}, Centroid {obj2}: {centroid2}")
    
    return position

def save_brightness_and_side_data(df_wide_brightness_and_background, cleaned_segments, filename, final_data_dir):
    # Get side position from first frame
    position = get_relative_position(next(iter(cleaned_segments.values())))

    # Add position column to dataframe
    df_wide_brightness_and_background['side_position'] = position

    # Extract base filename from full path and add suffix
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(final_data_dir, base_filename + ".csv")

    # Save dataframe as CSV
    df_wide_brightness_and_background.to_csv(output_filename, index=False)

    print(df_wide_brightness_and_background.describe())
    print("side_position unique values:", df_wide_brightness_and_background['side_position'].unique())
    print(df_wide_brightness_and_background['side_position'].value_counts())
    print(f"Data saved to: {output_filename}")

    return df_wide_brightness_and_background

df_wide_brightness_and_side = save_brightness_and_side_data(df_wide_brightness_and_background, raw_segments, filename, final_data_dir)

 """

def merge_brightness_and_angles_csvs(final_data_dir):
    """
    Merge CSV files in the final_data directory.
    Keep all columns from riabrightnessextraction files and only
    angle_degrees and angle_degrees_smoothed_3frame from headsegmentation_head_angles files.
    
    Parameters:
        final_data_dir: Directory containing the CSV files to merge
    
    Returns:
        dict: Dictionary with core_name as key and merged DataFrame as value
    """
    # Find all brightness extraction files
    brightness_files = glob.glob(os.path.join(final_data_dir, "*riabrightnessextraction.csv"))
    
    merged_data = {}
    
    for brightness_file in brightness_files:
        # Extract core name from brightness file
        base_filename = os.path.splitext(os.path.basename(brightness_file))[0]
        core_name = base_filename.replace("_riabrightnessextraction", "")
        
        # Look for corresponding head angles file
        angles_file = os.path.join(final_data_dir, f"{core_name}_headsegmentation_head_angles.csv")
        
        if not os.path.exists(angles_file):
            print(f"Warning: No corresponding angles file found for {core_name}")
            continue
        
        # Load both CSV files
        print(f"Merging data for {core_name}...")
        brightness_df = pd.read_csv(brightness_file)
        angles_df = pd.read_csv(angles_file)
        
        # Select only the required columns from angles file
        angles_subset = angles_df[['frame', 'angle_degrees', 'angle_degrees_smoothed_3frame']].copy()
        
        # Merge on frame column
        merged_df = pd.merge(brightness_df, angles_subset, on='frame', how='left')
        
        # Save merged file
        output_filename = os.path.join(final_data_dir, f"{core_name}_merged_brightness_angles.csv")
        merged_df.to_csv(output_filename, index=False)
        
        print(f"Merged data saved to: {output_filename}")
        print(f"Shape: {merged_df.shape}")
        print(f"Columns: {list(merged_df.columns)}")
        print()
        
        merged_data[core_name] = merged_df
    
    return merged_data

# Run the merge function
print("Starting CSV merge process...")
merged_results = merge_brightness_and_angles_csvs(final_data_dir)
print(f"Successfully merged {len(merged_results)} datasets")






