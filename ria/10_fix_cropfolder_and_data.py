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

def load_cleaned_segments_from_h5(filename):
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        nb_frames = 0
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            nb_frames += 1

    
    print(f"{nb_frames} frames loaded from {filename}")
    return cleaned_segments

def get_random_unprocessed_video(cleaned_aligned_segments_dir, final_data_dir):
    all_videos = [os.path.splitext(d)[0] for d in os.listdir(cleaned_aligned_segments_dir)]
    unprocessed_videos = [
        video for video in all_videos
        if not any(video[:video.find('crop')] in f 
                  for f in os.listdir(final_data_dir))
    ]
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(cleaned_aligned_segments_dir, random.choice(unprocessed_videos) + ".h5")

cleaned_aligned_segments_dir = '/home/lilly/phd/ria/data_analyzed/AG_WT/cleaned_aligned_segments'
final_data_dir = '/home/lilly/phd/ria/data_analyzed/AG_WT/final_data'
VIDEO_FRAMES_BASE_DIR = "/home/lilly/phd/ria/data_foranalysis/AG_WT/riacrop"
final_data_fixcrop_dir = '/home/lilly/phd/ria/data_analyzed/AG_WT/final_data_fixcropfolder'

filename = get_random_unprocessed_video(cleaned_aligned_segments_dir, final_data_fixcrop_dir)
cleaned_segments = load_cleaned_segments_from_h5(filename)



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

def load_image(frame_idx: int, video_frames_dir: str) -> np.ndarray:
    """Loads a specific frame image from the video's frame directory."""
    image_path = os.path.join(video_frames_dir, f"{frame_idx:06d}.jpg")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
    return image

def count_mask_pixels(masks):
    pixel_counts = {}
    for obj_id, mask in masks.items():
        pixel_counts[obj_id] = np.sum(mask)
    return pixel_counts

def calculate_mean_values_and_pixel_counts(image, masks, background_coordinates):
    mean_values = {}
    pixel_counts = count_mask_pixels(masks)
    
    # Calculate mean background value
    bg_pixel_values = image[background_coordinates[:, 0], background_coordinates[:, 1]]
    mean_values['background'] = np.mean(bg_pixel_values)
    
    # Calculate mean value for each mask
    for obj_id, mask in masks.items():
        mask_pixel_values = image[mask.squeeze()]
        mean_values[obj_id] = np.mean(mask_pixel_values)
    
    return mean_values, pixel_counts

def create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts):
    data = {'frame': []}
    
    all_objects = set()
    for frame_data in mean_values.values():
        all_objects.update(frame_data.keys())
    all_objects.remove('background')
    
    for obj in all_objects:
        data[obj] = []
        data[f"{obj}_bg_corrected"] = []
        data[f"{obj}_pixel_count"] = []
    
    for frame_idx, frame_data in mean_values.items():
        data['frame'].append(frame_idx)
        bg_value = frame_data['background']
        frame_pixel_counts = pixel_counts[frame_idx]
        for obj in all_objects:
            obj_value = frame_data.get(obj, np.nan)
            data[obj].append(obj_value)
            
            if pd.notnull(obj_value):
                bg_corrected = obj_value - bg_value
            else:
                bg_corrected = np.nan
            data[f"{obj}_bg_corrected"].append(bg_corrected)
            
            data[f"{obj}_pixel_count"].append(frame_pixel_counts.get(obj, 0))
    
    df = pd.DataFrame(data)
    
    return df

def process_cleaned_segments(cleaned_segments):
    first_frame = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame.values()))
    image_shape = first_mask.shape

    mean_values = {}
    pixel_counts = {}

    # Determine the current video's frame directory from the global filename
    base_h5_filename = os.path.basename(filename) # filename is global
    # Extract the part of the filename up to and including "_crop"
    video_name_part = base_h5_filename.split('_crop')[0] + '_crop'
    current_video_frames_dir = os.path.join(VIDEO_FRAMES_BASE_DIR, video_name_part)
    print(f"Loading images from: {current_video_frames_dir}")

    for frame_idx, frame_masks in tqdm(cleaned_segments.items(), desc="Processing frames"):
        bg_coordinates = get_background_sample(frame_masks, image_shape)
        image = load_image(frame_idx, current_video_frames_dir)
        
        if image is None:
            print(f"Warning: Could not load image for frame {frame_idx}")
            continue
        
        frame_mean_values, frame_pixel_counts = calculate_mean_values_and_pixel_counts(image, frame_masks, bg_coordinates)
        mean_values[frame_idx] = frame_mean_values
        pixel_counts[frame_idx] = frame_pixel_counts

    df_wide_bg_corrected = create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts)
    df_wide_bg_corrected.columns = df_wide_bg_corrected.columns.astype(str)

    if 'background' not in df_wide_bg_corrected.columns:
        background_values = [frame_data['background'] for frame_data in mean_values.values()]
        df_wide_bg_corrected['background'] = background_values

    background_column = ['background']
    original_columns = [col for col in df_wide_bg_corrected.columns if not col.endswith('_bg_corrected') and not col.endswith('_pixel_count') and col != 'frame']
    original_columns = [col for col in original_columns if col != 'background']
    bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_bg_corrected')]
    pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]

    all_columns = ['frame'] + background_column + original_columns + bg_corrected_columns + pixel_count_columns

    print(df_wide_bg_corrected[all_columns].describe())

    return df_wide_bg_corrected

df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments)	




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
    # Get centroids for objects 2 and 4
    centroid2 = get_centroid(first_frame[2])
    centroid4 = get_centroid(first_frame[4])
    
    if centroid2 is None or centroid4 is None:
        return "One or both objects not found in frame"
    
    # Compare x-coordinates of centroids
    if centroid4[0] < centroid2[0]:
        return "left"
    else:
        return "right"
    
def save_brightness_and_side_data(df_wide_brightness_and_background, cleaned_segments, h5_filename, existing_data_dir, new_output_dir):
    # Get side position from first frame
    position = get_relative_position(next(iter(cleaned_segments.values())))

    # Start with a copy of the new calculations
    current_calculations_df = df_wide_brightness_and_background.copy()
    current_calculations_df['side_position'] = position

    # Determine base filename from the input H5 filename
    base_h5_stem = os.path.splitext(os.path.basename(h5_filename))[0]
    
    # Path to the existing CSV in existing_data_dir
    existing_csv_name = base_h5_stem + "_headangles.csv"
    existing_csv_path = os.path.join(existing_data_dir, existing_csv_name)

    final_df = current_calculations_df.copy() # This will be our base

    if os.path.exists(existing_csv_path):
        print(f"Found existing CSV: {existing_csv_path}")
        old_df = pd.read_csv(existing_csv_path)
        
        # Ensure 'frame' column exists in both dataframes for merging
        if 'frame' not in old_df.columns:
            print(f"Warning: 'frame' column not found in {existing_csv_path}. Cannot merge additional data.")
        elif 'frame' not in final_df.columns:
             print(f"Warning: 'frame' column not found in new calculations. Cannot merge additional data.")
        else:
            # Identify columns in old_df that are not in final_df (new calculations)
            # These are the columns to carry over from old_df
            cols_to_carry_over = [col for col in old_df.columns if col not in final_df.columns and col != 'frame']
            
            if cols_to_carry_over:
                print(f"Carrying over columns from old CSV: {cols_to_carry_over}")
                
                # Attempt to align 'frame' dtypes before merging
                if final_df['frame'].dtype != old_df['frame'].dtype:
                    try:
                        print(f"Attempting to align 'frame' dtypes for merge: new is {final_df['frame'].dtype}, old is {old_df['frame'].dtype}")
                        old_df['frame'] = old_df['frame'].astype(final_df['frame'].dtype)
                    except Exception as e:
                        print(f"Warning: Could not align 'frame' column types for merging ({e}). Merge may behave unexpectedly.")
                
                # Merge these columns into final_df, aligning on 'frame'
                final_df = pd.merge(final_df, old_df[['frame'] + cols_to_carry_over], on='frame', how='left')
            else:
                print("No unique columns to carry over, or old CSV columns are fully covered by new calculations.")
            
    else:
        print(f"No existing CSV found at: {existing_csv_path}. Saving new calculations only.")

    # Create the output directory if it doesn't exist
    os.makedirs(new_output_dir, exist_ok=True)

    # Construct the output filename
    # The stem is from the H5/original CSV, then add "_fixfolder.csv"
    output_filename_stem = base_h5_stem 
    output_filename = os.path.join(new_output_dir, output_filename_stem + "_fixfolder.csv")

    # Save dataframe as CSV
    final_df.to_csv(output_filename, index=False)

    print(f"\nFinal data saved to: {output_filename}")
    print("\nFinal DataFrame structure:")
    final_df.info()
    print("\nFinal DataFrame summary statistics (including non-numeric):")
    print(final_df.describe(include='all'))
    
    if 'side_position' in final_df.columns:
        print("\nside_position unique values:", final_df['side_position'].unique())
        print("side_position value_counts:\n", final_df['side_position'].value_counts())

    return final_df


# Update the script's execution part:
df_final_updated = save_brightness_and_side_data(
    df_wide_brightness_and_background, 
    cleaned_segments, 
    filename,  # This is the h5 filename
    final_data_fixcrop_dir, 
    final_data_fixcrop_dir
)