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
from scipy import stats
import glob


# Define the base directory for video frames
VIDEO_FRAMES_BASE_DIR = "/home/lilly/phd/ria/data_foranalysis/AG_WT/riacrop"

#region [SAM brightest]
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

# Define the specific file path
filename = "/home/lilly/phd/ria/data_analyzed/AG_WT/cleaned_aligned_segments/AG_WT-MMH99_10s_20190320_03_crop_riasegmentation_cleanedalignedsegments.h5"

# Load the cleaned segments from the specified file
cleaned_segments = load_cleaned_segments_from_h5(filename)

# Determine the specific video's frames directory
video_name_for_frames = os.path.basename(filename).split('_riasegmentation_')[0]
current_video_frames_dir = os.path.join(VIDEO_FRAMES_BASE_DIR, video_name_for_frames)



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
    stats_dict = {} 
    pixel_counts = count_mask_pixels(masks)
    
    # Calculate mean background value
    bg_pixel_values = image[background_coordinates[:, 0], background_coordinates[:, 1]]
    if bg_pixel_values.size > 0:
        stats_dict['background'] = np.mean(bg_pixel_values)
    else:
        stats_dict['background'] = np.nan
    
    # Calculate stats for each mask
    for obj_id, mask in masks.items():
        mask_pixel_values = image[mask.squeeze()]
        
        obj_stats = {}
        if mask_pixel_values.size > 0:
            obj_stats['mean'] = np.mean(mask_pixel_values)
            obj_stats['std'] = np.std(mask_pixel_values)
            obj_stats['median'] = np.median(mask_pixel_values)
            obj_stats['min'] = np.min(mask_pixel_values)
            obj_stats['max'] = np.max(mask_pixel_values)
            mode_result = stats.mode(mask_pixel_values, keepdims=False)
            obj_stats['mode'] = mode_result.mode 
        else:
            obj_stats['mean'] = np.nan
            obj_stats['std'] = np.nan
            obj_stats['median'] = np.nan
            obj_stats['min'] = np.nan
            obj_stats['max'] = np.nan
            obj_stats['mode'] = np.nan
            
        stats_dict[obj_id] = obj_stats
    
    return stats_dict, pixel_counts

def create_wide_format_table_with_bg_correction_and_pixel_count(stats_values_over_frames, pixel_counts):
    data = {'frame': []}
    
    all_objects = set()
    for frame_stats_dict in stats_values_over_frames.values():
        for key in frame_stats_dict.keys():
            if key != 'background':
                all_objects.add(key)
    
    stat_names = ['mean', 'std', 'median', 'min', 'max', 'mode']
    for obj in all_objects:
        for stat_name in stat_names:
            data[f"{obj}_{stat_name}"] = []
        data[f"{obj}_mean_bg_corrected"] = [] 
        data[f"{obj}_pixel_count"] = []
    
    for frame_idx, frame_stats_dict in stats_values_over_frames.items():
        data['frame'].append(frame_idx)
        bg_value = frame_stats_dict.get('background', np.nan)
        frame_pixel_counts = pixel_counts[frame_idx]
        for obj in all_objects:
            obj_specific_stats = frame_stats_dict.get(obj, {}) 
            
            obj_mean_value = obj_specific_stats.get('mean', np.nan)
            
            for stat_name in stat_names:
                data[f"{obj}_{stat_name}"].append(obj_specific_stats.get(stat_name, np.nan))
            
            if pd.notnull(obj_mean_value) and pd.notnull(bg_value):
                bg_corrected_mean = obj_mean_value - bg_value
            else:
                bg_corrected_mean = np.nan
            data[f"{obj}_mean_bg_corrected"].append(bg_corrected_mean)
            
            data[f"{obj}_pixel_count"].append(frame_pixel_counts.get(obj, 0))
    
    df = pd.DataFrame(data)
    
    return df

def process_cleaned_segments(cleaned_segments, video_frames_dir: str):
    first_frame = next(iter(cleaned_segments.values()))
    first_mask = next(iter(first_frame.values()))
    image_shape = first_mask.shape

    stats_data_all_frames = {} 
    pixel_counts = {}

    for frame_idx, frame_masks in tqdm(cleaned_segments.items(), desc="Processing frames"):
        bg_coordinates = get_background_sample(frame_masks, image_shape)
        image = load_image(frame_idx, video_frames_dir)
        
        if image is None:
            # Warning already printed by load_image, so just continue
            continue
        
        frame_stats_values, frame_pixel_counts = calculate_mean_values_and_pixel_counts(image, frame_masks, bg_coordinates)
        stats_data_all_frames[frame_idx] = frame_stats_values
        pixel_counts[frame_idx] = frame_pixel_counts

    df_wide_bg_corrected = create_wide_format_table_with_bg_correction_and_pixel_count(stats_data_all_frames, pixel_counts)
    df_wide_bg_corrected.columns = df_wide_bg_corrected.columns.astype(str)

    if 'background' not in df_wide_bg_corrected.columns:
        background_values = [frame_data.get('background', np.nan) for frame_data in stats_data_all_frames.values()]
        df_wide_bg_corrected['background'] = background_values

    frame_col = ['frame']
    background_main_col = ['background']

    raw_stat_columns = [
        col for col in df_wide_bg_corrected.columns 
        if not col.endswith('_mean_bg_corrected') and \
           not col.endswith('_pixel_count') and \
           col not in frame_col and \
           col not in background_main_col
    ]
    raw_stat_columns.sort() 

    mean_bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_mean_bg_corrected')]
    mean_bg_corrected_columns.sort() 

    pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]
    pixel_count_columns.sort() 

    all_columns_for_describe = frame_col + background_main_col + raw_stat_columns + mean_bg_corrected_columns + pixel_count_columns
    
    # Ensure all columns from the dataframe are included and in the specified order for describe
    final_columns_for_describe = [col for col in all_columns_for_describe if col in df_wide_bg_corrected.columns]
    for col in df_wide_bg_corrected.columns: # Add any potentially missed columns
        if col not in final_columns_for_describe:
            final_columns_for_describe.append(col)


    print(df_wide_bg_corrected[final_columns_for_describe].describe())

    return df_wide_bg_corrected

df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments, current_video_frames_dir)

# Save the DataFrame to the specified directory
output_dir = "/home/lilly/phd/ria/benchmarks/brightest/data/sam"

# Extract video name from filename
video_name = os.path.basename(filename).split('.')[0]
output_path = os.path.join(output_dir, f"{video_name}_brightest_sam.csv")
df_wide_brightness_and_background.to_csv(output_path, index=False)

#endregion


#region [Create Long Format DataFrame from SAM and Fiji]

def merge_fiji_xlsx_to_wide(fiji_xlsx_directory: str) -> pd.DataFrame:
    """
    Opens all XLSX files in the given directory, prefixes their column names
    with the file's base name, and merges them into a single wide DataFrame.
    A 'frame' column (1-based index) is added as the first column.

    Args:
        fiji_xlsx_directory (str): Path to the directory containing Fiji XLSX files.

    Returns:
        pd.DataFrame: A wide DataFrame with merged data, or an empty DataFrame
                      if no files are found or an error occurs.
    """
    xlsx_files = glob.glob(os.path.join(fiji_xlsx_directory, "*.xlsx"))
    
    if not xlsx_files:
        print(f"No XLSX files found in '{fiji_xlsx_directory}'.")
        return pd.DataFrame()
        
    list_of_dfs = []
    print(f"Found {len(xlsx_files)} XLSX files to merge:")
    
    for file_path in xlsx_files:
        try:
            print(f"Processing '{os.path.basename(file_path)}'...")
            df = pd.read_excel(file_path)
            
            # Get the base name of the file (without .xlsx extension)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Add the base name as a prefix to all column names
            df_prefixed = df.add_prefix(f"{base_name}_")
            
            list_of_dfs.append(df_prefixed)
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue
            
    if not list_of_dfs:
        print("No DataFrames were successfully processed.")
        return pd.DataFrame()
        
    try:
        # Concatenate all DataFrames horizontally.
        # This assumes corresponding rows across files.
        # If row counts differ, NaN values will be introduced.
        merged_df = pd.concat(list_of_dfs, axis=1)
        
        # Add 'frame' column (1-based index)
        if not merged_df.empty:
            merged_df.insert(0, 'frame', np.arange(1, len(merged_df) + 1))
            
        print("Successfully merged DataFrames and added 'frame' column.")
        return merged_df
    except Exception as e:
        print(f"Error concatenating DataFrames or adding 'frame' column: {e}")
        return pd.DataFrame()

# Example placeholder - replace with your actual path
FIJI_DATA_DIRECTORY = "/home/lilly/phd/ria/benchmarks/brightest/data/fiji" 
merged_fiji_df = merge_fiji_xlsx_to_wide(FIJI_DATA_DIRECTORY)

if not merged_fiji_df.empty:
    print("\nHead of merged Fiji DataFrame:")
    print(merged_fiji_df.head())
    # You might want to save it or process it further
    # merged_fiji_df.to_csv("merged_fiji_data_wide.csv", index=False)


def merge_sam_and_fiji_dfs(df_sam: pd.DataFrame, df_fiji: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the SAM DataFrame and the Fiji DataFrame on the 'frame' column.

    Args:
        df_sam (pd.DataFrame): DataFrame containing SAM data with a 'frame' column.
        df_fiji (pd.DataFrame): DataFrame containing Fiji data with a 'frame' column.

    Returns:
        pd.DataFrame: A merged DataFrame, or an empty DataFrame if an error occurs
                      or one of the input DataFrames is empty.
    """
    if df_sam.empty or df_fiji.empty:
        print("One or both DataFrames are empty. Cannot merge.")
        if df_sam.empty and not df_fiji.empty:
            return df_fiji # Or handle as an error/return empty
        elif not df_sam.empty and df_fiji.empty:
            return df_sam # Or handle as an error/return empty
        return pd.DataFrame()

    try:
        # Ensure 'frame' column is of the same type if necessary, though merge usually handles it.
        # df_sam['frame'] = df_sam['frame'].astype(int)
        # df_fiji['frame'] = df_fiji['frame'].astype(int)
        
        merged_df = pd.merge(df_sam, df_fiji, on='frame', how='outer')
        # 'outer' merge to keep all frames from both, filling with NaN if a frame is in one but not the other.
        # Use 'inner' if you only want frames present in both.
        # Use 'left' to keep all frames from df_sam and matching from df_fiji.
        
        print("Successfully merged SAM and Fiji DataFrames.")
        return merged_df
    except KeyError:
        print("Error: 'frame' column not found in one or both DataFrames.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred during merging: {e}")
        return pd.DataFrame()

# Merge SAM and Fiji data
combined_df = merge_sam_and_fiji_dfs(df_wide_brightness_and_background, merged_fiji_df)

if not combined_df.empty:
    print("\nHead of combined SAM and Fiji DataFrame:")
    print(combined_df.head())
    print("\nInfo of combined SAM and Fiji DataFrame:")
    combined_df.info()
    
    # Example: Save the fully combined DataFrame
    combined_output_path = os.path.join(output_dir, f"{video_name}_sam_fiji_combined.csv")
    combined_df.to_csv(combined_output_path, index=False)
    # print(f"Combined data saved to {combined_output_path}")

#endregion


#region [Plot results]

def normalize_column(series: pd.Series) -> pd.Series:
    """Normalizes a pandas Series to the range [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.zeros_like(series.values), index=series.index) # Or handle as all 0.5, or raise error
    return (series - min_val) / (max_val - min_val)

# Ensure combined_df is available and not empty
if not combined_df.empty:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Data for subplots
    plot_data = [
        {"sam_col": "2_max", "fiji_col": "fiji_nrd_max", "title": "NRD Comparison (Normalized Max Intensity)"},
        {"sam_col": "3_max", "fiji_col": "fiji_nrv_max", "title": "NRV Comparison (Normalized Max Intensity)"},
        {"sam_col": "4_max", "fiji_col": "fiji_loop_max", "title": "Loop Comparison (Normalized Max Intensity)"}
    ]

    for i, data in enumerate(plot_data):
        ax = axes[i]
        sam_col_name = data["sam_col"]
        fiji_col_name = data["fiji_col"]

        # Check if columns exist before trying to normalize and plot
        if sam_col_name in combined_df.columns and fiji_col_name in combined_df.columns:
            norm_sam = normalize_column(combined_df[sam_col_name])
            norm_fiji = normalize_column(combined_df[fiji_col_name])

            ax.plot(combined_df['frame'].values, norm_sam.values, label=f'SAM: Normalized {sam_col_name}')
            ax.plot(combined_df['frame'].values, norm_fiji.values, label=f'Fiji: Normalized {fiji_col_name}', linestyle='--')
            
            ax.set_ylabel("Normalized Max Intensity")
            ax.set_title(data["title"])
            ax.legend()
            ax.grid(True)
        else:
            missing_cols = []
            if sam_col_name not in combined_df.columns:
                missing_cols.append(sam_col_name)
            if fiji_col_name not in combined_df.columns:
                missing_cols.append(fiji_col_name)
            ax.text(0.5, 0.5, f"Data for columns: {', '.join(missing_cols)} not found", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(data["title"])
            print(f"Warning: Columns {', '.join(missing_cols)} not found for subplot {i+1}")


    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    
    # Save the plot
    plot_output_dir = "/home/lilly/phd/ria/benchmarks/brightest/sampleplots"
    plot_output_path = os.path.join(plot_output_dir, f"{video_name}_sam_fiji_comparison_plot.png")
    plt.savefig(plot_output_path)
    print(f"Comparison plot saved to {plot_output_path}")
    
    plt.show()
    plt.close()

else:
    print("Combined DataFrame is empty. Cannot generate plot.")

#endregion

