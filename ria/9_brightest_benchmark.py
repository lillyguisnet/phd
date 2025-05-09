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


# --- Configuration (can be adjusted by user) ---
# Root directory of the project. All other paths are derived from this.
PROJECT_ROOT_DIR = "/home/lilly/phd/ria" 

# Base directories for different types of data, derived from PROJECT_ROOT_DIR
# These point to the parent directories that would contain group-specific folders (like AG_WT)
ANALYZED_DATA_PARENT_DIR = os.path.join(PROJECT_ROOT_DIR, "data_analyzed")
RAW_FRAMES_PARENT_DIR = os.path.join(PROJECT_ROOT_DIR, "data_foranalysis")
BENCHMARKS_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, "benchmarks/brightest")

# Output directories for this benchmark script's results
# These are specific to the "brightest" benchmark
SAM_CSV_OUTPUT_DIR = os.path.join(BENCHMARKS_ROOT_DIR, "data/sam")
PLOTS_OUTPUT_DIR = os.path.join(BENCHMARKS_ROOT_DIR, "sampleplots")

# Ensure output directories exist at the start
os.makedirs(SAM_CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)


# Define the base directory for video frames
# VIDEO_FRAMES_BASE_DIR = "/home/lilly/phd/ria/data_foranalysis/AG_WT/riacrop" # Will be derived

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
# filename = "/home/lilly/phd/ria/data_analyzed/AG_WT/cleaned_aligned_segments/AG_WT-MMH99_10s_20190320_03_crop_riasegmentation_cleanedalignedsegments.h5" # Will be derived

# Load the cleaned segments from the specified file
# cleaned_segments = load_cleaned_segments_from_h5(filename) # Moved into main processing function

# Determine the specific video's frames directory
# video_name_for_frames = os.path.basename(filename).split('_riasegmentation_')[0] # Will be derived
# current_video_frames_dir = os.path.join(VIDEO_FRAMES_BASE_DIR, video_name_for_frames) # Will be derived



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
        data['frame'].append(frame_idx + 1)
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

# df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments, current_video_frames_dir) # Moved

# Save the DataFrame to the specified directory
# output_dir = "/home/lilly/phd/ria/benchmarks/brightest/data/sam" # Will use SAM_CSV_OUTPUT_DIR

# Extract video name from filename
# video_name = os.path.basename(filename).split('.')[0] # Will be derived as output_filename_stem
# output_path = os.path.join(output_dir, f"{video_name}_brightest_sam.csv") # Will be constructed
# df_wide_brightness_and_background.to_csv(output_path, index=False) # Moved

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
# FIJI_DATA_DIRECTORY = "/home/lilly/phd/ria/benchmarks/brightest/data/fiji" # Will be an argument
# merged_fiji_df = merge_fiji_xlsx_to_wide(FIJI_DATA_DIRECTORY) # Moved

# if not merged_fiji_df.empty:
#     print("\nHead of merged Fiji DataFrame:")
#     print(merged_fiji_df.head())
    # You might want to save it or process it further
    # merged_fiji_df.to_csv("merged_fiji_data_wide.csv", index=False)


def merge_sam_and_fiji_dfs(df_sam: pd.DataFrame, df_fiji: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the SAM DataFrame and the Fiji DataFrame on the 'frame' column.
    It's expected that df_fiji is not empty when this function is called by process_video_dataset.
    If df_sam is empty, a warning is printed, and the merge proceeds (resulting in df_fiji).
    """
    # This is a safeguard; process_video_dataset should ensure df_fiji is not empty.
    if df_fiji.empty:
        print("CRITICAL ERROR in merge_sam_and_fiji_dfs: Fiji DataFrame is unexpectedly empty. This should have been caught by the caller. Aborting merge.")
        return pd.DataFrame()

    # sam_was_empty = False # This logic is now less relevant here, as caller ensures df_sam is not empty.
    # if df_sam.empty:
    #     print("Warning: SAM DataFrame is empty. The 'combined' DataFrame will effectively be the Fiji data after merge.")
    #     sam_was_empty = True
    #     # pd.merge with how='outer' handles one df being empty correctly if 'on' key is present in non-empty one.
    # This check is now a safeguard for an unexpected state, as process_video_dataset should ensure df_sam is not empty.
    if df_sam.empty:
        print("CRITICAL ERROR in merge_sam_and_fiji_dfs: SAM DataFrame is unexpectedly empty. This should have been caught by the caller. Aborting merge.")
        return pd.DataFrame()

    try:
        # Using suffixes for robustness in case of any unexpected column name overlaps,
        # though current naming conventions for SAM (e.g., '2_mean') and Fiji (e.g., 'fiji_nrd_Mean')
        # should prevent clashes on data columns. The 'frame' column is the merge key.
        merged_df = pd.merge(df_sam, df_fiji, on='frame', how='outer', suffixes=('_sam_data', '_fiji_data'))
        
        print("Successfully merged SAM and Fiji DataFrames.")
        # if sam_was_empty: # No longer needed here.
        #     print("Note: SAM data was empty; the merged DataFrame primarily reflects Fiji data.")
        return merged_df
    except KeyError:
        # This implies 'frame' column was missing, which is critical.
        # merge_fiji_xlsx_to_wide should add 'frame' to fiji_df.
        # process_cleaned_segments should ensure 'frame' in sam_df (if not empty).
        print("CRITICAL Error: 'frame' column not found in one or both DataFrames during merge. Cannot combine.")
        return pd.DataFrame()
    except Exception as e:
        print(f"CRITICAL Error: An unexpected error occurred during SAM and Fiji DataFrame merging: {e}")
        return pd.DataFrame()

# Merge SAM and Fiji data
# combined_df = merge_sam_and_fiji_dfs(df_wide_brightness_and_background, merged_fiji_df) # Moved

# if not combined_df.empty:
#     print("\nHead of combined SAM and Fiji DataFrame:")
#     print(combined_df.head())
#     print("\nInfo of combined SAM and Fiji DataFrame:")
#     combined_df.info()
    
    # Example: Save the fully combined DataFrame
    # combined_output_path = os.path.join(output_dir, f"{video_name}_sam_fiji_combined.csv") # Moved
    # combined_df.to_csv(combined_output_path, index=False) # Moved
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

def generate_comparison_plot(combined_df: pd.DataFrame, plot_output_file_path: str, video_id: str):
    """
    Generates and saves a comparison plot from the combined DataFrame.
    Args:
        combined_df (pd.DataFrame): DataFrame containing merged SAM and Fiji data.
        plot_output_file_path (str): Full path to save the generated plot.
        video_id (str): The ID of the video being analyzed, used for the plot title.
    """
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot generate plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Comparison Plot for Video: {video_id}", fontsize=16)

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
            ax.plot(combined_df['frame'].values, norm_fiji.values, label=f'Fiji: Normalized {fiji_col_name}')
            
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
            
            log_message = f"Data for columns: {', '.join(missing_cols)} not found for plot: {data['title']}"
            if combined_df.empty or not any(col in combined_df.columns for col in [sam_col_name, fiji_col_name]):
                 ax.text(0.5, 0.5, log_message, 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else: # Plot available data if one of the pair is missing but others exist
                if sam_col_name in combined_df.columns:
                    norm_sam = normalize_column(combined_df[sam_col_name])
                    ax.plot(combined_df['frame'].values, norm_sam.values, label=f'SAM: Normalized {sam_col_name}')
                if fiji_col_name in combined_df.columns: # Should not happen if the first if failed for fiji_col_name
                    norm_fiji = normalize_column(combined_df[fiji_col_name])
                    ax.plot(combined_df['frame'].values, norm_fiji.values, label=f'Fiji: Normalized {fiji_col_name}', linestyle='--')
                ax.legend()

            ax.set_title(data["title"])
            print(f"Warning: {log_message}")


    axes[-1].set_xlabel("Frame")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    plt.savefig(plot_output_file_path)
    print(f"Comparison plot saved to {plot_output_file_path}")
    
    # plt.show() # Optionally disable interactive showing if running in batch
    plt.close(fig)

#endregion



# --- Main Processing Function ---
def process_video_dataset(fiji_video_id_input_dir: str, group_prefix: str):
    """
    Processes a single video dataset: loads SAM data, Fiji data, merges them,
    saves CSVs, and generates comparison plots.

    Args:
        fiji_video_id_input_dir (str): Path to the directory containing Fiji XLSX files 
                                       for a specific video ID.
                                       e.g., "/home/lilly/phd/ria/benchmarks/brightest/data/fiji/MMH99_10s_20190320_03"
        group_prefix (str): The group prefix (e.g., "AG_WT") used in H5 file and 
                            raw frame directory naming conventions.
    """
    video_id = os.path.basename(fiji_video_id_input_dir)
    print(f"\n--- Processing dataset for Video ID: {video_id} (Group: {group_prefix}) ---")

    # --- Construct paths based on video_id and group_prefix ---
    h5_filename_stem = f"{group_prefix}-{video_id}_crop_riasegmentation_cleanedalignedsegments"
    cleaned_segments_h5_file = os.path.join(
        ANALYZED_DATA_PARENT_DIR, group_prefix, "cleaned_aligned_segments", f"{h5_filename_stem}.h5"
    )

    video_frames_subdir_name = f"{group_prefix}-{video_id}_crop"
    current_video_frames_dir = os.path.join(
        RAW_FRAMES_PARENT_DIR, group_prefix, "riacrop", video_frames_subdir_name
    )

    # --- Input validation ---
    if not os.path.isfile(cleaned_segments_h5_file):
        print(f"Error: Cleaned segments H5 file not found: {cleaned_segments_h5_file}")
        return
    if not os.path.isdir(current_video_frames_dir):
        print(f"Error: Video frames directory not found: {current_video_frames_dir}")
        return
    if not os.path.isdir(fiji_video_id_input_dir):
        print(f"Error: Fiji data directory not found: {fiji_video_id_input_dir}")
        return

    print(f"Using H5 file: {cleaned_segments_h5_file}")
    print(f"Using video frames from: {current_video_frames_dir}")
    print(f"Using Fiji data from: {fiji_video_id_input_dir}")

    # --- Part 1: SAM brightest processing ---
    print("\n[Phase 1: Processing SAM brightness data]")
    cleaned_segments = load_cleaned_segments_from_h5(cleaned_segments_h5_file)
    if not cleaned_segments:
        print(f"CRITICAL ERROR: Failed to load cleaned segments from {cleaned_segments_h5_file} (H5 file might be corrupt or empty). Aborting processing for Video ID: {video_id}.")
        return

    df_wide_brightness_and_background = process_cleaned_segments(cleaned_segments, current_video_frames_dir)
    if df_wide_brightness_and_background.empty:
        print(f"CRITICAL ERROR: SAM brightness processing for Video ID '{video_id}' (from H5 file '{cleaned_segments_h5_file}') resulted in an empty DataFrame. This indicates no valid SAM data could be extracted or processed. Aborting processing for this video.")
        return
    
    sam_csv_output_path = os.path.join(SAM_CSV_OUTPUT_DIR, f"{h5_filename_stem}_brightest_sam.csv")
    df_wide_brightness_and_background.to_csv(sam_csv_output_path, index=False)
    print(f"SAM brightness data saved to {sam_csv_output_path}")

    # --- Part 2: Merge Fiji XLSX ---
    print("\n[Phase 2: Processing Fiji data]")
    merged_fiji_df = merge_fiji_xlsx_to_wide(fiji_video_id_input_dir)
    if merged_fiji_df.empty:
        print(f"CRITICAL ERROR: Fiji data processing from '{fiji_video_id_input_dir}' resulted in an empty DataFrame (no .xlsx files found or files were unreadable). Aborting processing for Video ID: {video_id}.")
        return 
    else:
        print(f"Successfully processed Fiji data from {fiji_video_id_input_dir}.")

    # --- Part 3: Merge SAM and Fiji DataFrames ---
    print("\n[Phase 3: Merging SAM and Fiji data]")
    # At this point, merged_fiji_df is guaranteed to be non-empty.
    # df_wide_brightness_and_background (SAM data) might be empty, which is handled by merge_sam_and_fiji_dfs.
    combined_df = merge_sam_and_fiji_dfs(df_wide_brightness_and_background, merged_fiji_df)

    if not combined_df.empty:
        # This means the merge was successful. 
        # If SAM data was empty, merge_sam_and_fiji_dfs would have printed a warning, 
        # and combined_df would effectively be merged_fiji_df.
        print("Successfully created combined DataFrame.")
        # Note about SAM data being empty is now printed within merge_sam_and_fiji_dfs
        print("\nHead of combined DataFrame:")
        print(combined_df.head(3))
        
        combined_csv_output_path = os.path.join(SAM_CSV_OUTPUT_DIR, f"{h5_filename_stem}_sam_fiji_combined.csv")
        combined_df.to_csv(combined_csv_output_path, index=False)
        print(f"Combined data saved to {combined_csv_output_path}")
    else:
        # This 'else' will be reached if merge_sam_and_fiji_dfs returned an empty DataFrame,
        # indicating a critical failure within the merge operation itself (e.g., KeyError, other exception),
        # as an empty merged_fiji_df (the main Fiji data source) is caught in Phase 2.
        sam_status = "empty" if df_wide_brightness_and_background.empty else "present"
        # merged_fiji_df is guaranteed not empty here, so no need to check its status for this message.
        print(f"CRITICAL ERROR: Failed to create a combined DataFrame for Video ID: {video_id} (SAM data was {sam_status}). This usually indicates a problem during the merge operation itself (e.g., missing 'frame' column or other mismatch). Review messages from the merge function. Aborting further steps for this video.")
        return 

    # --- Part 4: Plot results ---
    print("\n[Phase 4: Generating comparison plot]")
    if not combined_df.empty:
        plot_output_file_path = os.path.join(PLOTS_OUTPUT_DIR, f"{h5_filename_stem}_sam_fiji_comparison_plot.png")
        generate_comparison_plot(combined_df, plot_output_file_path, video_id)
    else:
        print("Plotting skipped as combined data is empty.")
    
    print(f"--- Finished processing for Video ID: {video_id} ---")




# --- User: Define the specific dataset to process here ---
# Example:
fiji_directory_for_video = "/home/lilly/phd/ria/benchmarks/brightest/data/fiji/MMH99_10s_20190320_03"
data_group_prefix = "AG_WT"

# To process a different dataset, change the two variables above.
# For example, for a different video and group:
# fiji_directory_for_video = "/path/to/your/benchmarks/brightest/data/fiji/ANOTHER_VIDEO_ID"
# data_group_prefix = "ANOTHER_GROUP"
if not fiji_directory_for_video or not data_group_prefix:
    print("Error: Please set 'fiji_directory_for_video' and 'data_group_prefix' in the script.")
else:
    process_video_dataset(fiji_directory_for_video, data_group_prefix)




#region [Compute agreement between Fiji and SAM per compartment per worm]









#endregion
