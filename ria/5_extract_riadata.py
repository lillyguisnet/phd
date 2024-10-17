import h5py
import numpy as np
import os
import cv2
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import distance_transform_edt
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

filename = '/home/lilly/phd/ria/data_analyzed/cleaned_segments/AG-MMH122_10s_20190830_04_crop_riasegmentation_cleanedsegments.h5'

cleaned_segments = load_cleaned_segments_from_h5(filename)



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
    image_path = f"/home/lilly/phd/ria/data_foranalysis/riacrop/AG-MMH122_10s_20190830_04_crop/{frame_idx:06d}.jpg"
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
    df.set_index('frame', inplace=True)
    
    return df

# Main script
filename = '/home/lilly/phd/ria/data_analyzed/cleaned_segments/AG-MMH122_10s_20190830_04_crop_riasegmentation_cleanedsegments.h5'
cleaned_segments = load_cleaned_segments_from_h5(filename)

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

df_wide_bg_corrected = create_wide_format_table_with_bg_correction_and_pixel_count(mean_values, pixel_counts)
df_wide_bg_corrected.columns = df_wide_bg_corrected.columns.astype(str)

if 'background' not in df_wide_bg_corrected.columns:
    background_values = [frame_data['background'] for frame_data in mean_values.values()]
    df_wide_bg_corrected['background'] = background_values

background_column = ['background']
original_columns = [col for col in df_wide_bg_corrected.columns if not col.endswith('_bg_corrected') and not col.endswith('_pixel_count')]
original_columns = [col for col in original_columns if col != 'background']
bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_bg_corrected')]
pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]

all_columns = background_column + original_columns + bg_corrected_columns + pixel_count_columns

print("\nBasic statistics for all values (including background and pixel counts):")
print(df_wide_bg_corrected[all_columns].describe())

correlation_matrix = df_wide_bg_corrected[all_columns].corr()
print("\nCorrelation matrix:")
print(correlation_matrix)


### Normalize the data
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)

# Identify background-corrected columns and pixel count columns
bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if isinstance(col, str) and col.endswith('_bg_corrected')]
pixel_count_columns = [col for col in df_wide_bg_corrected.columns if isinstance(col, str) and col.endswith('_pixel_count')]

# Create normalized columns for background-corrected and pixel count columns
for col in bg_corrected_columns + pixel_count_columns:
    norm_col_name = f"{col}_normalized"
    df_wide_bg_corrected[norm_col_name] = normalize(df_wide_bg_corrected[col])

# Update column lists
normalized_columns = [col for col in df_wide_bg_corrected.columns if isinstance(col, str) and col.endswith('_normalized')]

all_columns = df_wide_bg_corrected.columns.tolist()

print(df_wide_bg_corrected.describe())

# Correlation matrix for all values
correlation_matrix = df_wide_bg_corrected.corr()
print(correlation_matrix)



### Plot the data
# Identify normalized columns
normalized_columns = [col for col in df_wide_bg_corrected.columns if isinstance(col, str) and col.endswith('_normalized')]

# Create a color palette for the objects
color_palette = sns.color_palette("husl", len(normalized_columns))

# Create the first plot (Normalized Background-Corrected Values)
plt.figure(figsize=(40, 8))

# Convert index to numpy array
x = df_wide_bg_corrected.index.to_numpy()

# Define new labels
new_labels = {
    '2': '2 (nrd)',
    '3': '3 (nrv)',
    '4': '4 (loop)'
}

for i, column in enumerate(normalized_columns):
    obj_id = column.replace('_bg_corrected_normalized', '')
    label = new_labels.get(obj_id, obj_id)
    plt.plot(x, df_wide_bg_corrected[column].to_numpy(), 
             label=label, 
             color=color_palette[i], linewidth=2)

plt.xlabel('Frame Number', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title('Normalized Background-Corrected Values Over Frames', fontsize=18)
plt.legend(title='Objects', title_fontsize='14', fontsize='12', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim(x.min(), x.max())
plt.xticks(np.arange(x.min(), x.max() + 1, 10), fontsize=12)

plt.tight_layout()
plt.savefig('normalized_values_over_frames.png', dpi=300, bbox_inches='tight')
plt.close()



# Create the second plot (Normalized Background-Corrected Values and Pixel Counts)
# Create subplots for each object
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 24), sharex=True)
axes = [ax1, ax2, ax3]

# Identify pixel count columns and normalized background-corrected columns
pixel_count_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_pixel_count')]
normalized_bg_corrected_columns = [col for col in df_wide_bg_corrected.columns if col.endswith('_bg_corrected_normalized')]

# Create plots for each object
for i, obj_id in enumerate(['2', '3', '4']):
    ax = axes[i]
    ax2 = ax.twinx()

    # Plot normalized pixel counts
    pixel_count_col = f'{obj_id}_pixel_count'
    normalized_pixel_counts = df_wide_bg_corrected[pixel_count_col] / df_wide_bg_corrected[pixel_count_col].max()
    ax.plot(x, normalized_pixel_counts.to_numpy(), 
            label=f"{new_labels[obj_id]} (Normalized Pixel Count)", 
            color=color_palette[i], linewidth=2, linestyle='--')

    # Plot normalized background-corrected values
    bg_corrected_col = f'{obj_id}_bg_corrected_normalized'
    ax2.plot(x, df_wide_bg_corrected[bg_corrected_col].to_numpy(), 
             label=f"{new_labels[obj_id]} (Normalized BG-Corrected)", 
             color=color_palette[i], linewidth=2)

    ax.set_ylabel('Normalized Pixel Count', fontsize=14)
    ax2.set_ylabel('Normalized BG-Corrected Value', fontsize=14)
    ax.set_title(f'Object {new_labels[obj_id]}', fontsize=18)

    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize='12', loc='center left', bbox_to_anchor=(1, 0.5))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(x.min(), x.max())

# Set common x-axis label
axes[-1].set_xlabel('Frame Number', fontsize=14)

# Set common x-axis ticks
axes[-1].set_xticks(np.arange(x.min(), x.max() + 1, 10))
axes[-1].tick_params(axis='x', labelsize=12)

plt.tight_layout()
plt.savefig('normalized_values_and_pixel_counts_over_frames.png', dpi=300, bbox_inches='tight')
plt.close()




###Comapre to Fiji

fiji_data = pd.read_excel('/home/lilly/phd/ria/MMH122_10s_20190830_04.xlsx')

# Normalize the columns nrD, nrV and loop
columns_to_normalize = ['nrD', 'nrV', 'loop']
for col in columns_to_normalize:
    norm_col_name = f"{col}_normalized"
    fiji_data[norm_col_name] = normalize(fiji_data[col])



# Create a color palette for the objects
color_palette = sns.color_palette("husl", len(columns_to_normalize))

# Create the plot
plt.figure(figsize=(40, 8))

# Convert 'Frame' to numpy array for x-axis
x = fiji_data['Frame'].to_numpy()

# Define new labels
new_labels = {
    'nrD': 'nrD (2)',
    'nrV': 'nrV (3)',
    'loop': 'loop (4)'
}

for i, column in enumerate(columns_to_normalize):
    norm_col_name = f"{column}_normalized"
    label = new_labels.get(column, column)
    plt.plot(x, fiji_data[norm_col_name].to_numpy(), 
             label=label, 
             color=color_palette[i], linewidth=2)

plt.xlabel('Frame Number', fontsize=14)
plt.ylabel('Normalized Value', fontsize=14)
plt.title('Normalized Values Over Frames (Fiji Data)', fontsize=18)
plt.legend(title='Objects', title_fontsize='14', fontsize='12', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim(x.min(), x.max())
plt.xticks(np.arange(x.min(), x.max() + 1, 10), fontsize=12)

plt.tight_layout()
plt.savefig('normalized_values_over_frames_fiji.png', dpi=300, bbox_inches='tight')
plt.close()




###Combine the methods
# Create a color palette for the methods
color_palette = sns.color_palette("husl", 2)  # 2 colors for 2 methods

# Create the plot with 3 subplots (one for each object)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(40, 24), sharex=True)

# Convert 'Frame' to numpy array for x-axis
x = fiji_data['Frame'].to_numpy()

# Define new labels and data pairs
objects_data = [
    ('nrD (2)', 'nrD_normalized', '2_bg_corrected_normalized', ax1),
    ('nrV (3)', 'nrV_normalized', '3_bg_corrected_normalized', ax2),
    ('loop (4)', 'loop_normalized', '4_bg_corrected_normalized', ax3)
]

for obj_name, fiji_col, our_col, ax in objects_data:
    # Plot Fiji data
    ax.plot(x, fiji_data[fiji_col].to_numpy(), 
            label=f'{obj_name} - Fiji', 
            color=color_palette[0], linewidth=2, linestyle='-')

    # Plot our method data
    ax.plot(x, df_wide_bg_corrected[our_col].to_numpy(), 
            label=f'{obj_name} - Our Method', 
            color=color_palette[1], linewidth=2, linestyle='--')

    ax.set_ylabel('Normalized Value', fontsize=14)
    ax.set_title(f'Comparison of Normalized Values for {obj_name}', fontsize=18)
    ax.legend(title='Methods', title_fontsize='14', fontsize='12', loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(x.min(), x.max())
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.xlabel('Frame Number', fontsize=14)
plt.xticks(np.arange(x.min(), x.max() + 1, 10))

plt.tight_layout()
plt.savefig('comparison_normalized_values_fiji_vs_our_method_split.png', dpi=300, bbox_inches='tight')
plt.close()
