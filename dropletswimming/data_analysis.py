import h5py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# region [Data analysis]

###Combine all the h5 files into a single dataframe###

def load_h5_to_dict(filename):
    with h5py.File(filename, 'r') as h5file:
        return _recursively_load_dict_contents_from_group(h5file)

def _recursively_load_dict_contents_from_group(h5file):
    ans = {}
    for key, item in h5file.items():
        if isinstance(item, h5py.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            if all(k.isdigit() for k in item.keys()):
                # This is a list
                ans[key] = [_recursively_load_dict_contents_from_group(item[k]) if isinstance(item[k], h5py.Group)
                            else item[k][()] for k in sorted(item.keys(), key=int)]
            else:
                ans[key] = _recursively_load_dict_contents_from_group(item)
    return ans

def extract_info_from_filename(filename):
    base_filename = os.path.basename(filename)
    parts = base_filename.split('-')
    viscosity = parts[0]
    condition = parts[1]
    id = parts[2]
    return viscosity, condition, id

def dict_to_dataframe(data_dict, filename, ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = []
    elif isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]
    
    max_length = max(len(value) if isinstance(value, (list, np.ndarray)) else 1 
                     for key, value in data_dict.items() if key not in ignore_keys)

    viscosity, condition, id = extract_info_from_filename(filename)

    df_dict = {
        'viscosity': [viscosity] * max_length,
        'condition': [condition] * max_length,
        'id': [id] * max_length
    }
    
    for key, value in data_dict.items():
        if key not in ignore_keys:
            if isinstance(value, (list, np.ndarray)):
                if len(value) == max_length:
                    df_dict[key] = value
                elif len(value) < max_length:
                    df_dict[key] = np.pad(value, (0, max_length - len(value)), 
                                          mode='constant', constant_values=np.nan)
                else:
                    df_dict[key] = value[:max_length]
            else:
                df_dict[key] = [value] * max_length

    return pd.DataFrame(df_dict)

def process_h5_files(folder_path, ignore_keys=None, max_files=None):
    all_dataframes = []
    
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if max_files is not None:
        h5_files = h5_files[:max_files]
    
    for filename in h5_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")
        data_dict = load_h5_to_dict(file_path)
        df = dict_to_dataframe(data_dict, filename, ignore_keys=ignore_keys)
        all_dataframes.append(df)
    
    return pd.concat(all_dataframes, ignore_index=True)

# Usage
folder_path = '/home/maxime/prg/phd/dropletswimming/data_analyzed/final_shapenalysis'
ignore_keys = ['masks', 'fps', 'smooth_points']
max_files = 30  # Set this to the number of files you want to process, or None for all files

combined_df = process_h5_files(folder_path, ignore_keys=ignore_keys, max_files=max_files)

# Save the combined DataFrame to a CSV file
output_file = 'combined_data_sample.csv'
combined_df.to_csv(output_file, index=False)
print(f"Combined data saved to {output_file}")

# Display some information about the combined DataFrame
print(f"Combined DataFrame shape: {combined_df.shape}")
print("\nColumn names:")
print(combined_df.columns.tolist())
print("\nFirst few rows:")
print(combined_df.head())

# If you want to see which files were processed
processed_files = combined_df['id'].unique()
print("\nProcessed files:")
print(processed_files)
 
### Detect when the worm is turned on the z-axis by finding frames where worm length is smaller than 1 std of the mean worm length
def detect_z_axis_turns(df):
    # Group by unique video ID
    grouped = df.groupby(['viscosity', 'condition', 'id'])
    
    # Function to detect z-axis turns for each group
    def detect_turns(group):
        mean_length = group['worm_lengths'].mean()
        std_length = group['worm_lengths'].std()
        threshold = mean_length - std_length
        return pd.DataFrame({
            'z_axis_turn': group['worm_lengths'] < threshold,
            'video_id': f"{group['viscosity'].iloc[0]}_{group['condition'].iloc[0]}_{group['id'].iloc[0]}"
        })

    # Apply the function to each group
    result = grouped.apply(detect_turns).reset_index(level=[0,1,2], drop=True)
    
    # Add the results as new columns
    df['z_axis_turn'] = result['z_axis_turn']
    df['video_id'] = result['video_id']

    return df

# Apply the function to detect z-axis turns
combined_df = detect_z_axis_turns(combined_df)

# Print summary of detected z-axis turns
total_turns = combined_df['z_axis_turn'].sum()
total_frames = len(combined_df)
print(f"Detected {total_turns} frames with potential z-axis turns out of {total_frames} total frames")
print(f"Percentage of frames with z-axis turns: {(total_turns/total_frames)*100:.2f}%") ###13.4%


#For each video, calculate % of frames in z-axis turn, c-shape, s-shape and straight

# Group by video ID
combined_df['shape'] = combined_df['shape'].str.decode('utf-8')
combined_df['shape'] = combined_df['shape'].astype('string')
grouped = combined_df.groupby('video_id')

# Function to calculate percentages for each shape
def calculate_shape_percentages(group):
    total_frames = len(group)
    z_axis_turns = group['z_axis_turn'].sum()
    
    # Count other shapes only if it's not a z-axis turn
    c_shapes = ((group['shape'] == 'C-shape') & (~group['z_axis_turn'])).sum()
    s_shapes = ((group['shape'] == 'S-shape') & (~group['z_axis_turn'])).sum()
    straight = ((group['shape'] == 'Straight') & (~group['z_axis_turn'])).sum()

    return pd.Series({
        'z_axis_turn_percentage': (z_axis_turns / total_frames) * 100,
        'c_shape_percentage': (c_shapes / total_frames) * 100,
        's_shape_percentage': (s_shapes / total_frames) * 100,
        'straight_percentage': (straight / total_frames) * 100,
        'viscosity': group['viscosity'].iloc[0],
        'condition': group['condition'].iloc[0],
        'id': group['id'].iloc[0]
    })

# Apply the function to each group
shape_percentages = grouped.apply(calculate_shape_percentages).reset_index()

""" # Convert the dictionary in column '0' to separate columns
shape_percentages = pd.concat([shape_percentages, shape_percentages[0].apply(pd.Series)], axis=1)
shape_percentages = shape_percentages.drop(columns=[0]) """

# Ensure all required columns are present
required_columns = ['z_axis_turn_percentage', 'c_shape_percentage', 's_shape_percentage', 'straight_percentage']
for column in required_columns:
    if column not in shape_percentages.columns:
        shape_percentages[column] = 0.0

# Add the percentage columns to the original dataframe
combined_df = combined_df.merge(shape_percentages, on='video_id', how='left')

combined_df['c_shape_percentage'].unique()

# endregion [Data analysis]


###Plots###

#loaded_dict.keys()
#dict_keys([
# 'frames',
# 'shape',
# 'smoothed_worm_lengths', 'worm_lengths'
# 'avg_amplitudes', 'smoothed_avg_amplitudes',
# 'max_amplitudes', 'smoothed_max_amplitudes', 
# 'curvatures', ###Curvature along worm length by frame
# 'curvature_time_series', ###Normalized (by video) mean curvature value for the frame 
# 'dominant_spatial_freqs', ###Most prominent spatial frequency (cycles per pixel) in the worm's shape for each frame. ~spread
# 'interpolated_freqs', ###Dominant temporal frequency by frame. Oscillation rate, how often a full cycle is completed
# 'f', ###The different oscillation frequencies (cycles per second) present in the worm's movement
# 'psd', ###Power spectral density, importance of each of these frequencies in the overall movement pattern
# 'smoothed_wavelengths', 'wavelengths', ###Number of pixels along the worm to make 1 wave
# 'smoothed_wave_numbers', 'wave_numbers', ###Number of wavelengths per worm length
# 'normalized_wavelengths', 'smoothed_normalized_wavelengths', ###Wavelength as a proportion of the worm's length


# region [Shape percentage plots]

# Create a figure with subplots for each shape
fig, axs = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('Percentage of frames in each shape for each condition', fontsize=16)

shapes = ['straight_percentage', 's_shape_percentage', 'c_shape_percentage', 'z_axis_turn_percentage']
titles = ['Straight', 'S-shape', 'C-shape', 'Z-axis Turn']

# Use shape_percentages DataFrame which has one row per video_id
# Find the global max value across all shapes and add a 10% buffer
y_max = max(shape_percentages[shapes].max())
y_max_with_buffer = y_max * 1.1  # Add 10% buffer

for i, (shape, title) in enumerate(zip(shapes, titles)):
    sns.boxplot(x='condition', y=shape, data=shape_percentages, order=sorted(shape_percentages['condition'].unique()), ax=axs[i], boxprops=dict(alpha=0.3))
    sns.swarmplot(x='condition', y=shape, data=shape_percentages, order=sorted(shape_percentages['condition'].unique()), ax=axs[i], color="black", size=5, alpha=1)
    axs[i].set_title(f'Percentage of {title} frames')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylabel(f'Percentage of {title} frames')
    axs[i].tick_params(axis='x', rotation=45)
    axs[i].set_ylim(0, y_max_with_buffer)  # Set the y-axis limits from 0 to the buffered max

plt.tight_layout()
plt.savefig('shape_percentage_boxplot.png')
plt.close()

# endregion [Shape percentage plots]

# region [Max amplitude plots]

###### Boxplot of average amplitude (for each frame) per condition ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]

# Calculate mean values for condition "a"
mean_a_avg = filtered_df[filtered_df['condition'] == 'a']['avg_amplitudes'].mean()
mean_a_smoothed = filtered_df[filtered_df['condition'] == 'a']['smoothed_avg_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_avg_amplitudes'] = (filtered_df['avg_amplitudes'] - mean_a_avg) / mean_a_avg
filtered_df['normalized_smoothed_avg_amplitudes'] = (filtered_df['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed
filtered_df = filtered_df.rename(columns={'condition_x': 'condition'})

# Create a figure with subplots for normalized avg amplitude and normalized smoothed avg amplitude
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Normalized Average Amplitude and Smoothed Average Amplitude per Condition', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(filtered_df['normalized_avg_amplitudes'].min(), filtered_df['normalized_smoothed_avg_amplitudes'].min())
y_max = max(filtered_df['normalized_avg_amplitudes'].max(), filtered_df['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

# Boxplot for normalized avg_amplitudes
sns.boxplot(x='condition', y='normalized_avg_amplitudes', data=filtered_df, order=sorted(filtered_df['condition'].unique()), ax=axs[0])
axs[0].set_title('Normalized Average Amplitude (excluding z-axis turns)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Normalized Average Amplitude')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(y_min_with_buffer, y_max_with_buffer)

# Boxplot for normalized smoothed_avg_amplitudes
sns.boxplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=filtered_df, order=sorted(filtered_df['condition'].unique()), ax=axs[1])
axs[1].set_title('Normalized Smoothed Average Amplitude (excluding z-axis turns)')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Normalized Smoothed Average Amplitude')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('normalized_avgamplitude_boxplots_no_zaxis_turns.png')
plt.close()

###### Box plot of absolute average amplitude value in video ######

# Get the maximum value for each "video_id" for 'avg_amplitudes' and 'smoothed_avg_amplitudes'
# Calculate avg amplitudes per video
avg_amplitudes_per_video = combined_df[~combined_df['z_axis_turn']].groupby('video_id').agg({
    'avg_amplitudes': 'max',
    'smoothed_avg_amplitudes': 'max',
    'condition_x': 'first'  # Keep the condition for each video_id
}).reset_index()
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'condition_x': 'condition'})

# Calculate mean values for condition "a"
mean_a_avg = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['avg_amplitudes'].mean()
mean_a_smoothed = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['smoothed_avg_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
avg_amplitudes_per_video['normalized_avg_amplitudes'] = (avg_amplitudes_per_video['avg_amplitudes'] - mean_a_avg) / mean_a_avg
avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'] = (avg_amplitudes_per_video['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed

# Create a figure with subplots for normalized avg amplitude and normalized smoothed avg amplitude
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Normalized Maximum Average Amplitude per Video', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(avg_amplitudes_per_video['normalized_avg_amplitudes'].min(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].min())
y_max = max(avg_amplitudes_per_video['normalized_avg_amplitudes'].max(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

# Boxplot for normalized max_amplitudes
sns.boxplot(x='condition', y='normalized_max_amplitudes', data=max_amplitudes_per_video, 
            order=sorted(max_amplitudes_per_video['condition'].unique()), ax=axs[0])
sns.swarmplot(x='condition', y='normalized_max_amplitudes', data=max_amplitudes_per_video, 
              order=sorted(max_amplitudes_per_video['condition'].unique()), ax=axs[0], color=".25", size=5, alpha=0.5)
axs[0].set_title('Normalized Max Amplitude (per video)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Normalized Max Amplitude')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(y_min_with_buffer, y_max_with_buffer)

# Boxplot for normalized smoothed_max_amplitudes
sns.boxplot(x='condition', y='normalized_smoothed_max_amplitudes', data=max_amplitudes_per_video, 
            order=sorted(max_amplitudes_per_video['condition'].unique()), ax=axs[1])
sns.swarmplot(x='condition', y='normalized_smoothed_max_amplitudes', data=max_amplitudes_per_video, 
              order=sorted(max_amplitudes_per_video['condition'].unique()), ax=axs[1], color=".25", size=5, alpha=0.5)
axs[1].set_title('Normalized Smoothed Max Amplitude (per video)')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Normalized Smoothed Max Amplitude')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('normalized_max_amplitude_per_video_boxplots.png')
plt.close()

###### Boxplot of average max amplitude value in video ######
#Use only values when z_axis_turn is False

# Calculate and plot average max amplitude per video
# Calculate average amplitudes per video
avg_amplitudes_per_video = combined_df[~combined_df['z_axis_turn']].groupby('video_id').agg({
    'max_amplitudes': 'mean',
    'smoothed_max_amplitudes': 'mean',
    'condition_x': 'first'  # Keep the condition for each video_id
}).reset_index()
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'condition_x': 'condition'})

# Calculate mean values for condition "a"
mean_a_max = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['max_amplitudes'].mean()
mean_a_smoothed = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['smoothed_max_amplitudes'].mean()

# Normalize values by making the value relative to the mean value for condition "a"
avg_amplitudes_per_video['normalized_max_amplitudes'] = (avg_amplitudes_per_video['max_amplitudes']-mean_a_max) / mean_a_max
avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'] = (avg_amplitudes_per_video['smoothed_max_amplitudes']-mean_a_smoothed) / mean_a_smoothed

# Create a figure with subplots for average max amplitude and average smoothed max amplitude
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Normalized Average Maximum Amplitude per Video', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(avg_amplitudes_per_video['normalized_max_amplitudes'].min(), avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'].min())
y_max = max(avg_amplitudes_per_video['normalized_max_amplitudes'].max(), avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

# Boxplot for normalized average max_amplitudes
sns.boxplot(x='condition', y='normalized_max_amplitudes', data=avg_amplitudes_per_video, 
            order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[0])
sns.swarmplot(x='condition', y='normalized_max_amplitudes', data=avg_amplitudes_per_video, 
              order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[0], color=".25", size=5, alpha=0.5)
axs[0].set_title('Normalized Average Max Amplitude (per video)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Normalized Average Max Amplitude')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(y_min_with_buffer, y_max_with_buffer)

# Boxplot for normalized average smoothed_max_amplitudes
sns.boxplot(x='condition', y='normalized_smoothed_max_amplitudes', data=avg_amplitudes_per_video, 
            order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[1])
sns.swarmplot(x='condition', y='normalized_smoothed_max_amplitudes', data=avg_amplitudes_per_video, 
              order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[1], color=".25", size=5, alpha=0.5)
axs[1].set_title('Normalized Average Smoothed Max Amplitude (per video)')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Normalized Average Smoothed Max Amplitude')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('normalized_avg_max_amplitude_per_video_boxplots.png')
plt.close()



###### Boxplot of max amplitude for each frame ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]

# Calculate mean values for condition "a"
mean_a_max = filtered_df[filtered_df['condition'] == 'a']['max_amplitudes'].mean()
mean_a_smoothed = filtered_df[filtered_df['condition'] == 'a']['smoothed_max_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_max_amplitudes'] = (filtered_df['max_amplitudes'] - mean_a_max) / mean_a_max
filtered_df['normalized_smoothed_max_amplitudes'] = (filtered_df['smoothed_max_amplitudes'] - mean_a_smoothed) / mean_a_smoothed
filtered_df = filtered_df.rename(columns={'condition_x': 'condition'})

# Get unique conditions
conditions = sorted(filtered_df['condition'].unique())

# Create a figure with 4x2 subplots, one row for each condition, two columns for smoothed and non-smoothed
fig, axs = plt.subplots(4, 2, figsize=(42, 24))  # Increased width from 24 to 32
fig.suptitle('Normalized Max Amplitudes by Frame for Each Condition', fontsize=16)

for i, condition in enumerate(conditions):
    condition_df = filtered_df[filtered_df['condition'] == condition]
    
    # Create boxplot for normalized max amplitudes, split by frame
    sns.boxplot(x='frames', y='normalized_max_amplitudes', data=condition_df, ax=axs[i, 0])
    
    # Customize the subplot
    axs[i, 0].set_title(f'Condition: {condition} - Non-smoothed')
    axs[i, 0].set_xlabel('Frame')
    axs[i, 0].set_ylabel('Normalized Max Amplitude')
    axs[i, 0].set_xticks(range(0, 601, 10))
    axs[i, 0].set_xticklabels(range(0, 601, 10))
    axs[i, 0].tick_params(axis='x', rotation=45)
    
    # Create boxplot for normalized smoothed max amplitudes, split by frame
    sns.boxplot(x='frames', y='normalized_smoothed_max_amplitudes', data=condition_df, ax=axs[i, 1])
    
    # Customize the subplot
    axs[i, 1].set_title(f'Condition: {condition} - Smoothed')
    axs[i, 1].set_xlabel('Frame')
    axs[i, 1].set_ylabel('Normalized Smoothed Max Amplitude')
    axs[i, 1].set_xticks(range(0, 601, 10))
    axs[i, 1].set_xticklabels(range(0, 601, 10))
    axs[i, 1].tick_params(axis='x', rotation=45)

### Adjust layout and save the figure
plt.tight_layout()
plt.savefig('normalized_max_amplitudes_by_frame_and_condition_boxplot.png', dpi=300, bbox_inches='tight')  # Increased DPI and added bbox_inches for better quality
plt.close()


########## Using percentage of max amplitude possible (half worm length) ##########

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
# Calculate amplitudes as a percentage of half the worm length per frame
filtered_df.loc[:, 'max_amplitude_percentage'] = filtered_df['max_amplitudes'] / ((filtered_df['worm_lengths'] / 2))

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition_x'].unique())
viscosities = sorted(filtered_df['viscosity_x'].unique())


###### Boxplot of max amplitude for each frame as a proportion of max amplitude possible (half worm length) ######

# Create a figure with subplots for each condition and viscosity
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(20, 5*len(conditions)), sharex=False, sharey=True)
fig.suptitle('Max Amplitude Percentage by Frame, Condition, and Viscosity', fontsize=16)

for i, condition in enumerate(conditions):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition_x'] == condition) & (filtered_df['viscosity_x'] == viscosity)]
        
        sns.boxplot(x='frames', y='max_amplitude_percentage', data=subset, ax=axs[i, j])
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame')
        axs[i, j].set_ylabel('Max Amplitude Percentage')
        axs[i, j].set_ylim(0, 100)
        
        if viscosity == 'ngm':
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 100))
            axs[i, j].set_xticklabels(range(0, 601, 100))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 50))
            axs[i, j].set_xticklabels(range(0, 301, 50))
        
        axs[i, j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('max_amplitude_percentage_by_frame_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of max amplitude percentage by condition and viscosity ######

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Max Amplitude Percentage by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity_x'] == viscosity]
    
    sns.boxplot(x='condition_x', y='max_amplitude_percentage', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Max Amplitude Percentage')
    axs[j].set_ylim(0, 100)
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('max_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of average max amplitude percentage by condition and viscosity ######

# Calculate the average max_amplitude_percentage for each video_id
averaged_df = filtered_df.groupby(['video_id', 'condition_x', 'viscosity_x'], as_index=False)['max_amplitude_percentage'].mean()

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Average Max Amplitude Percentage by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = averaged_df[averaged_df['viscosity_x'] == viscosity]
    
    sns.boxplot(x='condition_x', y='max_amplitude_percentage', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Average Max Amplitude Percentage')
    axs[j].set_ylim(0, 100)
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('avg_max_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of absolute maximum max amplitude percentage per video ######

# Calculate the maximum max_amplitude_percentage for each video
max_amplitude_per_video = filtered_df.groupby(['video_id', 'condition_x', 'viscosity_x'])['max_amplitude_percentage'].max().reset_index()

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Maximum Max Amplitude Percentage per Video by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = max_amplitude_per_video[max_amplitude_per_video['viscosity_x'] == viscosity]
    
    sns.boxplot(x='condition_x', y='max_amplitude_percentage', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Maximum Max Amplitude Percentage')
    axs[j].set_ylim(0, 100)
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('max_max_amplitude_percentage_per_video_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



# endregion

# region [Average amplitude plots]

###### Boxplot of average amplitude (for each frame) per condition ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition'})

# Calculate mean values for condition "a"
mean_a_avg = filtered_df[filtered_df['condition'] == 'a']['avg_amplitudes'].mean()
mean_a_smoothed = filtered_df[filtered_df['condition'] == 'a']['smoothed_avg_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_avg_amplitudes'] = (filtered_df['avg_amplitudes'] - mean_a_avg) / mean_a_avg
filtered_df['normalized_smoothed_avg_amplitudes'] = (filtered_df['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed


# Create a figure with subplots for normalized avg amplitude and normalized smoothed avg amplitude
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Normalized Average Amplitude and Smoothed Average Amplitude per Condition', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(filtered_df['normalized_avg_amplitudes'].min(), filtered_df['normalized_smoothed_avg_amplitudes'].min())
y_max = max(filtered_df['normalized_avg_amplitudes'].max(), filtered_df['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

# Boxplot for normalized avg_amplitudes
sns.boxplot(x='condition', y='normalized_avg_amplitudes', data=filtered_df, order=sorted(filtered_df['condition'].unique()), ax=axs[0])
axs[0].set_title('Normalized Average Amplitude (excluding z-axis turns)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Normalized Average Amplitude')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(y_min_with_buffer, y_max_with_buffer)

# Boxplot for normalized smoothed_avg_amplitudes
sns.boxplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=filtered_df, order=sorted(filtered_df['condition'].unique()), ax=axs[1])
axs[1].set_title('Normalized Smoothed Average Amplitude (excluding z-axis turns)')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Normalized Smoothed Average Amplitude')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('normalized_avgamplitude_boxplots_no_zaxis_turns.png')
plt.close()


###### Boxplot of average avg amplitude value per video ######
#Use only values when z_axis_turn is False

# Calculate and plot average avg amplitude per video
# Calculate average amplitudes per video
avg_amplitudes_per_video = combined_df[~combined_df['z_axis_turn']].groupby('video_id').agg({
    'avg_amplitudes': 'mean',
    'smoothed_avg_amplitudes': 'mean',
    'condition_x': 'first'  # Keep the condition for each video_id
}).reset_index()
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'condition_x': 'condition'})

# Calculate mean values for condition "a"
mean_a_avg = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['avg_amplitudes'].mean()
mean_a_smoothed = avg_amplitudes_per_video[avg_amplitudes_per_video['condition'] == 'a']['smoothed_avg_amplitudes'].mean()

# Normalize values by making the value relative to the mean value for condition "a"
avg_amplitudes_per_video['normalized_avg_amplitudes'] = (avg_amplitudes_per_video['avg_amplitudes']-mean_a_avg) / mean_a_avg
avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'] = (avg_amplitudes_per_video['smoothed_avg_amplitudes']-mean_a_smoothed) / mean_a_smoothed

# Create a figure with subplots for average avg amplitude and average smoothed avg amplitude
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Normalized Average Average Amplitude per Video', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(avg_amplitudes_per_video['normalized_avg_amplitudes'].min(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].min())
y_max = max(avg_amplitudes_per_video['normalized_avg_amplitudes'].max(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

# Boxplot for normalized average avg_amplitudes
sns.boxplot(x='condition', y='normalized_avg_amplitudes', data=avg_amplitudes_per_video, 
            order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[0])
sns.swarmplot(x='condition', y='normalized_avg_amplitudes', data=avg_amplitudes_per_video, 
              order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[0], color=".25", size=5, alpha=0.5)
axs[0].set_title('Normalized Average Avg Amplitude (per video)')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Normalized Average Avg Amplitude')
axs[0].tick_params(axis='x', rotation=45)
axs[0].set_ylim(y_min_with_buffer, y_max_with_buffer)

# Boxplot for normalized average smoothed_avg_amplitudes
sns.boxplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=avg_amplitudes_per_video, 
            order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[1])
sns.swarmplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=avg_amplitudes_per_video, 
              order=sorted(avg_amplitudes_per_video['condition'].unique()), ax=axs[1], color=".25", size=5, alpha=0.5)
axs[1].set_title('Normalized Average Smoothed Avg Amplitude (per video)')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Normalized Average Smoothed Avg Amplitude')
axs[1].tick_params(axis='x', rotation=45)
axs[1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('normalized_avg_avg_amplitude_per_video_boxplots.png')
plt.close()

###### Boxplot of avg amplitude for each frame ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition'})

# Calculate mean values for condition "a"
mean_a_avg = filtered_df[filtered_df['condition'] == 'a']['avg_amplitudes'].mean()
mean_a_smoothed = filtered_df[filtered_df['condition'] == 'a']['smoothed_avg_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_avg_amplitudes'] = (filtered_df['avg_amplitudes'] - mean_a_avg) / mean_a_avg
filtered_df['normalized_smoothed_avg_amplitudes'] = (filtered_df['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed

# Get unique conditions
conditions = sorted(filtered_df['condition'].unique())

# Create a figure with 4x2 subplots, one row for each condition, two columns for smoothed and non-smoothed
fig, axs = plt.subplots(4, 2, figsize=(42, 24))  # Increased width from 24 to 32
fig.suptitle('Normalized Avg Amplitudes by Frame for Each Condition', fontsize=16)

for i, condition in enumerate(conditions):
    condition_df = filtered_df[filtered_df['condition'] == condition]
    
    # Create boxplot for normalized avg amplitudes, split by frame
    sns.boxplot(x='frames', y='normalized_avg_amplitudes', data=condition_df, ax=axs[i, 0])
    
    # Customize the subplot
    axs[i, 0].set_title(f'Condition: {condition} - Non-smoothed')
    axs[i, 0].set_xlabel('Frame')
    axs[i, 0].set_ylabel('Normalized Avg Amplitude')
    axs[i, 0].set_xticks(range(0, 601, 10))
    axs[i, 0].set_xticklabels(range(0, 601, 10))
    axs[i, 0].tick_params(axis='x', rotation=45)
    
    # Create boxplot for normalized smoothed avg amplitudes, split by frame
    sns.boxplot(x='frames', y='normalized_smoothed_avg_amplitudes', data=condition_df, ax=axs[i, 1])
    
    # Customize the subplot
    axs[i, 1].set_title(f'Condition: {condition} - Smoothed')
    axs[i, 1].set_xlabel('Frame')
    axs[i, 1].set_ylabel('Normalized Smoothed Avg Amplitude')
    axs[i, 1].set_xticks(range(0, 601, 10))
    axs[i, 1].set_xticklabels(range(0, 601, 10))
    axs[i, 1].tick_params(axis='x', rotation=45)

### Adjust layout and save the figure
plt.tight_layout()
plt.savefig('normalized_avg_amplitudes_by_frame_and_condition_boxplot.png', dpi=300, bbox_inches='tight')  # Increased DPI and added bbox_inches for better quality
plt.close()



########## Using percentage of max amplitude possible (half worm length) ##########

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
# Calculate amplitudes as a percentage of half the worm length per frame
filtered_df.loc[:, 'avg_amplitude_percentage'] = filtered_df['avg_amplitudes'] / ((filtered_df['worm_lengths'] / 2))

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition_x'].unique())
viscosities = sorted(filtered_df['viscosity_x'].unique())


###### Boxplot of avg amplitude for each frame as a proportion of max amplitude possible (half worm length) ######

# Create a figure with subplots for each condition and viscosity
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(20, 5*len(conditions)), sharex=False, sharey=True)
fig.suptitle('Avg Amplitude Percentage by Frame, Condition, and Viscosity', fontsize=16)

for i, condition in enumerate(conditions):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition_x'] == condition) & (filtered_df['viscosity_x'] == viscosity)]
        
        sns.boxplot(x='frames', y='avg_amplitude_percentage', data=subset, ax=axs[i, j])
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame')
        axs[i, j].set_ylabel('Avg Amplitude Percentage')
        axs[i, j].set_ylim(0, 1)
        
        if viscosity == 'ngm':
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 100))
            axs[i, j].set_xticklabels(range(0, 601, 100))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 50))
            axs[i, j].set_xticklabels(range(0, 301, 50))
        
        axs[i, j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('avg_amplitude_percentage_by_frame_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of max amplitude percentage by condition and viscosity ######

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Avg Amplitude Percentage by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity_x'] == viscosity]
    
    sns.boxplot(x='condition_x', y='avg_amplitude_percentage', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Avg Amplitude Percentage')
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('avg_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of average max amplitude percentage by condition and viscosity ######

# Calculate the average max_amplitude_percentage for each video_id
averaged_df = filtered_df.groupby(['video_id', 'condition_x', 'viscosity_x'], as_index=False)['avg_amplitude_percentage'].mean()

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Average Avg Amplitude Percentage by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = averaged_df[averaged_df['viscosity_x'] == viscosity]
    
    sns.boxplot(x='condition_x', y='avg_amplitude_percentage', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Average Avg Amplitude Percentage')
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('avg_avg_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



# endregion

# region [Curvature_1d plots]

### ----> Maybe not useful

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Calculate mean values for condition "a"
mean_a_curvature = filtered_df[filtered_df['condition'] == 'a']['curvature_time_series'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_curvature_ts'] = (filtered_df['curvature_time_series'] - mean_a_curvature) / mean_a_curvature

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition'].unique())
viscosities = sorted(filtered_df['viscosity'].unique())


###### Boxplot of normalized curvature_1d by frame ######

# Create a figure with subplots, one for each condition and viscosity combination
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(30*len(viscosities), 6*len(conditions)), sharex=False, sharey=True)
fig.suptitle('Normalized Curvature Time Series by Frame for Each Condition and Viscosity', fontsize=16)

for i, condition in enumerate(conditions):
    for j, viscosity in enumerate(viscosities):
        condition_visc_df = filtered_df[(filtered_df['condition'] == condition) & (filtered_df['viscosity'] == viscosity)]
        
        # Create boxplot for normalized curvature time series, split by frame
        sns.boxplot(x='frames', y='normalized_curvature_ts', data=condition_visc_df, ax=axs[i, j])
        
        # Customize the subplot
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_ylabel('Normalized Curvature')
        
        # Set x-axis limits and ticks based on viscosity
        if viscosity == "ngm":
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 100))
            axs[i, j].set_xticklabels(range(0, 601, 100))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 50))
            axs[i, j].set_xticklabels(range(0, 301, 50))
        
        axs[i, j].tick_params(axis='x', rotation=45)

        # Only set y-label for leftmost plots
        if j == 0:
            axs[i, j].set_ylabel('Normalized Curvature')
        else:
            axs[i, j].set_ylabel('')

        # Only set x-label for bottom plots
        if i == len(conditions) - 1:
            axs[i, j].set_xlabel('Frame')
        else:
            axs[i, j].set_xlabel('')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('normalized_curvature_ts_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



###### Boxplot of normalized curvature_ts by condition and viscosity ######

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Normalized Curvature Time Series by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='normalized_curvature_ts', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Normalized Curvature')
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('normalized_curvature_ts_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()






# endregion

# region [Curvature plots]

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Calculate average and max curvature for each frame in each video
filtered_df['avg_curvature'] = filtered_df['curvatures'].apply(lambda x: np.mean(x))
filtered_df['max_curvature'] = filtered_df['curvatures'].apply(lambda x: np.max(x))

# Normalize curvature values between 0 and 1 where the theoretical maximum is 0.5
filtered_df['normalized_avg_curvature'] = filtered_df['avg_curvature'] / 0.5
filtered_df['normalized_max_curvature'] = filtered_df['max_curvature'] / 0.5

# Clip values to ensure they're between 0 and 1
filtered_df['normalized_avg_curvature'] = filtered_df['normalized_avg_curvature'].clip(0, 1)
filtered_df['normalized_max_curvature'] = filtered_df['normalized_max_curvature'].clip(0, 1)

# Find the maximum "normalized_max_curvature" for each "video_id"
max_curvature_by_video = filtered_df.groupby(['video_id', 'condition', 'viscosity'])['normalized_max_curvature'].max().reset_index()

# Get unique conditions and viscosities, with conditions sorted alphabetically
conditions = sorted(filtered_df['condition'].unique())
viscosities = filtered_df['viscosity'].unique()


###### Boxplot of curvature by frame and condition/viscosity ######

# Create a figure with subplots for each condition and viscosity
# Plot for average curvature
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(20, 6*len(conditions)), sharex=True, sharey=True)
fig.suptitle('Normalized Average Curvature by Frame, Condition, and Viscosity', fontsize=16)

for i, condition in enumerate(sorted(conditions)):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition'] == condition) & (filtered_df['viscosity'] == viscosity)]
        
        sns.boxplot(x='frames', y='normalized_avg_curvature', data=subset, ax=axs[i, j], color='lightblue', fliersize=1)
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame' if i == len(conditions) - 1 else '')
        axs[i, j].set_ylabel('Normalized Average Curvature' if j == 0 else '')
        axs[i, j].tick_params(axis='x', rotation=45)
        
        # Set x-axis limit and ticks
        if viscosity == "ngm":
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 10))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 10))
        
        # Only show every 10th tick label
        for label in axs[i, j].xaxis.get_ticklabels()[1::10]:
            label.set_visible(False)
        
        # Set y-axis limit
        axs[i, j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('normalized_avg_curvature_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



# Plot for max curvature
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(20, 6*len(conditions)), sharex=True, sharey=True)
fig.suptitle('Normalized Max Curvature by Frame, Condition, and Viscosity', fontsize=16)

for i, condition in enumerate(sorted(conditions)):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition'] == condition) & (filtered_df['viscosity'] == viscosity)]
        
        sns.boxplot(x='frames', y='normalized_max_curvature', data=subset, ax=axs[i, j], color='lightgreen', fliersize=1)
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame' if i == len(conditions) - 1 else '')
        axs[i, j].set_ylabel('Normalized Max Curvature' if j == 0 else '')
        axs[i, j].tick_params(axis='x', rotation=45)
        
        # Set x-axis limit and ticks
        if viscosity == "ngm":
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 10))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 10))
        
        # Only show every 10th tick label
        for label in axs[i, j].xaxis.get_ticklabels()[1::10]:
            label.set_visible(False)
        
        # Set y-axis limit
        axs[i, j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('normalized_max_curvature_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



##### Boxplot of normalized_avg_curvature by condition, split by viscosity horizontally
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Normalized Average Curvature by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity'] == viscosity]
    
    # Sort the conditions alphabetically
    sorted_conditions = sorted(subset['condition'].unique())
    
    sns.boxplot(x='condition', y='normalized_avg_curvature', data=subset, ax=axs[j], order=sorted_conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Normalized Average Curvature' if j == 0 else '')
    axs[j].tick_params(axis='x', rotation=45)
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('normalized_avg_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


##### Boxplot of average normalized_avg_curvature by condition, split by viscosity horizontally
# First, calculate the average value for each video_id
avg_by_video = filtered_df.groupby(['viscosity', 'condition', 'video_id'])['normalized_avg_curvature'].mean().reset_index()

fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Average Normalized Average Curvature by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = avg_by_video[avg_by_video['viscosity'] == viscosity]
    
    # Sort the conditions alphabetically
    sorted_conditions = sorted(subset['condition'].unique())
    
    sns.boxplot(x='condition', y='normalized_avg_curvature', data=subset, ax=axs[j], order=sorted_conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Average Normalized Average Curvature' if j == 0 else '')
    axs[j].tick_params(axis='x', rotation=45)
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('avg_normalized_avg_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


##### Boxplot of normalized_max_curvature by condition, split by viscosity horizontally
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Normalized Max Curvature by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity'] == viscosity]
    
    # Sort the conditions alphabetically
    sorted_conditions = sorted(subset['condition'].unique())
    
    sns.boxplot(x='condition', y='normalized_max_curvature', data=subset, ax=axs[j], order=sorted_conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Normalized Max Curvature' if j == 0 else '')
    axs[j].tick_params(axis='x', rotation=45)
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('normalized_max_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


##### Boxplot of average normalized_max_curvature by condition, split by viscosity horizontally
# First, calculate the average value for each video_id
avg_by_video = filtered_df.groupby(['viscosity', 'condition', 'video_id'])['normalized_max_curvature'].mean().reset_index()

fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Average Normalized Max Curvature by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = avg_by_video[avg_by_video['viscosity'] == viscosity]
    
    # Sort the conditions alphabetically
    sorted_conditions = sorted(subset['condition'].unique())
    
    sns.boxplot(x='condition', y='normalized_max_curvature', data=subset, ax=axs[j], order=sorted_conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Average Normalized Max Curvature' if j == 0 else '')
    axs[j].tick_params(axis='x', rotation=45)
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('avg_normalized_max_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


###### Boxplot of maximum normalized_max_curvature by video by condition and viscosity ######
conditions = sorted(max_curvature_by_video['condition'].unique())
viscosities = max_curvature_by_video['viscosity'].unique()

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Maximum Normalized Curvature by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = max_curvature_by_video[max_curvature_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='normalized_max_curvature', data=subset, ax=axs[j])
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Maximum Normalized Curvature' if j == 0 else '')
    axs[j].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('max_normalized_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


# endregion


# region [dominant_spatial_freqs]

combined_df


# endregion