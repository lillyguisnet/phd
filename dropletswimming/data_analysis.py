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
folder_path = 'C:/Users/aurel/Documents/GitHub/phd/dropletswimming/final_shapenalysis'
ignore_keys = ['masks', 'fps', 'smooth_points']
#max_files = 30  # Set this to the number of files you want to process, or None for all files

combined_df = process_h5_files(folder_path, ignore_keys=ignore_keys)

# Save the combined DataFrame to a CSV file
output_file = 'combined_data_sample.csv'
#combined_df.to_csv(output_file, index=False)
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

#Load the combined dataframe from the csv file
combined_df = pd.read_csv('combined_data_sample.csv')

 
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
print(f"Percentage of frames with z-axis turns: {(total_turns/total_frames)*100:.2f}%") ###13.68%


#For each video, calculate % of frames in z-axis turn, c-shape, s-shape and straight

# Group by video ID

# Convert 'shape' column to string type
import ast

# Clean up the shape column
combined_df['shape'] = combined_df['shape'].apply(lambda x: ast.literal_eval(x).decode('utf-8'))

print("After cleanup:")
print(combined_df['shape'].dtype)
print(combined_df['shape'].unique())


grouped = combined_df.groupby('video_id')

# Function to calculate percentages for each shape
def calculate_shape_percentages(group):
    total_frames = len(group)
    z_axis_turns = group['z_axis_turn'].sum()
    non_z_axis_frames = total_frames - z_axis_turns
    
    # Count other shapes only if it's not a z-axis turn
    c_shapes = ((group['shape'] == 'C-shape') & (~group['z_axis_turn'])).sum()
    s_shapes = ((group['shape'] == 'S-shape') & (~group['z_axis_turn'])).sum()
    straight = ((group['shape'] == 'Straight') & (~group['z_axis_turn'])).sum()

    return pd.Series({
        'z_axis_turn_percentage': (z_axis_turns / total_frames) * 100,
        'c_shape_percentage': (c_shapes / total_frames) * 100,
        's_shape_percentage': (s_shapes / total_frames) * 100,
        'straight_percentage': (straight / total_frames) * 100,
        'c_shape_percentage_excluding_z': (c_shapes / non_z_axis_frames) * 100 if non_z_axis_frames > 0 else 0,
        's_shape_percentage_excluding_z': (s_shapes / non_z_axis_frames) * 100 if non_z_axis_frames > 0 else 0,
        'straight_percentage_excluding_z': (straight / non_z_axis_frames) * 100 if non_z_axis_frames > 0 else 0,
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
required_columns = ['z_axis_turn_percentage', 'c_shape_percentage', 's_shape_percentage', 'straight_percentage', 'c_shape_percentage_excluding_z', 's_shape_percentage_excluding_z', 'straight_percentage_excluding_z']
for column in required_columns:
    if column not in shape_percentages.columns:
        shape_percentages[column] = 0.0

# Add the percentage columns to the original dataframe
combined_df = combined_df.merge(shape_percentages, on='video_id', how='left')

combined_df['c_shape_percentage'].unique()

#Save the combined dataframe to a csv file
combined_df.to_csv('combined_data_sample_with_shape_percentages.csv', index=False)

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
# Get unique viscosities
viscosities = sorted(shape_percentages['viscosity'].unique())

# Create a figure with subplots for each shape and viscosity
fig, axs = plt.subplots(len(viscosities), 4, figsize=(24, 6*len(viscosities)))
fig.suptitle('Percentage of frames in each shape for each condition and viscosity', fontsize=16)

shapes = ['straight_percentage', 's_shape_percentage', 'c_shape_percentage', 'z_axis_turn_percentage']
titles = ['Straight', 'S-shape', 'C-shape', 'Z-axis Turn']

# Find the global max value across all shapes and add a 10% buffer
y_max = max(shape_percentages[shapes].max())
y_max_with_buffer = y_max * 1.1  # Add 10% buffer

for j, viscosity in enumerate(viscosities):
    viscosity_data = shape_percentages[shape_percentages['viscosity'] == viscosity]
    
    for i, (shape, title) in enumerate(zip(shapes, titles)):
        ax = axs[j, i] if len(viscosities) > 1 else axs[i]
        sns.boxplot(x='condition', y=shape, data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=ax, boxprops=dict(alpha=0.3))
        sns.swarmplot(x='condition', y=shape, data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=ax, color="black", size=5, alpha=1)
        ax.set_title(f'Percentage of {title} frames (Viscosity: {viscosity})')
        ax.set_xlabel('Condition')
        ax.set_ylabel(f'Percentage of {title} frames')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, y_max_with_buffer)  # Set the y-axis limits from 0 to the buffered max

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/shape/shape_percentage_boxplot_by_viscosity.png')
plt.close()


#### Percentage of frames in each shape excluding z-axis turns ####

viscosities = sorted(shape_percentages['viscosity'].unique())
viscosities = [v for v in viscosities if v != "visc05"] + ["visc05"]

# Create a figure with subplots for each shape and viscosity
fig, axs = plt.subplots(len(viscosities), 4, figsize=(24, 6*len(viscosities)))
fig.suptitle('Percentage of frames in each shape for each condition and viscosity', fontsize=16)

shapes = ['straight_percentage_excluding_z', 's_shape_percentage_excluding_z', 'c_shape_percentage_excluding_z', 'z_axis_turn_percentage']
titles = ['Straight', 'S-shape', 'C-shape', 'Z-axis Turn']

# Find the global max value across all shapes and add a 10% buffer
y_max = max(shape_percentages[shapes].max())
y_max_with_buffer = y_max * 1.1  # Add 10% buffer

for j, viscosity in enumerate(viscosities):
    viscosity_data = shape_percentages[shape_percentages['viscosity'] == viscosity]
    
    for i, (shape, title) in enumerate(zip(shapes, titles)):
        ax = axs[j, i] if len(viscosities) > 1 else axs[i]
        sns.boxplot(x='condition', y=shape, data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=ax, boxprops=dict(alpha=0.3))
        sns.swarmplot(x='condition', y=shape, data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=ax, color="black", size=5, alpha=1)
        ax.set_title(f'Percentage of {title} frames (Viscosity: {viscosity})')
        ax.set_xlabel('Condition')
        ax.set_ylabel(f'Percentage of {title} frames')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, y_max_with_buffer)  # Set the y-axis limits from 0 to the buffered max

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/shape/shape_percentage_boxplot_by_viscosity_excluding_z.png')
plt.close()

# endregion [Shape percentage plots]

# region [Max amplitude plots]

###### Boxplot of average max amplitude value in video ######
#Use only values when z_axis_turn is False

# Calculate and plot average max amplitude per video
# Calculate average amplitudes per video
avg_amplitudes_per_video = combined_df[~combined_df['z_axis_turn']].groupby('video_id').agg({
    'max_amplitudes': 'mean',
    'smoothed_max_amplitudes': 'mean',
    'condition_x': 'first',  # Keep the condition for each video_id
    'viscosity_x': 'first'  # Add viscosity to the groupby
}).reset_index()
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'condition_x': 'condition'})
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'viscosity_x': 'viscosity'})

# Calculate mean values for condition "a" and viscosity "ngm"
ngm_data = avg_amplitudes_per_video[(avg_amplitudes_per_video['condition'] == 'a') & (avg_amplitudes_per_video['viscosity'] == 'ngm')]
mean_a_max = ngm_data['max_amplitudes'].mean()
mean_a_smoothed = ngm_data['smoothed_max_amplitudes'].mean()

# Normalize values by making the value relative to the mean value for condition "a"
avg_amplitudes_per_video['normalized_max_amplitudes'] = (avg_amplitudes_per_video['max_amplitudes']-mean_a_max) / mean_a_max
avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'] = (avg_amplitudes_per_video['smoothed_max_amplitudes']-mean_a_smoothed) / mean_a_smoothed

# Get unique viscosities and sort them
viscosities = sorted(avg_amplitudes_per_video['viscosity'].unique())
# Move 'visc05' to the end if it exists
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(len(viscosities), 2, figsize=(16, 6*len(viscosities)))
fig.suptitle('Normalized Average Maximum Amplitude per Video', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(avg_amplitudes_per_video['normalized_max_amplitudes'].min(), avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'].min())
y_max = max(avg_amplitudes_per_video['normalized_max_amplitudes'].max(), avg_amplitudes_per_video['normalized_smoothed_max_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_amplitudes_per_video[avg_amplitudes_per_video['viscosity'] == viscosity]
    
    # Boxplot for normalized average max_amplitudes
    sns.boxplot(x='condition', y='normalized_max_amplitudes', data=viscosity_data, 
                order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 0])
    sns.swarmplot(x='condition', y='normalized_max_amplitudes', data=viscosity_data, 
                  order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 0], color=".25", size=5, alpha=0.5)
    axs[i, 0].set_title(f'Normalized Average Max Amplitude (Viscosity: {viscosity})')
    axs[i, 0].set_xlabel('Condition')
    axs[i, 0].set_ylabel('Normalized Average Max Amplitude')
    axs[i, 0].tick_params(axis='x')
    axs[i, 0].set_ylim(y_min_with_buffer, y_max_with_buffer)

    # Boxplot for normalized average smoothed_max_amplitudes
    sns.boxplot(x='condition', y='normalized_smoothed_max_amplitudes', data=viscosity_data, 
                order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 1])
    sns.swarmplot(x='condition', y='normalized_smoothed_max_amplitudes', data=viscosity_data, 
                  order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 1], color=".25", size=5, alpha=0.5)
    axs[i, 1].set_title(f'Normalized Average Smoothed Max Amplitude (Viscosity: {viscosity})')
    axs[i, 1].set_xlabel('Condition')
    axs[i, 1].set_ylabel('Normalized Average Smoothed Max Amplitude')
    axs[i, 1].tick_params(axis='x', rotation=45)
    axs[i, 1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/normalized_avg_max_amplitude_per_video_boxplots_by_viscosity.png')
plt.close()




###### Boxplot of max amplitude for each frame ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]

# Calculate mean values for condition "a" for each viscosity
viscosities = filtered_df['viscosity_x'].unique()

# Calculate global min and max for y-axis limits
global_min = float('inf')
global_max = float('-inf')

for viscosity in viscosities:
    viscosity_df = filtered_df[filtered_df['viscosity_x'] == viscosity]
    
    # Calculate mean values for condition "a" when viscosity is "ngm"
    ngm_df = filtered_df[filtered_df['viscosity_x'] == 'ngm']
    mean_a_max = ngm_df[ngm_df['condition_x'] == 'a']['max_amplitudes'].mean()
    mean_a_smoothed = ngm_df[ngm_df['condition_x'] == 'a']['smoothed_max_amplitudes'].mean()

    # Normalize the values relative to the mean of condition "a" for this viscosity
    viscosity_df['normalized_max_amplitudes'] = (viscosity_df['max_amplitudes'] - mean_a_max) / mean_a_max
    viscosity_df['normalized_smoothed_max_amplitudes'] = (viscosity_df['smoothed_max_amplitudes'] - mean_a_smoothed) / mean_a_smoothed
    viscosity_df = viscosity_df.rename(columns={'condition_x': 'condition'})

    # Update global min and max
    global_min = min(global_min, viscosity_df['normalized_max_amplitudes'].min(), viscosity_df['normalized_smoothed_max_amplitudes'].min())
    global_max = max(global_max, viscosity_df['normalized_max_amplitudes'].max(), viscosity_df['normalized_smoothed_max_amplitudes'].max())

# Add a 10% buffer to the y-axis limits
y_range = global_max - global_min
y_min = global_min - 0.1 * y_range
y_max = global_max + 0.1 * y_range

for viscosity in viscosities:
    viscosity_df = filtered_df[filtered_df['viscosity_x'] == viscosity]
    
    # Calculate mean values for condition "a" when viscosity is "ngm"
    ngm_df = filtered_df[filtered_df['viscosity_x'] == 'ngm']
    mean_a_max = ngm_df[ngm_df['condition_x'] == 'a']['max_amplitudes'].mean()
    mean_a_smoothed = ngm_df[ngm_df['condition_x'] == 'a']['smoothed_max_amplitudes'].mean()

    # Normalize the values relative to the mean of condition "a" for this viscosity
    viscosity_df['normalized_max_amplitudes'] = (viscosity_df['max_amplitudes'] - mean_a_max) / mean_a_max
    viscosity_df['normalized_smoothed_max_amplitudes'] = (viscosity_df['smoothed_max_amplitudes'] - mean_a_smoothed) / mean_a_smoothed
    viscosity_df = viscosity_df.rename(columns={'condition_x': 'condition'})

    # Get unique conditions for this viscosity
    conditions = sorted(viscosity_df['condition'].unique())

    # Create a figure with 4x2 subplots, one row for each condition, two columns for smoothed and non-smoothed
    fig, axs = plt.subplots(4, 2, figsize=(42, 24))
    fig.suptitle(f'Normalized Max Amplitudes by Frame for Each Condition (Viscosity: {viscosity})', fontsize=16)

    for i, condition in enumerate(conditions):
        condition_df = viscosity_df[viscosity_df['condition'] == condition]
        
        # Create boxplot for normalized max amplitudes, split by frame
        sns.boxplot(x='frames', y='normalized_max_amplitudes', data=condition_df, ax=axs[i, 0])
        
        # Customize the subplot
        axs[i, 0].set_title(f'Condition: {condition} - Non-smoothed')
        axs[i, 0].set_xlabel('Frame')
        axs[i, 0].set_ylabel('Normalized Max Amplitude')
        if viscosity == 'ngm':
            axs[i, 0].set_xticks(range(0, 601, 10))
            axs[i, 0].set_xticklabels(range(0, 601, 10))
        else:
            axs[i, 0].set_xlim(0, 300)
            axs[i, 0].set_xticks(range(0, 301, 10))
            axs[i, 0].set_xticklabels(range(0, 301, 10))
        axs[i, 0].tick_params(axis='x', rotation=45)
        axs[i, 0].set_ylim(y_min, y_max)
        
        # Create boxplot for normalized smoothed max amplitudes, split by frame
        sns.boxplot(x='frames', y='normalized_smoothed_max_amplitudes', data=condition_df, ax=axs[i, 1])
        
        # Customize the subplot
        axs[i, 1].set_title(f'Condition: {condition} - Smoothed')
        axs[i, 1].set_xlabel('Frame')
        axs[i, 1].set_ylabel('Normalized Smoothed Max Amplitude')
        if viscosity == 'ngm':
            axs[i, 1].set_xticks(range(0, 601, 10))
            axs[i, 1].set_xticklabels(range(0, 601, 10))
        else:
            axs[i, 1].set_xlim(0, 300)
            axs[i, 1].set_xticks(range(0, 301, 10))
            axs[i, 1].set_xticklabels(range(0, 301, 10))
        axs[i, 1].tick_params(axis='x', rotation=45)
        axs[i, 1].set_ylim(y_min, y_max)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/normalized_max_amplitudes_by_frame_and_condition_boxplot_{viscosity}.png', dpi=300, bbox_inches='tight')
    plt.close()





########## Using percentage of max amplitude possible (half worm length) ##########

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
# Calculate amplitudes as a percentage of half the worm length per frame
filtered_df.loc[:, 'max_amplitude_percentage'] = filtered_df['max_amplitudes'] / ((filtered_df['worm_lengths'] / 2))

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition_x'].unique())
viscosities = sorted(filtered_df['viscosity_x'].unique())
viscosities = [visc for visc in viscosities if visc != 'visc05'] + ['visc05']


###### Boxplot of max amplitude for each frame as a proportion of max amplitude possible (half worm length) ######

# Calculate the total width for the figure
total_width = 20 + 5  # 20 for non-ngm viscosities, 5 for the extra width of ngm

# Create a figure with subplots for each condition and viscosity
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(total_width, 5*len(conditions)), sharex=False, sharey=True, gridspec_kw={'width_ratios': [2 if v == 'ngm' else 1 for v in viscosities]})
fig.suptitle('Max Amplitude Percentage by Frame, Condition, and Viscosity', fontsize=16)

for i, condition in enumerate(conditions):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition_x'] == condition) & (filtered_df['viscosity_x'] == viscosity)]
        
        sns.boxplot(x='frames', y='max_amplitude_percentage', data=subset, ax=axs[i, j])
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame')
        axs[i, j].set_ylabel('Max Amplitude Percentage')
        axs[i, j].set_ylim(0, 1)
        
        if viscosity == 'ngm':
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 100))
            axs[i, j].set_xticklabels(range(0, 601, 100))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 50))
            axs[i, j].set_xticklabels(range(0, 301, 50))
        
        axs[i, j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/max_amplitude_percentage_by_frame_condition_viscosity.png', dpi=300, bbox_inches='tight')
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
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/max_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
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
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/avg_max_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
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
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/maxamplitude/max_max_amplitude_percentage_per_video_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



# endregion

# region [Average amplitude plots]

###### Boxplot of average_amplitude (for each frame) per condition ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Calculate mean values for condition "a" and viscosity "ngm"
ngm_data = filtered_df[(filtered_df['condition'] == 'a') & (filtered_df['viscosity'] == 'ngm')]
mean_a_avg = ngm_data['avg_amplitudes'].mean()
mean_a_smoothed = ngm_data['smoothed_avg_amplitudes'].mean()

# Normalize the values relative to the mean of condition "a"
filtered_df['normalized_avg_amplitudes'] = (filtered_df['avg_amplitudes'] - mean_a_avg) / mean_a_avg
filtered_df['normalized_smoothed_avg_amplitudes'] = (filtered_df['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed

# Get unique viscosities and sort them
viscosities = sorted(filtered_df['viscosity'].unique())
# Move 'visc05' to the end if it exists
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(len(viscosities), 2, figsize=(16, 6*len(viscosities)))
fig.suptitle('Normalized Average Amplitude and Smoothed Average Amplitude per Condition and Viscosity', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(filtered_df['normalized_avg_amplitudes'].min(), filtered_df['normalized_smoothed_avg_amplitudes'].min())
y_max = max(filtered_df['normalized_avg_amplitudes'].max(), filtered_df['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

for i, viscosity in enumerate(viscosities):
    viscosity_data = filtered_df[filtered_df['viscosity'] == viscosity]
    
    # Boxplot for normalized avg_amplitudes
    sns.boxplot(x='condition', y='normalized_avg_amplitudes', data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 0])
    axs[i, 0].set_title(f'Normalized Average Amplitude (Viscosity: {viscosity})')
    axs[i, 0].set_xlabel('Condition')
    axs[i, 0].set_ylabel('Normalized Average Amplitude')
    axs[i, 0].tick_params(axis='x', rotation=45)
    axs[i, 0].set_ylim(y_min_with_buffer, y_max_with_buffer)

    # Boxplot for normalized smoothed_avg_amplitudes
    sns.boxplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=viscosity_data, order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 1])
    axs[i, 1].set_title(f'Normalized Smoothed Average Amplitude (Viscosity: {viscosity})')
    axs[i, 1].set_xlabel('Condition')
    axs[i, 1].set_ylabel('Normalized Smoothed Average Amplitude')
    axs[i, 1].tick_params(axis='x', rotation=45)
    axs[i, 1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/normalized_avgamplitude_boxplots_no_zaxis_turns_by_viscosity.png')
plt.close()




###### Boxplot of average avg amplitude value per video ######
#Use only values when z_axis_turn is False

# Calculate and plot average avg amplitude per video
# Calculate average amplitudes per video
avg_amplitudes_per_video = combined_df[~combined_df['z_axis_turn']].groupby('video_id').agg({
    'avg_amplitudes': 'mean',
    'smoothed_avg_amplitudes': 'mean',
    'condition_x': 'first',  # Keep the condition for each video_id
    'viscosity_x': 'first'  # Add viscosity to the groupby
}).reset_index()
avg_amplitudes_per_video = avg_amplitudes_per_video.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Calculate mean values for condition "a" and viscosity "ngm"
ngm_data = filtered_df[(filtered_df['condition'] == 'a') & (filtered_df['viscosity'] == 'ngm')]
mean_a_avg = ngm_data['avg_amplitudes'].mean()
mean_a_smoothed = ngm_data['smoothed_avg_amplitudes'].mean()

# Normalize values by making the value relative to the mean value for condition "a"
avg_amplitudes_per_video['normalized_avg_amplitudes'] = (avg_amplitudes_per_video['avg_amplitudes']-mean_a_avg) / mean_a_avg
avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'] = (avg_amplitudes_per_video['smoothed_avg_amplitudes']-mean_a_smoothed) / mean_a_smoothed

# Get unique viscosities and sort them
viscosities = sorted(avg_amplitudes_per_video['viscosity'].unique())
# Move 'visc05' to the end if it exists
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(len(viscosities), 2, figsize=(16, 6*len(viscosities)))
fig.suptitle('Normalized Average Average Amplitude per Video', fontsize=16)

# Calculate the global min and max values across both normalized amplitude columns and add a 10% buffer
y_min = min(avg_amplitudes_per_video['normalized_avg_amplitudes'].min(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].min())
y_max = max(avg_amplitudes_per_video['normalized_avg_amplitudes'].max(), avg_amplitudes_per_video['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_amplitudes_per_video[avg_amplitudes_per_video['viscosity'] == viscosity]
    
    # Boxplot for normalized average avg_amplitudes
    sns.boxplot(x='condition', y='normalized_avg_amplitudes', data=viscosity_data, 
                order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 0])
    sns.swarmplot(x='condition', y='normalized_avg_amplitudes', data=viscosity_data, 
                  order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 0], color=".25", size=5, alpha=0.5)
    axs[i, 0].set_title(f'Normalized Average Avg Amplitude (Viscosity: {viscosity})')
    axs[i, 0].set_xlabel('Condition')
    axs[i, 0].set_ylabel('Normalized Average Avg Amplitude')
    axs[i, 0].tick_params(axis='x', rotation=45)
    axs[i, 0].set_ylim(y_min_with_buffer, y_max_with_buffer)

    # Boxplot for normalized average smoothed_avg_amplitudes
    sns.boxplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=viscosity_data, 
                order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 1])
    sns.swarmplot(x='condition', y='normalized_smoothed_avg_amplitudes', data=viscosity_data, 
                  order=sorted(viscosity_data['condition'].unique()), ax=axs[i, 1], color=".25", size=5, alpha=0.5)
    axs[i, 1].set_title(f'Normalized Average Smoothed Avg Amplitude (Viscosity: {viscosity})')
    axs[i, 1].set_xlabel('Condition')
    axs[i, 1].set_ylabel('Normalized Average Smoothed Avg Amplitude')
    axs[i, 1].tick_params(axis='x', rotation=45)
    axs[i, 1].set_ylim(y_min_with_buffer, y_max_with_buffer)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/normalized_avg_avg_amplitude_per_video_boxplots_by_viscosity.png')
plt.close()






###### Boxplot of avg amplitude for each frame ######

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
filtered_df = filtered_df.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Calculate mean values for condition "a" and viscosity "ngm"
ngm_data = filtered_df[(filtered_df['condition'] == 'a') & (filtered_df['viscosity'] == 'ngm')]
mean_a_avg = ngm_data['avg_amplitudes'].mean()
mean_a_smoothed = ngm_data['smoothed_avg_amplitudes'].mean()

# Normalize values by making the value relative to the mean value for condition "a"
filtered_df['normalized_avg_amplitudes'] = (filtered_df['avg_amplitudes'] - mean_a_avg) / mean_a_avg
filtered_df['normalized_smoothed_avg_amplitudes'] = (filtered_df['smoothed_avg_amplitudes'] - mean_a_smoothed) / mean_a_smoothed

# Get unique viscosities and sort them
viscosities = sorted(filtered_df['viscosity'].unique())
# Move 'visc05' to the end if it exists
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')

# Get unique conditions
conditions = sorted(filtered_df['condition'].unique())

# Calculate global y-axis limits
y_min = min(filtered_df['normalized_avg_amplitudes'].min(), filtered_df['normalized_smoothed_avg_amplitudes'].min())
y_max = max(filtered_df['normalized_avg_amplitudes'].max(), filtered_df['normalized_smoothed_avg_amplitudes'].max())
y_range = y_max - y_min
y_min_with_buffer = y_min - 0.1 * y_range
y_max_with_buffer = y_max + 0.1 * y_range

for viscosity in viscosities:
    viscosity_df = filtered_df[filtered_df['viscosity'] == viscosity]
    
    # Create a figure with 4x2 subplots, one row for each condition, two columns for smoothed and non-smoothed
    fig, axs = plt.subplots(4, 2, figsize=(42, 24))
    fig.suptitle(f'Normalized Avg Amplitudes by Frame for Each Condition (Viscosity: {viscosity})', fontsize=16)

    for i, condition in enumerate(conditions):
        condition_df = viscosity_df[viscosity_df['condition'] == condition]
        
        # Create boxplot for normalized avg amplitudes, split by frame
        sns.boxplot(x='frames', y='normalized_avg_amplitudes', data=condition_df, ax=axs[i, 0])
        
        # Customize the subplot
        axs[i, 0].set_title(f'Condition: {condition} - Non-smoothed')
        axs[i, 0].set_xlabel('Frame')
        axs[i, 0].set_ylabel('Normalized Avg Amplitude')
        if viscosity == 'ngm':
            axs[i, 0].set_xlim(0, 600)
            axs[i, 0].set_xticks(range(0, 601, 50))
            axs[i, 0].set_xticklabels(range(0, 601, 50))
        else:
            axs[i, 0].set_xlim(0, 300)
            axs[i, 0].set_xticks(range(0, 301, 25))
            axs[i, 0].set_xticklabels(range(0, 301, 25))
        axs[i, 0].tick_params(axis='x', rotation=45)
        axs[i, 0].set_ylim(y_min_with_buffer, y_max_with_buffer)
        
        # Create boxplot for normalized smoothed avg amplitudes, split by frame
        sns.boxplot(x='frames', y='normalized_smoothed_avg_amplitudes', data=condition_df, ax=axs[i, 1])
        
        # Customize the subplot
        axs[i, 1].set_title(f'Condition: {condition} - Smoothed')
        axs[i, 1].set_xlabel('Frame')
        axs[i, 1].set_ylabel('Normalized Smoothed Avg Amplitude')
        if viscosity == 'ngm':
            axs[i, 1].set_xlim(0, 600)
            axs[i, 1].set_xticks(range(0, 601, 50))
            axs[i, 1].set_xticklabels(range(0, 601, 50))
        else:
            axs[i, 1].set_xlim(0, 300)
            axs[i, 1].set_xticks(range(0, 301, 25))
            axs[i, 1].set_xticklabels(range(0, 301, 25))
        axs[i, 1].tick_params(axis='x', rotation=45)
        axs[i, 1].set_ylim(y_min_with_buffer, y_max_with_buffer)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/normalized_avg_amplitudes_by_frame_and_condition_boxplot_{viscosity}.png', dpi=300, bbox_inches='tight')
    plt.close()



########## Using percentage of max amplitude possible (half worm length) ##########

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]
# Calculate amplitudes as a percentage of half the worm length per frame
filtered_df.loc[:, 'avg_amplitude_percentage'] = filtered_df['avg_amplitudes'] / ((filtered_df['worm_lengths'] / 2))

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition_x'].unique())
viscosities = sorted(filtered_df['viscosity_x'].unique())
viscosities = [visc for visc in viscosities if visc != 'visc05'] + ['visc05']


###### Boxplot of avg amplitude for each frame as a proportion of max amplitude possible (half worm length) ######

# Calculate the total width of the figure
total_width = sum(2 if visc == 'ngm' else 1 for visc in viscosities)

# Create a figure with subplots for each condition and viscosity
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(20 * (total_width / len(viscosities)), 5*len(conditions)), sharex=False, sharey=True, gridspec_kw={'width_ratios': [2 if visc == 'ngm' else 1 for visc in viscosities]})
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
        
        axs[i, j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/avg_amplitude_percentage_by_frame_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()




###### Boxplot of avg amplitude percentage by condition and viscosity ######

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
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/avg_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



###### Boxplot of average avg amplitude percentage by condition and viscosity ######

# Calculate the average avg_amplitude_percentage for each video_id
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
    axs[j].tick_params(axis='x')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/avgamplitude/avg_avg_amplitude_percentage_by_condition_viscosity.png', dpi=300, bbox_inches='tight')
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

# Calculate average and max curvature for each frame in each video (every row) ("curvatures" is an array of curvature values along the worm length for each frame)
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

# Get unique conditions and viscosities, with conditions sorted alphabetically and putting "visc05" at the end
conditions = sorted(filtered_df['condition'].unique())
viscosities = filtered_df['viscosity'].unique()
viscosities = [visc for visc in viscosities if visc != 'visc05'] + ['visc05']




###### Boxplot of curvature by frame and condition/viscosity ######

# Create a figure with subplots for each condition and viscosity
# Plot for average curvature
# Calculate the width ratios for the subplots
width_ratios = [2 if v == 'ngm' else 1 for v in viscosities]

fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(30, 6*len(conditions)), 
                        sharex=True, sharey=True, gridspec_kw={'width_ratios': width_ratios})
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
            axs[i, j].set_xticks(range(0, 601, 20))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 20))
        
        # Only show every 20th tick label
        for label in axs[i, j].xaxis.get_ticklabels()[1::20]:
            label.set_visible(False)
        
        # Set y-axis limit
        axs[i, j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/normalized_avg_curvature_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



# Plot for max curvature
# Calculate the width ratios for the subplots
width_ratios = [2 if v == 'ngm' else 1 for v in viscosities]

fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(30, 6*len(conditions)), 
                        sharex=True, sharey=True, gridspec_kw={'width_ratios': width_ratios})
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
            axs[i, j].set_xticks(range(0, 601, 20))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 20))
        
        # Only show every 20th tick label
        for label in axs[i, j].xaxis.get_ticklabels()[1::20]:
            label.set_visible(False)
        
        # Set y-axis limit
        axs[i, j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/normalized_max_curvature_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
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
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/normalized_avg_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
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
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/avg_normalized_avg_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
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
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/normalized_max_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
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
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/avg_normalized_max_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()




###### Boxplot of maximum normalized_max_curvature by video by condition and viscosity ######
conditions = sorted(max_curvature_by_video['condition'].unique())
viscosities = list(max_curvature_by_video['viscosity'].unique())

# Move 'visc05' to the end if it exists
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')

# Create a figure with subplots for each viscosity
fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Maximum Normalized Curvature in Video by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = max_curvature_by_video[max_curvature_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='normalized_max_curvature', data=subset, ax=axs[j], order=conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Maximum Normalized Curvature by Video' if j == 0 else '')
    axs[j].tick_params(axis='x')
    axs[j].set_ylim(bottom=0)  # Set minimum y-axis limit to 0

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/curvature/max_normalized_curvature_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()


# endregion


# region [dominant_spatial_freqs]

# ----> Maybe check s-shapes only?

# Filter out z-axis turns
filtered_df = combined_df[~combined_df['z_axis_turn']]

# Calculate the normalization factor (mean value for condition "a" and viscosity "ngm")
norm_factor = filtered_df[(filtered_df['condition_x'] == 'a') & (filtered_df['viscosity_x'] == 'ngm')]['dominant_spatial_freqs'].mean()
filtered_df = filtered_df.rename(columns={'condition_x': 'condition', 'viscosity_x': 'viscosity'})

# Normalize the dominant_spatial_freqs
filtered_df['normalized_dominant_spatial_freqs'] = (filtered_df['dominant_spatial_freqs'] - norm_factor) / norm_factor

# Get unique conditions and viscosities
conditions = sorted(filtered_df['condition'].unique())
viscosities = filtered_df['viscosity'].unique().tolist()
if 'visc05' in viscosities:
    viscosities.remove('visc05')
    viscosities.append('visc05')


###### Boxplot of normalized_dominant_spatial_freqs by frame, condition, and viscosity ######

# Calculate the width ratios for the subplots
width_ratios = [2 if v == 'ngm' else 1 for v in viscosities]

# Create a figure with subplots for each condition and viscosity
fig, axs = plt.subplots(len(conditions), len(viscosities), figsize=(30, 6*len(conditions)), 
                        sharex=False, sharey=True, gridspec_kw={'width_ratios': width_ratios})
fig.suptitle('Normalized Dominant Spatial Frequencies by Frame, Condition, and Viscosity', fontsize=16)

# Calculate y-axis limits
y_min = filtered_df['normalized_dominant_spatial_freqs'].min()
y_max = filtered_df['normalized_dominant_spatial_freqs'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, condition in enumerate(conditions):
    for j, viscosity in enumerate(viscosities):
        subset = filtered_df[(filtered_df['condition'] == condition) & (filtered_df['viscosity'] == viscosity)]
        
        sns.boxplot(x='frames', y='normalized_dominant_spatial_freqs', data=subset, ax=axs[i, j], color='lightblue', fliersize=1)
        
        axs[i, j].set_title(f'Condition: {condition}, Viscosity: {viscosity}')
        axs[i, j].set_xlabel('Frame' if i == len(conditions) - 1 else '')
        axs[i, j].set_ylabel('Normalized Dominant Spatial Frequencies' if j == 0 else '')
        axs[i, j].tick_params(axis='x', rotation=45)
        
        # Set x-axis limit and ticks
        if viscosity == "ngm":
            axs[i, j].set_xlim(0, 600)
            axs[i, j].set_xticks(range(0, 601, 50))
        else:
            axs[i, j].set_xlim(0, 300)
            axs[i, j].set_xticks(range(0, 301, 30))
        
        # Only show every 10th tick label
        for label in axs[i, j].xaxis.get_ticklabels()[::10]:
            label.set_visible(True)
        for label in axs[i, j].xaxis.get_ticklabels():
            if label not in axs[i, j].xaxis.get_ticklabels()[::10]:
                label.set_visible(False)
        
        # Set y-axis limit
        axs[i, j].set_ylim(y_limits)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/spatialfreq/normalized_dominant_spatial_freqs_by_frame_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()




###### Boxplot of normalized_dominant_spatial_freqs by condition and viscosity ######

# Calculate y-axis limits for all subplots
y_min = filtered_df['normalized_dominant_spatial_freqs'].min()
y_max = filtered_df['normalized_dominant_spatial_freqs'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Normalized Dominant Spatial Frequencies by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = filtered_df[filtered_df['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='normalized_dominant_spatial_freqs', data=subset, ax=axs[j], order=conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Normalized Dominant Spatial Frequencies' if j == 0 else '')
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(y_limits)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/spatialfreq/normalized_dominant_spatial_freqs_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



###### Boxplot of mean normalized_dominant_spatial_freqs by condition and viscosity ######

# Calculate mean values for each video_id
mean_df = filtered_df.groupby(['video_id', 'condition', 'viscosity'])['normalized_dominant_spatial_freqs'].mean().reset_index()

# Calculate y-axis limits for all subplots
y_min = mean_df['normalized_dominant_spatial_freqs'].min()
y_max = mean_df['normalized_dominant_spatial_freqs'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

fig, axs = plt.subplots(1, len(viscosities), figsize=(20, 6), sharey=True)
fig.suptitle('Mean Normalized Dominant Spatial Frequencies by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    subset = mean_df[mean_df['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='normalized_dominant_spatial_freqs', data=subset, ax=axs[j], order=conditions)
    
    axs[j].set_title(f'Viscosity: {viscosity}')
    axs[j].set_xlabel('Condition')
    axs[j].set_ylabel('Mean Normalized Dominant Spatial Frequencies' if j == 0 else '')
    axs[j].tick_params(axis='x')
    
    # Set y-axis limit
    axs[j].set_ylim(y_limits)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/spatialfreq/mean_normalized_dominant_spatial_freqs_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()




###### Boxplot of maximum normalized_dominant_spatial_freqs by condition and viscosity ######
# Calculate maximum values for each video_id
max_df = filtered_df.groupby(['video_id', 'condition', 'viscosity'])['normalized_dominant_spatial_freqs'].max().reset_index()

# Calculate minimum values for each video_id
min_df = filtered_df.groupby(['video_id', 'condition', 'viscosity'])['normalized_dominant_spatial_freqs'].min().reset_index()

# Calculate y-axis limits for all subplots
y_min = min(max_df['normalized_dominant_spatial_freqs'].min(), min_df['normalized_dominant_spatial_freqs'].min())
y_max = max(max_df['normalized_dominant_spatial_freqs'].max(), min_df['normalized_dominant_spatial_freqs'].max())
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

fig, axs = plt.subplots(2, len(viscosities), figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle('Maximum and Minimum Normalized Dominant Spatial Frequencies by Condition and Viscosity', fontsize=16)

for j, viscosity in enumerate(viscosities):
    # Maximum plot
    max_subset = max_df[max_df['viscosity'] == viscosity]
    sns.boxplot(x='condition', y='normalized_dominant_spatial_freqs', data=max_subset, ax=axs[0, j], order=conditions)
    sns.swarmplot(x='condition', y='normalized_dominant_spatial_freqs', data=max_subset, ax=axs[0, j], order=conditions, color='.25', size=4)
    
    axs[0, j].set_title(f'Viscosity: {viscosity}')
    axs[0, j].set_xlabel('')
    axs[0, j].set_ylabel('Maximum Normalized Dominant Spatial Frequencies' if j == 0 else '')
    axs[0, j].tick_params(axis='x', rotation=45)
    axs[0, j].set_ylim(y_limits)

    # Minimum plot
    min_subset = min_df[min_df['viscosity'] == viscosity]
    sns.boxplot(x='condition', y='normalized_dominant_spatial_freqs', data=min_subset, ax=axs[1, j], order=conditions)
    sns.swarmplot(x='condition', y='normalized_dominant_spatial_freqs', data=min_subset, ax=axs[1, j], order=conditions, color='.25', size=4)
    
    axs[1, j].set_xlabel('Condition')
    axs[1, j].set_ylabel('Minimum Normalized Dominant Spatial Frequencies' if j == 0 else '')
    axs[1, j].tick_params(axis='x', rotation=45)
    axs[1, j].set_ylim(y_limits)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/spatialfreq/max_min_normalized_dominant_spatial_freqs_by_condition_viscosity_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()



# endregion


# region [interpolated_freqs]


### Recalculate interpolated_freqs with the global curvature values

import numpy as np
from scipy.signal import welch, find_peaks
import ast

def string_to_numpy(s):
    """Convert a string representation of a numpy array to an actual numpy array."""
    # Remove square brackets and split the string into individual numbers
    numbers = s.strip('[]').split()
    # Convert the list of strings to a numpy array of floats
    return np.array([float(num) for num in numbers])

def calculate_comparable_frequencies(all_curvatures, fps, nperseg, noverlap):
    # Flatten all curvatures into a single 1D array
    all_curvature_1d = np.concatenate([np.mean(c, axis=1) for c in all_curvatures])
    
    # Global normalization
    global_mean = np.mean(all_curvature_1d)
    global_std = np.std(all_curvature_1d)
    
    all_interpolated_freqs = []
    
    for curvatures in all_curvatures:
        # Normalize curvature using global mean and std
        curvature_1d = np.mean(curvatures, axis=1)
        curvature_1d = (curvature_1d - global_mean) / global_std
        
        # Calculate PSD for the entire signal
        f, psd = welch(curvature_1d, fs=fps, nperseg=nperseg, noverlap=noverlap)
        
        # Sliding window analysis
        dominant_freqs = []
        time_points = []
        
        for i in range(0, len(curvature_1d), nperseg - noverlap):
            end = min(i + nperseg, len(curvature_1d))
            segment = curvature_1d[i:end]
            
            if len(segment) < nperseg:
                # Pad the last segment if it's shorter than nperseg
                segment = np.pad(segment, (0, nperseg - len(segment)), mode='edge')
            
            f_segment, psd_segment = welch(segment, fs=fps, nperseg=nperseg, noverlap=noverlap)
            
            peaks, _ = find_peaks(psd_segment, height=np.max(psd_segment) * 0.1)
            if len(peaks) > 0:
                dominant_freq_idx = peaks[np.argmax(psd_segment[peaks])]
                dominant_freqs.append(f_segment[dominant_freq_idx])
            else:
                # Instead of appending 0, use the frequency with maximum power
                dominant_freqs.append(f_segment[np.argmax(psd_segment)])
            
            time_points.append(i / fps)
        
        # Ensure we have a frequency for the last data point
        if len(time_points) < len(curvature_1d):
            time_points.append((len(curvature_1d) - 1) / fps)
            dominant_freqs.append(dominant_freqs[-1])
        
        # Interpolate frequencies
        frame_numbers = np.arange(len(curvatures))
        interpolated_freqs = np.interp(frame_numbers / fps, time_points, dominant_freqs)
        
        all_interpolated_freqs.append(interpolated_freqs)
    
    return all_interpolated_freqs


# Usage
fps = 10  # frames per second
nperseg = 30  # window size for Welch's method
noverlap = 25  # overlap between windows

# Prepare all_curvatures list
all_curvatures = []
video_ids = []
for video_id, group in combined_df.groupby('video_id'):
    curvatures = np.array([string_to_numpy(c) for c in group['curvatures']])
    all_curvatures.append(curvatures)
    video_ids.append(video_id)

comparable_interpolated_freqs = calculate_comparable_frequencies(all_curvatures, fps, nperseg, noverlap)


# Create a dataframe to store the results
import pandas as pd

# Create an empty list to store the data
data_list = []

# Iterate through each video and its frequencies
for video_id, freqs in zip(video_ids, comparable_interpolated_freqs):
    # For each frequency value, create a row with video_id, frame number, and frequency value
    for frame, freq in enumerate(freqs):
        data_list.append({
            'video_id': video_id,
            'frame': frame,
            'interpolated_freq': freq
        })

# Create the DataFrame from the list of dictionaries
new_interpolated_freqs_df = pd.DataFrame(data_list)

# Sort the DataFrame by video_id and frame
new_interpolated_freqs_df = new_interpolated_freqs_df.sort_values(['video_id', 'frame'])

# Reset the index
new_interpolated_freqs_df = new_interpolated_freqs_df.reset_index(drop=True)

def extract_info_from_filename(filename):
    base_filename = os.path.basename(filename)
    parts = base_filename.split('_')
    viscosity = parts[0]
    condition = parts[1]
    id = parts[2]
    return viscosity, condition, id

# Extract viscosity and condition from video_id
new_interpolated_freqs_df[['viscosity', 'condition', 'id']] = new_interpolated_freqs_df['video_id'].apply(lambda x: pd.Series(extract_info_from_filename(x)))

#Save the new_interpolated_freqs_df to a csv file
new_interpolated_freqs_df.to_csv('new_interpolated_freqs_df.csv', index=False)



###### Dot plot of interpolated_freqs by frame for 5 random video_id ######

# Select 5 random video_ids
random_video_ids = np.random.choice(new_interpolated_freqs_df['video_id'].unique(), 5, replace=False)

# Create a figure with subplots for each random video
fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True, sharey=True)
fig.suptitle('Interpolated Frequencies by Frame for 5 Random Videos', fontsize=16)

# Calculate global y-limits
y_min = new_interpolated_freqs_df['interpolated_freq'].min()
y_max = new_interpolated_freqs_df['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, video_id in enumerate(random_video_ids):
    video_data = new_interpolated_freqs_df[new_interpolated_freqs_df['video_id'] == video_id]
    
    axs[i].scatter(video_data['frame'], video_data['interpolated_freq'], alpha=0.6, s=10)
    axs[i].set_ylabel('Oscillation rate (Hz)')
    axs[i].set_title(f'Video ID: {video_id}')
    axs[i].set_ylim(y_limits)
    
    # Add condition and viscosity information to the plot
    condition = video_data['condition'].iloc[0]
    viscosity = video_data['viscosity'].iloc[0]
    axs[i].text(0.02, 0.95, f'Condition: {condition}, Viscosity: {viscosity}', 
                transform=axs[i].transAxes, fontsize=10, verticalalignment='top')

axs[-1].set_xlabel('Frame')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/random_videos_interpolated_freqs_dot_plot.png', dpi=300, bbox_inches='tight')
plt.close()




###### Boxplot of interpolated freq by frame, split by viscosity and condition ######
# Sort conditions alphabetically
sorted_conditions = sorted(new_interpolated_freqs_df['condition'].unique())

# Get unique viscosities and sort them, putting visc05 last
viscosities = sorted([v for v in new_interpolated_freqs_df['viscosity'].unique() if v != 'visc05']) + ['visc05']

# Create subplots with increased width
fig, axs = plt.subplots(len(sorted_conditions), len(viscosities), 
                        figsize=(8*len(viscosities), 6*len(sorted_conditions)), 
                        gridspec_kw={'width_ratios': [2 if v == 'ngm' else 1 for v in viscosities]})
fig.suptitle('Interpolated Frequencies by Frame, Condition, and Viscosity', fontsize=16)

# Calculate global y-limits
y_min = new_interpolated_freqs_df['interpolated_freq'].min()
y_max = new_interpolated_freqs_df['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, condition in enumerate(sorted_conditions):
    for j, viscosity in enumerate(viscosities):
        data = new_interpolated_freqs_df[(new_interpolated_freqs_df['condition'] == condition) & 
                                         (new_interpolated_freqs_df['viscosity'] == viscosity)]
        
        sns.boxplot(x='frame', y='interpolated_freq', data=data, ax=axs[i, j])
        
        # Set y-limit to be the same for all subplots
        axs[i, j].set_ylim(y_limits)
        
        # Set x-limit to 300 when viscosity is not 'ngm'
        if viscosity != 'ngm':
            axs[i, j].set_xlim(0, 300)
        
        # Set x-ticks and labels only every 50 frames
        x_ticks = axs[i, j].get_xticks()[::50]
        axs[i, j].set_xticks(x_ticks)
        axs[i, j].set_xticklabels([f'{int(x)}' if x % 50 == 0 else '' for x in x_ticks])
        
        # Set labels
        if j == 0:
            axs[i, j].set_ylabel(f'{condition}\nOscillation rate (Hz)')
        if i == len(sorted_conditions) - 1:
            axs[i, j].set_xlabel('Frame')
        
        axs[i, j].set_title(f'Viscosity: {viscosity}')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/interpolated_freqs_boxplot_by_frame_condition_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()




###### Box plot of interpolated frequencies by condition and viscosity ######

# Sort conditions alphabetically
sorted_conditions = sorted(new_interpolated_freqs_df['condition'].unique())

# Get unique viscosities and sort them (assuming 'visc05' should be rightmost)
viscosities = sorted(new_interpolated_freqs_df['viscosity'].unique(), key=lambda x: x if x != 'visc05' else 'z')

# Create subplots
fig, axs = plt.subplots(1, len(viscosities), figsize=(5*len(viscosities), 6), sharey=True)
fig.suptitle('Interpolated Frequencies by Condition and Viscosity', fontsize=16)

# Calculate global y-limits
y_min = new_interpolated_freqs_df['interpolated_freq'].min()
y_max = new_interpolated_freqs_df['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, viscosity in enumerate(viscosities):
    viscosity_data = new_interpolated_freqs_df[new_interpolated_freqs_df['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='interpolated_freq', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i])
    
    axs[i].set_title(f'Viscosity: {viscosity}')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylim(y_limits)
    
    if i == 0:
        axs[i].set_ylabel('Oscillation rate (Hz)')
    else:
        axs[i].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i].set_xticklabels(axs[i].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/interpolated_freqs_boxplot_by_condition_and_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()




###### Box plot of average interpolated frequencies by condition and viscosity ######

# Calculate average interpolated frequency by video_id
avg_freq_by_video = new_interpolated_freqs_df.groupby(['condition', 'viscosity', 'video_id'])['interpolated_freq'].mean().reset_index()

# Create subplots
fig, axs = plt.subplots(1, len(viscosities), figsize=(5*len(viscosities), 6), sharey=True)
fig.suptitle('Average Interpolated Frequencies by Condition and Viscosity', fontsize=16)

# Calculate global y-limits for the new plot
y_min = avg_freq_by_video['interpolated_freq'].min()
y_max = avg_freq_by_video['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_freq_by_video[avg_freq_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='interpolated_freq', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i])
    
    axs[i].set_title(f'Viscosity: {viscosity}')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylim(y_limits)
    
    if i == 0:
        axs[i].set_ylabel('Average Oscillation rate (Hz)')
    else:
        axs[i].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i].set_xticklabels(axs[i].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/avg_interpolated_freqs_boxplot_by_condition_and_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()




###### Box plot of average interpolated frequencies by condition and viscosity (swim only) ######

# Calculate average interpolated frequency by video_id, filtering for values >= 1
avg_freq_by_video = new_interpolated_freqs_df[new_interpolated_freqs_df['interpolated_freq'] >= 1].groupby(['condition', 'viscosity', 'video_id'])['interpolated_freq'].mean().reset_index()

# Create subplots
fig, axs = plt.subplots(1, len(viscosities), figsize=(5*len(viscosities), 6), sharey=True)
fig.suptitle('Average Interpolated Frequencies by Condition and Viscosity (≥ 1 Hz)', fontsize=16)

# Calculate global y-limits for the new plot
y_min = avg_freq_by_video['interpolated_freq'].min()
y_max = avg_freq_by_video['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (max(1, y_min - y_buffer), y_max + y_buffer)  # Ensure lower limit is at least 1

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_freq_by_video[avg_freq_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='interpolated_freq', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i])
    
    axs[i].set_title(f'Viscosity: {viscosity}')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylim(y_limits)
    
    if i == 0:
        axs[i].set_ylabel('Average Oscillation rate (Hz)')
    else:
        axs[i].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i].set_xticklabels(axs[i].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/avg_interpolated_freqs_boxplot_by_condition_and_viscosity_filtered.png', dpi=300, bbox_inches='tight')
plt.close()

###### Box plot of average interpolated frequencies by condition and viscosity (quiescent only) ######

# Calculate average interpolated frequency by video_id, filtering for values >= 1
avg_freq_by_video = new_interpolated_freqs_df[new_interpolated_freqs_df['interpolated_freq'] < 0.4].groupby(['condition', 'viscosity', 'video_id'])['interpolated_freq'].mean().reset_index()

# Create subplots
fig, axs = plt.subplots(1, len(viscosities), figsize=(5*len(viscosities), 6), sharey=True)
fig.suptitle('Average Interpolated Frequencies by Condition and Viscosity (< 0.4 Hz)', fontsize=16)

# Calculate global y-limits for the new plot
y_min = avg_freq_by_video['interpolated_freq'].min()
y_max = avg_freq_by_video['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (max(0, y_min - y_buffer), y_max + y_buffer)  # Ensure lower limit is at least 1

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_freq_by_video[avg_freq_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='interpolated_freq', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i])
    
    axs[i].set_title(f'Viscosity: {viscosity}')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylim(y_limits)
    
    if i == 0:
        axs[i].set_ylabel('Average Oscillation rate (Hz)')
    else:
        axs[i].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i].set_xticklabels(axs[i].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/avg_interpolated_freqs_boxplot_by_condition_and_viscosity_quiescent.png', dpi=300, bbox_inches='tight')
plt.close()



###### Histogram of interpolated freq, split by viscosity and condition ######

# Sort conditions alphabetically
sorted_conditions = sorted(new_interpolated_freqs_df['condition'].unique())

# Get unique viscosities and sort them, putting visc05 last
viscosities = sorted([v for v in new_interpolated_freqs_df['viscosity'].unique() if v != 'visc05']) + ['visc05']

# Create subplots
fig, axs = plt.subplots(len(viscosities), len(sorted_conditions), figsize=(5*len(sorted_conditions), 4*len(viscosities)), sharex=True, sharey=True)
fig.suptitle('Histogram of Interpolated Frequencies by Condition and Viscosity', fontsize=16)

for i, viscosity in enumerate(viscosities):
    for j, condition in enumerate(sorted_conditions):
        data = new_interpolated_freqs_df[(new_interpolated_freqs_df['viscosity'] == viscosity) & 
                                         (new_interpolated_freqs_df['condition'] == condition)]
        
        sns.histplot(data=data, x='interpolated_freq', ax=axs[i, j], kde=True)
        
        if i == len(viscosities) - 1:
            axs[i, j].set_xlabel('Interpolated Frequency (Hz)')
        else:
            axs[i, j].set_xlabel('')
        
        if j == 0:
            axs[i, j].set_ylabel(f'Viscosity: {viscosity}')
        else:
            axs[i, j].set_ylabel('')
        
        if i == 0:
            axs[i, j].set_title(condition)

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/interpolated_freqs_histogram_by_condition_and_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



###### Box plot of maximum interpolated frequencies by condition and viscosity ######

# Calculate maximum interpolated frequency by video_id
max_freq_by_video = new_interpolated_freqs_df.groupby(['condition', 'viscosity', 'video_id'])['interpolated_freq'].max().reset_index()

# Create subplots
fig, axs = plt.subplots(1, len(viscosities), figsize=(5*len(viscosities), 6), sharey=True)
fig.suptitle('Maximum Interpolated Frequencies by Condition and Viscosity', fontsize=16)

# Calculate global y-limits for the new plot
y_min = max_freq_by_video['interpolated_freq'].min()
y_max = max_freq_by_video['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, viscosity in enumerate(viscosities):
    viscosity_data = max_freq_by_video[max_freq_by_video['viscosity'] == viscosity]
    
    sns.boxplot(x='condition', y='interpolated_freq', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i])
    
    axs[i].set_title(f'Viscosity: {viscosity}')
    axs[i].set_xlabel('Condition')
    axs[i].set_ylim(y_limits)
    
    if i == 0:
        axs[i].set_ylabel('Maximum Oscillation rate (Hz)')
    else:
        axs[i].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i].set_xticklabels(axs[i].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/max_interpolated_freqs_boxplot_by_condition_and_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()



###### Line plot of interpolated freq by frame, split by viscosity and condition ######

# Sort conditions alphabetically
sorted_conditions = sorted(new_interpolated_freqs_df['condition'].unique())

# Get unique viscosities and sort them, putting visc05 last
viscosities = sorted([v for v in new_interpolated_freqs_df['viscosity'].unique() if v != 'visc05']) + ['visc05']

# Calculate the total width of the figure
total_width = sum([2 if v == 'ngm' else 1 for v in viscosities]) * 7  # Increased from 5 to 7

# Create subplots with adjusted widths
fig, axs = plt.subplots(len(sorted_conditions), len(viscosities), 
                        figsize=(total_width, 5*len(sorted_conditions)), 
                        sharex='col', sharey=True,
                        gridspec_kw={'width_ratios': [2 if v == 'ngm' else 1 for v in viscosities]})
fig.suptitle('Interpolated Frequencies by Frame, Condition, and Viscosity', fontsize=16)

# Calculate global y-limits
y_min = new_interpolated_freqs_df['interpolated_freq'].min()
y_max = new_interpolated_freqs_df['interpolated_freq'].max()
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, condition in enumerate(sorted_conditions):
    for j, viscosity in enumerate(viscosities):
        subset = new_interpolated_freqs_df[(new_interpolated_freqs_df['condition'] == condition) & 
                                           (new_interpolated_freqs_df['viscosity'] == viscosity)]
        
        for video_id in subset['video_id'].unique():
            video_data = subset[subset['video_id'] == video_id]
            axs[i, j].plot(video_data['frame'], video_data['interpolated_freq'], alpha=0.6, linewidth=1)
            axs[i, j].scatter(video_data['frame'], video_data['interpolated_freq'], alpha=0.6, s=10)
        
        axs[i, j].set_ylim(y_limits)
        
        if viscosity != 'ngm':
            axs[i, j].set_xlim(0, 300)
        else:
            axs[i, j].set_xlim(0, 600)  # Double the x-axis range for 'ngm'
        
        if i == len(sorted_conditions) - 1:
            axs[i, j].set_xlabel('Frame')
        if j == 0:
            axs[i, j].set_ylabel('Oscillation rate (Hz)')
        
        if i == 0:
            axs[i, j].set_title(f'Viscosity: {viscosity}')
        
        # Add condition label to the left of each row
        if j == 0:
            axs[i, j].text(-0.15, 0.5, f'Condition: {condition}', 
                           rotation=90, transform=axs[i, j].transAxes, 
                           verticalalignment='center', horizontalalignment='center')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/temporalfreq/interpolated_freqs_line_plot_by_condition_and_viscosity.png', dpi=300, bbox_inches='tight')
plt.close()




# endregion


# region [Wavelength]


###### Box plot of wavelength by condition and viscosity ######
# Filter data for S-shape
s_shape_df = combined_df[combined_df['shape'] == 'S-shape']

# Sort conditions alphabetically
sorted_conditions = sorted(s_shape_df['condition_x'].unique())

# Get unique viscosities and sort them (assuming 'visc05' should be rightmost)
viscosities = sorted(s_shape_df['viscosity_x'].unique(), key=lambda x: x if x != 'visc05' else 'z')

# Create subplots for wavelengths and smoothed wavelengths
fig, axs = plt.subplots(len(viscosities), 2, figsize=(12, 5*len(viscosities)), sharey='col')
fig.suptitle('Wavelengths and Smoothed Wavelengths by Condition and Viscosity (S-shape)', fontsize=16)

# Calculate global y-limits for wavelengths
y_min_wavelengths = s_shape_df['wavelengths'].min()
y_max_wavelengths = s_shape_df['wavelengths'].max()
y_range_wavelengths = y_max_wavelengths - y_min_wavelengths
y_buffer_wavelengths = y_range_wavelengths * 0.1
y_limits_wavelengths = (y_min_wavelengths - y_buffer_wavelengths, y_max_wavelengths + y_buffer_wavelengths)

# Calculate global y-limits for smoothed wavelengths
y_min_smoothed = s_shape_df['smoothed_wavelengths'].min()
y_max_smoothed = s_shape_df['smoothed_wavelengths'].max()
y_range_smoothed = y_max_smoothed - y_min_smoothed
y_buffer_smoothed = y_range_smoothed * 0.1
y_limits_smoothed = (y_min_smoothed - y_buffer_smoothed, y_max_smoothed + y_buffer_smoothed)

for i, viscosity in enumerate(viscosities):
    viscosity_data = s_shape_df[s_shape_df['viscosity_x'] == viscosity]
    
    # Plot wavelengths
    sns.boxplot(x='condition_x', y='wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 0])
    
    axs[i, 0].set_title(f'Viscosity: {viscosity}')
    axs[i, 0].set_xlabel('')
    axs[i, 0].set_ylim(y_limits_wavelengths)
    
    if i == len(viscosities) - 1:
        axs[i, 0].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 0].set_ylabel('Wavelength (pixels)')
    else:
        axs[i, 0].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), ha='right')
    
    # Plot smoothed wavelengths
    sns.boxplot(x='condition_x', y='smoothed_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 1])
    
    axs[i, 1].set_title(f'Viscosity: {viscosity}')
    axs[i, 1].set_xlabel('')
    axs[i, 1].set_ylim(y_limits_smoothed)
    
    if i == len(viscosities) - 1:
        axs[i, 1].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 1].set_ylabel('Smoothed Wavelength (pixels)')
    else:
        axs[i, 1].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/wavelength/wavelengths_and_smoothed_wavelengths_boxplot_by_condition_and_viscosity_S_shape.png', dpi=300, bbox_inches='tight')
plt.close()



###### Box plot of normalized wavelengths by condition and viscosity ######

# Normalize wavelengths for S-shape data
s_shape_df['normalized_wavelengths'] = (s_shape_df['wavelengths'] - s_shape_df['worm_lengths']) / s_shape_df['worm_lengths']
s_shape_df['normalized_smoothed_wavelengths'] = (s_shape_df['smoothed_wavelengths'] - s_shape_df['worm_lengths']) / s_shape_df['worm_lengths']

# Create subplots for normalized wavelengths and normalized smoothed wavelengths
fig, axs = plt.subplots(len(viscosities), 2, figsize=(12, 5*len(viscosities)), sharey='col')
fig.suptitle('Normalized Wavelengths and Smoothed Wavelengths by Condition and Viscosity (S-shape)', fontsize=16)

# Calculate global y-limits for normalized wavelengths
y_min_normalized = s_shape_df['normalized_wavelengths'].min()
y_max_normalized = s_shape_df['normalized_wavelengths'].max()
y_range_normalized = y_max_normalized - y_min_normalized
y_buffer_normalized = y_range_normalized * 0.1
y_limits_normalized = (y_min_normalized - y_buffer_normalized, y_max_normalized + y_buffer_normalized)

# Calculate global y-limits for normalized smoothed wavelengths
y_min_normalized_smoothed = s_shape_df['normalized_smoothed_wavelengths'].min()
y_max_normalized_smoothed = s_shape_df['normalized_smoothed_wavelengths'].max()
y_range_normalized_smoothed = y_max_normalized_smoothed - y_min_normalized_smoothed
y_buffer_normalized_smoothed = y_range_normalized_smoothed * 0.1
y_limits_normalized_smoothed = (y_min_normalized_smoothed - y_buffer_normalized_smoothed, y_max_normalized_smoothed + y_buffer_normalized_smoothed)

for i, viscosity in enumerate(viscosities):
    viscosity_data = s_shape_df[s_shape_df['viscosity_x'] == viscosity]
    
    # Plot normalized wavelengths
    sns.boxplot(x='condition_x', y='normalized_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 0])
    
    axs[i, 0].set_title(f'Viscosity: {viscosity}')
    axs[i, 0].set_xlabel('')
    axs[i, 0].set_ylim(y_limits_normalized)
    
    if i == len(viscosities) - 1:
        axs[i, 0].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 0].set_ylabel('Normalized Wavelength')
    else:
        axs[i, 0].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), ha='right')
    
    # Plot normalized smoothed wavelengths
    sns.boxplot(x='condition_x', y='normalized_smoothed_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 1])
    
    axs[i, 1].set_title(f'Viscosity: {viscosity}')
    axs[i, 1].set_xlabel('')
    axs[i, 1].set_ylim(y_limits_normalized_smoothed)
    
    if i == len(viscosities) - 1:
        axs[i, 1].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 1].set_ylabel('Normalized Smoothed Wavelength')
    else:
        axs[i, 1].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/wavelength/normalized_wavelengths_and_smoothed_wavelengths_boxplot_by_condition_and_viscosity_S_shape.png', dpi=300, bbox_inches='tight')
plt.close()




###### Box plot of average wavelengths by condition and viscosity ######


# Calculate average values per video_id for S-shape data
avg_df = s_shape_df.groupby(['video_id', 'condition_x', 'viscosity_x']).agg({
    'wavelengths': 'mean',
    'smoothed_wavelengths': 'mean'
}).reset_index()

# Create subplots for average wavelengths and average smoothed wavelengths
fig, axs = plt.subplots(len(viscosities), 2, figsize=(12, 5*len(viscosities)), sharey='col')
fig.suptitle('Average Wavelengths and Smoothed Wavelengths by Condition and Viscosity (S-shape)', fontsize=16)

# Calculate global y-limits for average wavelengths
y_min_wavelengths = avg_df['wavelengths'].min()
y_max_wavelengths = avg_df['wavelengths'].max()
y_range_wavelengths = y_max_wavelengths - y_min_wavelengths
y_buffer_wavelengths = y_range_wavelengths * 0.1
y_limits_wavelengths = (y_min_wavelengths - y_buffer_wavelengths, y_max_wavelengths + y_buffer_wavelengths)

# Calculate global y-limits for average smoothed wavelengths
y_min_smoothed = avg_df['smoothed_wavelengths'].min()
y_max_smoothed = avg_df['smoothed_wavelengths'].max()
y_range_smoothed = y_max_smoothed - y_min_smoothed
y_buffer_smoothed = y_range_smoothed * 0.1
y_limits_smoothed = (y_min_smoothed - y_buffer_smoothed, y_max_smoothed + y_buffer_smoothed)

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_df[avg_df['viscosity_x'] == viscosity]
    
    # Plot average wavelengths
    sns.boxplot(x='condition_x', y='wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 0])
    
    axs[i, 0].set_title(f'Viscosity: {viscosity}')
    axs[i, 0].set_xlabel('')
    axs[i, 0].set_ylim(y_limits_wavelengths)
    
    if i == len(viscosities) - 1:
        axs[i, 0].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 0].set_ylabel('Average Wavelength (pixels)')
    else:
        axs[i, 0].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), ha='right')
    
    # Plot average smoothed wavelengths
    sns.boxplot(x='condition_x', y='smoothed_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 1])
    
    axs[i, 1].set_title(f'Viscosity: {viscosity}')
    axs[i, 1].set_xlabel('')
    axs[i, 1].set_ylim(y_limits_smoothed)
    
    if i == len(viscosities) - 1:
        axs[i, 1].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 1].set_ylabel('Average Smoothed Wavelength (pixels)')
    else:
        axs[i, 1].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/wavelength/average_wavelengths_and_smoothed_wavelengths_boxplot_by_condition_and_viscosity_S_shape.png', dpi=300, bbox_inches='tight')
plt.close()





###### Box plot of average normalized wavelengths by condition and viscosity ######

# Calculate average values per video_id for normalized wavelengths (S-shape)
avg_normalized_df = s_shape_df.groupby(['video_id', 'condition_x', 'viscosity_x']).agg({
    'normalized_wavelengths': 'mean',
    'normalized_smoothed_wavelengths': 'mean'
}).reset_index()

# Create subplots for average normalized wavelengths and average normalized smoothed wavelengths
fig, axs = plt.subplots(len(viscosities), 2, figsize=(12, 5*len(viscosities)), sharey='col')
fig.suptitle('Average Normalized Wavelengths and Smoothed Wavelengths by Condition and Viscosity (S-shape)', fontsize=16)

# Calculate global y-limits for average normalized wavelengths
y_min_avg_normalized = avg_normalized_df['normalized_wavelengths'].min()
y_max_avg_normalized = avg_normalized_df['normalized_wavelengths'].max()
y_range_avg_normalized = y_max_avg_normalized - y_min_avg_normalized
y_buffer_avg_normalized = y_range_avg_normalized * 0.1
y_limits_avg_normalized = (y_min_avg_normalized - y_buffer_avg_normalized, y_max_avg_normalized + y_buffer_avg_normalized)

# Calculate global y-limits for average normalized smoothed wavelengths
y_min_avg_normalized_smoothed = avg_normalized_df['normalized_smoothed_wavelengths'].min()
y_max_avg_normalized_smoothed = avg_normalized_df['normalized_smoothed_wavelengths'].max()
y_range_avg_normalized_smoothed = y_max_avg_normalized_smoothed - y_min_avg_normalized_smoothed
y_buffer_avg_normalized_smoothed = y_range_avg_normalized_smoothed * 0.1
y_limits_avg_normalized_smoothed = (y_min_avg_normalized_smoothed - y_buffer_avg_normalized_smoothed, y_max_avg_normalized_smoothed + y_buffer_avg_normalized_smoothed)

for i, viscosity in enumerate(viscosities):
    viscosity_data = avg_normalized_df[avg_normalized_df['viscosity_x'] == viscosity]
    
    # Plot average normalized wavelengths
    sns.boxplot(x='condition_x', y='normalized_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 0])
    
    axs[i, 0].set_title(f'Viscosity: {viscosity}')
    axs[i, 0].set_xlabel('')
    axs[i, 0].set_ylim(y_limits_avg_normalized)
    
    if i == len(viscosities) - 1:
        axs[i, 0].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 0].set_ylabel('Average Normalized Wavelength')
    else:
        axs[i, 0].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), ha='right')
    
    # Plot average normalized smoothed wavelengths
    sns.boxplot(x='condition_x', y='normalized_smoothed_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 1])
    
    axs[i, 1].set_title(f'Viscosity: {viscosity}')
    axs[i, 1].set_xlabel('')
    axs[i, 1].set_ylim(y_limits_avg_normalized_smoothed)
    
    if i == len(viscosities) - 1:
        axs[i, 1].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 1].set_ylabel('Average Normalized Smoothed Wavelength')
    else:
        axs[i, 1].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/wavelength/average_normalized_wavelengths_and_smoothed_wavelengths_boxplot_by_condition_and_viscosity_S_shape.png', dpi=300, bbox_inches='tight')
plt.close()



###### Box plot of maximum normalized wavelengths by condition and viscosity ######

# Calculate maximum values per video_id for normalized wavelengths (S-shape)
max_normalized_df = s_shape_df.groupby(['video_id', 'condition_x', 'viscosity_x']).agg({
    'normalized_wavelengths': 'max',
    'normalized_smoothed_wavelengths': 'max'
}).reset_index()

# Create subplots for maximum normalized wavelengths and maximum normalized smoothed wavelengths
fig, axs = plt.subplots(len(viscosities), 2, figsize=(12, 5*len(viscosities)), sharey='row')
fig.suptitle('Maximum Normalized Wavelengths and Smoothed Wavelengths by Condition and Viscosity (S-shape)', fontsize=16)

# Calculate global y-limits for both normalized and smoothed normalized wavelengths
y_min = min(max_normalized_df['normalized_wavelengths'].min(), max_normalized_df['normalized_smoothed_wavelengths'].min())
y_max = max(max_normalized_df['normalized_wavelengths'].max(), max_normalized_df['normalized_smoothed_wavelengths'].max())
y_range = y_max - y_min
y_buffer = y_range * 0.1
y_limits = (y_min - y_buffer, y_max + y_buffer)

for i, viscosity in enumerate(viscosities):
    viscosity_data = max_normalized_df[max_normalized_df['viscosity_x'] == viscosity]
    
    # Plot maximum normalized wavelengths
    sns.boxplot(x='condition_x', y='normalized_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 0])
    
    axs[i, 0].set_title(f'Viscosity: {viscosity}')
    axs[i, 0].set_xlabel('')
    axs[i, 0].set_ylim(y_limits)
    
    if i == len(viscosities) - 1:
        axs[i, 0].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 0].set_ylabel('Maximum Normalized Wavelength')
    else:
        axs[i, 0].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 0].set_xticklabels(axs[i, 0].get_xticklabels(), ha='right')
    
    # Plot maximum normalized smoothed wavelengths
    sns.boxplot(x='condition_x', y='normalized_smoothed_wavelengths', data=viscosity_data, 
                order=sorted_conditions, ax=axs[i, 1])
    
    axs[i, 1].set_title(f'Viscosity: {viscosity}')
    axs[i, 1].set_xlabel('')
    axs[i, 1].set_ylim(y_limits)
    
    if i == len(viscosities) - 1:
        axs[i, 1].set_xlabel('Condition')
    
    if i == 0:
        axs[i, 1].set_ylabel('Maximum Normalized Smoothed Wavelength')
    else:
        axs[i, 1].set_ylabel('')
    
    # Rotate x-axis labels for better readability
    axs[i, 1].set_xticklabels(axs[i, 1].get_xticklabels(), ha='right')

plt.tight_layout()
plt.savefig('C:/Users/aurel/Documents/GitHub/phd/dropletswimming/plots/wavelength/maximum_normalized_wavelengths_and_smoothed_wavelengths_boxplot_by_condition_and_viscosity_S_shape.png', dpi=300, bbox_inches='tight')
plt.close()




# endregion



