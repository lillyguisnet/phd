import os
import random
import shutil
import csv
import pandas as pd
import numpy as np # Import numpy for NaN handling if necessary
import matplotlib.pyplot as plt
import seaborn as sns # For improved aesthetics, optional



#region [1) Copy random images]
def copy_random_images(source_dir, dest_dir, num_images, csv_path):
    """
    Copies a specified number of random JPG images from a source directory
    (and its subdirectories) to a destination directory.
    Also records the folder name (worm) and frame name for each copied image
    into a CSV file.

    Args:
        source_dir (str): The path to the source directory.
        dest_dir (str): The path to the destination directory.
        num_images (int): The number of random images to copy.
        csv_path (str): The path to the CSV file for logging.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    # Prepare CSV file
    csv_file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, 'a', newline='') as csvfile: # Ensure 'a' for append mode
            csv_writer = csv.writer(csvfile)
            if not csv_file_exists:
                csv_writer.writerow(["worm", "frame"]) # Write header only if file is new
    except IOError as e:
        print(f"Error preparing CSV file {csv_path}: {e}")
        return # Stop if CSV cannot be opened

    jpg_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_files.append(os.path.join(root, file))

    if not jpg_files:
        print(f"No JPG files found in {source_dir}")
        return

    if len(jpg_files) < num_images:
        print(f"Warning: Found only {len(jpg_files)} JPG files, which is less than the requested {num_images}. Copying all found files.")
        selected_files = jpg_files
    else:
        selected_files = random.sample(jpg_files, num_images)

    copied_count = 0
    for file_path in selected_files:
        try:
            # Extract worm and frame information
            worm_name = os.path.basename(os.path.dirname(file_path))
            original_image_basename = os.path.basename(file_path)
            frame_name_for_csv = os.path.splitext(original_image_basename)[0]

            # Construct new destination file name to include the folder name
            new_dest_file_name = f"{worm_name}_{original_image_basename}"
            destination_file_path = os.path.join(dest_dir, new_dest_file_name)

            shutil.copy(file_path, destination_file_path)
            
            # Append to CSV
            with open(csv_path, 'a', newline='') as csvfile: # Ensure 'a' for append mode
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([worm_name, frame_name_for_csv])
            copied_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Successfully copied {copied_count} images to {dest_dir}")
    if copied_count > 0:
        print(f"Logged information for {copied_count} images to {csv_path}")


source_directory = "/home/lilly/phd/ria/data_foranalysis/AG_WT/videotojpg"
destination_directory = "/home/lilly/phd/ria/benchmarks/headangle/randomjpg"
csv_output_file = "/home/lilly/phd/ria/benchmarks/headangle/human_angle.csv"
number_of_images_to_copy = 50
copy_random_images(source_directory, destination_directory, number_of_images_to_copy, csv_output_file)
#endregion



#region [2) Fetch corresponding angles from fiji and sam]

def fetch_and_merge_angles(human_csv_path, fiji_sam_data_path):
    """
    Fetches angles from Fiji and SAM data sources and merges them with human-annotated angles,
    including worm orientation (side_position).

    Args:
        human_csv_path (str): Path to the CSV file with human angles (worm, frame, human_angle).
        fiji_sam_data_path (str): Path to the CSV file with Fiji and SAM processed data.

    Returns:
        pandas.DataFrame: DataFrame with columns ['worm', 'frame', 'human_angle', 'fiji_angle', 'sam_angle', 'side_position'].
    """
    # Load human data
    try:
        human_df = pd.read_csv(human_csv_path)
    except FileNotFoundError:
        print(f"Error: Human angles CSV file not found at {human_csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading human angles CSV {human_csv_path}: {e}")
        return pd.DataFrame()

    # Prepare human_df for merging
    # Rename original 'worm' column to keep it, create 'key_worm' for merging
    human_df.rename(columns={'worm': 'original_worm_name'}, inplace=True)
    # Assuming worm names in human_angle.csv might have a prefix like 'AG_WT-'
    # that needs to be removed to match the worm names in fiji_sam_data.csv
    human_df['key_worm'] = human_df['original_worm_name'].str.replace('AG_WT-', '', n=1, regex=False)
    try:
        human_df['frame'] = human_df['frame'].astype(int)
    except ValueError as e:
        print(f"Error converting 'frame' column in human data to integer: {e}")
        print("Please ensure 'frame' column in human_angle.csv contains valid numbers.")
        return pd.DataFrame()


    # Load Fiji and SAM data
    try:
        fiji_sam_df = pd.read_csv(fiji_sam_data_path)
        print(f"DEBUG: Columns in fiji_sam_df ({fiji_sam_data_path}): {fiji_sam_df.columns.tolist()}") # DEBUG
    except FileNotFoundError:
        print(f"Error: Fiji/SAM data CSV file not found at {fiji_sam_data_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading Fiji/SAM data CSV {fiji_sam_data_path}: {e}")
        return pd.DataFrame()

    try:
        fiji_sam_df['frame'] = fiji_sam_df['frame'].astype(int)
    except ValueError as e:
        print(f"Error converting 'frame' column in Fiji/SAM data to integer: {e}")
        print("Please ensure 'frame' column in merged_fiji_sam_normalized_data.csv contains valid numbers.")
        return pd.DataFrame()


    # Prepare Fiji angles
    fiji_angles_df = fiji_sam_df[fiji_sam_df['source'] == 'fiji'][['worm', 'frame', 'norm_angle']].copy()
    fiji_angles_df.rename(columns={'worm': 'key_worm', 'norm_angle': 'fiji_angle'}, inplace=True)

    # Merge human data with Fiji angles
    merged_df = pd.merge(human_df, fiji_angles_df, on=['key_worm', 'frame'], how='left')

    # Prepare SAM angles
    sam_angles_df = fiji_sam_df[fiji_sam_df['source'] == 'sam'][['worm', 'frame', 'angle_degrees_corrected']].copy()
    sam_angles_df.rename(columns={'worm': 'key_worm', 'angle_degrees_corrected': 'sam_angle'}, inplace=True)

    # Merge with SAM angles
    merged_df = pd.merge(merged_df, sam_angles_df, on=['key_worm', 'frame'], how='left')
    print(f"DEBUG: Columns in merged_df after SAM merge: {merged_df.columns.tolist()}") # DEBUG

    # Extract and merge side_position
    # Assuming side_position is consistent for a given worm/frame in fiji_sam_data_path
    # Taking unique worm, frame, side_position combinations
    if 'side_position' in fiji_sam_df.columns:
        print("DEBUG: 'side_position' found in fiji_sam_df.columns. Proceeding with merge.") # DEBUG
        side_positions_df = fiji_sam_df[['worm', 'frame', 'side_position']].copy()
        side_positions_df.rename(columns={'worm': 'key_worm'}, inplace=True)
        print(f"DEBUG: Columns in side_positions_df before dropna/drop_duplicates: {side_positions_df.columns.tolist()}") # DEBUG
        print(f"DEBUG: side_positions_df head before dropna:\n{side_positions_df.head()}") # DEBUG

        side_positions_df = side_positions_df.dropna(subset=['side_position'])
        print(f"DEBUG: side_positions_df head after dropna(subset=['side_position']):\n{side_positions_df.head()}") # DEBUG
        print(f"DEBUG: side_positions_df shape after dropna: {side_positions_df.shape}") # DEBUG

        side_positions_df = side_positions_df.drop_duplicates(subset=['key_worm', 'frame'], keep='first')
        print(f"DEBUG: Columns in side_positions_df before merging to merged_df: {side_positions_df.columns.tolist()}") # DEBUG
        print(f"DEBUG: side_positions_df head after drop_duplicates:\n{side_positions_df.head()}") # DEBUG
        print(f"DEBUG: side_positions_df shape after drop_duplicates: {side_positions_df.shape}") # DEBUG

        merged_df = pd.merge(merged_df, side_positions_df, on=['key_worm', 'frame'], how='left')
        print(f"DEBUG: Columns in merged_df after side_position merge attempt: {merged_df.columns.tolist()}") # DEBUG
    else:
        print(f"Warning: 'side_position' column not found in {fiji_sam_data_path}. Cannot adjust human angles for orientation.") # This should NOT print based on your error
        merged_df['side_position'] = np.nan # Add an empty column if not present
        print(f"DEBUG: Columns in merged_df after adding NaN 'side_position': {merged_df.columns.tolist()}") # DEBUG


    # Select and rename final columns
    final_columns_list = ['original_worm_name', 'frame', 'human_angle', 'fiji_angle', 'sam_angle', 'side_position']
    print(f"DEBUG: Desired final columns: {final_columns_list}") # DEBUG
    # Ensure all selected columns exist in merged_df before selection
    final_columns_present = [col for col in final_columns_list if col in merged_df.columns]
    print(f"DEBUG: Actual final columns to be selected (must be in merged_df.columns): {final_columns_present}") # DEBUG
    
    if not merged_df.empty:
        final_df = merged_df[final_columns_present]
        final_df = final_df.rename(columns={'original_worm_name': 'worm'})
    else: # Handle empty merged_df case
        print("DEBUG: merged_df is empty before final selection. Creating empty final_df.") #DEBUG
        final_df = pd.DataFrame(columns=[col.replace('original_worm_name', 'worm') for col in final_columns_present])


    return final_df

# Paths for the data files
# csv_output_file is already defined in your script for human angles
# human_angles_csv = csv_output_file (assuming it's in scope or you pass it)
fiji_sam_angles_csv = "/home/lilly/phd/merged_fiji_sam_normalized_data.csv"

# Ensure human_angles_csv path is correctly referenced
# If this script part is run after the first region, csv_output_file will be defined.
# Otherwise, you might need to define it explicitly here if running this part standalone.
if 'csv_output_file' in globals():
    human_angles_csv = csv_output_file
else:
    # Fallback or error if csv_output_file is not defined (e.g. if running this section alone)
    human_angles_csv = "/home/lilly/phd/ria/benchmarks/headangle/human_angle.csv"
    print(f"Warning: 'csv_output_file' not found in global scope, using default: {human_angles_csv}")


# Fetch and merge the angles
comparison_df = fetch_and_merge_angles(human_angles_csv, fiji_sam_angles_csv)

if not comparison_df.empty:
    print("\nComparison DataFrame (first 5 rows):")
    print(comparison_df.head())

    # You can save this DataFrame to a new CSV if needed:
    # output_comparison_csv = "/home/lilly/phd/ria/benchmarks/headangle/comparison_angles.csv"
    # comparison_df.to_csv(output_comparison_csv, index=False)
    # print(f"\nComparison data saved to {output_comparison_csv}")
else:
    print("\nFailed to generate comparison DataFrame.")




#endregion


#region [3) Make angles comparable]

def normalize_angles(df):
    """
    Normalizes 'human_angle', 'fiji_angle', and 'sam_angle' in the DataFrame.
    - Human angle: adjusted to 'human_angle_normalized'. If 'side_position' is 'left',
                   angle is flipped (180 - angle). Otherwise, it's copied. Range [0, 180].
    - Fiji angle: 'fiji_angle' from [-1, 1] to 'fiji_angle_normalized' [0, 180].
    - SAM angle: 'sam_angle' from [-90, 90] to 'sam_angle_normalized' [0, 180].

    Args:
        df (pd.DataFrame): DataFrame with 'human_angle', 'fiji_angle', 'sam_angle',
                           and 'side_position' columns.

    Returns:
        pd.DataFrame: DataFrame with normalized angles in new columns
                      ('human_angle_normalized', 'fiji_angle_normalized', 'sam_angle_normalized').
    """
    df_copy = df.copy()

    # Normalize human_angle based on side_position
    if 'human_angle' in df_copy.columns and 'side_position' in df_copy.columns:
        df_copy['human_angle_normalized'] = df_copy.apply(
            lambda row: 180 - row['human_angle'] if pd.notnull(row['human_angle']) and row['side_position'] == 'left' else row['human_angle'],
            axis=1
        )
    elif 'human_angle' in df_copy.columns:
        print("Warning: 'side_position' column not found or 'human_angle' missing. Human angles will not be adjusted for orientation, copying 'human_angle' to 'human_angle_normalized'.")
        df_copy['human_angle_normalized'] = df_copy['human_angle']
    else:
        print("Warning: 'human_angle' column not found. Cannot create 'human_angle_normalized'.")
        df_copy['human_angle_normalized'] = np.nan


    # Normalize Fiji angle
    # Formula: new_angle = old_angle * 90 + 90
    if 'fiji_angle' in df_copy.columns:
        df_copy['fiji_angle_normalized'] = df_copy['fiji_angle'].apply(lambda x: x * 90 + 90 if pd.notnull(x) else np.nan)
    else:
        print("Warning: 'fiji_angle' column not found. Cannot create 'fiji_angle_normalized'.")
        df_copy['fiji_angle_normalized'] = np.nan

    # Normalize SAM angle
    # Formula: new_angle = old_angle + 90
    if 'sam_angle' in df_copy.columns:
        df_copy['sam_angle_normalized'] = df_copy['sam_angle'].apply(lambda x: x + 90 if pd.notnull(x) else np.nan)
    else:
        print("Warning: 'sam_angle' column not found. Cannot create 'sam_angle_normalized'.")
        df_copy['sam_angle_normalized'] = np.nan
    
    return df_copy

if not comparison_df.empty:
    print("\nComparison DataFrame (first 5 rows) before normalization:")
    print(comparison_df.head())

    comparison_df_normalized = normalize_angles(comparison_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

    print("\nComparison DataFrame (first 5 rows) after normalization:")
    print(comparison_df_normalized.head())

    # You can save this DataFrame to a new CSV if needed:
    # output_comparison_csv = "/home/lilly/phd/ria/benchmarks/headangle/comparison_angles_normalized.csv"
    # comparison_df_normalized.to_csv(output_comparison_csv, index=False)
    # print(f"\nNormalized comparison data saved to {output_comparison_csv}")
else:
    print("\nFailed to generate comparison DataFrame, skipping normalization.")

comparison_df_normalized.describe()
comparison_df_normalized.to_csv("/home/lilly/phd/ria/benchmarks/headangle/comparison_angles_normalized.csv", index=False)


#endregion


#region [4) Plot the angles]

import matplotlib.pyplot as plt
import seaborn as sns # For improved aesthetics, optional
import pandas as pd
import numpy as np

def plot_angles_comparison(df):
    """
    Plots human, Fiji (normalized), and SAM (normalized) angles for comparison.

    Args:
        df (pd.DataFrame): DataFrame containing 'worm', 'frame', 'human_angle',
                           'fiji_angle_normalized', and 'sam_angle_normalized'.
    """
    if df.empty:
        print("DataFrame is empty. Cannot generate plot.")
        return

    # Prepare data for plotting
    plot_df = df.copy()
    plot_df['worm_frame'] = plot_df['worm'] + "_" + plot_df['frame'].astype(str)

    # Sort by human_angle_normalized (previously was human_angle)
    if 'human_angle_normalized' in plot_df.columns:
        plot_df = plot_df.sort_values(by='human_angle_normalized')
    elif 'human_angle' in plot_df.columns: # Fallback if normalized not present
        plot_df = plot_df.sort_values(by='human_angle')


    # Convert pandas Series to NumPy arrays for plotting
    x_values = plot_df['worm_frame'].to_numpy()
    # Use human_angle_normalized for plotting
    human_angles = plot_df['human_angle_normalized'].to_numpy() if 'human_angle_normalized' in plot_df.columns else plot_df['human_angle'].to_numpy()
    fiji_angles = plot_df['fiji_angle_normalized'].to_numpy()
    sam_angles = plot_df['sam_angle_normalized'].to_numpy()

    # Create the plot
    plt.figure(figsize=(15, 7)) # Adjust figure size as needed

    # Plotting each angle series (markers only)
    # Update label for human angle
    plt.plot(x_values, human_angles, label='Human Angle (Orient. Corrected)', marker='o', linestyle='None', zorder=3)
    plt.plot(x_values, fiji_angles, label='Fiji Angle (Normalized)', marker='x', linestyle='None', zorder=3)
    plt.plot(x_values, sam_angles, label='SAM Angle (Normalized)', marker='s', linestyle='None', zorder=3)

    # Add vertical lines for comparison at each x-point
    # x_indices will be 0, 1, 2, ... corresponding to x_values
    x_indices = range(len(x_values))
    for i in x_indices:
        angles_at_point = []
        if pd.notnull(human_angles[i]): # human_angles is now from human_angle_normalized
            angles_at_point.append(human_angles[i])
        if pd.notnull(fiji_angles[i]):
            angles_at_point.append(fiji_angles[i])
        if pd.notnull(sam_angles[i]):
            angles_at_point.append(sam_angles[i])
        
        if len(angles_at_point) > 1: # Only draw line if there are at least two points to connect
            min_angle = min(angles_at_point)
            max_angle = max(angles_at_point)
            plt.vlines(x=i, ymin=min_angle, ymax=max_angle, colors='grey', linestyles='dotted', alpha=0.7, zorder=1)


    # Customize the plot
    plt.xlabel("Worm_Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Comparison of Human, Fiji, and SAM Head Angles")
    plt.xticks(rotation=90) # Rotate x-axis labels for readability
    plt.legend()
    # Remove general grid
    # plt.grid(True, linestyle='--', alpha=0.7) 
    # Add a specific horizontal line at 90 degrees
    plt.axhline(y=90, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, zorder=0)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    # Show the plot
    plt.show()
    plt.savefig("/home/lilly/phd/ria/benchmarks/headangle/comparison_angles_plot.png")
    plt.close()

# Assuming comparison_df_normalized is your final DataFrame from the previous step
if 'comparison_df_normalized' in globals() and not comparison_df_normalized.empty:
    plot_angles_comparison(comparison_df_normalized)
else:
    print("\n'comparison_df_normalized' not found or is empty. Attempting to load from CSV for plotting.")
    try:
        fallback_df = pd.read_csv("/home/lilly/phd/ria/benchmarks/headangle/comparison_angles_normalized.csv")
        if not fallback_df.empty:
           print("\nUsing fallback data (loaded from CSV) for plotting.")
           # Ensure 'human_angle' exists for sorting, and other necessary columns are present
           if 'human_angle' in fallback_df.columns and \
              'worm' in fallback_df.columns and \
              'frame' in fallback_df.columns and \
              'fiji_angle_normalized' in fallback_df.columns and \
              'sam_angle_normalized' in fallback_df.columns:
               plot_angles_comparison(fallback_df)
           else:
               print("\nFallback DataFrame is missing required columns for plotting.")
        else:
           print("\nFallback data from CSV is empty.")
    except FileNotFoundError:
        print("\nFallback CSV '/home/lilly/phd/ria/benchmarks/headangle/comparison_angles_normalized.csv' not found.")
    except Exception as e:
        print(f"\nError loading or processing fallback CSV for plotting: {e}")


#endregion


#region [5) Calculate some metrics]

## Calculate the difference between human_angle_normalized and fiji_angle_normalized/sam_angle_normalized
def calculate_angle_differences(df_input):
    """
    Calculates the difference between human_angle_normalized and
    fiji_angle_normalized/sam_angle_normalized.
    Assumes 'human_angle_normalized', 'fiji_angle_normalized', 
    and 'sam_angle_normalized' columns exist in the input DataFrame.

    Args:
        df_input (pd.DataFrame): DataFrame with 'human_angle_normalized',
                                 'fiji_angle_normalized', and 'sam_angle_normalized'.

    Returns:
        pd.DataFrame: A new DataFrame with additional columns 'diff_human_fiji' 
                      and 'diff_human_sam'. The original DataFrame is not modified.
    """
    df_with_diff = df_input.copy() # Work on a copy to not modify the original DataFrame

    # Direct calculation assuming columns exist
    df_with_diff['diff_human_fiji'] = df_with_diff['fiji_angle_normalized'] - df_with_diff['human_angle_normalized']
    df_with_diff['diff_human_sam'] = df_with_diff['sam_angle_normalized'] - df_with_diff['human_angle_normalized']
        
    return df_with_diff

# Assuming 'comparison_df_normalized' is available from previous steps.
# Per user instruction, complex fallbacks are removed. If 'comparison_df_normalized'
# is not suitable (e.g., missing, empty, or lacks required columns),
# this section will raise an error.

df_angle_differences = calculate_angle_differences(comparison_df_normalized)

print("\nDataFrame with angle differences (first 5 rows):")
print(df_angle_differences.head())


output_diff_csv = "/home/lilly/phd/ria/benchmarks/headangle/df_angle_differences.csv"
df_angle_differences.to_csv(output_diff_csv, index=False)

df_angle_differences.describe().to_csv("/home/lilly/phd/ria/benchmarks/headangle/df_angle_differences_describe.csv", index=False)


# Calculate and display mean absolute differences
if 'diff_human_fiji' in df_angle_differences.columns and 'diff_human_sam' in df_angle_differences.columns:
    abs_diff_fiji = df_angle_differences['diff_human_fiji'].abs()
    abs_diff_sam = df_angle_differences['diff_human_sam'].abs()

    abs_mean_diff_fiji = abs_diff_fiji.mean()
    abs_mean_diff_sam = abs_diff_sam.mean()
    
    abs_median_diff_fiji = abs_diff_fiji.median()
    abs_median_diff_sam = abs_diff_sam.median()

    abs_min_diff_fiji = abs_diff_fiji.min()
    abs_min_diff_sam = abs_diff_sam.min()

    abs_max_diff_fiji = abs_diff_fiji.max()
    abs_max_diff_sam = abs_diff_sam.max()

    abs_std_diff_fiji = abs_diff_fiji.std()
    abs_std_diff_sam = abs_diff_sam.std()

    print("\n--- Absolute Difference Statistics (Human vs Fiji) ---")
    print(f"Mean Absolute Difference: {abs_mean_diff_fiji:.2f} degrees")
    print(f"Median Absolute Difference: {abs_median_diff_fiji:.2f} degrees")
    print(f"Min Absolute Difference: {abs_min_diff_fiji:.2f} degrees")
    print(f"Max Absolute Difference: {abs_max_diff_fiji:.2f} degrees")
    print(f"STD of Absolute Difference: {abs_std_diff_fiji:.2f} degrees")

    print("\n--- Absolute Difference Statistics (Human vs SAM) ---")
    print(f"Mean Absolute Difference: {abs_mean_diff_sam:.2f} degrees")
    print(f"Median Absolute Difference: {abs_median_diff_sam:.2f} degrees")
    print(f"Min Absolute Difference: {abs_min_diff_sam:.2f} degrees")
    print(f"Max Absolute Difference: {abs_max_diff_sam:.2f} degrees")
    print(f"STD of Absolute Difference: {abs_std_diff_sam:.2f} degrees")

    # Save these metrics to a separate file
    abs_diff_summary = pd.Series({
        'abs_mean_diff_human_fiji': abs_mean_diff_fiji,
        'abs_median_diff_human_fiji': abs_median_diff_fiji,
        'abs_min_diff_human_fiji': abs_min_diff_fiji,
        'abs_max_diff_human_fiji': abs_max_diff_fiji,
        'abs_std_diff_human_fiji': abs_std_diff_fiji,
        'abs_mean_diff_human_sam': abs_mean_diff_sam,
        'abs_median_diff_human_sam': abs_median_diff_sam,
        'abs_min_diff_human_sam': abs_min_diff_sam,
        'abs_max_diff_human_sam': abs_max_diff_sam,
        'abs_std_diff_human_sam': abs_std_diff_sam
    })
    summary_output_path = "/home/lilly/phd/ria/benchmarks/headangle/df_angle_differences_abs_stats.csv" # Renamed for clarity
    try:
        abs_diff_summary.to_csv(summary_output_path, header=['value'], index_label='metric')
        print(f"\nAbsolute difference statistics summary saved to {summary_output_path}")
    except Exception as e:
        print(f"\nError saving absolute difference statistics summary: {e}")
else:
    print("\nWarning: Difference columns ('diff_human_fiji', 'diff_human_sam') not found. Cannot calculate absolute difference statistics.")



##Categorize angle bins
#Make the bins
def add_angle_bins(df_input):
    """
    Categorizes angles into 4 bins and 6 bins.

    Args:
        df_input (pd.DataFrame): DataFrame with 'human_angle_normalized',
                                 'fiji_angle_normalized', and 'sam_angle_normalized'.

    Returns:
        pd.DataFrame: DataFrame with new columns for binned angles.
    """
    df_with_bins = df_input.copy()

    angle_columns = {
        'human': 'human_angle_normalized',
        'fiji': 'fiji_angle_normalized',
        'sam': 'sam_angle_normalized'
    }

    # Define bins and labels
    bins_4 = [0, 45, 90, 135, 180]
    labels_4 = [1, 2, 3, 4]

    bins_6 = [0, 30, 60, 90, 120, 150, 180]
    labels_6 = [1, 2, 3, 4, 5, 6]

    for prefix, col_name in angle_columns.items():
        if col_name in df_with_bins.columns:
            # 4 bins
            df_with_bins[f'{prefix}_angle_bin4'] = pd.cut(
                df_with_bins[col_name],
                bins=bins_4,
                labels=labels_4,
                right=True,        # (0, 45], (45, 90] ... (135, 180]
                include_lowest=True # Makes the first interval [0, 45]
            )
            
            # 6 bins
            df_with_bins[f'{prefix}_angle_bin6'] = pd.cut(
                df_with_bins[col_name],
                bins=bins_6,
                labels=labels_6,
                right=True,
                include_lowest=True
            )
        else:
            print(f"Warning: Column '{col_name}' not found. Skipping binning for {prefix}.")
            df_with_bins[f'{prefix}_angle_bin4'] = np.nan
            df_with_bins[f'{prefix}_angle_bin6'] = np.nan
            
    return df_with_bins

# Assuming 'df_angle_differences' is available and contains the normalized angle columns.
# If not, this will print warnings from within the function.
df_binned_angles = add_angle_bins(df_angle_differences)

print("\nDataFrame with binned angles (first 5 rows):")
print(df_binned_angles.head())

# Save the DataFrame with binned angles
output_binned_csv = "/home/lilly/phd/ria/benchmarks/headangle/df_binned_angles.csv"
df_binned_angles.to_csv(output_binned_csv, index=False)
print(f"\nDataFrame with binned angles saved to {output_binned_csv}")

df_binned_angles.describe(include='all').to_csv("/home/lilly/phd/ria/benchmarks/headangle/df_binned_angles_describe.csv", index=True)

#Output the binned results
def generate_and_save_binned_angle_counts(df, output_csv_path):
    """
    Prints the count of values in each bin for human, Fiji, and SAM angles,
    collects these counts into a DataFrame, and saves it to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with binned angle columns 
                           (e.g., 'human_angle_bin4', 'fiji_angle_bin6').
        output_csv_path (str): Path to save the binned counts CSV.
    """
    sources = ['human', 'fiji', 'sam']
    bin_types = [4, 6]
    all_counts_data = []

    print("\n--- Binned Angle Counts ---")

    for source in sources:
        for bins in bin_types:
            col_name = f'{source}_angle_bin{bins}'
            if col_name in df.columns:
                # Ensure the column is treated as categorical to get all defined bins
                # even if some have zero counts, especially after pd.cut
                # However, value_counts on a categorical series with include_lowest=True
                # and right=True from pd.cut should list all bins that could possibly have values.
                # If a bin has 0 actual occurrences among non-NaN original values, 
                # it will be listed by value_counts().sort_index() on a pd.Categorical.
                
                # Convert to string to handle potential CategoricalDtype issues with NaN if any
                # and to ensure .cat.categories can be accessed if we wanted to ensure all bins are present.
                # For now, value_counts().sort_index() should be sufficient for bins that have data.
                
                counts = df[col_name].value_counts().sort_index()
                
                print(f"\nCounts for {source.capitalize()} - {bins} Bins (Column: {col_name}):")
                if not counts.empty:
                    for bin_label, count_val in counts.items():
                        print(f"  Bin {bin_label}: {count_val}")
                        all_counts_data.append({
                            'source': source,
                            'bin_type': f'{bins}_bins',
                            'bin_label': bin_label,
                            'count': count_val
                        })
                else:
                    print("  No data or all values are NaN in this column.")
                    # If we want to represent all possible bins even with 0 counts,
                    # we would need to iterate through expected labels_4/labels_6 here.
                    # For now, only bins with counts or resulting from pd.cut are added.
            else:
                print(f"\nWarning: Column '{col_name}' not found. Cannot count bins for {source.capitalize()} - {bins} Bins.")

    if all_counts_data:
        counts_df = pd.DataFrame(all_counts_data)
        try:
            counts_df.to_csv(output_csv_path, index=False)
            print(f"\nBinned angle counts saved to {output_csv_path}")
        except Exception as e:
            print(f"\nError saving binned angle counts to CSV: {e}")
    else:
        print("\nNo binned angle counts data to save.")


# Call the function with the DataFrame containing binned angles
binned_counts_csv_path = "/home/lilly/phd/ria/benchmarks/headangle/binned_angle_counts.csv"
if 'df_binned_angles' in globals() and not df_binned_angles.empty:
    generate_and_save_binned_angle_counts(df_binned_angles, binned_counts_csv_path)
else:
    print("\n'df_binned_angles' not found or is empty. Skipping binned angle counts generation and saving.")


#endregion



