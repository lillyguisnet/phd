import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the directories
final_data_dir = "/home/lilly/phd/ria/data_analyzed/AG_WT/final_data_fixcropfolder"
fiji_analysis_dir = "/home/lilly/phd/ria/data_analyzed/AG_WT/fiji_analysis"

#region [1) check column names]

# Get a list of files in each directory
try:
    final_data_files = [f for f in os.listdir(final_data_dir) if f.endswith('.csv')]
    fiji_analysis_files = [f for f in os.listdir(fiji_analysis_dir) if f.endswith('.xlsx')]
except FileNotFoundError as e:
    print(f"Error: Directory not found - {e}")
    exit()

# Check if there are files in the directories
if not final_data_files:
    print(f"No CSV files found in {final_data_dir}")
else:
    # Construct the full path to the first CSV file
    first_csv_file_path = os.path.join(final_data_dir, final_data_files[0])
    try:
        # Read the CSV file
        csv_df = pd.read_csv(first_csv_file_path)
        print(f"Column names in {final_data_files[0]}:")
        print(csv_df.columns.tolist())
    except Exception as e:
        print(f"Error reading {first_csv_file_path}: {e}")

print("-" * 30) # Separator

if not fiji_analysis_files:
    print(f"No XLSX files found in {fiji_analysis_dir}")
else:
    # Construct the full path to the first XLSX file
    first_xlsx_file_path = os.path.join(fiji_analysis_dir, fiji_analysis_files[0])
    try:
        # Read the XLSX file
        xlsx_df = pd.read_excel(first_xlsx_file_path)
        print(f"Column names in {fiji_analysis_files[0]}:")
        print(xlsx_df.columns.tolist())
    except Exception as e:
        print(f"Error reading {first_xlsx_file_path}: {e}")



#['frame', '2', '2_bg_corrected', '2_pixel_count', '3', '3_bg_corrected', '3_pixel_count', '4', '4_bg_corrected', '4_pixel_count', 'background', 'side_position', 'angle_degrees_corrected', 'object_id', 'angle_degrees', 'bend_location', 'bend_magnitude', 'bend_position_y', 'bend_position_x', 'head_mag', 'body_mag', 'is_noise_peak', 'peak_deviation', 'window_size_used', 'error', 'is_straight', 'has_warning']

#['Worm', 'Ventral Side', 'Frame', 'Background', '--', 'X', 'Y', 'XM', 'YM', 'Major', 'Minor', 'Angle', '--.1', 'nrD', 'nrV', 'loop']

#endregion [check column names]


#region [2) merge fiji files and prepare for analysis]

##Merge fiji files

# Columns to keep and their new names
columns_to_keep = {
    'Worm': 'worm',
    'Ventral Side': 'ventral_side',
    'Frame': 'frame',
    'Background': 'background',
    'Angle': 'angle',
    'nrD': 'nrd',
    'nrV': 'nrv',
    'loop': 'loop'
}

all_fiji_data = []

if not fiji_analysis_files:
    print(f"No XLSX files found in {fiji_analysis_dir} to process.")
else:
    for file_name in fiji_analysis_files:
        file_path = os.path.join(fiji_analysis_dir, file_name)
        try:
            df = pd.read_excel(file_path)

            # Fill NaN values for 'Worm' and 'Ventral Side' from the first row
            if not df.empty:
                if 'Worm' in df.columns:
                    first_worm_value = df['Worm'].iloc[0]
                    df['Worm'] = df['Worm'].fillna(first_worm_value)
                if 'Ventral Side' in df.columns:
                    first_ventral_side_value = df['Ventral Side'].iloc[0]
                    df['Ventral Side'] = df['Ventral Side'].fillna(first_ventral_side_value)

            # Check row count
            expected_rows = 613
            if len(df) != expected_rows:
                print(f"Warning: File {file_name} has {len(df)} rows, expected {expected_rows}.")

            # Select and rename columns
            # Ensure all keys in columns_to_keep are present in df before selection
            current_columns_to_select = [col for col in columns_to_keep.keys() if col in df.columns]
            if len(current_columns_to_select) != len(columns_to_keep.keys()):
                missing_cols = set(columns_to_keep.keys()) - set(df.columns)
                print(f"Warning: File {file_name} is missing columns: {missing_cols}. Skipping these columns.")
            
            df = df[current_columns_to_select]
            df = df.rename(columns=columns_to_keep)
            
            all_fiji_data.append(df)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    if all_fiji_data:
        merged_fiji_df = pd.concat(all_fiji_data, ignore_index=True)
        print("\nSuccessfully merged Fiji data.")
        print("First 5 rows of the merged data:")
        print(merged_fiji_df.head())
        print("\nInfo of the merged data:")
        merged_fiji_df.info()

        # Print summary statistics for specified columns
        cols_for_stats = ['background', 'nrd', 'nrv', 'loop', 'angle']
        # Check if all columns exist before describing
        existing_cols_for_stats = [col for col in cols_for_stats if col in merged_fiji_df.columns]
        
        if existing_cols_for_stats:
            print(f"\nSummary statistics for columns: {existing_cols_for_stats}")
            print(merged_fiji_df[existing_cols_for_stats].describe())
        else:
            print("\nNone of the specified columns for statistics found in the merged DataFrame.")
            
    else:
        print("No data was processed from Fiji files.")



##Prepare for analysis

#Normalize and flip head angle
if 'merged_fiji_df' in locals() and not merged_fiji_df.empty:
    if 'worm' in merged_fiji_df.columns and 'angle' in merged_fiji_df.columns:
        # Define the normalization function
        def normalize_angle(series):
            min_val = series.min()
            max_val = series.max()
            if min_val == max_val: # Avoid division by zero if all values are the same
                return pd.Series([0] * len(series), index=series.index) # Or handle as appropriate
            # Normalize to [0, 1] first
            normalized_0_1 = (series - min_val) / (max_val - min_val)
            # Then scale to [-1, 1]
            normalized_neg1_1 = normalized_0_1 * 2 - 1
            return normalized_neg1_1

        # Apply the normalization grouped by 'worm'
        merged_fiji_df['norm_angle'] = merged_fiji_df.groupby('worm')['angle'].transform(normalize_angle)

        print("\nAdded 'norm_angle' column.")
        # print("First 5 rows with 'norm_angle':") # Moved for clarity
        # print(merged_fiji_df[['worm', 'angle', 'norm_angle']].head())

        # Adjust sign of norm_angle based on ventral_side
        if 'ventral_side' in merged_fiji_df.columns:
            # Ensure ventral_side is numeric if it's not already (e.g., if read as object)
            merged_fiji_df['ventral_side'] = pd.to_numeric(merged_fiji_df['ventral_side'], errors='coerce')
            
            # Apply sign adjustment.
            # The previous logic (flipping when ventral_side == 1) was found to be inverted,
            # leading to anti-correlation where correlation was expected.
            # New logic: Flip sign when ventral_side is -1.
            # This assumes that ventral_side == 1 means the angle is already oriented correctly after normalization,
            # and ventral_side == -1 means it needs to be flipped.
            merged_fiji_df.loc[merged_fiji_df['ventral_side'] == -1, 'norm_angle'] *= -1
            
            print("\nAdjusted 'norm_angle' sign based on 'ventral_side' (corrected logic).")
            print("First 5 rows with 'worm', 'angle', 'ventral_side', and adjusted 'norm_angle':")
            print(merged_fiji_df[['worm', 'angle', 'ventral_side', 'norm_angle']].head())
            
            print("\nSummary statistics for 'norm_angle' after sign adjustment:")
            print(merged_fiji_df['norm_angle'].describe())
        else:
            print("\n'ventral_side' column not found. Cannot adjust 'norm_angle' sign.")
            print("First 5 rows with 'norm_angle' (sign not adjusted):") # Print the unadjusted if ventral_side is missing
            print(merged_fiji_df[['worm', 'angle', 'norm_angle']].head())
            print("\nSummary statistics for 'norm_angle' (sign not adjusted):")
            print(merged_fiji_df['norm_angle'].describe())
    else:
        print("\n'worm' or 'angle' column not found in merged_fiji_df. Cannot create 'norm_angle'.")
else:
    print("\nmerged_fiji_df does not exist or is empty. Cannot create 'norm_angle'.")


#Normalize compartment values
if 'merged_fiji_df' in locals() and not merged_fiji_df.empty:
    # Create a copy to avoid modifying the original merged_fiji_df directly for these new normalizations
    fiji_df_normalized = merged_fiji_df.copy()
    print("\nCreated a copy of merged_fiji_df named fiji_df_normalized.")

    # Define the 0-1 normalization function
    def normalize_0_1(series):
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val: # Avoid division by zero if all values are the same
            return pd.Series([0] * len(series), index=series.index) # Or assign a default, e.g., 0.5 or NaN
        return (series - min_val) / (max_val - min_val)

    columns_to_normalize = ['nrd', 'nrv', 'loop']
    new_column_names = {
        'nrd': 'nrd_norm',
        'nrv': 'nrv_norm',
        'loop': 'loop_norm'
    }

    if 'worm' in fiji_df_normalized.columns:
        for col in columns_to_normalize:
            if col in fiji_df_normalized.columns:
                new_col_name = new_column_names[col]
                fiji_df_normalized[new_col_name] = fiji_df_normalized.groupby('worm')[col].transform(normalize_0_1)
                print(f"Added '{new_col_name}' column to fiji_df_normalized.")
            else:
                print(f"Warning: Column '{col}' not found in fiji_df_normalized. Cannot normalize.")
        
        print("\nFirst 5 rows of fiji_df_normalized with new normalized columns:")
        columns_to_show = ['worm'] + columns_to_normalize + list(new_column_names.values())
        # Ensure only existing columns are selected for printing
        existing_columns_to_show = [c for c in columns_to_show if c in fiji_df_normalized.columns]
        print(fiji_df_normalized[existing_columns_to_show].head())

        print("\nSummary statistics for new normalized columns in fiji_df_normalized:")
        # Ensure only existing new columns are described
        existing_new_cols = [nc for nc in new_column_names.values() if nc in fiji_df_normalized.columns]
        if existing_new_cols:
            print(fiji_df_normalized[existing_new_cols].describe())
        else:
            print("No new normalized columns were created to describe.")

    else:
        print("\n'worm' column not found in fiji_df_normalized. Cannot perform grouped normalization.")
else:
    print("\nmerged_fiji_df does not exist or is empty. Cannot normalize compartment values.")


#Save df
fiji_df_normalized.to_csv('fiji_df_normalized.csv', index=False)

#endregion [merge fiji files and prepare for analysis]


#region [3) merge sam files and prepare for analysis]

##Merge sam files
try:
    final_data_files = [f for f in os.listdir(final_data_dir) if f.endswith('.csv')]
except FileNotFoundError:
    print(f"Error: Directory not found - {final_data_dir}")
    final_data_files = [] # Ensure final_data_files is defined

all_sam_data = []

if not final_data_files:
    print(f"No CSV files found in {final_data_dir} to process.")
else:
    print(f"Found {len(final_data_files)} CSV files to merge from {final_data_dir}.")
    for file_name in final_data_files:
        file_path = os.path.join(final_data_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            
            # Attempt to extract a worm_id from the filename
            # Assumes filename format like "wormID_details.csv" or "wormID.csv"
            worm_id = file_name.split('.')[0] 
            df['worm_id'] = worm_id
            
            all_sam_data.append(df)
            print(f"Successfully processed and added data from {file_name} with worm_id: {worm_id}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    if all_sam_data:
        merged_sam_df = pd.concat(all_sam_data, ignore_index=True)
        print("\nSuccessfully merged SAM data.")
        print(f"Total rows in merged_sam_df: {len(merged_sam_df)}")
        print("First 5 rows of the merged SAM data:")
        print(merged_sam_df.head())
        print("\nInfo of the merged SAM data:")
        merged_sam_df.info()
        
        # Optional: Save the merged SAM data
        # merged_sam_df.to_csv('merged_sam_data.csv', index=False)
        # print("\nMerged SAM data saved to merged_sam_data.csv")
    else:
        print("No data was processed from SAM (final_data_dir) files.")


##Summary statistics for background and corrected values
print("\nSummary statistics for background and corrected values:")
print(merged_sam_df[['background', '2_bg_corrected', '3_bg_corrected', '4_bg_corrected']].describe())


##Check distributions

if 'merged_sam_df' in locals() and not merged_sam_df.empty:
    # Define the columns for the histograms
    columns_to_plot_bg_corrected = ["background", "2_bg_corrected", "3_bg_corrected", "4_bg_corrected"]
    columns_to_plot_raw = ["background", "2", "3", "4"]
    
    if 'worm_id' not in merged_sam_df.columns:
        print("Error: 'worm_id' column not found in merged_sam_df. Cannot create plots.")
    else:
        unique_worm_ids_list = merged_sam_df['worm_id'].unique()
        palette = sns.color_palette(n_colors=len(unique_worm_ids_list))
        worm_color_map = {worm_id: color for worm_id, color in zip(unique_worm_ids_list, palette)}

        # --- Calculate shared X-axis limits for corresponding plots ---
        shared_x_limits = []
        for i in range(len(columns_to_plot_bg_corrected)): # Assumes 4 plots per figure
            col_name_fig1 = columns_to_plot_bg_corrected[i]
            col_name_fig2 = columns_to_plot_raw[i]
            
            series_to_combine = []
            
            if col_name_fig1 in merged_sam_df.columns:
                s1 = merged_sam_df[col_name_fig1].dropna()
                if not s1.empty:
                    series_to_combine.append(s1)
            
            if col_name_fig2 in merged_sam_df.columns:
                s2 = merged_sam_df[col_name_fig2].dropna()
                if not s2.empty:
                    series_to_combine.append(s2)

            if not series_to_combine:
                shared_x_limits.append((None, None))
                continue

            # Concatenate all available series for the pair and drop duplicates
            # This handles cases where col_name_fig1 == col_name_fig2 (e.g., "background")
            # or where they are different but represent corresponding measures.
            combined_values_for_pair = pd.concat(series_to_combine, ignore_index=True).drop_duplicates()

            if combined_values_for_pair.empty:
                shared_x_limits.append((None, None))
                continue
                
            min_x = combined_values_for_pair.min()
            max_x = combined_values_for_pair.max()
            
            if min_x == max_x: # If all unique values for the pair are the same
                lim_min_x = min_x - 0.5 
                lim_max_x = max_x + 0.5
            else:
                range_val = max_x - min_x
                padding = range_val * 0.02 # 2% padding of the range on each side
                lim_min_x = min_x - padding
                lim_max_x = max_x + padding
            shared_x_limits.append((lim_min_x, lim_max_x))
        
        # --- Figure 1: Background Corrected Values ---
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()
        max_density_overall_bg_corrected = 0

        for i, col_name in enumerate(columns_to_plot_bg_corrected):
            ax = axes[i]
            current_x_lims_for_plot = shared_x_limits[i]

            if col_name in merged_sam_df.columns:
                sns.histplot(data=merged_sam_df, x=col_name, hue='worm_id', ax=ax, 
                             kde=False, element="step", stat="density", common_norm=False, 
                             alpha=0.6, legend=False, palette=worm_color_map)
                ax.set_title(f'Histogram of {col_name}')
                ax.set_xlabel(col_name)
                ax.set_ylabel('Density')
                
                if current_x_lims_for_plot[0] is not None and current_x_lims_for_plot[1] is not None:
                    ax.set_xlim(current_x_lims_for_plot)

                for worm_id_val in unique_worm_ids_list:
                    worm_specific_data = merged_sam_df[
                        (merged_sam_df['worm_id'] == worm_id_val) & 
                        (merged_sam_df[col_name].notna())
                    ][col_name]
                    if not worm_specific_data.empty:
                        max_val_for_worm = worm_specific_data.max()
                        ax.axvline(x=max_val_for_worm, color=worm_color_map[worm_id_val], 
                                   linestyle='--', linewidth=1.2, alpha=0.7)

                current_max_density_subplot = ax.get_ylim()[1]
                if current_max_density_subplot > max_density_overall_bg_corrected:
                    max_density_overall_bg_corrected = current_max_density_subplot
            else:
                print(f"Warning: Column '{col_name}' not found. Skipping this plot.")
                if i < len(axes): 
                    ax.set_title(f'Column {col_name} not found')
                    ax.text(0.5, 0.5, 'Data not available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    if current_x_lims_for_plot[0] is not None and current_x_lims_for_plot[1] is not None:
                         ax.set_xlim(current_x_lims_for_plot) # Apply x-lim even if data for this specific plot is missing, for consistency

        # --- Figure 2: Raw Values ---
        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(18, 12))
        axes_raw = axes_raw.flatten()
        max_density_overall_raw = 0

        for i, col_name in enumerate(columns_to_plot_raw):
            ax = axes_raw[i]
            current_x_lims_for_plot = shared_x_limits[i]

            if col_name in merged_sam_df.columns:
                sns.histplot(data=merged_sam_df, x=col_name, hue='worm_id', ax=ax,
                             kde=False, element="step", stat="density", common_norm=False,
                             alpha=0.6, legend=False, palette=worm_color_map)
                ax.set_title(f'Histogram of {col_name} (Raw)')
                ax.set_xlabel(col_name)
                ax.set_ylabel('Density')

                if current_x_lims_for_plot[0] is not None and current_x_lims_for_plot[1] is not None:
                    ax.set_xlim(current_x_lims_for_plot)

                for worm_id_val in unique_worm_ids_list:
                    worm_specific_data = merged_sam_df[
                        (merged_sam_df['worm_id'] == worm_id_val) &
                        (merged_sam_df[col_name].notna())
                    ][col_name]
                    if not worm_specific_data.empty:
                        max_val_for_worm = worm_specific_data.max()
                        ax.axvline(x=max_val_for_worm, color=worm_color_map[worm_id_val],
                                   linestyle='--', linewidth=1.2, alpha=0.7)
                
                current_max_density_subplot_raw = ax.get_ylim()[1]
                if current_max_density_subplot_raw > max_density_overall_raw:
                    max_density_overall_raw = current_max_density_subplot_raw
            else:
                print(f"Warning: Column '{col_name}' not found. Skipping this raw plot.")
                if i < len(axes_raw):
                    ax.set_title(f'Column {col_name} not found')
                    ax.text(0.5, 0.5, 'Data not available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    if current_x_lims_for_plot[0] is not None and current_x_lims_for_plot[1] is not None:
                        ax.set_xlim(current_x_lims_for_plot) # Apply x-lim even if data for this specific plot is missing

        # Determine the global maximum y-axis density
        global_max_y_density = 0
        if max_density_overall_bg_corrected > 0 : 
            global_max_y_density = max(global_max_y_density, max_density_overall_bg_corrected)
        if max_density_overall_raw > 0: 
             global_max_y_density = max(global_max_y_density, max_density_overall_raw)


        # Apply global y-axis limits to Figure 1
        if global_max_y_density > 0: 
            for ax_j in axes:
                if ax_j.has_data(): 
                    ax_j.set_ylim(0, global_max_y_density * 1.05)
        
        plt.figure(fig.number) 
        plt.tight_layout(rect=[0, 0, 1, 0.96]) 
        plt.suptitle('Histograms of Background and Corrected Values by Worm ID (Shared X/Y Scales)', fontsize=16)
        plt.savefig('histograms_background_corrected_values_by_worm_id_shared_scales.png', bbox_inches='tight')
        print("\nPlots saved to histograms_background_corrected_values_by_worm_id_shared_scales.png")
        plt.show()

        # Apply global y-axis limits to Figure 2
        if global_max_y_density > 0: 
            for ax_j in axes_raw:
                if ax_j.has_data():
                    ax_j.set_ylim(0, global_max_y_density * 1.05)

        plt.figure(fig_raw.number) 
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle('Histograms of Raw Compartment Values by Worm ID (Shared X/Y Scales)', fontsize=16)
        plt.savefig('histograms_raw_compartment_values_by_worm_id_shared_scales.png', bbox_inches='tight')
        print("\nPlots saved to histograms_raw_compartment_values_by_worm_id_shared_scales.png")
        plt.show()

else:
    print("merged_sam_df does not exist or is empty. Cannot create histogram plots.")



##Normalize compartment values

if 'merged_sam_df' in locals() and not merged_sam_df.empty:
    # Create a copy to avoid modifying the original merged_sam_df directly
    sam_df_normalized_overall = merged_sam_df.copy()
    print("\nCreated a copy of merged_sam_df named sam_df_normalized_overall.")

    # Define the 0-1 normalization function
    def normalize_0_1_overall(series):
        min_val = series.min()
        max_val = series.max()
        if min_val == max_val: # Avoid division by zero if all values are the same
            # If all values are the same, normalization could result in 0, 0.5, or NaN.
            # Here, we'll return a series of 0s. Adjust if a different behavior is needed.
            return pd.Series([0] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)

    columns_to_normalize_overall = ['2_bg_corrected', '3_bg_corrected', '4_bg_corrected']
    new_column_names_overall = {
        '2_bg_corrected': '2_bg_corrected_norm_overall',
        '3_bg_corrected': '3_bg_corrected_norm_overall',
        '4_bg_corrected': '4_bg_corrected_norm_overall'
    }

    for col in columns_to_normalize_overall:
        if col in sam_df_normalized_overall.columns:
            new_col_name = new_column_names_overall[col]
            sam_df_normalized_overall[new_col_name] = normalize_0_1_overall(sam_df_normalized_overall[col])
            print(f"Added '{new_col_name}' column to sam_df_normalized_overall (normalized across entire dataset).")
        else:
            print(f"Warning: Column '{col}' not found in sam_df_normalized_overall. Cannot normalize.")
    
    print("\nFirst 5 rows of sam_df_normalized_overall with new overall normalized columns:")
    columns_to_show_overall = ['worm_id'] + columns_to_normalize_overall + list(new_column_names_overall.values())
    # Ensure only existing columns are selected for printing
    existing_columns_to_show_overall = [c for c in columns_to_show_overall if c in sam_df_normalized_overall.columns]
    print(sam_df_normalized_overall[existing_columns_to_show_overall].head())

    print("\nSummary statistics for new overall normalized columns in sam_df_normalized_overall:")
    # Ensure only existing new columns are described
    existing_new_cols_overall = [nc for nc in new_column_names_overall.values() if nc in sam_df_normalized_overall.columns]
    if existing_new_cols_overall:
        print(sam_df_normalized_overall[existing_new_cols_overall].describe())
    else:
        print("No new overall normalized columns were created to describe.")

else:
    print("\nmerged_sam_df does not exist or is empty. Cannot perform overall normalization.")


##Save df
sam_df_normalized_overall.to_csv('sam_df_normalized_overall.csv', index=False)

#endregion [merge sam files and prepare for analysis]


#region [4) merge fiji and sam files]

if 'fiji_df_normalized' in locals() and 'sam_df_normalized_overall' in locals():
    print("\nStarting merge of fiji_df_normalized and sam_df_normalized_overall...")

    # Make copies to avoid modifying the original dataframes
    fiji_df_copy = fiji_df_normalized.copy()
    sam_df_copy = sam_df_normalized_overall.copy()

    # Add source column
    fiji_df_copy['source'] = 'fiji'
    sam_df_copy['source'] = 'sam'
    print("Added 'source' column to both DataFrame copies.")

    # Standardize column names in sam_df_copy
    sam_rename_map = {
        'worm_id': 'worm',  # Standardize worm identifier
        '2_bg_corrected_norm_overall': 'nrd_norm',
        '3_bg_corrected_norm_overall': 'nrv_norm',
        '4_bg_corrected_norm_overall': 'loop_norm'
    }
    
    # Check if columns to be renamed exist in sam_df_copy before renaming
    actual_sam_rename_map = {}
    missing_cols_in_sam_for_rename = []
    for k_orig, k_new in sam_rename_map.items():
        if k_orig in sam_df_copy.columns:
            actual_sam_rename_map[k_orig] = k_new
        else:
            missing_cols_in_sam_for_rename.append(k_orig)

    if missing_cols_in_sam_for_rename:
        print(f"Warning: The following columns intended for renaming were not found in the SAM data copy: {missing_cols_in_sam_for_rename}")

    if actual_sam_rename_map:
        sam_df_copy = sam_df_copy.rename(columns=actual_sam_rename_map)
        print(f"Renamed columns in SAM data copy: {actual_sam_rename_map}")
    else:
        print("No columns were renamed in the SAM data copy (either not found or map was empty).")


    # Concatenate the dataframes
    # pd.concat handles columns that are not present in both DataFrames by filling with NaN
    # sort=False is used to maintain column order as much as possible and prevent future warnings
    merged_df_fiji_sam = pd.concat([fiji_df_copy, sam_df_copy], ignore_index=True, sort=False)

    print("\nSuccessfully merged Fiji and SAM data.")

    # Clean worm names
    if 'worm' in merged_df_fiji_sam.columns:
        print("\nOriginal unique worm names (sample before any cleaning):")
        # Print a sample that might contain the AG_WT prefix and MMG
        sample_uncleaned_names = []
        if len(merged_df_fiji_sam['worm'].unique()) > 25: # Check if there are enough unique names
             sample_uncleaned_names = list(merged_df_fiji_sam['worm'].unique()[-5:]) + list(merged_df_fiji_sam['worm'].unique()[:5])
        else:
            sample_uncleaned_names = list(merged_df_fiji_sam['worm'].unique())
        print(sample_uncleaned_names)


        def clean_worm_name(name):
            if isinstance(name, str):
                cleaned_name = name
                if "AG_WT-" in cleaned_name and "_crop" in cleaned_name:
                    # Extract the part between "AG_WT-" and "_crop"
                    # Example: AG_WT-MMH99_10s_20190221_02_crop... -> MMH99_10s_20190221_02
                    try:
                        name_part = cleaned_name.split("AG_WT-")[1]
                        cleaned_name = name_part.split("_crop")[0]
                    except IndexError:
                        # In case the split doesn't work as expected, return original
                        pass # Keep 'cleaned_name' as it was
                
                # Correct MMG to MMH
                if "MMG" in cleaned_name:
                    cleaned_name = cleaned_name.replace("MMG", "MMH")
                
                return cleaned_name
            return name

        merged_df_fiji_sam['worm'] = merged_df_fiji_sam['worm'].apply(clean_worm_name)
        print("\nCleaned unique worm names (sample after general cleaning):")
        print(merged_df_fiji_sam['worm'].unique()[:10]) 
        
        # --- Specific correction for 'MMH99_10s_20190305_01' ---
        specific_old_name = 'MMH99_10s_20190305_01'
        specific_new_name = 'MMH99_10s_20190305_03'
        
        # Check if the specific old name exists before attempting replacement
        if specific_old_name in merged_df_fiji_sam['worm'].unique():
            merged_df_fiji_sam['worm'] = merged_df_fiji_sam['worm'].replace(specific_old_name, specific_new_name)
            print(f"\nApplied specific correction: Replaced '{specific_old_name}' with '{specific_new_name}'.")
            print("Unique worm names (sample after specific correction):")
            print(merged_df_fiji_sam['worm'].unique()[:10]) 
        else:
            print(f"\nSpecific name '{specific_old_name}' not found for replacement. Skipping this specific correction.")
        # --- End specific correction ---

        # Specifically check for MMG absence
        mmg_after_cleaning = [n for n in merged_df_fiji_sam['worm'].unique() if isinstance(n, str) and "MMG" in n]
        if not mmg_after_cleaning:
            print("Confirmed: No 'MMG' found in unique worm names after cleaning.")
        else:
            print(f"Warning: 'MMG' still present in unique worm names: {mmg_after_cleaning}")
        
        print(f"Total unique worm names after cleaning: {merged_df_fiji_sam['worm'].nunique()}")

        # --- Begin: Check for unpaired worm IDs ---
        if 'source' in merged_df_fiji_sam.columns:
            print("\n--- Checking for unpaired worm IDs ---")
            
            fiji_worms = set(merged_df_fiji_sam[merged_df_fiji_sam['source'] == 'fiji']['worm'].unique())
            sam_worms = set(merged_df_fiji_sam[merged_df_fiji_sam['source'] == 'sam']['worm'].unique())

            print(f"Unique cleaned worm IDs from Fiji source: {len(fiji_worms)}")
            print(f"Unique cleaned worm IDs from SAM source: {len(sam_worms)}")

            only_in_fiji = fiji_worms - sam_worms
            only_in_sam = sam_worms - fiji_worms
            
            if only_in_fiji:
                print(f"Worm IDs found ONLY in Fiji data: {only_in_fiji}")
            if only_in_sam:
                print(f"Worm IDs found ONLY in SAM data: {only_in_sam}")
            
            if not only_in_fiji and not only_in_sam:
                print("All cleaned worm IDs are present in both Fiji and SAM sources.")
            
            # Cross-check by counting sources per worm_id in the merged dataframe
            worm_source_counts = merged_df_fiji_sam.groupby('worm')['source'].nunique()
            unpaired_worms = worm_source_counts[worm_source_counts != 2]
            
            if not unpaired_worms.empty:
                print("\nWorm IDs in merged data not associated with exactly 2 sources (fiji & sam):")
                print(unpaired_worms)
            else:
                print("\nAll worm IDs in merged data are correctly paired with both sources.")
            print("--- End of unpaired worm ID check ---")
        else:
            print("\nWarning: 'source' column not found. Cannot perform unpaired worm ID check.")
        # --- End: Check for unpaired worm IDs ---

    else:
        print("Warning: 'worm' column not found for cleaning.")


    print("\nFirst 5 rows of the merged data (after cleaning worm names):")
    print(merged_df_fiji_sam.head())
    print("\nLast 5 rows of the merged data:")
    print(merged_df_fiji_sam.tail())
    
    print("\nInfo of the merged data:")
    # Using verbose=True to ensure all columns are listed, especially for wide DataFrames
    merged_df_fiji_sam.info(verbose=True, show_counts=True)

    # Verify source column distribution
    if 'source' in merged_df_fiji_sam.columns:
        print("\nValue counts for 'source' column:")
        print(merged_df_fiji_sam['source'].value_counts())
    else:
        print("\nWarning: 'source' column not found in the merged DataFrame.")
        
    # Sanity check for key renamed/common columns
    key_cols_to_check = ['worm', 'frame', 'nrd_norm', 'nrv_norm', 'loop_norm', 'background']
    print("\nChecking presence and non-null counts for key columns in merged data:")
    for col in key_cols_to_check:
        if col in merged_df_fiji_sam.columns:
            non_null_count = merged_df_fiji_sam[col].notna().sum()
            print(f"Column '{col}': Present, Non-null count: {non_null_count}")
        else:
            print(f"Column '{col}': NOT FOUND in merged data.")


    # Save the fully merged dataframe
    output_filename = 'merged_fiji_sam_normalized_data.csv'
    try:
        merged_df_fiji_sam.to_csv(output_filename, index=False)
        print(f"\nMerged data saved to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving merged data to '{output_filename}': {e}")

else:
    missing_dfs = []
    if 'fiji_df_normalized' not in locals():
        missing_dfs.append('fiji_df_normalized')
    if 'sam_df_normalized_overall' not in locals():
        missing_dfs.append('sam_df_normalized_overall')
    print(f"\nCannot merge: One or both required DataFrames ({', '.join(missing_dfs)}) not found in the current scope.")

#endregion [4) merge fiji and sam files]


#region [5) Plot per worm]

# Check if merged_df_fiji_sam exists and is properly populated
if 'merged_df_fiji_sam' not in locals() or merged_df_fiji_sam.empty:
    print("\nmerged_df_fiji_sam is not available or empty. Skipping per-worm plotting.")
else:
    print("\nStarting per-worm plotting...")
    
    output_plot_dir = "/home/lilly/phd/ria/plot_analysis/per_worm_intranorm_roll5"    #### <------------------
    try:
        os.makedirs(output_plot_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_plot_dir}")
    except OSError as e:
        print(f"Error creating directory {output_plot_dir}: {e}. Skipping per-worm plotting.")
        merged_df_fiji_sam = pd.DataFrame() 

    if not merged_df_fiji_sam.empty: 
        required_cols = ['worm', 'frame', 
                         'nrd_norm', 'nrv_norm', 'loop_norm', 
                         'source', 'norm_angle', 'angle_degrees_corrected',
                         '2_bg_corrected', '3_bg_corrected', '4_bg_corrected' # Added for new traces
                        ]
        missing_cols = [col for col in required_cols if col not in merged_df_fiji_sam.columns]

        if missing_cols:
            print(f"Error: Missing required columns in merged_df_fiji_sam for plotting: {missing_cols}. Skipping plotting.")
        else:
            unique_worms = merged_df_fiji_sam['worm'].unique()
            print(f"Found {len(unique_worms)} unique worms to plot.")

            # Define normalization function for per-worm series
            def normalize_series_0_1(series_in):
                series = pd.to_numeric(series_in.copy(), errors='coerce')
                if series.isna().all() or series.empty:
                    return pd.Series([np.nan] * len(series_in), index=series_in.index)
                
                min_val = series.min()
                max_val = series.max()

                if pd.isna(min_val) or pd.isna(max_val): # Should be covered by the first check
                    return pd.Series([np.nan] * len(series_in), index=series_in.index)

                if min_val == max_val:
                    return series.apply(lambda x: 0.5 if pd.notna(x) else np.nan)
                
                normalized = (series - min_val) / (max_val - min_val)
                return normalized

            for worm_id_obj in unique_worms:
                if pd.isna(worm_id_obj):
                    print("Encountered a NaN worm_id. Skipping this entry.")
                    continue
                worm_id = str(worm_id_obj)

                print(f"Generating plot for worm: {worm_id}...")
                worm_data = merged_df_fiji_sam[merged_df_fiji_sam['worm'] == worm_id].copy()

                if worm_data.empty:
                    print(f"No data found for worm {worm_id}. Skipping.")
                    continue
                
                if 'frame' not in worm_data.columns or worm_data['frame'].dropna().empty:
                    print(f"Skipping worm {worm_id}: 'frame' column missing, empty, or all NaN for the worm.")
                    continue

                worm_data_sorted = worm_data.sort_values(by='frame')

                fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True) # 4 subplots now
                fig.suptitle(f"Worm: {worm_id}", fontsize=16)

                any_data_plotted_for_worm = False
                shade_regions = [(0, 100), (200, 300), (400, 500)]
                rolling_window_size = 3

                # --- Head Bend Subplot (axes[0]) ---
                ax_hb = axes[0]
                ax_hb_twin = ax_hb.twinx()
                
                ax_hb.set_ylabel('Head Bend (Fiji, norm. angle)', color='blue')
                ax_hb.set_ylim(-1, 1)
                ax_hb.tick_params(axis='y', labelcolor='blue')

                ax_hb_twin.set_ylabel('Head Bend (SAM, degrees)', color='green')
                ax_hb_twin.set_ylim(-90, 90)
                ax_hb_twin.tick_params(axis='y', labelcolor='green')

                lines_hb, labels_hb = [], [] # These will be populated by get_legend_handles_labels
                plotted_hb_fiji = False
                plotted_hb_sam = False

                # Plot Fiji norm_angle
                fiji_hb_data = worm_data_sorted[worm_data_sorted['source'] == 'fiji']
                if 'norm_angle' in fiji_hb_data.columns and not fiji_hb_data['norm_angle'].dropna().empty:
                    valid_fiji_hb_data = fiji_hb_data.dropna(subset=['frame', 'norm_angle'])
                    if not valid_fiji_hb_data.empty:
                        frames_np = valid_fiji_hb_data['frame'].to_numpy()
                        norm_angle_series = valid_fiji_hb_data['norm_angle']
                        
                        # Rolling average
                        norm_angle_avg = norm_angle_series.rolling(window=rolling_window_size, center=True, min_periods=1).mean().to_numpy()
                        ax_hb.plot(frames_np, norm_angle_avg,
                                   linestyle='-', linewidth=1.2, markersize=0, color='blue', label='Fiji (norm. angle)')
                        
                        any_data_plotted_for_worm = True
                        plotted_hb_fiji = True
                
                # Plot SAM angle_degrees_corrected
                sam_hb_data = worm_data_sorted[worm_data_sorted['source'] == 'sam']
                if 'angle_degrees_corrected' in sam_hb_data.columns and not sam_hb_data['angle_degrees_corrected'].dropna().empty:
                    valid_sam_hb_data = sam_hb_data.dropna(subset=['frame', 'angle_degrees_corrected'])
                    if not valid_sam_hb_data.empty:
                        frames_np = valid_sam_hb_data['frame'].to_numpy()
                        angle_deg_corr_series = valid_sam_hb_data['angle_degrees_corrected']

                        # Rolling average
                        angle_deg_corr_avg = angle_deg_corr_series.rolling(window=rolling_window_size, center=True, min_periods=1).mean().to_numpy()
                        ax_hb_twin.plot(frames_np, angle_deg_corr_avg,
                                        linestyle='-', linewidth=1.2, markersize=0, color='green', label='SAM (degrees)')
                        
                        any_data_plotted_for_worm = True
                        plotted_hb_sam = True

                if plotted_hb_fiji or plotted_hb_sam:
                    # Combine legends from both y-axes for ax_hb
                    lns_combined = []
                    labs_combined = []
                    if ax_hb.has_data():
                        l_ax, la_ax = ax_hb.get_legend_handles_labels()
                        lns_combined.extend(l_ax)
                        labs_combined.extend(la_ax)
                    if ax_hb_twin.has_data():
                        l_twin, la_twin = ax_hb_twin.get_legend_handles_labels()
                        lns_combined.extend(l_twin)
                        labs_combined.extend(la_twin)
                    if lns_combined: # Check if there's anything to put in the legend
                         ax_hb.legend(lns_combined, labs_combined, loc='best', fontsize='small')
                else:
                    ax_hb.text(0.5, 0.5, 'Head bend data\nnot available', 
                               horizontalalignment='center', verticalalignment='center', 
                               transform=ax_hb.transAxes)

                for start, end in shade_regions:
                    ax_hb.axvspan(start, end, color='lightblue', alpha=0.3, lw=0, zorder=0)


                # --- Intensity Subplots (axes[1], axes[2], axes[3]) ---
                plot_info = [
                    ('nrd_norm', '2_bg_corrected', 'Normalized Dorsal Intensity', 'blue'),
                    ('nrv_norm', '3_bg_corrected', 'Normalized Ventral Intensity', 'blue'), 
                    ('loop_norm', '4_bg_corrected', 'Normalized Loop Intensity', 'blue')
                ]
                sam_intensity_color = 'purple'


                for i_plot, (norm_col_name, sam_raw_col_name, y_label_text, fiji_color) in enumerate(plot_info):
                    ax_intensity = axes[i_plot+1] # Start from axes[1]
                    ax_intensity.set_ylabel(y_label_text)
                    ax_intensity.set_ylim(0, 1)
                    
                    lines_intensity_subplot, labels_intensity_subplot = [], [] # For this specific subplot's legend
                    plotted_anything_on_this_intensity_ax = False

                    # Plot Fiji data for intensity (norm_col_name)
                    fiji_intensity_data = worm_data_sorted[worm_data_sorted['source'] == 'fiji']
                    if norm_col_name in fiji_intensity_data.columns and not fiji_intensity_data[norm_col_name].dropna().empty:
                        valid_fiji_intensity = fiji_intensity_data.dropna(subset=['frame', norm_col_name])
                        if not valid_fiji_intensity.empty:
                            frames_np = valid_fiji_intensity['frame'].to_numpy()
                            fiji_intensity_series = valid_fiji_intensity[norm_col_name]
                            
                            # Rolling average
                            fiji_intensity_avg = fiji_intensity_series.rolling(window=rolling_window_size, center=True, min_periods=1).mean().to_numpy()
                            line_avg, = ax_intensity.plot(frames_np, fiji_intensity_avg, 
                                                          linestyle='-', linewidth=1.2, markersize=0, color=fiji_color, label=f'Fiji ({norm_col_name})')
                            lines_intensity_subplot.append(line_avg)
                            labels_intensity_subplot.append(f'Fiji ({norm_col_name})')
                            
                            any_data_plotted_for_worm = True
                            plotted_anything_on_this_intensity_ax = True
                    
                    sam_intensity_data = worm_data_sorted[worm_data_sorted['source'] == 'sam'] 
                    if sam_raw_col_name in sam_intensity_data.columns and not sam_intensity_data[sam_raw_col_name].dropna().empty:
                        series_to_normalize = sam_intensity_data[sam_raw_col_name]
                        normalized_sam_raw_trace_series = normalize_series_0_1(series_to_normalize)
                        
                        plot_df_trace = pd.DataFrame({
                            'frame': sam_intensity_data['frame'], 
                            'trace_value': normalized_sam_raw_trace_series
                        }).dropna(subset=['frame', 'trace_value'])

                        if not plot_df_trace.empty:
                            frames_np = plot_df_trace['frame'].to_numpy()
                            sam_trace_series = plot_df_trace['trace_value']
                            
                            # Rolling average
                            sam_trace_avg = sam_trace_series.rolling(window=rolling_window_size, center=True, min_periods=1).mean().to_numpy()
                            line_avg, = ax_intensity.plot(frames_np, sam_trace_avg, 
                                                          linestyle='-', linewidth=1.2, markersize=0, color=sam_intensity_color, 
                                                          label=f'SAM ({sam_raw_col_name} worm-norm)')
                            lines_intensity_subplot.append(line_avg)
                            labels_intensity_subplot.append(f'SAM ({sam_raw_col_name} worm-norm)')
                            
                            any_data_plotted_for_worm = True
                            plotted_anything_on_this_intensity_ax = True

                    if plotted_anything_on_this_intensity_ax:
                        ax_intensity.legend(lines_intensity_subplot, labels_intensity_subplot, loc='best', fontsize='small')
                    else:
                        ax_intensity.text(0.5, 0.5, f'Data for {norm_col_name} or {sam_raw_col_name}\nnot available', 
                                          horizontalalignment='center', verticalalignment='center', 
                                          transform=ax_intensity.transAxes)
                    
                    for start, end in shade_regions:
                        ax_intensity.axvspan(start, end, color='lightblue', alpha=0.3, lw=0, zorder=0)

                if not any_data_plotted_for_worm:
                     print(f"No data available to plot for any metric for worm {worm_id}. Figure may be blank except for axes and shading.")
                
                axes[-1].set_xlabel('Frame') # xlabel on the last subplot (axes[3])
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

                filename = os.path.join(output_plot_dir, f"{worm_id}.png")
                
                try:
                    plt.savefig(filename)
                except Exception as e:
                    print(f"Error saving plot for worm {worm_id}: {e}")
                finally:
                    plt.close(fig) 

            print(f"\nFinished generating per-worm plots. Check the directory: {output_plot_dir}")

#endregion [5) Plot per worm]







""" 
['frame', 
'2', '2_bg_corrected', '2_pixel_count',
'3', '3_bg_corrected', '3_pixel_count',
'4', '4_bg_corrected', '4_pixel_count',
'background',
'side_position',
'angle_degrees_corrected',
'object_id',
'angle_degrees',
'bend_location', 'bend_magnitude', 'bend_position_y', 'bend_position_x', 'head_mag', 'body_mag',
'is_noise_peak', 'peak_deviation', 'window_size_used', 'error', 'is_straight', 'has_warning']
 """

def print_example_rows_line_by_line(df):
    if 'merged_df_fiji_sam' not in globals() or df.empty:
        print("merged_df_fiji_sam is not loaded or is empty. Cannot print examples.")
        return

    # No need for pd.set_option for this format

    print("\n--- Example rows from FIJI source (line by line) ---")
    fiji_examples = df[df['source'] == 'fiji'].head(3)
    if not fiji_examples.empty:
        for index, row in fiji_examples.iterrows():
            print(f"\nFiji Row Index: {index}")
            for col_name, value in row.items():
                print(f"  {col_name}: {value}")
    else:
        print("No data found for source 'fiji'.")

    print("\n--- Example rows from SAM source (line by line) ---")
    sam_examples = df[df['source'] == 'sam'].head(3)
    if not sam_examples.empty:
        for index, row in sam_examples.iterrows():
            print(f"\nSAM Row Index: {index}")
            for col_name, value in row.items():
                print(f"  {col_name}: {value}")
    else:
        print("No data found for source 'sam'.")

# To use this, you would call it after your main script has run and populated merged_df_fiji_sam:
# print_example_rows_line_by_line(merged_df_fiji_sam)

# If you are running this snippet completely separately, uncomment the line below
# and make sure 'merged_fiji_sam_normalized_data.csv' is in the correct path.
# merged_df_fiji_sam = pd.read_csv('merged_fiji_sam_normalized_data.csv')
# print_example_rows_line_by_line(merged_df_fiji_sam)

def print_matching_fiji_sam_rows(df, num_examples=3):
    if 'merged_df_fiji_sam' not in globals() or df.empty:
        print("merged_df_fiji_sam is not loaded or is empty. Cannot print examples.")
        return
    
    if 'worm' not in df.columns or 'frame' not in df.columns or 'source' not in df.columns:
        print("Error: DataFrame must contain 'worm', 'frame', and 'source' columns.")
        return

    fiji_data = df[df['source'] == 'fiji']
    sam_data = df[df['source'] == 'sam']

    if fiji_data.empty or sam_data.empty:
        print("Data for one or both sources (Fiji, SAM) is missing.")
        return

    common_worms = pd.Series(list(set(fiji_data['worm'].unique()) & set(sam_data['worm'].unique())))
    common_worms = common_worms.dropna().unique() # Ensure no NaN worms and unique

    if len(common_worms) == 0:
        print("No common worm IDs found between Fiji and SAM sources after cleaning.")
        return

    print(f"Found {len(common_worms)} common worms. Will try to find matching frames for comparison.")
    
    examples_found = 0
    for worm_id in common_worms:
        if examples_found >= num_examples:
            break

        if pd.isna(worm_id): # Skip if worm_id itself is NaN
            continue

        fiji_worm_data = fiji_data[fiji_data['worm'] == worm_id]
        sam_worm_data = sam_data[sam_data['worm'] == worm_id]

        if fiji_worm_data.empty or sam_worm_data.empty:
            continue

        common_frames = pd.Series(list(set(fiji_worm_data['frame'].dropna().unique()) & set(sam_worm_data['frame'].dropna().unique())))
        common_frames = common_frames.dropna().unique() # Ensure no NaN frames

        if len(common_frames) == 0:
            # print(f"No common frames for worm {worm_id}.") # Optional: uncomment for more verbose output
            continue
        
        # Take up to 'num_examples - examples_found' frames for this worm
        frames_to_show_for_this_worm = common_frames[:(num_examples - examples_found)]

        for frame_val in frames_to_show_for_this_worm:
            if examples_found >= num_examples:
                break
            
            print(f"\n--- Comparing Worm: {worm_id}, Frame: {frame_val} ---")

            fiji_row = fiji_worm_data[fiji_worm_data['frame'] == frame_val]
            sam_row = sam_worm_data[sam_worm_data['frame'] == frame_val]

            if not fiji_row.empty:
                print("\n--- FIJI Data ---")
                for index, data in fiji_row.iterrows():
                    for col_name, value in data.items():
                        print(f"  {col_name}: {value}")
            else:
                print(f"\n--- FIJI Data --- \n  No Fiji data for worm {worm_id}, frame {frame_val}")
            
            if not sam_row.empty:
                print("\n--- SAM Data ---")
                for index, data in sam_row.iterrows():
                    for col_name, value in data.items():
                        print(f"  {col_name}: {value}")
            else:
                print(f"\n--- SAM Data --- \n  No SAM data for worm {worm_id}, frame {frame_val}")
            
            print("-" * 40)
            examples_found += 1
            
    if examples_found == 0:
        print("\nCould not find any worms with common frames across both Fiji and SAM sources to display.")
    elif examples_found < num_examples:
        print(f"\nFound and displayed {examples_found} matching instance(s). Less than the requested {num_examples}.")


# To use this, you would call it after your main script has run and populated merged_df_fiji_sam:
# print_matching_fiji_sam_rows(merged_df_fiji_sam, num_examples=100) # Show 2 examples

# If you are running this snippet completely separately, uncomment the line below
# and make sure 'merged_fiji_sam_normalized_data.csv' is in the correct path.
# merged_df_fiji_sam = pd.read_csv('merged_fiji_sam_normalized_data.csv')
# print_matching_fiji_sam_rows(merged_df_fiji_sam, num_examples=2)


