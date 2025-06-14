import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np

def calculate_cross_correlation(s1_data, s2_data, series_name_for_debug="Series"):
    if len(s1_data) < 2 or len(s2_data) < 2:
        print(f"  Skipping {series_name_for_debug} due to insufficient data points (s1: {len(s1_data)}, s2: {len(s2_data)})")
        return None, None, np.nan, np.nan
    s1_demeaned = s1_data - np.mean(s1_data)
    s2_demeaned = s2_data - np.mean(s2_data)
    if np.std(s1_demeaned) == 0 or np.std(s2_demeaned) == 0:
        print(f"  Skipping {series_name_for_debug} due to zero standard deviation in one or both signals after de-meaning.")
        return None, None, np.nan, np.nan
    cross_corr_values = np.correlate(s1_demeaned, s2_demeaned, mode='full')
    norm_factor = np.sqrt(np.sum(s1_demeaned**2) * np.sum(s2_demeaned**2))
    if norm_factor == 0:
        print(f"  Skipping {series_name_for_debug} due to zero normalization factor.")
        normalized_cross_corr = np.full_like(cross_corr_values, np.nan, dtype=float)
    else:
        normalized_cross_corr = cross_corr_values / norm_factor
    lags = np.arange(-len(s1_demeaned) + 1, len(s2_demeaned))
    if len(normalized_cross_corr) == 0:
        print(f"  Skipping {series_name_for_debug} due to empty correlation results.")
        max_abs_corr, lag_at_max_abs_corr = np.nan, np.nan
    else:
        max_abs_corr_idx = np.argmax(np.abs(normalized_cross_corr))
        max_abs_corr = normalized_cross_corr[max_abs_corr_idx]
        if max_abs_corr_idx < len(lags):
            lag_at_max_abs_corr = lags[max_abs_corr_idx]
        else:
            print(f"  Warning: max_abs_corr_idx {max_abs_corr_idx} is out of bounds for lags array of length {len(lags)} for {series_name_for_debug}.")
            lag_at_max_abs_corr = np.nan
    return lags, normalized_cross_corr, max_abs_corr, lag_at_max_abs_corr

# Define the directory containing the CSV files
data_dir = "/home/lilly/phd/riverchip/data_analyzed/final_data"

# List the CSV files in the directory
csv_files = [
    "data_original-MMH223_20250519_05_merged_brightness_angles.csv",
    "data_original-MMH223_20250519_02_merged_brightness_angles.csv"
]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through the CSV files
for file_name in csv_files:
    # Construct the full file path
    file_path = os.path.join(data_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the group name from the file name (e.g., "MMH223_20250519_05")
    # This assumes the group identifier is between "data_original-" and "_merged_brightness_angles.csv"
    group_name = file_name.replace("data_original-", "").replace("_merged_brightness_angles.csv", "")
    
    # Add a 'worm_id' column to the DataFrame
    df['worm_id'] = group_name
    
    # Extract strain from worm_id
    df['strain'] = df['worm_id'].apply(lambda x: x.split('_')[0])
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list
merged_df = pd.concat(dfs, ignore_index=True)

# Display the first few rows of the merged DataFrame and its shape
print("Merged DataFrame head:")
print(merged_df.head())
print("\nMerged DataFrame shape:")
print(merged_df.shape)

# Display the unique values in the 'worm_id' column to verify
print("\nUnique worm_ids:")
print(merged_df['worm_id'].unique())

# Display the unique values in the 'strain' column to verify
print("\nUnique strains:")
print(merged_df['strain'].unique())

#### Normalize brightness columns (2_top10 and 3_top10) to 0-1 range
print("\nNormalizing 2_top10 and 3_top10 columns...")
for col_to_norm in ['2_top10', '3_top10']:
    min_val = merged_df[col_to_norm].min()
    max_val = merged_df[col_to_norm].max()
    if max_val - min_val > 0:
        merged_df[col_to_norm] = (merged_df[col_to_norm] - min_val) / (max_val - min_val)
    else:
        merged_df[col_to_norm] = 0 
    print(f"  Column '{col_to_norm}' normalized. Min: {merged_df[col_to_norm].min():.2f}, Max: {merged_df[col_to_norm].max():.2f}")

#### Count % of frames in each angle category per strain

# Define a function to categorize angles
def categorize_angle(angle):
    if -10 <= angle <= 10:
        return "straight"
    elif angle < -10:
        return "in odor"
    elif angle > 10:
        return "in buffer"
    else:
        return None # Should not happen if data is clean

# Apply the categorization function to the 'angle_degrees_smoothed_3frame' column
merged_df['angle_category'] = merged_df['angle_degrees_smoothed_3frame'].apply(categorize_angle)

# Calculate the percentage of frames in each category, grouped by strain
print("\nAngle category percentages per strain:")

# Group by strain and angle_category and count occurrences
category_counts = merged_df.groupby(['strain', 'angle_category']).size().unstack(fill_value=0)

# Calculate total frames per strain
total_frames_per_strain = category_counts.sum(axis=1)

# Calculate percentages
category_percentages = category_counts.apply(lambda x: (x / total_frames_per_strain) * 100, axis=0)

# Reorder columns to match request if necessary and fill NaN with 0 for strains that might be missing a category
category_percentages = category_percentages.reindex(columns=['straight', 'in odor', 'in buffer'], fill_value=0)

print(category_percentages)


#### Plotting

# Horizontal violin plot
print("\nGenerating violin plot...")
plt.figure(figsize=(10, 8))
ax = sns.violinplot(x='angle_degrees_smoothed_3frame', y='strain', data=merged_df, orient='h', inner=None, palette="pastel")

# Make violins transparent
for c in ax.collections:
    if isinstance(c, matplotlib.collections.PolyCollection):
        c.set_alpha(0.4) # Adjust alpha for violin transparency

# Overlay data points
sns.stripplot(x='angle_degrees_smoothed_3frame', y='strain', data=merged_df, ax=ax, orient='h', color='gray', size=2, jitter=True, alpha=0.3)

# Add background shading for the -10 to 10 region
ax.axvspan(-10, 10, color='lightgray', alpha=0.3, zorder=0)

plt.title('Distribution of Smoothed Angles by Strain')
plt.xlabel('Angle Degrees (Smoothed 3frame)')
plt.ylabel('Strain')
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
plt.savefig('/home/lilly/phd/riverchip/violin_plot.png')


#### Cross-correlation calculations (for 2_top10) - AVERAGING APPLIED

print("\nCalculating cross-correlations for angle_degrees_smoothed_3frame vs. 2_top10 (with averaging)...")

results_cross_corr = {}

for strain_name, strain_df in merged_df.groupby('strain'):
    print(f"\nProcessing strain for '2_top10': {strain_name}")
    
    all_individual_results_for_strain = []
    for worm_id, worm_df in strain_df.groupby('worm_id'):
        s1_worm = worm_df['angle_degrees_smoothed_3frame'].to_numpy()
        s2_worm = worm_df['2_top10'].to_numpy()
        ind_lags, ind_corr_values, _, _ = calculate_cross_correlation(
            s1_worm, s2_worm, series_name_for_debug=f"Worm {worm_id} (2_top10)"
        )
        all_individual_results_for_strain.append({
            'worm_id': worm_id, 'lags': ind_lags, 'corr_values': ind_corr_values
        })

    overall_lags_avg, overall_corr_values_avg = None, None
    overall_max_corr_avg, overall_max_lag_avg = np.nan, np.nan
    valid_individual_corr_values_list = []
    reference_lags_for_strain, reference_length_for_strain = None, None

    if any(res['lags'] is not None and res['corr_values'] is not None for res in all_individual_results_for_strain):
        for res in all_individual_results_for_strain:
            if res['lags'] is not None and res['corr_values'] is not None:
                reference_lags_for_strain, reference_length_for_strain = res['lags'], len(res['lags'])
                break
    
    if reference_lags_for_strain is not None:
        for res in all_individual_results_for_strain:
            if (res['lags'] is not None and res['corr_values'] is not None and
                len(res['lags']) == reference_length_for_strain and
                np.array_equal(res['lags'], reference_lags_for_strain) and
                len(res['corr_values']) == reference_length_for_strain):
                valid_individual_corr_values_list.append(res['corr_values'])
            elif res['lags'] is not None or res['corr_values'] is not None:
                print(f"  WARNING: Strain {strain_name}, Worm {res['worm_id']} (2_top10) has data structure different from reference. Skipping for averaging.")
        
        if valid_individual_corr_values_list:
            overall_lags_avg = reference_lags_for_strain
            # Filter out None arrays before vstacking
            arrays_to_stack = [arr for arr in valid_individual_corr_values_list if arr is not None and arr.ndim == 1]
            if arrays_to_stack: # only proceed if list is not empty
                stacked_corr_values = np.vstack(arrays_to_stack)
                if stacked_corr_values.size > 0:
                    overall_corr_values_avg = np.nanmean(stacked_corr_values, axis=0)
                    if overall_corr_values_avg is not None and len(overall_corr_values_avg) > 0 and not np.all(np.isnan(overall_corr_values_avg)):
                        max_abs_corr_idx = np.argmax(np.abs(overall_corr_values_avg))
                        overall_max_corr_avg = overall_corr_values_avg[max_abs_corr_idx]
                        overall_max_lag_avg = overall_lags_avg[max_abs_corr_idx]
                    print(f"  Overall Strain (Averaged for '2_top10'): {strain_name} Max abs corr: {overall_max_corr_avg:.4f} at lag {overall_max_lag_avg:.0f}")
                else: print(f"  INFO: No valid arrays to stack for averaging for strain {strain_name} (2_top10) after filtering.")
            else: print(f"  INFO: No valid correlation arrays to stack for averaging for strain {strain_name} (2_top10).")
        else: print(f"  INFO: No individual correlations with consistent structure for strain {strain_name} (2_top10).")
    else: print(f"  INFO: No valid individual correlations for strain {strain_name} (2_top10) to find reference for averaging.")

    results_cross_corr[strain_name] = {
        'overall': {'lags': overall_lags_avg, 'corr_values': overall_corr_values_avg, 'max_corr': overall_max_corr_avg, 'max_lag': overall_max_lag_avg},
        'individuals': all_individual_results_for_strain
    }
print("\nCross-correlation calculations (2_top10, averaged) complete.")


#### Plotting cross-correlations (for 2_top10)

print("\nPlotting cross-correlation functions with individual worm traces (2_top10)...")
plt.figure(figsize=(12, 8))

try:
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
except AttributeError:
    print(" INFO: Using default color list due to rcParams structure.")
    color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

all_overall_data_missing = True

for i, (strain_name, strain_data) in enumerate(results_cross_corr.items()):
    has_plotted_individual_for_strain = False
    for individual_corr_data in strain_data['individuals']:
        if (individual_corr_data['lags'] is not None and individual_corr_data['corr_values'] is not None and 
            len(individual_corr_data['lags']) > 0 and len(individual_corr_data['corr_values']) > 0 and 
            not np.all(np.isnan(individual_corr_data['corr_values']))):
            plt.plot(individual_corr_data['lags'], individual_corr_data['corr_values'], linewidth=1, alpha=0.3, color='blue', zorder=1)
            has_plotted_individual_for_strain = True
    if has_plotted_individual_for_strain: print(f"  INFO: Plotted individual worm traces in BLUE for strain '{strain_name}'.")
    else: print(f"  INFO: No valid individual worm traces to plot for strain '{strain_name}'.")

    overall_corr_data = strain_data['overall']
    if (overall_corr_data['lags'] is not None and overall_corr_data['corr_values'] is not None and 
        len(overall_corr_data['lags']) > 0 and len(overall_corr_data['corr_values']) > 0 and
        not np.all(np.isnan(overall_corr_data['corr_values']))):
        strain_color_to_use = color_cycle[i % len(color_cycle)]
        max_corr_val, max_lag_val = overall_corr_data['max_corr'], overall_corr_data['max_lag']
        label_text = f"Strain {strain_name} (Overall Max: {max_corr_val:.2f} at lag {max_lag_val:.0f})"
        if np.isnan(max_corr_val) or np.isnan(max_lag_val): label_text = f"Strain {strain_name} (Overall - data insufficient for max)"
        plt.plot(overall_corr_data['lags'], overall_corr_data['corr_values'], label=label_text, linewidth=2.5, alpha=1.0, color=strain_color_to_use, zorder=10)
        plt.axvline(x=0, color=strain_color_to_use, linestyle='-', linewidth=1.5, alpha=0.6, zorder=9)
        if not np.isnan(max_lag_val): plt.axvline(x=max_lag_val, color=strain_color_to_use, linestyle=':', linewidth=1.5, alpha=0.8, zorder=9)
        print(f"  INFO: Plotted overall (prominent) line for strain '{strain_name}' in color {strain_color_to_use}.")
        all_overall_data_missing = False
    else: print(f"  CRITICAL_INFO: Did NOT plot overall line for strain '{strain_name}' because its correlation data was missing, empty, or all NaN.")

if all_overall_data_missing and not results_cross_corr: print("  CRITICAL_INFO: No cross-correlation data was available at all (results_cross_corr is empty).")
elif all_overall_data_missing: print("  CRITICAL_INFO: No overall strain correlation lines were plotted for ANY strain. Legend will be empty.")

plt.title('Cross-correlation: angle_degrees_smoothed_3frame vs. 2_top10')
plt.xlabel('Lag (frames)')
plt.ylabel('Normalized Cross-correlation')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
if not all_overall_data_missing: plt.legend(loc='best', fontsize='small')
else: print(" INFO: Legend skipped as no overall strain lines were plotted.")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig('/home/lilly/phd/riverchip/cross_correlation_plot_with_individuals.png')
print("\nCross-correlation plotting for 2_top10 complete.")


#### Cross-correlation calculations for 3_top10 - AVERAGING APPLIED

print("\nCalculating cross-correlations for angle_degrees_smoothed_3frame vs. 3_top10 (with averaging)...")

results_cross_corr_3top10 = {}

for strain_name, strain_df in merged_df.groupby('strain'):
    print(f"\nProcessing strain for '3_top10': {strain_name}")
    
    all_individual_results_for_strain_3top10 = []
    for worm_id, worm_df in strain_df.groupby('worm_id'):
        s1_worm = worm_df['angle_degrees_smoothed_3frame'].to_numpy()
        s2_worm = worm_df['3_top10'].to_numpy()
        ind_lags, ind_corr_values, _, _ = calculate_cross_correlation(
            s1_worm, s2_worm, series_name_for_debug=f"Worm {worm_id} (3_top10)"
        )
        all_individual_results_for_strain_3top10.append({
            'worm_id': worm_id, 'lags': ind_lags, 'corr_values': ind_corr_values
        })

    overall_lags_avg_3top10, overall_corr_values_avg_3top10 = None, None
    overall_max_corr_avg_3top10, overall_max_lag_avg_3top10 = np.nan, np.nan
    valid_individual_corr_values_list_3top10 = []
    reference_lags_for_strain_3top10, reference_length_for_strain_3top10 = None, None

    if any(res['lags'] is not None and res['corr_values'] is not None for res in all_individual_results_for_strain_3top10):
        for res in all_individual_results_for_strain_3top10:
            if res['lags'] is not None and res['corr_values'] is not None:
                reference_lags_for_strain_3top10, reference_length_for_strain_3top10 = res['lags'], len(res['lags'])
                break

    if reference_lags_for_strain_3top10 is not None:
        for res in all_individual_results_for_strain_3top10:
            if (res['lags'] is not None and res['corr_values'] is not None and
                len(res['lags']) == reference_length_for_strain_3top10 and
                np.array_equal(res['lags'], reference_lags_for_strain_3top10) and
                len(res['corr_values']) == reference_length_for_strain_3top10):
                valid_individual_corr_values_list_3top10.append(res['corr_values'])
            elif res['lags'] is not None or res['corr_values'] is not None:
                print(f"  WARNING: Strain {strain_name}, Worm {res['worm_id']} (3_top10) has data structure different from reference. Skipping for averaging.")

        if valid_individual_corr_values_list_3top10:
            overall_lags_avg_3top10 = reference_lags_for_strain_3top10
            # Filter out None arrays before vstacking
            arrays_to_stack_3top10 = [arr for arr in valid_individual_corr_values_list_3top10 if arr is not None and arr.ndim == 1]
            if arrays_to_stack_3top10: # only proceed if list is not empty
                stacked_corr_values_3top10 = np.vstack(arrays_to_stack_3top10)
                if stacked_corr_values_3top10.size > 0:
                    overall_corr_values_avg_3top10 = np.nanmean(stacked_corr_values_3top10, axis=0)
                    if overall_corr_values_avg_3top10 is not None and len(overall_corr_values_avg_3top10) > 0 and not np.all(np.isnan(overall_corr_values_avg_3top10)):
                        max_abs_corr_idx_3top10 = np.argmax(np.abs(overall_corr_values_avg_3top10))
                        overall_max_corr_avg_3top10 = overall_corr_values_avg_3top10[max_abs_corr_idx_3top10]
                        overall_max_lag_avg_3top10 = overall_lags_avg_3top10[max_abs_corr_idx_3top10]
                    print(f"  Overall Strain (Averaged for '3_top10'): {strain_name} Max abs corr: {overall_max_corr_avg_3top10:.4f} at lag {overall_max_lag_avg_3top10:.0f}")
                else: print(f"  INFO: No valid arrays to stack for averaging for strain {strain_name} (3_top10) after filtering.")
            else: print(f"  INFO: No valid correlation arrays to stack for averaging for strain {strain_name} (3_top10).")
        else: print(f"  INFO: No individual correlations with consistent structure for strain {strain_name} (3_top10).")
    else: print(f"  INFO: No valid individual correlations for strain {strain_name} (3_top10) to find reference for averaging.")

    results_cross_corr_3top10[strain_name] = {
        'overall': {'lags': overall_lags_avg_3top10, 'corr_values': overall_corr_values_avg_3top10, 'max_corr': overall_max_corr_avg_3top10, 'max_lag': overall_max_lag_avg_3top10},
        'individuals': all_individual_results_for_strain_3top10
    }
print("\nCross-correlation calculations (3_top10, averaged) complete.")


#### Plotting cross-correlations for 3_top10 (red traces)

print("\nPlotting cross-correlation functions for 3_top10 (red individual traces)...")
plt.figure(figsize=(12, 8))
# color_cycle is already defined
all_overall_data_missing_3top10 = True

for i, (strain_name, strain_data) in enumerate(results_cross_corr_3top10.items()):
    has_plotted_individual_for_strain = False
    for individual_corr_data in strain_data['individuals']:
        if (individual_corr_data['lags'] is not None and individual_corr_data['corr_values'] is not None and 
            len(individual_corr_data['lags']) > 0 and len(individual_corr_data['corr_values']) > 0 and 
            not np.all(np.isnan(individual_corr_data['corr_values']))):
            plt.plot(individual_corr_data['lags'], individual_corr_data['corr_values'], linewidth=1, alpha=0.3, color='red', zorder=1)
            has_plotted_individual_for_strain = True
    if has_plotted_individual_for_strain: print(f"  INFO: Plotted individual worm traces in RED for strain '{strain_name}' (3_top10 plot).")
    else: print(f"  INFO: No valid individual worm traces to plot for strain '{strain_name}' (3_top10 plot).")

    overall_corr_data = strain_data['overall']
    if (overall_corr_data['lags'] is not None and overall_corr_data['corr_values'] is not None and 
        len(overall_corr_data['lags']) > 0 and len(overall_corr_data['corr_values']) > 0 and
        not np.all(np.isnan(overall_corr_data['corr_values']))):
        strain_color_to_use = 'red' # Overall line is also red for this plot
        max_corr_val, max_lag_val = overall_corr_data['max_corr'], overall_corr_data['max_lag']
        label_text = f"Strain {strain_name} (Overall Max: {max_corr_val:.2f} at lag {max_lag_val:.0f})"
        if np.isnan(max_corr_val) or np.isnan(max_lag_val): label_text = f"Strain {strain_name} (Overall - data insufficient for max)"
        plt.plot(overall_corr_data['lags'], overall_corr_data['corr_values'], label=label_text, linewidth=2.5, alpha=1.0, color=strain_color_to_use, zorder=10)
        plt.axvline(x=0, color=strain_color_to_use, linestyle='-', linewidth=1.5, alpha=0.6, zorder=9)
        if not np.isnan(max_lag_val): plt.axvline(x=max_lag_val, color=strain_color_to_use, linestyle=':', linewidth=1.5, alpha=0.8, zorder=9)
        print(f"  INFO: Plotted overall (prominent) line for strain '{strain_name}' in color {strain_color_to_use} (3_top10 plot).")
        all_overall_data_missing_3top10 = False
    else: print(f"  CRITICAL_INFO: Did NOT plot overall line for strain '{strain_name}' (3_top10 plot) because its correlation data was missing, empty, or all NaN.")

if all_overall_data_missing_3top10 and not results_cross_corr_3top10: print("  CRITICAL_INFO: No cross-correlation data (3_top10) was available at all.")
elif all_overall_data_missing_3top10: print("  CRITICAL_INFO: No overall strain correlation lines were plotted for ANY strain (3_top10 plot). Legend will be empty.")

plt.title('Cross-correlation: angle_degrees_smoothed_3frame vs. 3_top10')
plt.xlabel('Lag (frames)')
plt.ylabel('Normalized Cross-correlation')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
if not all_overall_data_missing_3top10: plt.legend(loc='best', fontsize='small')
else: print(" INFO: Legend skipped as no overall strain lines were plotted (3_top10 plot).")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig('/home/lilly/phd/riverchip/cross_correlation_plot_3top10_with_individuals.png')
print("\nCross-correlation plotting for 3_top10 complete.")
