import os
import glob
import pickle
import pandas as pd
import numpy as np

def create_worm_metrics_df(metrics_dir='/home/lilly/phd/laeya/final_metrics'):
    """
    Create a dataframe containing summary metrics for all worms across all images.
    
    Args:
        metrics_dir (str): Directory containing the metrics pickle files
        
    Returns:
        pandas.DataFrame: DataFrame containing the specified metrics for all worms
    """
    # List to store all worm metrics
    all_worms = []
    
    # Get all pickle files in the metrics directory
    metric_files = glob.glob(os.path.join(metrics_dir, '*.pkl'))
    
    # Process each file
    for metric_file in metric_files:
        try:
            # Load the pickle file
            with open(metric_file, 'rb') as f:
                worm_metrics = pickle.load(f)
            
            # Extract the desired metrics from each worm
            for worm in worm_metrics:
                worm_data = {
                    'img_id': worm['img_id'],
                    'worm_id': worm['worm_id'],
                    'worm_area_px': worm['area'],
                    'worm_perimeter_px': worm['perimeter'], 
                    'worm_length_px': worm['pruned_medialaxis_length'],
                    'mean_wormradius_px': worm['mean_wormwidth'],
                    'midlength_wormradius_px': worm['mid_length_width']
                }
                all_worms.append(worm_data)
                
        except Exception as e:
            print(f"Error processing {metric_file}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_worms)
    
    # Sort by img_id and worm_id
    df = df.sort_values(['img_id', 'worm_id'])
    
    return df

def create_worm_length_profile_df(metrics_dir='/home/lilly/phd/laeya/final_metrics'):
    """
    Create a long format dataframe containing all metrics including the width profile along the length.
    
    Args:
        metrics_dir (str): Directory containing the metrics pickle files
        
    Returns:
        pandas.DataFrame: Long format DataFrame containing all metrics and width profiles
    """
    all_worm_data = []
    
    # Get all pickle files in the metrics directory
    metric_files = glob.glob(os.path.join(metrics_dir, '*.pkl'))
    
    # Process each file
    for metric_file in metric_files:
        try:
            # Load the pickle file
            with open(metric_file, 'rb') as f:
                worm_metrics = pickle.load(f)
            
            # Extract metrics from each worm
            for worm in worm_metrics:
                # Get the paired lists
                length_list = worm['medialaxis_length_list']
                distances = worm['medial_axis_distances_sorted']
                
                # Create a row for each pair of measurements
                for length, distance in zip(length_list, distances):
                    worm_data = {
                        'img_id': worm['img_id'],
                        'worm_id': worm['worm_id'],
                        'worm_area_px': worm['area'],
                        'worm_perimeter_px': worm['perimeter'],
                        'worm_length_px': worm['pruned_medialaxis_length'],
                        'mean_wormradius_px': worm['mean_wormwidth'],
                        'midlength_wormradius_px': worm['mid_length_width'],
                        'location_along_wormlength_px': length,
                        'radius_along_wormlength_px': distance
                    }
                    all_worm_data.append(worm_data)
                
        except Exception as e:
            print(f"Error processing {metric_file}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_worm_data)
    
    # Sort by img_id, worm_id, and medialaxis_length
    df = df.sort_values(['img_id', 'worm_id'])
    
    return df

# Create and save both dataframes
summary_df = create_worm_metrics_df()
print(f"Created summary dataframe with {len(summary_df)} worms")
print("\nFirst few rows of summary dataframe:")
print(summary_df.head())

profile_df = create_worm_length_profile_df()
print(f"\nCreated profile dataframe with {len(profile_df)} rows")
print("\nFirst few rows of profile dataframe:")
print(profile_df.head())

# Save both to CSV
summary_df.to_csv('/home/lilly/phd/laeya/worm_metrics_summary.csv', index=False)
profile_df.to_csv('/home/lilly/phd/laeya/worm_metrics_long.csv', index=False)
