"""
Output handling utilities for head angle extraction.
Handles saving results, generating plots, and creating videos.
"""

import matplotlib.pyplot as plt
import numpy as np
from .config import Config

def save_results(results_df, filename):
    """
    Save results to CSV with side correction if enabled.
    
    Args:
        results_df: DataFrame with results
        filename: Original input filename for naming
    """
    if Config.should_save_csv():
        Config.debug_print("Saving CSV results - PLACEHOLDER")
        # This will be implemented by extracting code from the original file
    else:
        Config.debug_print("CSV saving disabled")

def generate_plots(results_df, output_filename="head_angles_and_bends.png"):
    """
    Generate plots of head angles and bend positions if enabled.
    
    Args:
        results_df: DataFrame with results
        output_filename: Name of the output plot file
    """
    if not Config.should_generate_plots():
        Config.debug_print("Plot generation disabled")
        return
        
    if len(results_df) == 0:
        print("⚠️  No data to plot")
        return
        
    Config.debug_print("Generating plots")
    
    # Create plot of head angle and bend position
    plt.figure(figsize=(12, 6))

    # Create twin axes sharing x-axis
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Convert DataFrame to numpy arrays before plotting
    frame_data = results_df['frame'].to_numpy()
    angle_data = results_df['angle_degrees'].to_numpy() 
    bend_data = results_df['bend_location'].to_numpy()

    # Add shaded region between -3 and 3 degrees
    ax1.axhspan(-3, 3, color='gray', alpha=0.2, label='Straight Region')

    # Plot head angle on left y-axis
    l1, = ax1.plot(frame_data, angle_data, 'b.-', alpha=0.7, label='Head Angle')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Head Angle (degrees)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(-180, 180)  # Set y-axis limits for head angle

    # Plot bend position on right y-axis
    l2, = ax2.plot(frame_data, bend_data, 'r.-', alpha=0.7, label='Bend Position Y')
    ax2.set_ylabel('Bend Position Y', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add legend
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Head Angle and Bend Position Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"✅ Plot saved: {output_filename}")

def generate_video(head_segments, skeletons, results_df):
    """
    Generate video with overlays if enabled.
    
    Args:
        head_segments: Original head segments
        skeletons: Skeleton data
        results_df: Results DataFrame
    """
    if Config.should_generate_video():
        Config.debug_print("Generating video - PLACEHOLDER")
        # This will be implemented by extracting code from the original file
    else:
        Config.debug_print("Video generation disabled") 