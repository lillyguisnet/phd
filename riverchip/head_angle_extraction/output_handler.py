"""
Output handling utilities for head angle extraction.
Handles saving results, generating plots, and creating videos.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import pandas as pd
from .config import Config

def save_results(results_df, filename):
    """
    Save results to CSV.
    
    Args:
        results_df: DataFrame with results
        filename: Original input filename for naming
    """
    if Config.should_save_csv():
        # Extract base filename without extension
        base_name = os.path.splitext(os.path.basename(filename))[0]
        csv_filename = f"{base_name}_head_angles.csv"
        
        Config.debug_print(f"Saving CSV results to {csv_filename}")
        results_df.to_csv(csv_filename, index=False)
        print(f"✅ Results saved: {csv_filename}")
    else:
        Config.debug_print("CSV saving disabled")

def generate_plots(results_df, output_filename="head_angles.png"):
    """
    Generate plots of head angles if enabled.
    
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
    
    # Create plot of head angle only
    plt.figure(figsize=(12, 6))

    # Use single axis for head angle
    ax1 = plt.gca()

    # Convert DataFrame to numpy arrays before plotting
    frame_data = results_df['frame'].to_numpy()
    angle_data = results_df['angle_degrees'].to_numpy() 

    # Add shaded region between -3 and 3 degrees
    ax1.axhspan(-3, 3, color='gray', alpha=0.2, label='Straight Region')

    # Plot head angle
    l1, = ax1.plot(frame_data, angle_data, 'b.-', alpha=0.7, label='Head Angle')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Head Angle (degrees)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(-180, 180)  # Set y-axis limits for head angle

    # Add legend
    ax1.legend(loc='upper right')

    plt.title('Head Angle Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"✅ Plot saved: {output_filename}")

def create_layered_mask_video(image_dir, bottom_masks_dict, top_masks_dict, angles_df,
                           output_path, fps=10, bottom_alpha=0.5, top_alpha=0.7):
    """
    Create a video with mask overlays and angle values displayed at skeleton tips.
    
    Args:
        image_dir (str): Directory containing the input images
        bottom_masks_dict (dict): Dictionary of bottom layer masks
        top_masks_dict (dict): Dictionary of top layer masks
        angles_df (pd.DataFrame): DataFrame containing 'frame' and 'angle_degrees' columns
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        bottom_alpha (float): Transparency of bottom mask overlay (0-1)
        top_alpha (float): Transparency of top mask overlay (0-1)
    """
    # Predefined colors for different masks
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (128, 0, 128),  # Purple
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (0, 128, 128),  # Teal
        (255, 128, 0),  # Orange
        (255, 0, 128),  # Deep Pink
        (128, 255, 0),  # Lime
        (255, 255, 0),  # Yellow
        (0, 255, 128)   # Spring Green
    ]

    def create_mask_overlay(image, frame_masks, mask_colors, alpha):
        """Helper function to create a mask overlay"""
        overlay = np.zeros_like(image)
        
        for mask_id, mask in frame_masks.items():
            # Convert to binary mask if needed
            if mask.dtype != bool:
                mask = mask > 0.5
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            # Resize mask to match image dimensions
            mask_resized = cv2.resize(mask.astype(np.uint8), 
                                    (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask_resized == 1] = mask_colors[mask_id]
            
            # Add to overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 1, 0)
        
        return overlay

    def find_skeleton_tip(mask):
        """Find the tip (topmost point) of the skeleton mask"""
        if mask.dtype != bool:
            mask = mask > 0.5
        
        # Ensure mask is 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        # Find all points where mask is True
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
            
        # Find the topmost point
        top_idx = np.argmin(y_coords)
        return (y_coords[top_idx], x_coords[top_idx])

    def add_angle_text(image, angle, position, font_scale=0.7):
        """Add angle text at the given position with background"""
        if position is None or angle is None:
            return image
            
        y, x = position
        angle_text = f"{angle:.1f} deg"
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            angle_text, font, font_scale, font_thickness)
        
        # Adjust position to put text to the right of the tip point
        text_x = int(x + 30)  # Offset text to the right
        text_y = int(y)  # Keep same vertical level as tip
        
        # Add background to make text more readable
        padding = 5
        cv2.rectangle(image, 
                     (text_x - padding, text_y - text_height - padding),
                     (text_x + text_width + padding, text_y + padding),
                     (0, 0, 0), -1)  # Black background
        
        # Draw text
        cv2.putText(image, angle_text,
                   (text_x, text_y + text_height),
                   font, font_scale, (255, 255, 255),  # White text
                   font_thickness)
        
        return image

    # Get sorted list of image files and extract their frame numbers
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Extract frame numbers and create mapping
    frame_numbers = []
    for img_file in image_files:
        # Extract number from filename (assuming format like 000000.jpg)
        match = re.search(r'(\d+)', img_file)
        if match:
            frame_numbers.append((int(match.group(1)), img_file))
    
    # Sort by frame number
    frame_numbers.sort(key=lambda x: x[0])
    
    if not frame_numbers:
        raise ValueError(f"No image files found in {image_dir}")

    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(image_dir, frame_numbers[0][1]))
    if first_image is None:
        raise ValueError(f"Could not read first image: {frame_numbers[0][1]}")
    
    height, width, _ = first_image.shape

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create color mapping for mask IDs
    bottom_mask_ids = set()
    top_mask_ids = set()
    for masks in bottom_masks_dict.values():
        bottom_mask_ids.update(masks.keys())
    for masks in top_masks_dict.values():
        top_mask_ids.update(masks.keys())
    
    mid_point = len(COLORS) // 2
    bottom_colors = COLORS[:mid_point]
    top_colors = COLORS[mid_point:] + COLORS[:max(0, len(top_mask_ids) - len(COLORS) // 2)]
    
    bottom_mask_colors = {mask_id: bottom_colors[i % len(bottom_colors)] 
                         for i, mask_id in enumerate(bottom_mask_ids)}
    top_mask_colors = {mask_id: top_colors[i % len(top_colors)] 
                      for i, mask_id in enumerate(top_mask_ids)}

    # Convert angles DataFrame to dictionary for faster lookup
    angles_dict = angles_df.set_index('frame')['angle_degrees'].to_dict()

    # Process each frame
    for frame_number, image_file in frame_numbers:
        try:
            # Read image
            image_path = os.path.join(image_dir, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Could not read image: {image_file}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Print debug info for first few frames
            if frame_number < 5:
                print(f"Processing frame {frame_number}, file: {image_file}")
                print(f"Bottom masks available: {frame_number in bottom_masks_dict}")
                print(f"Top masks available: {frame_number in top_masks_dict}")
            
            # Start with original frame
            final_frame = frame.copy()
            
            # Apply bottom masks if available
            if frame_number in bottom_masks_dict:
                bottom_overlay = create_mask_overlay(frame, 
                                                  bottom_masks_dict[frame_number],
                                                  bottom_mask_colors, 
                                                  bottom_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, bottom_overlay, bottom_alpha, 0)
            
            # Apply top masks and find skeleton tip
            tip_position = None
            if frame_number in top_masks_dict:
                top_overlay = create_mask_overlay(frame,
                                               top_masks_dict[frame_number],
                                               top_mask_colors,
                                               top_alpha)
                final_frame = cv2.addWeighted(final_frame, 1, top_overlay, top_alpha, 0)
                
                # Find tip position from the first skeleton mask (assuming it's the main one)
                if top_masks_dict[frame_number]:
                    first_mask_id = next(iter(top_masks_dict[frame_number]))
                    tip_position = find_skeleton_tip(top_masks_dict[frame_number][first_mask_id])
            
            # Add angle text if available
            if frame_number in angles_dict and tip_position is not None:
                final_frame = add_angle_text(final_frame, angles_dict[frame_number], tip_position)

            # Write frame
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Error processing frame {frame_number} ({image_file}): {str(e)}")
            continue

    # Clean up
    out.release()
    print(f"Video saved to {output_path}")

def generate_video(head_segments, skeletons, results_df):
    """
    Generate video with overlays if enabled.
    
    Args:
        head_segments: Original head segments
        skeletons: Skeleton data
        results_df: Results DataFrame
    """
    if not Config.should_generate_video():
        Config.debug_print("Video generation disabled")
        return
        
    if len(results_df) == 0:
        print("⚠️  No data to generate video")
        return
        
    Config.debug_print("Generating video with head angle overlays")
    
    try:
        # Use the working video generation function
        video_filename = "head_angle_analysis.mp4"
        
        create_layered_mask_video(
            image_dir=Config.VIDEO_DIR,
            bottom_masks_dict=head_segments,
            top_masks_dict=skeletons,
            angles_df=results_df,
            output_path=video_filename,
            fps=Config.VIDEO_FPS,
            bottom_alpha=Config.BOTTOM_ALPHA,
            top_alpha=Config.TOP_ALPHA
        )
        
        print(f"✅ Video saved: {video_filename}")
        
    except Exception as e:
        print(f"❌ Error generating video: {e}")
        Config.debug_print(f"Video generation error details: {str(e)}") 