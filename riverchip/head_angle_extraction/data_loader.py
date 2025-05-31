"""
Data loading utilities for head angle extraction.
Handles loading H5 files and finding unprocessed videos.
"""

import os
import h5py
import random
from .config import Config

def load_head_segments(filename):
    """
    Load cleaned segments from H5 file.
    
    Args:
        filename: Path to the H5 file containing head segments
        
    Returns:
        Dictionary with frame indices as keys and inner dictionaries containing object masks
    """
    Config.debug_print(f"Loading segments from {filename}")
    cleaned_segments = {}
    with h5py.File(filename, 'r') as f:
        num_frames = f.attrs['num_frames']
        object_ids = f.attrs['object_ids']
        
        masks_group = f['masks']
        
        for frame_idx in range(num_frames):
            frame_data = {}
            for obj_id in object_ids:
                mask = (masks_group[str(obj_id)][frame_idx] > 0).astype(bool)
                frame_data[obj_id] = mask
            
            cleaned_segments[frame_idx] = frame_data
            
            # Reduced debug printing for cleaner output
            if frame_idx % 100 == 0 or frame_idx < 10:
                Config.debug_print(f"Loading frame {frame_idx}")
    
    print(f"✅ Cleaned segments loaded from {filename}")
    print(f"📊 Total frames: {len(cleaned_segments)}, Objects: {len(object_ids)}")
    return cleaned_segments

def get_unprocessed_videos(head_segmentation_dir=None, final_data_dir=None):
    """
    Get list of videos that haven't been processed yet.
    
    Args:
        head_segmentation_dir: Directory containing head segmentation files
        final_data_dir: Directory containing final processed data
        
    Returns:
        List of unprocessed video filenames
    """
    if head_segmentation_dir is None:
        head_segmentation_dir = Config.HEAD_SEGMENTATION_DIR
    if final_data_dir is None:
        final_data_dir = Config.FINAL_DATA_DIR
        
    # Get all head segmentation files
    all_videos = [f for f in os.listdir(head_segmentation_dir) if f.endswith("_headsegmentation.h5")]
    
    # For each video, check if it doesn't have a corresponding _headangles.csv file
    processable_videos = []
    for video in all_videos:
        # Extract base name by removing _headsegmentation.h5
        base_name = video.replace("_headsegmentation.h5", "")
        
        # Check if headangles.csv doesn't exist for this base name
        if not os.path.exists(os.path.join(final_data_dir, base_name + "_headangles.csv")):
            processable_videos.append(video)
    
    return processable_videos

def get_random_unprocessed_video(head_segmentation_dir=None, final_data_dir=None):
    """
    Get a random unprocessed video file.
    
    Args:
        head_segmentation_dir: Directory containing head segmentation files
        final_data_dir: Directory containing final processed data
        
    Returns:
        Full path to a random unprocessed video file
    """
    if head_segmentation_dir is None:
        head_segmentation_dir = Config.HEAD_SEGMENTATION_DIR
    if final_data_dir is None:
        final_data_dir = Config.FINAL_DATA_DIR
        
    processable_videos = get_unprocessed_videos(head_segmentation_dir, final_data_dir)
    
    if not processable_videos:
        raise ValueError("No videos found that need head angle processing.")
    
    return os.path.join(head_segmentation_dir, random.choice(processable_videos)) 