import h5py
import numpy as np
import os
import random

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

def get_random_unprocessed_video(raw_segments_dir, final_data_dir):
    all_videos = [os.path.splitext(d)[0] for d in os.listdir(raw_segments_dir)]
    
    # Get all files in final_data_dir (without extensions)
    final_data_files = [os.path.splitext(f)[0] for f in os.listdir(final_data_dir)]
    
    unprocessed_videos = []
    for video in all_videos:
        # Extract core name from segmentation file
        # Example: "data_original-hannah_crop_riasegmentation" -> "data_original-hannah"
        if '_crop_' in video:
            core_name = video[:video.find('_crop_')]
        else:
            # Fallback: use the whole name if no '_crop_' pattern found
            core_name = video
        
        # Find all matching files with this core name
        matching_files = [f for f in final_data_files if f.startswith(core_name)]
        
        # Check if this video has been processed by this brightness extraction script
        # It's unprocessed if:
        # 1. No matching files at all, OR
        # 2. Only matching files end with "_headsegmentation_head_angles" (from previous processing)
        is_processed_by_brightness_script = any(
            not f.endswith('_headsegmentation_head_angles') 
            for f in matching_files
        )
        
        if not is_processed_by_brightness_script:
            unprocessed_videos.append(video)
    
    if not unprocessed_videos:
        raise ValueError("All videos have been processed.")
    
    return os.path.join(raw_segments_dir, random.choice(unprocessed_videos) + ".h5")

def inspect_empty_masks(raw_segments, required_ids=[2, 3]):
    """
    Inspect raw_segments to find all empty or missing masks
    """
    print("=== INSPECTING RAW SEGMENTS FOR EMPTY/MISSING MASKS ===")
    
    frames = sorted(raw_segments.keys())
    print(f"Total frames: {len(frames)}")
    print(f"Required object IDs: {required_ids}")
    
    # Count missing/empty masks per object
    missing_count = {obj_id: 0 for obj_id in required_ids}
    empty_count = {obj_id: 0 for obj_id in required_ids}
    
    # Track which frames have issues
    missing_frames = {obj_id: [] for obj_id in required_ids}
    empty_frames = {obj_id: [] for obj_id in required_ids}
    
    for frame in frames:
        for obj_id in required_ids:
            # Check if mask is missing
            if obj_id not in raw_segments[frame]:
                missing_count[obj_id] += 1
                missing_frames[obj_id].append(frame)
                print(f"Frame {frame}: Object {obj_id} MISSING")
                continue
            
            # Check if mask is None
            mask = raw_segments[frame][obj_id]
            if mask is None:
                missing_count[obj_id] += 1
                missing_frames[obj_id].append(frame)
                print(f"Frame {frame}: Object {obj_id} is None")
                continue
            
            # Check if mask is empty (all False/0)
            if np.sum(mask) == 0:
                empty_count[obj_id] += 1
                empty_frames[obj_id].append(frame)
                print(f"Frame {frame}: Object {obj_id} is EMPTY (sum = {np.sum(mask)})")
                continue
            
            # Mask is present and non-empty - report pixel count
            pixel_count = np.sum(mask)
            print(f"Frame {frame}: Object {obj_id} has {pixel_count} pixels")
    
    print("\n=== SUMMARY ===")
    total_issues = 0
    for obj_id in required_ids:
        total_missing = missing_count[obj_id]
        total_empty = empty_count[obj_id]
        total_issues += total_missing + total_empty
        
        print(f"Object {obj_id}:")
        print(f"  - Missing masks: {total_missing}")
        print(f"  - Empty masks: {total_empty}")
        print(f"  - Total problematic frames: {total_missing + total_empty}")
        
        if missing_frames[obj_id]:
            print(f"  - Missing in frames: {missing_frames[obj_id][:10]}{'...' if len(missing_frames[obj_id]) > 10 else ''}")
        if empty_frames[obj_id]:
            print(f"  - Empty in frames: {empty_frames[obj_id][:10]}{'...' if len(empty_frames[obj_id]) > 10 else ''}")
    
    print(f"\nTotal issues found: {total_issues}")
    
    # Also check which object IDs are actually present
    all_object_ids = set()
    for frame_data in raw_segments.values():
        all_object_ids.update(frame_data.keys())
    
    print(f"\nAll object IDs found in data: {sorted(all_object_ids)}")
    
    return {
        'missing_count': missing_count,
        'empty_count': empty_count,
        'missing_frames': missing_frames,
        'empty_frames': empty_frames,
        'total_issues': total_issues
    }

# Load the data
raw_segments_dir = '/home/lilly/phd/riverchip/data_analyzed/ria_segmentation'
final_data_dir = '/home/lilly/phd/riverchip/data_analyzed/final_data'

filename = get_random_unprocessed_video(raw_segments_dir, final_data_dir)
print(f"Loading: {filename}")
raw_segments = load_cleaned_segments_from_h5(filename)

# Inspect the data
inspection_results = inspect_empty_masks(raw_segments, required_ids=[2, 3]) 