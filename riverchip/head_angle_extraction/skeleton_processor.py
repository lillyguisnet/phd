"""
Skeleton processing utilities for head angle extraction.
Handles skeleton generation and truncation with pure business logic.
"""

import numpy as np
import multiprocessing as mp
from skimage import morphology
from .config import Config
from .parallel_processor import ParallelProcessor

def get_skeleton(mask):
    """
    Generate skeleton from binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Skeleton as binary array
    """
    return morphology.skeletonize(mask)

def process_frame_skeletons_chunk(frame_data_chunk):
    """
    Process a chunk of frames to generate skeletons in parallel.
    
    Args:
        frame_data_chunk: List of (frame_idx, frame_data) tuples
    
    Returns:
        Dictionary with frame indices as keys and dictionaries of skeletons as values
    """
    chunk_skeletons = {}
    chunk_sizes = []
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_skeletons = {}
        
        for obj_id, mask in frame_data.items():
            # Generate skeleton from mask
            skeleton = get_skeleton(mask)
            frame_skeletons[obj_id] = skeleton
            
            # Track skeleton size (number of pixels)
            size = np.sum(skeleton)
            chunk_sizes.append(size)
        
        chunk_skeletons[frame_idx] = frame_skeletons
    
    return chunk_skeletons, chunk_sizes

def generate_skeletons(head_segments, processor=None):
    """
    Generate skeletons from head segments using parallel processing.
    
    Args:
        head_segments: Dictionary with frame indices as keys and inner dictionaries 
                      containing object masks as values
        processor: ParallelProcessor instance (optional)
                      
    Returns:
        Tuple containing:
        - Dictionary with frame indices as keys and dictionaries of skeletons as values
        - Dictionary with skeleton size statistics
    """
    if processor is None:
        processor = ParallelProcessor()
    
    # Convert to list of (frame_idx, frame_data) tuples
    frame_list = list(head_segments.items())
    
    # Process in chunks using the parallel processor
    chunk_results = processor.process_in_chunks(
        frame_list, 
        process_frame_skeletons_chunk,
        description="Skeleton Generation"
    )
    
    # Combine results
    skeletons = {}
    all_skeleton_sizes = []
    
    for chunk_skeletons, chunk_sizes in chunk_results:
        skeletons.update(chunk_skeletons)
        all_skeleton_sizes.extend(chunk_sizes)
    
    # Calculate statistics
    stats = {
        'min_size': np.min(all_skeleton_sizes),
        'max_size': np.max(all_skeleton_sizes), 
        'mean_size': np.mean(all_skeleton_sizes),
        'median_size': np.median(all_skeleton_sizes),
        'std_size': np.std(all_skeleton_sizes)
    }
    
    print("\n📊 Skeleton Statistics:")
    print(f"Minimum size: {stats['min_size']:.1f} pixels")
    print(f"Maximum size: {stats['max_size']:.1f} pixels") 
    print(f"Mean size: {stats['mean_size']:.1f} pixels")
    print(f"Median size: {stats['median_size']:.1f} pixels")
    print(f"Standard deviation: {stats['std_size']:.1f} pixels")
    
    return skeletons, stats

def truncate_skeleton_fixed_chunk(chunk_data):
    """
    Truncate skeletons for a chunk of frames using fixed pixel count from top.
    
    Args:
        chunk_data: Tuple of (frame_data_chunk, keep_pixels)
    
    Returns:
        Dictionary with truncated skeletons for the chunk
    """
    frame_data_chunk, keep_pixels = chunk_data
    
    truncated_chunk = {}
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_truncated = {}
        
        for obj_id, skeleton in frame_data.items():
            # Get the 2D array from the 3D input (taking the first channel)
            skeleton_2d = skeleton[0] if skeleton.ndim > 2 else skeleton
            
            # Find all non-zero points
            points = np.where(skeleton_2d)
            if len(points[0]) == 0:  # Empty skeleton
                frame_truncated[obj_id] = skeleton
                continue
            
            # Get the top point and bottom point
            y_min = np.min(points[0])
            y_max = np.max(points[0])
            original_height = y_max - y_min
            
            # Calculate cutoff point - EXACT SAME LOGIC AS WORKING VERSION
            cutoff_point = y_min + keep_pixels + 1
            
            # Create truncated skeleton
            truncated = skeleton.copy()
            if skeleton.ndim > 2:
                truncated[0, cutoff_point:, :] = False  # Using False since it's boolean type
            else:
                truncated[cutoff_point:, :] = False
            
            # Verify the truncation
            new_points = np.where(truncated[0] if truncated.ndim > 2 else truncated)
            if len(new_points[0]) > 0:
                new_height = np.max(new_points[0]) - np.min(new_points[0])
            else:
                new_height = 0
            
            if Config.DEBUG_MODE:
                print(f"Frame {frame_idx}, Object {obj_id}:")
                print(f"Original height: {original_height}")
                print(f"New height: {new_height}")
                print(f"Top point: {y_min}")
                print(f"Cutoff point: {cutoff_point}")
                print("------------------------")
            
            frame_truncated[obj_id] = truncated
            
        truncated_chunk[frame_idx] = frame_truncated
    
    return truncated_chunk

def conservative_truncate_skeleton_chunk(chunk_data):
    """
    Conservatively truncate skeletons for a chunk of frames, only removing noise at extremities.
    
    Args:
        chunk_data: Tuple of (frame_data_chunk, tail_trim_pixels)
    
    Returns:
        Dictionary with conservatively truncated skeletons for the chunk
    """
    frame_data_chunk, tail_trim_pixels = chunk_data
    
    truncated_chunk = {}
    
    for frame_idx, frame_data in frame_data_chunk:
        frame_truncated = {}
        
        for obj_id, skeleton in frame_data.items():
            # Get the 2D array from the 3D input (taking the first channel)
            skeleton_2d = skeleton[0] if skeleton.ndim > 2 else skeleton
            
            # Find all non-zero points
            points = np.where(skeleton_2d)
            if len(points[0]) == 0:  # Empty skeleton
                frame_truncated[obj_id] = skeleton
                continue
            
            # Get the top point and bottom point
            y_min = np.min(points[0])
            y_max = np.max(points[0])
            original_height = y_max - y_min
            
            # Only trim a small amount from the tail (bottom) to remove noise
            # Keep at least 90% of the original skeleton
            max_trim = min(tail_trim_pixels, int(original_height * 0.1))
            cutoff_point = y_max - max_trim
            
            # Create conservatively truncated skeleton
            truncated = skeleton.copy()
            if skeleton.ndim > 2:
                truncated[0, cutoff_point:, :] = False  # Using False since it's boolean type
            else:
                truncated[cutoff_point:, :] = False
            
            frame_truncated[obj_id] = truncated
            
        truncated_chunk[frame_idx] = frame_truncated
    
    return truncated_chunk

def truncate_skeletons(skeletons, method="fixed", processor=None, **kwargs):
    """
    Truncate skeletons using specified method.
    
    Args:
        skeletons: Dictionary of skeleton data
        method: Truncation method ("fixed", "conservative", etc.)
        processor: ParallelProcessor instance (optional)
        **kwargs: Method-specific parameters
        
    Returns:
        Dictionary with truncated skeletons
    """
    if processor is None:
        processor = ParallelProcessor()
    
    if method == "fixed":
        keep_pixels = kwargs.get('keep_pixels', 400)  # Default to 400 like working version
        
        print(f"🎯 Fixed truncation: Keeping {keep_pixels} pixels from top")
        
        # Convert to list of (frame_idx, frame_data) tuples
        frame_list = list(skeletons.items())
        
        # Prepare chunk data with parameters
        chunk_data_list = [(chunk, keep_pixels) for chunk in 
                          [frame_list[i:i + max(1, len(frame_list) // (processor.num_processes * Config.CHUNK_SIZE_MULTIPLIER))] 
                           for i in range(0, len(frame_list), max(1, len(frame_list) // (processor.num_processes * Config.CHUNK_SIZE_MULTIPLIER)))]]
        
        # Process chunks
        try:
            if processor.num_processes == 1:
                Config.debug_print("Running truncation in sequential mode")
                results = [truncate_skeleton_fixed_chunk(chunk_data) for chunk_data in chunk_data_list]
            else:
                with mp.Pool(processes=processor.num_processes) as pool:
                    print("⚡ Starting parallel fixed skeleton truncation...")
                    results = pool.map(truncate_skeleton_fixed_chunk, chunk_data_list)
                    print("✅ Fixed skeleton truncation completed!")
        except Exception as e:
            print(f"❌ Error in parallel processing: {e}")
            # Fallback to sequential processing
            print("🔄 Falling back to sequential processing...")
            results = [truncate_skeleton_fixed_chunk(chunk_data) for chunk_data in chunk_data_list]
        
        # Combine results
        truncated_skeletons = {}
        for chunk_result in results:
            truncated_skeletons.update(chunk_result)
        
        print(f"🎉 Fixed truncation complete! Processed {len(truncated_skeletons)} frames using {processor.num_processes} cores.")
        return truncated_skeletons
    
    elif method == "conservative":
        tail_trim_pixels = kwargs.get('tail_trim_pixels', Config.TAIL_TRIM_PIXELS)
        
        print(f"🎯 Conservative truncation: Maximum tail trim = {tail_trim_pixels} pixels (keeping ~90% of skeleton)")
        
        # Convert to list of (frame_idx, frame_data) tuples
        frame_list = list(skeletons.items())
        
        # Prepare chunk data with parameters
        chunk_data_list = [(chunk, tail_trim_pixels) for chunk in 
                          [frame_list[i:i + max(1, len(frame_list) // (processor.num_processes * Config.CHUNK_SIZE_MULTIPLIER))] 
                           for i in range(0, len(frame_list), max(1, len(frame_list) // (processor.num_processes * Config.CHUNK_SIZE_MULTIPLIER)))]]
        
        # Process chunks
        try:
            if processor.num_processes == 1:
                Config.debug_print("Running truncation in sequential mode")
                results = [conservative_truncate_skeleton_chunk(chunk_data) for chunk_data in chunk_data_list]
            else:
                with mp.Pool(processes=processor.num_processes) as pool:
                    print("⚡ Starting parallel conservative skeleton truncation...")
                    results = pool.map(conservative_truncate_skeleton_chunk, chunk_data_list)
                    print("✅ Conservative skeleton truncation completed!")
        except Exception as e:
            print(f"❌ Error in parallel processing: {e}")
            # Fallback to sequential processing
            print("🔄 Falling back to sequential processing...")
            results = [conservative_truncate_skeleton_chunk(chunk_data) for chunk_data in chunk_data_list]
        
        # Combine results
        truncated_skeletons = {}
        for chunk_result in results:
            truncated_skeletons.update(chunk_result)
        
        print(f"🎉 Conservative truncation complete! Processed {len(truncated_skeletons)} frames using {processor.num_processes} cores.")
        return truncated_skeletons
    
    else:
        raise ValueError(f"Unknown truncation method: {method}") 