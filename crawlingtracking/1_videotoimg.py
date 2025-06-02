import cv2
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def get_supported_video_extensions():
    return ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']

def check_video_readability(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "OpenCV cannot open the video file. The format might be unsupported or the file might be corrupted."
    
    ret, frame = cap.read()
    if not ret:
        return False, "OpenCV can open the file, but cannot read frames from it. The video might be empty or corrupted."
    
    cap.release()
    return True, ""

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg']

def process_single_frame(args):
    """Process a single frame - designed to be called in parallel"""
    frame_number, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height = args
    
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    if (new_width, new_height) != (orig_width, orig_height):
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    frame_path = video_output_dir / f"{frame_number:06d}.{output_format}"
    
    if output_format == 'jpg':
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:  # png
        cv2.imwrite(str(frame_path), frame)
    
    return str(frame_path)

def process_video_for_sam2(video_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False, num_processes=None):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if num_processes is None:
        num_processes = min(cpu_count(), 48)  # Use all available cores up to 48
    
    if output_format not in ['png', 'jpg']:
        raise ValueError("output_format must be either 'png' or 'jpg'")
    
    if video_path.suffix.lower() not in get_supported_video_extensions():
        print(f"Warning: The file extension {video_path.suffix} might not be supported. Attempting to process anyway.")
    
    is_readable, error_message = check_video_readability(video_path)
    if not is_readable:
        raise ValueError(f"Cannot process video: {error_message}")
    
    # Get the subfolder name of the video path
    sub_folder_name = video_path.parent.name
    video_name = video_path.stem
    # Create the new folder name
    new_folder_name = f"{sub_folder_name}-{video_name}"
    video_output_dir = output_dir / new_folder_name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()  # Close the capture early since we'll open new ones in parallel processes
    
    if max_dimension and (orig_width > max_dimension or orig_height > max_dimension):
        scale = max_dimension / max(orig_width, orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    else:
        new_width, new_height = orig_width, orig_height
    
    if fps is None:
        fps = video_fps
    
    frame_interval = max(int(video_fps / fps), 1)
    
    # Check for existing frames in all supported formats
    existing_frames = []
    for ext in get_supported_image_extensions():
        existing_frames.extend(video_output_dir.glob(f"*{ext}"))
    existing_frames.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
    
    existing_frame_numbers = [int(re.search(r'\d+', f.stem).group()) for f in existing_frames]
    expected_frame_numbers = set(range(0, total_frames, frame_interval))
    
    inconsistencies = []
    
    if existing_frames and not force_reprocess:
        missing_frame_numbers = sorted(expected_frame_numbers - set(existing_frame_numbers))
        if missing_frame_numbers:
            inconsistencies.append(f"Found {len(missing_frame_numbers)} missing frames. They will be processed.")
        extra_frame_numbers = sorted(set(existing_frame_numbers) - expected_frame_numbers)
        if extra_frame_numbers:
            inconsistencies.append(f"Found {len(extra_frame_numbers)} unexpected frames. They will be ignored.")
    else:
        missing_frame_numbers = sorted(expected_frame_numbers)
        if force_reprocess:
            inconsistencies.append("Reprocessing all frames as requested.")
        else:
            print("Processing new video.")
    
    frame_paths = []
    new_frames = 0
    converted_frames = 0
    existing_frames_count = 0
    
    if force_reprocess:
        print(f"Reprocessing all frames using {num_processes} processes")
        frames_to_process = sorted(expected_frame_numbers)
        
        # Prepare arguments for parallel processing
        process_args = [(fn, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height) 
                       for fn in frames_to_process]
        
        # Process frames in parallel
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_frame, process_args), 
                               total=len(process_args), desc="Processing frames"))
        
        frame_paths = [path for path in results if path is not None]
        new_frames = len(frame_paths)
        
    else:
        existing_frame_dict = {int(re.search(r'\d+', f.stem).group()): f for f in existing_frames}
        frames_to_process = []
        
        for fn in sorted(expected_frame_numbers):
            if fn in existing_frame_dict:
                existing_frame = existing_frame_dict[fn]
                if existing_frame.suffix[1:] != output_format:
                    if convert_existing:
                        # Convert the existing frame to the desired format
                        img = cv2.imread(str(existing_frame))
                        new_frame_path = existing_frame.with_suffix(f".{output_format}")
                        cv2.imwrite(str(new_frame_path), img)
                        existing_frame.unlink()  # Remove the old frame
                        frame_paths.append(str(new_frame_path))
                        converted_frames += 1
                    else:
                        frame_paths.append(str(existing_frame))
                        existing_frames_count += 1
                else:
                    frame_paths.append(str(existing_frame))
                    existing_frames_count += 1
            else:
                frames_to_process.append(fn)
        
        if frames_to_process:
            print(f"Processing {len(frames_to_process)} new frames using {num_processes} processes")
            # Prepare arguments for parallel processing
            process_args = [(fn, video_path, video_output_dir, output_format, new_width, new_height, orig_width, orig_height) 
                           for fn in frames_to_process]
            
            # Process frames in parallel
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap(process_single_frame, process_args), 
                                   total=len(process_args), desc="Processing new frames"))
            
            new_frame_paths = [path for path in results if path is not None]
            frame_paths.extend(new_frame_paths)
            new_frames = len(new_frame_paths)
            
            # Check for failed frames
            failed_frames = len(frames_to_process) - len(new_frame_paths)
            if failed_frames > 0:
                inconsistencies.append(f"Failed to process {failed_frames} frames")
    
    frame_paths.sort(key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    
    return frame_paths, new_height, new_width, inconsistencies, {
        'new_frames': new_frames,
        'converted_frames': converted_frames,
        'existing_frames': existing_frames_count
    }

def process_video(video_path, output_dir, num_processes=None):
    try:
        frame_paths, video_height, video_width, inconsistencies, frame_stats = process_video_for_sam2(
            video_path, output_dir, output_format='jpg', num_processes=num_processes)

        print(f"Frame processing summary:")
        print(f"- New frames: {frame_stats['new_frames']}")
        print(f"- Converted frames: {frame_stats['converted_frames']}")
        print(f"- Existing frames (unchanged): {frame_stats['existing_frames']}")
        print(f"Total frames: {len(frame_paths)}")
        print(f"Video dimensions: {video_width}x{video_height}")
        print(f"Frames saved in: {Path(output_dir) / Path(video_path).stem}")
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
        else:
            print("No inconsistencies found.")
    
    except ValueError as e:
        print(f"Error processing video: {e}")



video_path = "/home/lilly/phd/crawlingtracking/data_original/food/a-02252022134507-0000.avi"
output_dir = "/home/lilly/phd/crawlingtracking/data_foranalysis/food"

# Use all available cores (up to 48) for parallel processing
# You can specify a specific number of processes by passing num_processes parameter
# e.g., process_video(video_path, output_dir, num_processes=24)
print(f"Using {min(cpu_count(), 48)} CPU cores for parallel processing")
process_video(video_path, output_dir)

###FOOD###
"/home/lilly/phd/crawlingtracking/data_original/food/a-02252022132222-0000.avi"
"/home/lilly/phd/crawlingtracking/data_original/food/a-02252022134507-0000.avi"