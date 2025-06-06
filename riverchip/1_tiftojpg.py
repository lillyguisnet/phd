import cv2
import numpy as np
from pathlib import Path
import re
import tifffile
import logging	
import random
import tqdm

def get_supported_video_extensions():
    return ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv', '.gif']

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

def check_file_readability(file_path):
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                if len(tif.pages) == 0:
                    return False, "The TIF file is empty or corrupted."
            return True, ""
        except Exception as e:
            return False, f"Error reading TIF file: {str(e)}"
    elif file_path.suffix.lower() == '.gif':
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return False, "OpenCV cannot open the GIF file."
            ret = cap.read()[0]
            cap.release()
            return ret, "" if ret else "Cannot read frames from the GIF file."
        except Exception as e:
            return False, f"Error reading GIF file: {str(e)}"
    else:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "OpenCV cannot open the file. The format might be unsupported or the file might be corrupted."
        
        ret, frame = cap.read()
        if not ret:
            return False, "OpenCV can open the file, but cannot read frames from it. The file might be empty or corrupted."
        
        cap.release()
        return True, ""

def process_file_for_sam2(file_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False, duration_seconds=None, start_time_seconds=0, flip_vertical=False):
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    
    if output_format not in ['png', 'jpg']:
        raise ValueError("output_format must be either 'png' or 'jpg'")
    
    is_readable, error_message = check_file_readability(file_path)
    if not is_readable:
        raise ValueError(f"Cannot process file: {error_message}")
    
    # Get the subfolder name of the file path
    sub_folder_name = file_path.parent.name
    file_name = file_path.stem
    # Create the new folder name
    new_folder_name = f"{sub_folder_name}-{file_name}"
    file_output_dir = output_dir / new_folder_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # For TIF stacks, get global min/max before processing frames
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        with tifffile.TiffFile(str(file_path)) as tif:
            # Read all frames to get global min/max
            print("Calculating global min/max across all frames...")
            global_min = float('inf')
            global_max = float('-inf')
            for page in tif.pages:
                frame = page.asarray()
                frame_min = np.min(frame)
                frame_max = np.max(frame)
                global_min = min(global_min, frame_min)
                global_max = max(global_max, frame_max)
            print(f"Global range: {global_min} to {global_max}")

            total_frames_in_file = len(tif.pages)
            first_page = tif.pages[0]
            orig_width, orig_height = first_page.shape[1], first_page.shape[0]

            def process_tif_frame(frame):
                if frame.dtype == np.uint16:
                    # Linear scaling preserving relative intensities
                    # First subtract the global minimum to start from zero
                    # Then scale to use more of the 8-bit range while maintaining exact proportions
                    frame_adjusted = frame - global_min
                    scaling_factor = 255.0 / (global_max - global_min)
                    frame_8bit = (frame_adjusted * scaling_factor).astype(np.uint8)
                else:
                    frame_8bit = frame

                if len(frame_8bit.shape) == 2:
                    frame_8bit = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)

                return frame_8bit

            # For TIF stacks, assume a default FPS if not provided
            if fps is None:
                fps = 10  # Default fps for TIF stacks
            
            # Calculate frame range for TIF stacks
            if duration_seconds is not None:
                start_frame = int(start_time_seconds * fps)
                end_frame = int((start_time_seconds + duration_seconds) * fps)
                end_frame = min(end_frame, total_frames_in_file)
                start_frame = min(start_frame, total_frames_in_file - 1)
                total_frames = end_frame - start_frame
                frame_range = range(start_frame, end_frame)
                print(f"Processing TIF frames {start_frame} to {end_frame-1} (total: {total_frames} frames)")
            else:
                total_frames = total_frames_in_file
                frame_range = range(total_frames)
                print(f"Processing all {total_frames} TIF frames")

    else:
        # Process video
        cap = cv2.VideoCapture(str(file_path))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range for videos
        if duration_seconds is not None:
            if fps is None:
                fps = video_fps  # Use video's original FPS if not specified
            
            # Calculate which frames to extract based on desired FPS and duration
            start_frame = int(start_time_seconds * video_fps)
            end_time = start_time_seconds + duration_seconds
            end_frame = int(end_time * video_fps)
            end_frame = min(end_frame, total_frames_in_file)
            start_frame = min(start_frame, total_frames_in_file - 1)
            
            # Calculate frame interval based on desired FPS vs video FPS
            frame_interval = max(1, int(video_fps / fps))
            
            # Generate frame numbers to extract
            frame_range = range(start_frame, end_frame, frame_interval)
            total_frames = len(frame_range)
            
            print(f"Extracting {total_frames} frames from video:")
            print(f"  - Time range: {start_time_seconds}s to {end_time}s")
            print(f"  - Frame range: {start_frame} to {end_frame-1}")
            print(f"  - Frame interval: every {frame_interval} frames")
            print(f"  - Target FPS: {fps}, Video FPS: {video_fps}")
        else:
            if fps is None:
                fps = video_fps
            frame_interval = max(1, int(video_fps / fps))
            frame_range = range(0, total_frames_in_file, frame_interval)
            total_frames = len(frame_range)
            print(f"Processing all video frames with interval {frame_interval} (total: {total_frames} frames)")
    
    if max_dimension and (orig_width > max_dimension or orig_height > max_dimension):
        scale = max_dimension / max(orig_width, orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    else:
        new_width, new_height = orig_width, orig_height
    
    # Process frames based on the calculated frame range
    expected_frame_numbers = list(frame_range)
    
    # Check for existing frames in all supported formats
    existing_frames = []
    for ext in get_supported_image_extensions():
        existing_frames.extend(file_output_dir.glob(f"*{ext}"))
    existing_frames.sort(key=lambda x: int(re.search(r'\d+', x.stem).group()))
    
    existing_frame_numbers = [int(re.search(r'\d+', f.stem).group()) for f in existing_frames]
    
    inconsistencies = []
    
    if existing_frames and not force_reprocess:
        missing_frame_numbers = sorted(set(expected_frame_numbers) - set(existing_frame_numbers))
        if missing_frame_numbers:
            inconsistencies.append(f"Found {len(missing_frame_numbers)} missing frames. They will be processed.")
        extra_frame_numbers = sorted(set(existing_frame_numbers) - set(expected_frame_numbers))
        if extra_frame_numbers:
            inconsistencies.append(f"Found {len(extra_frame_numbers)} unexpected frames. They will be ignored.")
    else:
        missing_frame_numbers = sorted(expected_frame_numbers)
        if force_reprocess:
            inconsistencies.append("Reprocessing all frames as requested.")
        else:
            print("Processing new file.")
    
    frame_paths = []
    frame_stats = {
        'new_frames': 0,
        'converted_frames': 0,
        'existing_frames': 0,
        'total_processed_frames': 0
    }

    def process_frame(frame_number):
        nonlocal frame_stats
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            with tifffile.TiffFile(str(file_path)) as tif:
                frame = tif.pages[frame_number].asarray()
                frame = process_tif_frame(frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                return None

        if (new_width, new_height) != (orig_width, orig_height):
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Apply vertical flip if requested (flips top and bottom)
        if flip_vertical:
            frame = cv2.flip(frame, 0)  # 0 means flip around x-axis (vertical flip)

        frame_path = file_output_dir / f"{frame_number:06d}.{output_format}"

        if output_format == 'jpg':
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # png
            cv2.imwrite(str(frame_path), frame)

        frame_stats['new_frames'] += 1
        frame_stats['total_processed_frames'] += 1
        if frame_stats['total_processed_frames'] % 100 == 0:  # Print every 100 frames
            print(f"Processed frame: {frame_stats['total_processed_frames']}/{total_frames}")
        return str(frame_path)

    if force_reprocess:
        print("Reprocessing all frames")
        for fn in expected_frame_numbers:
            path = process_frame(fn)
            if path:
                frame_paths.append(path)
            else:
                inconsistencies.append(f"Failed to process frame {fn}")
    else:
        existing_frame_dict = {int(re.search(r'\d+', f.stem).group()): f for f in existing_frames}
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
                        frame_stats['converted_frames'] += 1
                    else:
                        frame_paths.append(str(existing_frame))
                        frame_stats['existing_frames'] += 1
                else:
                    frame_paths.append(str(existing_frame))
                    frame_stats['existing_frames'] += 1
                frame_stats['total_processed_frames'] += 1
                if frame_stats['total_processed_frames'] % 100 == 0:  # Print every 100 frames
                    print(f"Processed frame: {frame_stats['total_processed_frames']}/{total_frames}")
            else:
                path = process_frame(fn)
                if path:
                    frame_paths.append(path)
                else:
                    inconsistencies.append(f"Failed to process new frame {fn}")

    frame_paths.sort(key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))

    if 'cap' in locals():
        cap.release()

    return frame_paths, new_height, new_width, inconsistencies, frame_stats

def process_file(file_path, output_dir, force_reprocess=False, duration_seconds=None, start_time_seconds=0, fps=None, flip_vertical=False):
    try:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        frame_paths, file_height, file_width, inconsistencies, frame_stats = process_file_for_sam2(
            file_path, output_dir, fps=fps, output_format='jpg', force_reprocess=force_reprocess, 
            duration_seconds=duration_seconds, start_time_seconds=start_time_seconds, flip_vertical=flip_vertical)

        # Get the subfolder name of the file path
        sub_folder_name = file_path.parent.name
        file_name = file_path.stem
        # Create the correct folder name
        correct_folder_name = f"{sub_folder_name}-{file_name}"
        
        print(f"\nFrame processing summary:")
        print(f"- New frames: {frame_stats['new_frames']}")
        print(f"- Converted frames: {frame_stats['converted_frames']}")
        print(f"- Existing frames (unchanged): {frame_stats['existing_frames']}")
        print(f"Total frames processed: {frame_stats['total_processed_frames']}")
        print(f"Total frames in output: {len(frame_paths)}")
        print(f"File dimensions: {file_width}x{file_height}")
        print(f"Frames saved in: {output_dir / correct_folder_name}")
        if duration_seconds is not None:
            print(f"Duration processed: {duration_seconds} seconds starting at {start_time_seconds}s")
        if flip_vertical:
            print("Applied vertical flip to all frames")
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
        else:
            print("No inconsistencies found.")

    except ValueError as e:
        print(f"Error processing file: {e}")

def process_random_unprocessed_video(video_files_dir, output_dir, duration_seconds=None, start_time_seconds=0, fps=None, flip_vertical=False):
    video_files_dir = Path(video_files_dir)
    output_dir = Path(output_dir)

    # Get all video files
    all_videos = []
    for ext in get_supported_video_extensions() + get_supported_image_extensions():
        all_videos.extend(video_files_dir.glob(f"**/*{ext}"))

    # Get all processed videos
    processed_videos = set(dir.name.split('-')[1] for dir in output_dir.iterdir() if dir.is_dir())

    # Filter out processed videos
    unprocessed_videos = [video for video in all_videos if video.stem not in processed_videos]

    if not unprocessed_videos:
        print("All videos have been processed.")
        return

    # Select a random unprocessed video
    random_video = random.choice(unprocessed_videos)

    print(f"Processing video: {random_video}")
    process_file(random_video, output_dir, duration_seconds=duration_seconds, start_time_seconds=start_time_seconds, fps=fps, flip_vertical=flip_vertical)
    print(f"Done processing video: {random_video}")
    return random_video




video_files = "/home/lilly/phd/riverchip/data_original"
#"/home/lilly/phd/ria/tst_free/original"

save_jpg_dir = "/home/lilly/phd/riverchip/data_foranalysis/videotojpg"
#"/home/lilly/phd/ria/tst_free/tojpg"

# Example: Extract 60 seconds with vertical flip (top and bottom swapped)
vid = process_random_unprocessed_video(video_files, save_jpg_dir, duration_seconds=60, start_time_seconds=0, fps=10, flip_vertical=True)

# Extract entire video (original behavior)
#vid = process_random_unprocessed_video(video_files, save_jpg_dir)

# Alternative examples:
# Process a specific file with duration limit and vertical flip
# process_file("/path/to/video.mp4", save_jpg_dir, duration_seconds=60, start_time_seconds=0, fps=10, flip_vertical=True)
