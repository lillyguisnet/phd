"""
This script converts a video to a series of images.
The first step in the analysis pipeline is to convert the original videos to a series of jpg images.
This is necessary because the Segment Anything Model (SAM) requires images as input.
Images are automatically named using the frame number in the proper format and folder corresponding to the video file name.
"""

import cv2
import numpy as np
from pathlib import Path
import re

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


def process_video_for_sam2(video_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
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
    
    def process_frame(frame_number):
        nonlocal new_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None
        
        if (new_width, new_height) != (orig_width, orig_height):
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        frame_path = video_output_dir / f"{frame_number:06d}.{output_format}"
        
        if output_format == 'jpg':
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # png
            cv2.imwrite(str(frame_path), frame)
        
        new_frames += 1
        print(new_frames)
        return str(frame_path)
    
    if force_reprocess:
        print("Reprocessing all frames")
        frame_paths = [process_frame(fn) for fn in expected_frame_numbers if process_frame(fn) is not None]
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
                        converted_frames += 1
                    else:
                        frame_paths.append(str(existing_frame))
                        existing_frames_count += 1
                else:
                    frame_paths.append(str(existing_frame))
                    existing_frames_count += 1
            else:
                path = process_frame(fn)
                if path:
                    frame_paths.append(path)
                else:
                    inconsistencies.append(f"Failed to process new frame {fn}")
    
    frame_paths.sort(key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    
    cap.release()
    
    return frame_paths, new_height, new_width, inconsistencies, {
        'new_frames': new_frames,
        'converted_frames': converted_frames,
        'existing_frames': existing_frames_count
    }

""" 
def process_video_for_sam2(video_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
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
    
    def process_frame(frame_number):
        nonlocal new_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        for _ in range(3):  # Try up to 3 times
            try:
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {frame_number}")
                    return None
                
                if (new_width, new_height) != (orig_width, orig_height):
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                frame_path = video_output_dir / f"{frame_number:06d}.{output_format}"
                
                if output_format == 'jpg':
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:  # png
                    cv2.imwrite(str(frame_path), frame)
                
                new_frames += 1
                print(new_frames)
                return str(frame_path)
            
            except cv2.error as e:
                print(f"Warning: Error processing frame {frame_number}: {str(e)}")
                # Try to skip to the next frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + 1)
        
        # If we've tried 3 times and still failed, try to reconstruct the frame
        print(f"Attempting to reconstruct frame {frame_number}")
        prev_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) - 2
        next_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
        ret, prev_img = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        ret, next_img = cap.read()
        
        if ret and prev_img is not None and next_img is not None:
            reconstructed_frame = cv2.addWeighted(prev_img, 0.5, next_img, 0.5, 0)
            frame_path = video_output_dir / f"{frame_number:06d}_reconstructed.{output_format}"
            
            if output_format == 'jpg':
                cv2.imwrite(str(frame_path), reconstructed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:  # png
                cv2.imwrite(str(frame_path), reconstructed_frame)
            
            new_frames += 1
            print(f"Reconstructed frame {frame_number}")
            return str(frame_path)
        
        print(f"Failed to process or reconstruct frame {frame_number}")
        return None
    
    if force_reprocess:
        print("Reprocessing all frames")
        frame_paths = [process_frame(fn) for fn in expected_frame_numbers if process_frame(fn) is not None]
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
                        converted_frames += 1
                    else:
                        frame_paths.append(str(existing_frame))
                        existing_frames_count += 1
                else:
                    frame_paths.append(str(existing_frame))
                    existing_frames_count += 1
            else:
                path = process_frame(fn)
                if path:
                    frame_paths.append(path)
                else:
                    inconsistencies.append(f"Failed to process new frame {fn}")
    
    frame_paths.sort(key=lambda x: int(re.search(r'\d+', Path(x).stem).group()))
    
    cap.release()
    
    return frame_paths, new_height, new_width, inconsistencies, {
        'new_frames': new_frames,
        'converted_frames': converted_frames,
        'existing_frames': existing_frames_count
    }

 """
def process_video(video_path, output_dir):
    try:
        frame_paths, video_height, video_width, inconsistencies, frame_stats = process_video_for_sam2(
            video_path, output_dir, output_format='jpg')

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




video_path = '/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022122501-0000.avi'
output_dir = "/home/maxime/prg/phd/dropletswimming/data_foranalysis/visc05" ###<------

process_video(video_path, output_dir)

###visc05###
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022165538-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022172859-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03222022173400-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022111734-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022112259-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/a-03252022112831-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022180324-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022181147-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03222022182050-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022113446-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022114209-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/b-03252022114730-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022195635-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022200832-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03222022201347-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022115515-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022120041-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/c-03252022120644-0000.avi'

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022204030-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022204518-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03222022205045-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022121224-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022121746-0000.avi'
'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022122501-0000.avi'




#List all videos in folder
import os

ngm_path = "/home/maxime/prg/phd/dropletswimming/data_original/visc05"
video_files = [os.path.join(ngm_path, file) for file in os.listdir(ngm_path) if file.endswith(".avi")]
video_files.sort()

for full_path in video_files:
    print(f"'{full_path}'")


###ALL visc05###

'/home/maxime/prg/phd/dropletswimming/data_original/visc05/d-03252022122501-0000.avi'