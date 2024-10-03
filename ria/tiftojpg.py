import cv2
import numpy as np
from pathlib import Path
import re
import tifffile

def get_supported_video_extensions():
    return ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

def process_tif_frame(frame):
    # Check if the frame is 16-bit
    if frame.dtype == np.uint16:
        # Normalize to 0-255 range
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to 8-bit
        frame_8bit = frame_normalized.astype(np.uint8)
    else:
        frame_8bit = frame

    # Check if the image is grayscale (single channel)
    if len(frame_8bit.shape) == 2:
        # Convert to 3-channel grayscale
        frame_8bit = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2BGR)

    return frame_8bit

def check_file_readability(file_path):
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with tifffile.TiffFile(str(file_path)) as tif:
                if len(tif.pages) == 0:
                    return False, "The TIF file is empty or corrupted."
            return True, ""
        except Exception as e:
            return False, f"Error reading TIF file: {str(e)}"
    else:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            return False, "OpenCV cannot open the file. The format might be unsupported or the file might be corrupted."
        
        ret, frame = cap.read()
        if not ret:
            return False, "OpenCV can open the file, but cannot read frames from it. The file might be empty or corrupted."
        
        cap.release()
        return True, ""

def process_file_for_sam2(file_path, output_dir, fps=None, max_dimension=None, output_format='jpg', force_reprocess=False, convert_existing=False):
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
    
    if file_path.suffix.lower() in ['.tif', '.tiff']:
        # Process TIF stack
        with tifffile.TiffFile(str(file_path)) as tif:
            total_frames = len(tif.pages)
            first_page = tif.pages[0]
            orig_width, orig_height = first_page.shape[1], first_page.shape[0]
    else:
        # Process video
        cap = cv2.VideoCapture(str(file_path))
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
    
    if fps is None and 'video_fps' in locals():
        fps = video_fps
    elif fps is None:
        fps = 30  # Default fps for TIF stacks
    
    frame_interval = max(int(total_frames / (fps * (total_frames / 30))), 1)
    
    # Check for existing frames in all supported formats
    existing_frames = []
    for ext in get_supported_image_extensions():
        existing_frames.extend(file_output_dir.glob(f"*{ext}"))
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

        frame_path = file_output_dir / f"{frame_number:06d}.{output_format}"

        if output_format == 'jpg':
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # png
            cv2.imwrite(str(frame_path), frame)

        frame_stats['new_frames'] += 1
        frame_stats['total_processed_frames'] += 1
        print(f"Processed frame: {frame_stats['total_processed_frames']}")
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

def process_file(file_path, output_dir, force_reprocess=False):
    try:
        frame_paths, file_height, file_width, inconsistencies, frame_stats = process_file_for_sam2(
            file_path, output_dir, output_format='jpg', force_reprocess=force_reprocess)

        print(f"Frame processing summary:")
        print(f"- New frames: {frame_stats['new_frames']}")
        print(f"- Converted frames: {frame_stats['converted_frames']}")
        print(f"- Existing frames (unchanged): {frame_stats['existing_frames']}")
        print(f"Total frames processed: {frame_stats['total_processed_frames']}")
        print(f"Total frames in output: {len(frame_paths)}")
        print(f"File dimensions: {file_width}x{file_height}")
        print(f"Frames saved in: {Path(output_dir) / Path(file_path).stem}")
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
        else:
            print("No inconsistencies found.")

    except ValueError as e:
        print(f"Error processing file: {e}")

# Example usage
file_path = '/home/maxime/prg/phd/ria/MMH99_10s_20190813_03.tif'  # Can be a TIF stack or a video file
output_dir = "/home/maxime/prg/phd/ria/tstvideo"

process_file(file_path, output_dir, force_reprocess=True)