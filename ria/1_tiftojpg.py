import cv2
import numpy as np
from pathlib import Path
import re
import tifffile
import javabridge
import bioformats
import xml.etree.ElementTree as ET
import logging	
import random
import tqdm

def get_supported_video_extensions():
    return ['.avi', '.mp4', '.mov', '.mkv', '.wmv', '.flv']

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.nd2']

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

def process_nd2_frame(frame):
    # Convert to 8-bit if necessary
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert to 3-channel if grayscale
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]  # Take only the first 3 channels if there are more
    
    return frame

def check_file_readability(file_path):
    if file_path.suffix.lower() == '.nd2':
        try:
            with bioformats.ImageReader(str(file_path)) as reader:
                if reader.rdr.getImageCount() == 0:
                    return False, "The ND2 file is empty or corrupted."
            return True, ""
        except Exception as e:
            return False, f"Error reading ND2 file: {str(e)}"
    elif file_path.suffix.lower() in ['.tif', '.tiff']:
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
    
    if file_path.suffix.lower() == '.nd2':
        # Process ND2 file using bioformats
        with bioformats.ImageReader(str(file_path)) as reader:
            metadata = bioformats.get_omexml_metadata(str(file_path))
            ome = ET.fromstring(metadata)
            pixels = ome.find('.//ome:Pixels', {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'})
            total_frames = int(pixels.get('SizeT'))
            orig_width = int(pixels.get('SizeX'))
            orig_height = int(pixels.get('SizeY'))
            if fps is None:
                fps = float(pixels.find('.//ome:Plane', {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}).get('DeltaT', 30))
    elif file_path.suffix.lower() in ['.tif', '.tiff']:
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
        fps = 10  # Default fps for TIF stacks and ND2 files if not specified
    
    # Process all frames
    expected_frame_numbers = range(total_frames)
    
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
        if file_path.suffix.lower() == '.nd2':
            with bioformats.ImageReader(str(file_path)) as reader:
                frame = reader.read(t=frame_number, rescale=False)
                frame = process_nd2_frame(frame)
        elif file_path.suffix.lower() in ['.tif', '.tiff']:
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
        if frame_stats['total_processed_frames'] % 100 == 0:  # Print every 10 frames
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
                if frame_stats['total_processed_frames'] % 100 == 0:  # Print every 10 frames
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

def process_file(file_path, output_dir, force_reprocess=False):
    try:
        file_path = Path(file_path)
        output_dir = Path(output_dir)
        
        frame_paths, file_height, file_width, inconsistencies, frame_stats = process_file_for_sam2(
            file_path, output_dir, output_format='jpg', force_reprocess=force_reprocess)

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
        if inconsistencies:
            print("Inconsistencies found:")
            for inc in inconsistencies:
                print(f"- {inc}")
        else:
            print("No inconsistencies found.")

    except ValueError as e:
        print(f"Error processing file: {e}")

def process_random_unprocessed_video(video_files_dir, output_dir):
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
    process_file(random_video, output_dir)



# Example usage
video_files = "/home/maxime/prg/phd/ria/data_original/AG"
save_jpg_dir = "/home/maxime/prg/phd/ria/data_foranalysis/videotojpg"

process_random_unprocessed_video(video_files, save_jpg_dir)





###ND2 stuff


if __name__ == "__main__":
    # Start Java VM
    javabridge.start_vm(class_path=bioformats.JARS)

    try:
        # Example usage
        file_path = '/home/maxime/prg/phd/ria/nt03.nd2'
        output_dir = "/home/maxime/prg/phd/ria/tstvideo"

        process_file(file_path, output_dir)

    finally:
        # Kill Java VM
        javabridge.kill_vm()

# Example usage
file_path = '/home/maxime/prg/phd/ria/nt02.nd2'
output_dir = "/home/maxime/prg/phd/ria/tstvideo"

process_file(file_path, output_dir)


import javabridge
import bioformats

javabridge.start_vm(class_path=bioformats.JARS)

try:
    with bioformats.ImageReader('/home/maxime/prg/phd/ria/nt02.nd2') as reader:
        metadata = bioformats.get_omexml_metadata('/home/maxime/prg/phd/ria/nt02.nd2')
        print(metadata)
finally:
    javabridge.kill_vm()
