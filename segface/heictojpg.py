import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import os
from pillow_heif import register_heif_opener
from tqdm import tqdm

# Register HEIF opener with Pillow
register_heif_opener()

def get_supported_image_extensions():
    return ['.png', '.jpg', '.jpeg', '.heic']

def check_file_readability(file_path):
    try:
        if file_path.suffix.lower() == '.heic':
            # For HEIC, we can attempt to open it with Pillow
            Image.open(file_path).load()
        else:
            # For other image formats, try opening with OpenCV
            img = cv2.imread(str(file_path))
            if img is None:
                return False, "OpenCV cannot open the file. The format might be unsupported or the file might be corrupted."
        return True, ""
    except Exception as e:
        return False, f"Error reading image file: {str(e)}"

def process_image(input_path, output_path, output_format='jpg', max_dimension=None):
    """Process a single image file."""
    try:
        # Check if output file already exists
        if output_path.exists():
            return False, "File already exists"

        # Check if file is readable
        is_readable, error_message = check_file_readability(input_path)
        if not is_readable:
            return False, error_message

        # Open and process image
        img = Image.open(input_path)
        
        # Convert HEIC to RGB if necessary
        if input_path.suffix.lower() == '.heic' or img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if max_dimension is specified
        if max_dimension:
            orig_width, orig_height = img.size
            if orig_width > max_dimension or orig_height > max_dimension:
                scale = max_dimension / max(orig_width, orig_height)
                new_width = int(orig_width * scale)
                new_height = int(orig_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save the image
        if output_format.lower() == 'jpg':
            img.save(output_path, "JPEG", quality=95)
        else:  # png
            img.save(output_path, "PNG")

        return True, "Success"
    except Exception as e:
        return False, f"Error processing image: {str(e)}"

def process_directory(input_dir, output_dir, output_format='jpg', max_dimension=None):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in get_supported_image_extensions():
        image_files.extend(input_dir.glob(f"*{ext}"))

    # Process each file
    print(f"Processing {len(image_files)} files...")
    for file_path in tqdm(image_files):
        # Create output path with new extension
        output_path = output_dir / f"{file_path.stem}.{output_format}"
        
        # Process the image
        success, message = process_image(file_path, output_path, output_format, max_dimension)
        
        if not success and message != "File already exists":
            print(f"Failed to process {file_path}: {message}")

# Example usage
input_dir = "/home/lilly/phd/segface/faces"  # Path to your image files
output_dir = "/home/lilly/phd/segface/face_jpg"  # Path to save converted images

# Process all images in the directory
process_directory(input_dir, output_dir, output_format='jpg', max_dimension=None)