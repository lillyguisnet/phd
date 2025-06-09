import os
import glob
import cv2
import numpy as np
import re
from tqdm import tqdm

def extract_frame_number(filename):
    """Extract frame number from filename like MMH22320250523_010019.tif"""
    # Find the last sequence of digits before .tif
    match = re.search(r'(\d+)\.tif$', filename)
    if match:
        return int(match.group(1))
    return 0

def convert_tiff_to_jpg(source_dir, dest_dir, max_frames=600):
    """Convert first 600 TIFF frames to JPG for each folder"""
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all subdirectories in source directory
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(subdirs)} folders to process:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    for subdir in subdirs:
        print(f"\nProcessing folder: {subdir}")
        
        source_folder = os.path.join(source_dir, subdir)
        dest_folder = os.path.join(dest_dir, subdir)
        
        # Create destination subfolder
        os.makedirs(dest_folder, exist_ok=True)
        
        # Get all TIFF files in the folder
        tiff_pattern = os.path.join(source_folder, "*.tif")
        tiff_files = glob.glob(tiff_pattern)
        
        if not tiff_files:
            print(f"  No TIFF files found in {subdir}")
            continue
        
        # Sort files by frame number
        tiff_files.sort(key=lambda x: extract_frame_number(os.path.basename(x)))
        
        print(f"  Found {len(tiff_files)} TIFF files")
        
        # Process first 600 frames (or all if less than 600)
        frames_to_process = min(max_frames, len(tiff_files))
        print(f"  Converting first {frames_to_process} frames...")
        
        converted_count = 0
        for i, tiff_file in enumerate(tqdm(tiff_files[:frames_to_process], desc="  Converting")):
            try:
                # Read TIFF image using OpenCV with UNCHANGED flag to preserve bit depth
                img = cv2.imread(tiff_file, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"    Error: Could not read {tiff_file}")
                    continue
                
                # Debug info for first image
                if i == 0:
                    print(f"    First image info - Shape: {img.shape}, Dtype: {img.dtype}, Min: {img.min()}, Max: {img.max()}")
                
                # Convert to 8-bit using actual dynamic range of the image
                if img.dtype != np.uint8:
                    img_min = float(img.min())
                    img_max = float(img.max())
                    
                    if img_max > img_min:
                        # Normalize to 0-255 range using actual min/max values
                        img = ((img.astype(np.float32) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
                    else:
                        # If all pixels have the same value, create a uniform gray image
                        img = np.full_like(img, 128, dtype=np.uint8)
                
                # Create JPG filename with same base name but .jpg extension
                base_name = os.path.splitext(os.path.basename(tiff_file))[0]
                jpg_filename = f"{base_name}.jpg"
                jpg_path = os.path.join(dest_folder, jpg_filename)
                
                # Save as JPG with high quality
                cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                converted_count += 1
                        
            except Exception as e:
                print(f"    Error converting {tiff_file}: {str(e)}")
                continue
        
        print(f"  Successfully converted {converted_count} frames to {dest_folder}")

if __name__ == "__main__":
    # Define source and destination directories
    source_directory = r"Z:\Hannah's Data\raw_data\calcium imaging\Calcium Videos to Analyze_TIFF"
    destination_directory = r"Z:\Hannah's Data\raw_data\calcium imaging\jpgconvert"
    
    print("TIFF to JPG Converter")
    print("=" * 50)
    print(f"Source: {source_directory}")
    print(f"Destination: {destination_directory}")
    print(f"Max frames per folder: 600")
    print("=" * 50)
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory does not exist: {source_directory}")
        exit(1)
    
    # Run the conversion
    convert_tiff_to_jpg(source_directory, destination_directory, max_frames=600)
    
    print("\nConversion completed!") 