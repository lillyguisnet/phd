import os
from PIL import Image
import numpy as np
import tifffile

# Source and destination directories
src_dir = '/home/lilly/phd/laeya/data/data_original'
dst_dir = '/home/lilly/phd/laeya/data/data_jpg'

def convert_16bit_to_8bit(image):
    """Convert a 16-bit image to 8-bit using tifffile and proper scaling"""
    # Read with tifffile to properly handle different TIF formats
    img_array = tifffile.imread(image)
    
    # Calculate global min/max for proper scaling
    global_min = img_array.min()
    global_max = img_array.max()
    
    # Linear scaling preserving relative intensities
    # First subtract the global minimum to start from zero
    # Then scale to use the 8-bit range while maintaining exact proportions
    img_adjusted = img_array - global_min
    scaling_factor = 255.0 / (global_max - global_min)
    img_8bit = (img_adjusted * scaling_factor).astype(np.uint8)
    
    return Image.fromarray(img_8bit)

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Walk through all subdirectories
for root, dirs, files in os.walk(src_dir):
    # Get the relative path from src_dir
    rel_path = os.path.relpath(root, src_dir)
    
    # Create corresponding directory in destination
    dst_path = os.path.join(dst_dir, rel_path)
    os.makedirs(dst_path, exist_ok=True)
    
    # Process all .tif files in current directory
    for file in files:
        if file.lower().endswith('.tif'):
            # Construct full file paths
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, os.path.splitext(file)[0] + '.jpg')
            
            # Open and convert the image
            try:
                # Use tifffile for reading and our conversion function
                img = convert_16bit_to_8bit(src_file)
                
                # Save as JPEG
                img.save(dst_file, 'JPEG', quality=95)
                print(f"Converted: {src_file} -> {dst_file}")
            except Exception as e:
                print(f"Error converting {src_file}: {str(e)}")
