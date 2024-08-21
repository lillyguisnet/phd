import tifffile
import numpy as np
from PIL import Image
import os

def normalize_16bit_to_8bit(image_data):
    # Find the maximum pixel value in the image
    max_val = image_data.max()
    # Normalize the image data to 255
    normalized_data = (image_data / max_val) * 255.0
    # Convert to 8-bit unsigned integer
    return normalized_data.astype(np.uint8)

# Set the directory containing your .tif files
source_dir = '/home/maxime/prg/phd/dev/data_oro'

# Set the directory where you want to save the .png files
target_dir = '/home/maxime/prg/phd/dev/data_oro/data_oro_png'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


for file_name in os.listdir(source_dir):
    if file_name.endswith('.tif'):
        file_path = os.path.join(source_dir, file_name)
        target_file_name = os.path.splitext(file_name)[0] + '.png'
        target_file_path = os.path.join(target_dir, target_file_name)

        # Read the image using tifffile
        img = tifffile.imread(file_path)

            # Convert to 8-bit (if it's not already 8-bit)
        if img.dtype == np.uint16:
            img = normalize_16bit_to_8bit(img)
            #img = (img // 256).astype(np.uint8)
            #scale_factor = np.iinfo(np.uint8).max / np.iinfo(np.uint16).max
            #img = (img * scale_factor).astype(np.uint8)

            # Convert the NumPy array to a PIL Image and save as PNG
        Image.fromarray(img).save(target_file_path, 'PNG')

print('Conversion completed.')

num_files = len([name for name in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, name))])
png_files = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])


#Open img with cv2
import cv2
img = "/home/maxime/prg/phd/dev/data_oro/data_oro_png/a_20220403T185346 001.png"
imgor = cv2.imread(img)