import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import skimage
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label
from scipy.ndimage import convolve
import scipy
import pickle
import random
import shutil

sam_checkpoint = "/home/maxime/prg/phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
#default_generator = SamAutomaticMaskGenerator(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    #points_per_batch = 128,
    pred_iou_thresh=0.95,
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1500,  # Requires open-cv to run post-processing #Based on min egg size
)

"""
For each image:
    Get cutout image of each mask,
        except those that touch edge.
    Save all mask predictions in a pickle file.
    
"""

imgor = cv2.imread(mm[0])
img = cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(img)

h, w = masks[0]['segmentation'].shape

cutout = imgor*np.expand_dims(masks[1]['segmentation'], axis=-1)
cv2.imwrite('cutout.png', cutout)

bbox = masks[1]['bbox']
np.any(np.asarray(bbox) == 0)

wmask = masks[1]['segmentation']
wmask_01 = np.where(wmask == 1, 1, 0)

img_array = np.uint8(cropped_image) * 10
img = Image.fromarray(img_array)
img.save('cropped_image.png')

def is_on_edge(x, y, w, h, img_width, img_height):
    # Check left edge
    if x == 0:
        return True
    # Check top edge
    if y == 0:
        return True
    # Check right edge
    if x + w == img_width:
        return True
    # Check bottom edge
    if y + h == img_height:
        return True
    return False


###Multimask cutouts and save predictions
all_masks = [] 
for png in glob.glob("/home/maxime/prg/phd/dev/data_oro/data_oro_png/*.png"):

    imgor = cv2.imread(png)
    img = cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img)
    all_masks.append({'filename': png, 'masks': masks})
    hi, wi, _ = img.shape

    for maskpred in range(len(masks)):

        bbox = masks[maskpred]['bbox']

        if is_on_edge(*bbox, hi, wi) == True:
            continue
        
        cutout = imgor*np.expand_dims(masks[maskpred]['segmentation'], axis=-1)
        x, y, w, h = bbox
        cropped_image = cutout[y:y+h, x:x+w]
        savename = "c" + str(maskpred) + "_" + png.split(os.sep)[-1]
        saveloc = "/home/maxime/prg/phd/dev/oro/mask_cutouts/" + savename

        cv2.imwrite(saveloc, cropped_image)


# Save all_masks list as a pickle file
with open("/home/maxime/prg/phd/dev/oro/oro_allmaskspred.pkl", "wb") as f:
    pickle.dump(all_masks, f)


###Annontate in LB
#Check masks of some images
import pickle
import numpy as np

with open("/home/maxime/prg/phd/dev/oro/oro_allmaskspred.pkl", "rb") as f:
    all_masks = pickle.load(f)

#Select an image in the list based on 'filename' key
img = '/home/maxime/prg/phd/dev/data_oro/data_oro_png/a_20220403T203017 043.png'
for i in range(len(all_masks)):
    if all_masks[i]['filename'] == img:
        print(all_masks[i])
        print(len(all_masks[i]['masks']))
    
    

###Extract and save classification data
import ndjson

def process_ndjson_file(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)

    results = []
    for item in data:
        print(item)
        result = {'maskid': item['data_row']['external_id'], 'classid': ''}
        
        classification = item['projects']['cltiqbonj01a307y96ixdd1tz']['labels'][0]['annotations']['classifications'][0]['radio_answer']['name'].lower()

        result['classid'] = classification

        results.append(result)

    return results

# Replace 'your_file_path.ndjson' with the path to your ndjson file
file_path = "/home/maxime/prg/phd/dev/oro/Export v2 project - oro_allcutouts - 3_8_2024.ndjson"
processed_data = process_ndjson_file(file_path)
print(processed_data)

with open('/home/maxime/prg/phd/dev/oro/oro_cutouts_classification.pkl', 'wb') as file:
    pickle.dump(processed_data, file)





def split_and_copy_images(image_data, source_folder, train_folder, val_folder, val_percentage):
    # Create subfolders if they don't exist
    for folder in [train_folder, val_folder]:
        for subfolder in ['notworm', 'worm']:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

    # Split data into train and validation sets
    notworm_images = [img['maskid'] for img in image_data if img['classid'] == 'notworm']
    worm_images = [img['maskid'] for img in image_data if img['classid'] == 'worm']

    notworm_val = random.sample(notworm_images, int(len(notworm_images) * val_percentage))
    worm_val = random.sample(worm_images, int(len(worm_images) * val_percentage))

    # Function to copy files to the appropriate folder
    def copy_files(images, class_type, destination_folder):
        for maskid in images:
            source_file = os.path.join(source_folder, maskid)
            if not os.path.exists(source_file):
                print(f"File not found: {source_file}")
                continue
            destination = os.path.join(destination_folder, class_type, maskid)
            shutil.copy2(source_file, destination)

    # Copy files to train and val folders
    copy_files(set(notworm_images) - set(notworm_val), 'notworm', train_folder)
    copy_files(notworm_val, 'notworm', val_folder)
    copy_files(set(worm_images) - set(worm_val), 'worm', train_folder)
    copy_files(worm_val, 'worm', val_folder)


image_data = processed_data 
source_folder = "/home/maxime/prg/phd/dev/oro/mask_cutouts"
train_folder = "/home/maxime/prg/phd/dev/oro/mask_classification_split/train"
val_folder = "/home/maxime/prg/phd/dev/oro/mask_classification_split/val"
val_percentage = 0.2  # 20% of the images go to the validation set

split_and_copy_images(image_data, source_folder, train_folder, val_folder, val_percentage)





# Copy and rename all .png files in a directory
import shutil

def copy_and_rename_pngs(src_dir, dst_dir):
    """
    Copy all .png files from src_dir to dst_dir.
    """
    # Ensure the destination directory exists, create if it doesn't
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for filename in os.listdir(src_dir):
        if filename.endswith(".png"):
            # Change the first character to 'x'
            new_filename = filename.replace(" ", "_")
            
            # Construct the full source and destination paths
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, new_filename)
            
            # Copy the file to the new destination with the new name
            shutil.copy2(src_path, dst_path)
            print(f"Copied and renamed {filename} to {new_filename}")

# Example usage
source_directory = '/home/maxime/prg/phd/dev/data_oro/data_oro_png'
destination_directory = '/home/maxime/prg/phd/dev/oro/original_imgs_fixedname'
copy_and_rename_pngs(source_directory, destination_directory)