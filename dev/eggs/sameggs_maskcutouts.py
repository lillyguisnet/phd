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
default_generator = SamAutomaticMaskGenerator(sam)

"""
For each image:
    Get cutout image of each mask,
        except those that touch edge.
    
"""

mm=[]
with open("multilist.txt", "r") as file:
    for line in file:
        # Remove the newline character at the end of each line
        mm.append(line.rstrip('\n'))

imgor = cv2.imread(mm[0])
img = cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)
masks = default_generator.generate(img)

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


###Multimask cutouts
for png in glob.glob("/home/maxime/prg/phd/dev/data_eggs/*.png"):

    imgor = cv2.imread(png)
    img = cv2.cvtColor(imgor, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(img)
    hi, wi, _ = img.shape

    for maskpred in range(len(masks)):

        bbox = masks[maskpred]['bbox']

        if is_on_edge(*bbox, hi, wi) == True:
            continue
        
        cutout = imgor*np.expand_dims(masks[maskpred]['segmentation'], axis=-1)
        x, y, w, h = bbox
        cropped_image = cutout[y:y+h, x:x+w]
        savename = "c" + str(maskpred) + "_" + png.split(os.sep)[-1]
        saveloc = "/home/maxime/prg/phd/dev/eggs/mask_cutouts/" + savename

        cv2.imwrite(saveloc, cropped_image)





import ndjson

def process_ndjson_file(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)

    results = []
    for item in data:
        result = {'maskid': item['data_row']['external_id'], 'classid': '', 'wormpart': ''}
        
        classifications = item['projects']['cls0mnls703tz0754c8odbc5n']['labels'][0]['annotations']['classifications']

        for classification in classifications:
            if classification['name'] == 'id':
                classid = classification['radio_answer']['name'].lower()
                result['classid'] = classid
            elif classification['name'] == 'worm_part':
                wormpart_answers = classification.get('checklist_answers', [])
                if wormpart_answers:
                    wormpart = wormpart_answers[0]['name'].lower()
                    result['wormpart'] = wormpart

        # Apply conditional logic for 'wormpart'
        if not result['wormpart']:
            if result['classid'] == 'worm':
                result['wormpart'] = 'whole'
            elif result['classid'] == 'notworm':
                result['wormpart'] = ''

        results.append(result)

    return results

# Replace 'your_file_path.ndjson' with the path to your ndjson file
file_path = "/home/maxime/prg/phd/dev/sambody/Export v2 project - dev_cutouts - 1_30_2024.ndjson"
processed_data = process_ndjson_file(file_path)
print(processed_data)

with open('multimask_cutouts_classification.pkl', 'wb') as file:
    pickle.dump(processed_data, file)





def split_and_copy_images(image_data, source_folder, train_folder, val_folder, val_percentage):
    # Create subfolders if they don't exist
    for folder in [train_folder, val_folder]:
        for subfolder in ['notworm', 'worm_any']:
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
    copy_files(set(worm_images) - set(worm_val), 'worm_any', train_folder)
    copy_files(worm_val, 'worm_any', val_folder)

# Example usage
image_data = processed_data 
source_folder = "/home/maxime/prg/phd/dev/sambody/multimask_cutouts"
train_folder = "/home/maxime/prg/phd/dev/sambody/multimask_classification_training/train"
val_folder = "/home/maxime/prg/phd/dev/sambody/multimask_classification_training/val"
val_percentage = 0.2  # 20% of the images go to the validation set

split_and_copy_images(image_data, source_folder, train_folder, val_folder, val_percentage)


