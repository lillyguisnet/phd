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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def plotmasks(img, masks, savename):
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(savename + '.png', bbox_inches='tight', pad_inches=0)

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
default_generator = SamAutomaticMaskGenerator(sam)

manythings = []
with open("manythings.txt", "r") as file:
    for line in file:
        # Remove the newline character at the end of each line
        manythings.append(line.rstrip('\n'))


def image_preds(png):

    img = cv2.imread(png)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(img)

    listof_preds = []
    for pred in range(len(masks)):
        listof_preds.append(masks[pred])

    return listof_preds

def save_preds(path_list, picklename):

    listof_imgpreds = []

    for file in range(len(path_list)):
        listof_imgpreds.append(image_preds(path_list[file]))

    with open(picklename + ".pkl", "wb") as file:
        pickle.dump(listof_imgpreds, file)

    return

#save_preds(manythings, 'multimask_preds')

with open("multimask_preds.pkl", "rb") as file:
    mm = pickle.load(file)


###WORM
total_length = sum(len(sublist) for sublist in mm)
average_length = total_length / len(mm)

index_of_longest = max(enumerate(mm), key=lambda x: len(x[1]))[0]
longest_sublist = len(max(mm, key=len))

list_stab_worm = [sublist[1]['stability_score'] for sublist in mm]
av_stab_worm = sum(list_stab_worm)/len(list_stab_worm)
min_stab_worm = min(list_stab_worm)
max_stab_worm = max(list_stab_worm)

list_iou_worm = [sublist[1]['predicted_iou'] for sublist in mm]
av_iou_worm = sum(list_iou_worm)/len(list_iou_worm)
min_iou_worm = min(list_iou_worm)
max_iou_worm = max(list_iou_worm)


###BG
list_stab_bg = [sublist[0]['stability_score'] for sublist in mm]
av_stab_bg = sum(list_stab_bg)/len(list_stab_bg)
min_stab_bg = min(list_stab_bg)
max_stab_bg = max(list_stab_bg)

list_iou_bg = [sublist[0]['predicted_iou'] for sublist in mm]
av_iou_bg = sum(list_iou_bg)/len(list_iou_bg)
min_iou_bg = min(list_iou_bg)
max_iou_bg = max(list_iou_bg)


###ANO
list_stab_ano = []
for sublist in mm:
    # Loop through each dictionary from the third entry onwards
    for entry in sublist[2:]:
        list_stab_ano.append(entry['stability_score'])
av_stab_ano = sum(list_stab_ano)/len(list_stab_ano)
min_stab_ano = min(list_stab_ano)
max_stab_ano = max(list_stab_ano)

list_iou_ano = []
for sublist in mm:
    # Loop through each dictionary from the third entry onwards
    for entry in sublist[2:]:
        list_iou_ano.append(entry['predicted_iou'])
av_iou_ano = sum(list_iou_ano)/len(list_iou_ano)
min_iou_ano = min(list_iou_ano)
max_iou_ano = max(list_iou_ano)



for n in range(len(manythings)):
    png = random.choice(manythings)
    img = cv2.imread(manythings[n])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(img)
    savename = "/home/maxime/prg/phd/dev/sambody/tstwormmasks/" + manythings[n].split(os.sep)[-1]
    wmask = masks[1]['segmentation']
    wmask_01 = np.where(wmask == 1, 1, 0)
    img_array = np.uint8(wmask_01) * 150 
    img = Image.fromarray(img_array)
    img.save(savename)




img = cv2.imread("/home/maxime/prg/phd/dev/data/c_p3_01a_d5.png")

img = cv2.imread(manythings[129])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
masks = default_generator.generate(img)
print(len(masks))
#print(masks[0].keys())
#plotmasks(img, masks, 'masks_c_p1_03_d1.png')

bmask = masks[0]['segmentation']
wmask = masks[1]['segmentation']
nmask = masks[2]['segmentation']
omask = masks[3]['segmentation']
lmask = masks[4]['segmentation']
wmask = wmask.astype(int) - omask.astype(int)
bmask_02 = np.where(bmask, 2, 0)
wmask_01 = np.where(wmask, 1, 0)
nmask_02 = np.where(nmask, 2, 0)
omask_03 = np.where(omask, 3, 0)
lmask_03 = np.where(lmask, 3, 0)

merged_masks = wmask_01 + bmask_02 + omask_03

nbrows, nbcols = merged_masks.shape

#dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])

img_array = np.uint8(lmask_03) * 100 
img = Image.fromarray(img_array)
img.save('lmask_03.png')


""""
Next masks if:
    - Touches an edge
""""