import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

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
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    ax.imshow(img)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "/home/maxime/prg/phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#tst parameters
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    #points_per_batch = 128,
    pred_iou_thresh=0.6,
    stability_score_thresh=0.6,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  # Requires open-cv to run post-processing #Based on min egg size
)

#default_generator = SamAutomaticMaskGenerator(sam)

#Use first frame in pickle file
import pickle
with open("/home/maxime/prg/phd/dropletswimming/tst_frames.pkl", "rb") as f:
    frames = pickle.load(f)

image = frames[563]

#image = cv2.imread('/home/maxime/prg/phd/dev/data_gonad/a_p1_01_d1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

x = 500
image = image[x:-x, x:-x, :]

masks = mask_generator.generate(image)
#masks = default_generator.generate(image)


print(len(masks))
#print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('droplet.png', bbox_inches='tight', pad_inches=0)

##Default masks
imgs_path = "/home/maxime/prg/phd/dev/data_oro/data_oro_png/"

#default_generator = SamAutomaticMaskGenerator(sam)

import glob
import os

for img in glob.glob(imgs_path + "*.png"):
    print(img)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    #masks = default_generator.generate(image)
    print(len(masks))

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("/home/maxime/prg/phd/dev/oro/default_masks/def_" + img.split(os.sep)[-1], bbox_inches='tight', pad_inches=0)

