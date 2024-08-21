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
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#tst parameters
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch = 128,
    pred_iou_thresh=0.1,
    stability_score_thresh=0.1,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=50,  # Requires open-cv to run post-processing
)

image = cv2.imread('phd/dev/sambody/tstimgs/body_43.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = mask_generator.generate(image)


print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig('my_image9.png', bbox_inches='tight', pad_inches=0)

#Default masks for tstimgs
imgs_path = "phd/dev/sambody/tstimgs/"

default_generator = SamAutomaticMaskGenerator(sam)

import glob
import os

for img in glob.glob(imgs_path + "*.png"):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = default_generator.generate(image)
    print(len(masks))

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(imgs_path + 'defaultmasks/defaultmasks_' + img.split(os.sep)[-1], bbox_inches='tight', pad_inches=0)


#Investigate non-masked edges
worm12 = cv2.imread('phd/dev/sambody/tstimgs/body_12.png')
worm12 = cv2.cvtColor(worm12, cv2.COLOR_BGR2RGB)

masks12 = default_generator.generate(worm12)

wrmmask = masks12[1]['segmentation']

img_array = np.uint8(wrmmask) * 255 
img = Image.fromarray(img_array, 'L')
img.save('output.png')

print(len(masks12))
print(masks12[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(worm12)
show_anns(masks12)
plt.axis('off')
plt.savefig('worm12.png', bbox_inches='tight', pad_inches=0)

#Sobel filter
image = cv2.imread('phd/dev/sambody/tstimgs/body_43.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Sobel filter in the x direction
sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)

# Apply Sobel filter in the y direction
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)

# You can also combine both
sobel_combined = cv2.bitwise_or(sobelx, sobely)

# Optionally, save the result
cv2.imwrite('sobelx.png', sobelx)
cv2.imwrite('sobely.png', sobely)
cv2.imwrite('sobel_combined.png', sobel_combined)
