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

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



default_generator = SamAutomaticMaskGenerator(sam)


img = cv2.imread('/home/maxime/prg/phd/MedSAM/assets/body_8_512.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

masks = default_generator.generate(img)
print(len(masks))
print(masks[0].keys())

wmask = masks[1]['segmentation']
bmask = masks[0]['segmentation']

wmask_01 = wmask.astype(int)
bmask_02 = np.where(bmask, 2, 0)

merged_masks = wmask_01 + bmask_02

nbrows, nbcols = merged_masks.shape



###Add "seed" to large dead zones before filling

#Find general orientation of worm
    #If mean row repeats higher, "horizontal"
    #If mean col repeats higher, "vertical"

row_one, col_one = np.where(merged_masks == 1)

row_id, row_count = np.unique(row_one, return_counts = True)
col_id, col_count = np.unique(col_one, return_counts = True)

worm_orientation = "horizontal" if np.mean(row_count) >= np.mean(col_count) else "vertical"

#For 10 pixels from the border, fill with background

row_zero, col_zero = np.where(merged_masks == 0)

coord_zero = list(zip(row_zero, col_zero))

filled_masks = merged_masks.copy()

modified_zeros = np.zeros(len(coord_zero))

for zeros in range(len(coord_zero)):
    row, col = coord_zero[zeros]
    if row <= 10 or col <= 10 or row >= nbrows - 11 or col >= nbcols - 11:
        filled_masks[row, col] = 2
        modified_zeros[zeros] = 1
        print(zeros)

# Pair each coordinate with its modified flag and filter out modified coordinates
coord_zero_noedge = [coord for coord, modified in zip(coord_zero, modified_zeros) if not modified]

#mrow_zero, mcol_zero = np.where(filled_masks == 0)

#If worm "vertical":
    #Check pixels bording the continuous zeros in a row
        #If less than 20, do nothing
            #If 2 worms (curved worm), add background seed in middle
            #If 2 backgrounds, add worm seed in middle

mrow_zero = [coord[0] for coord in coord_zero_noedge]
mcol_zero = [coord[1] for coord in coord_zero_noedge]

mrow, mrow_count = np.unique(mrow_zero, return_counts = True)

local_row = 0
local_col = 0
continuous_zeros = []

for coord in range(len(coord_zero_noedge)):
    row = coord_zero_noedge[coord][0]
    col = coord_zero_noedge[coord][1]
    if row != local_row or col != local_col + 1:
        if len(continuous_zeros) > 60:
            middle = np.floor(len(continuous_zeros)/2)
            filled_masks[continuous_zeros[int(middle)]] = 1
        continuous_zeros = []
        continuous_zeros.append((row, col))
        local_row = row
        local_col = col
    else:
        continuous_zeros.append((row, col))
        local_col = col




    #if not on same row or if col != local_col + 1
        #Deal with current list
            #if length continuous_list <= 20, do nothing
            #if length continuous_list >20, find middle corrdinate and change to value of 1
        #Start new list
            #continuous_zeros = []
            #append new row, col
            #local_col = col
            #local_row = row
    
    #else (same row and col == local_col +1):
        #append to continuous zeros list
        #local_col = col
        





#If worm "horizontal":
    #Same but check continuous zeros in col



tailfill = filled_masks.copy()

###Fill direct neighbors with local value
while np.any(tailfill == 0):
    for i in range(nbrows):
        for j in range(nbcols):
            local_val = tailfill[i][j]
            if local_val != 0:
                if j-1 >= 0:
                    if tailfill[i][j-1] == 0:
                        tailfill[i][j-1] = local_val
                if i-1 >= 0:
                    if tailfill[i-1][j] == 0:
                        tailfill[i-1][j] = local_val
                if j+1 < nbcols:
                    if tailfill[i][j+1] == 0:
                        tailfill[i][j+1] = local_val
                if i+1 < nbrows:
                    if tailfill[i+1][j] == 0:
                        tailfill[i+1][j] = local_val




### This function will fill the zeros between ones and twos
def floodfill_zeros(array):
    rows, cols = array.shape
    for i in range(rows):
        start_index = -1
        end_index = -1
        for j in range(cols):
            if array[i, j] == 1 and start_index == -1:
                # Find the start of a zero sequence
                start_index = j
            elif array[i, j] == 2 and start_index != -1:
                # Find the end of a zero sequence
                end_index = j
                # Calculate how many zeros to fill with 1 and 2
                zero_count = end_index - start_index - 1
                half_zero_count = zero_count // 2
                # Fill half with 1 and half with 2
                array[i, start_index + 1:start_index + 1 + half_zero_count] = 1
                array[i, start_index + 1 + half_zero_count:end_index] = 2
                # Reset the indices
                start_index = -1
                end_index = -1

    return array

# Fill the zeros in the example array
filled_array = floodfill_zeros(merged_masks)




plt.figure(figsize=(20,20))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.savefig('multi.png', bbox_inches='tight', pad_inches=0)


#Save array as img
img_array = np.uint8(tailfill) * 100 
img = Image.fromarray(img_array)
img.save('outputtail.png')