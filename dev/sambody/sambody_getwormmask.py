import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torchvision
from torchvision import datasets, models, transforms
import time
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import skimage
from skimage.morphology import medial_axis, skeletonize
from skimage.measure import label
import scipy
from scipy.ndimage import convolve
import glob
import os
import pickle
import random
from skimage.measure import label
from scipy.ndimage import convolve
import pandas as pd
from sklearn.metrics import mean_squared_error

sam_checkpoint = "phd/samdownloadvith/sam_vit_h_4b8939.pth"
model_type = "vit_h"
samdevice = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=samdevice)
sam_generator = SamAutomaticMaskGenerator(sam)

classifdevice = torch.device("cuda:0")
""" worm_noworm_classif_model = torch.jit.load('/home/maxime/prg/phd/dev/sambody/worm_noworm_classifier_vith_perfect_tscript.pt')
worm_noworm_classif_model = worm_noworm_classif_model.to(classifdevice) """
classif_weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
worm_noworm_classif_model = torchvision.models.vit_h_14(weights=classif_weights)
num_ftrs = worm_noworm_classif_model.heads.head.in_features
worm_noworm_classif_model.heads.head = nn.Linear(num_ftrs, 2)
worm_noworm_classif_model = worm_noworm_classif_model.to(classifdevice)
worm_noworm_classif_model.load_state_dict(torch.load('/home/maxime/prg/phd/dev/sambody/worm_noworm_classifier_vith_perfect_weights.pth', map_location=classifdevice))


worm_noworm_classif_model.eval()
class_names = ["notworm", "worm_any"]
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

###Get a final worm mask for each image###
    #Get SAM masks for each image
    #Remove border masks
        #Class masks
            #If more than 1 worm mask
                #
        #Keep worm mask
    #Keep worm mask


###Get SAM masks for all images
all_imgs = "/home/maxime/prg/phd/dev/data/"
#all_imgs_masks = []

for path in glob.glob(all_imgs + "*.png"):
    print(path)
    image_or = cv2.imread(path)
    image = cv2.cvtColor(image_or, cv2.COLOR_BGR2RGB)
    masks = sam_generator.generate(image)
    all_imgs_masks.append({"img_id": path, "masks": masks})


with open("/home/maxime/prg/phd/dev/sambody/allimages_sam_allmasks.pkl", "wb") as file:
    pickle.dump(all_imgs_masks, file)

#lengths = [len(item["masks"]) for item in allsam]
#>>> Counter(lengths)
#Counter({2: 111, 3: 90, 4: 48, 5: 29, 6: 10, 7: 4, 8: 2, 1: 1, 15: 1})
    

###Filter masks
with open("/home/maxime/prg/phd/dev/sambody/allimages_sam_allmasks.pkl", 'rb') as file:
    allsam = pickle.load(file)

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


#cutout = cv2.imread("/home/maxime/prg/phd/dev/sambody/multimask_cutouts/c13_a_p1_01_d1.png")
def isworm_mask(original_image_path, sam_segmentation):

    cutout = cv2.imread(original_image_path)*np.expand_dims(sam_segmentation, axis=-1)
    cropped_cutout = cutout[y:y+h, x:x+w] #?????? Yields many worms to be notworm (?)
    #cv2.imwrite("crop.png",cropped_cutout)
    img = Image.fromarray(cropped_cutout)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(classifdevice)

    outputs = worm_noworm_classif_model(img)
    _, preds = torch.max(outputs, 1)
    model_classification = class_names[preds]

    if model_classification == "worm_any":
        return True
    if model_classification == "notworm":
        return False


##Get real worm mask
def get_real_worm(img_masks, thewormmask, imgw, imgh):  

    #Merge all found worm masks
    sum_masks = np.sum([img_masks[segs]['segmentation'] for segs in thewormmask], axis = 0)
    sum_masks[sum_masks != 0] = 1
    #Get external contours of blobs
    contours, hierarchy = cv2.findContours((sum_masks * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #If countours == 1, masks were worm submasks and were merged. Take the merged mask as worm.
    if len(contours) == 1:
        zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
        final_worm = cv2.drawContours(zeroimg, [contours[0]], -1, color=1, thickness=cv2.FILLED)

        return final_worm        
    
    #If countours > 1, take best fit quadratic mask as worm.
    else:
        #Get the worm blob with best quadratic fit
        quadfit_mse = []
        for i, contour in enumerate(contours):
            print(i, len(contour))
            if len(contour) < 200:
                quadfit_mse.append(1000000)
            else:
                #Draw the blob
                zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
                blobarr = cv2.drawContours(zeroimg, [contour], -1, color=1, thickness=cv2.FILLED)
                #Get blob medial axis
                medial_axis, distance = skimage.morphology.medial_axis(blobarr > 0, return_distance=True)
                print(np.sum(blobarr))
                print(np.sum(medial_axis))
                if np.sum(medial_axis) < 200:
                    quadfit_mse.append(1000000)
                else:
                    structuring_element = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
                    neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
                    end_points = np.where(neighbours == 11, 1, 0)
                    branch_points = np.where(neighbours > 12, 1, 0)
                    labeled_branches = label(branch_points, connectivity=2)
                    branch_indices = np.argwhere(labeled_branches > 0)
                    end_indices = np.argwhere(end_points > 0)
                    indices = np.concatenate((branch_indices, end_indices), axis=0)
                    paths = []
                    for start in range(len(indices)):
                        for end in range(len(indices)):
                            startid = tuple(indices[start])
                            endid = tuple(indices[end])
                            route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
                            length = len(route)
                            paths.append([startid, endid, length, route, weight])
                    longest_length = max(paths, key=lambda x: x[2])
                    pruned_mediala = np.zeros((imgh, imgw), dtype=np.uint8)
                    for coord in range(len(longest_length[3])):
                        pruned_mediala[longest_length[3][coord]] = 1
                    #Get distances along the medial axis
                    medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
                    #Medial axis length (length of worm)
                    medialaxis_length = 0 + np.arange(0, len(medial_axis_distances_sorted))
                    #Quadratic fit
                    coefs = np.polyfit(medialaxis_length, medial_axis_distances_sorted, 2)
                    p = np.poly1d(coefs)
                    y_pred = p(medialaxis_length)
                    mse = mean_squared_error(medial_axis_distances_sorted, y_pred)
                    quadfit_mse.append(mse)

        #Draw final worm as blob with the best quadratic fit
        wormcontour = contours[np.argmin(quadfit_mse)]
        zeroimg = np.zeros((imgh, imgw), dtype=np.uint8)
        final_worm = cv2.drawContours(zeroimg, [wormcontour], -1, color=1, thickness=cv2.FILLED)

        return final_worm



allimages_finalwormmask = []
manyworms = []
zeroworm = []

#/home/maxime/prg/phd/dev/sambody/final_worm_masks/fm_c_p1_02_d1.png

""" png = 184
original_image_path = img_path
sam_segmentation = img_masks[3]['segmentation'] """

for png in range(len(allsam)):
    print(png)
    img_path = allsam[png]['img_id']
    if img_path == "/home/maxime/prg/phd/dev/data/c_p1_02_d1.png":
        continue
    print(img_path)
    img_masks = allsam[png]['masks']
    imgh, imgw, _ = cv2.imread(img_path).shape

    thewormmask = []

    for maskpred in range(len(img_masks)):

        ###Remove border masks
        x, y, w, h = bbox = img_masks[maskpred]['bbox']
        if is_on_edge(*bbox, imgw, imgh) == True:
            continue

        ###Class masks
        #if isworm_mask(img_path, img_masks[maskpred]['segmentation']) == False:
            #continue

        if isworm_mask(img_path, img_masks[maskpred]['segmentation']) == True:
            thewormmask.append(maskpred)

    if len(thewormmask) > 1:
        real_worm = get_real_worm(img_masks, thewormmask, imgw, imgh)
        manyworms.append(img_path)
        print("Many: " + str(len(thewormmask)))
    
    if len(thewormmask) == 0:
        #Fix it
        zeroworm.append(img_path)
        print("No mask:" + str(len(img_masks)))
        continue  

    if len(thewormmask) == 1:
        real_worm = img_masks[thewormmask[0]]['segmentation'] 
    
    allimages_finalwormmask.append({"img_id": img_path, "final_wormmask": real_worm})


with open("/home/maxime/prg/phd/dev/sambody/final_worm_mask.pkl", "wb") as file:
    pickle.dump(allimages_finalwormmask, file)


for dict_ in allimages_finalwormmask:
    if dict_.get('img_id') == img_path:
        wmask_01 = dict_.get('final_wormmask')


###Test crap

max_dicts = 0
for item in allimages_finalwormmask:
    num_dicts = len(item["final_wormmask"])
    print(num_dicts)
    if num_dicts > max_dicts:
        max_dicts = num_dicts

#43 Many 8 noworm
#37 Many 6 noworm
    

medial_axis, distance = skimage.morphology.medial_axis(img_masks[3]['segmentation'] > 0, return_distance=True)

# Function to compute the medial axis and find branch and end points
    # Define a structuring element for the convolution operation to find neighbours
structuring_element = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)
    # Convolve skeleton with the structuring element to count neighbours
neighbours = convolve(medial_axis.astype(np.uint8), structuring_element, mode='constant', cval=0)
    # End points have one neighbour
end_points = np.where(neighbours == 11, 1, 0)
    # Branch points have more than two neighbours
branch_points = np.where(neighbours > 12, 1, 0)

labeled_branches = label(branch_points, connectivity=2)
# Extract the indices of branch and end points
branch_indices = np.argwhere(labeled_branches > 0)
end_indices = np.argwhere(end_points > 0)
# Include branch points as potential start/end points for routes
indices = np.concatenate((branch_indices, end_indices), axis=0)

paths = []
for start in range(len(indices)):
    for end in range(len(indices)):
        startid = tuple(indices[start])
        endid = tuple(indices[end])
        route, weight = skimage.graph.route_through_array(np.invert(medial_axis), startid, endid)
        length = len(route)
        paths.append([startid, endid, length, route, weight])


longest_length = max(paths, key=lambda x: x[2])

pruned_mediala = np.zeros(img_masks[3]['segmentation'].shape)

for coord in range(len(longest_length[3])):
    pruned_mediala[longest_length[3][coord]] = 1 

medial_axis_distances = distance[np.array(pruned_mediala, dtype=bool)]
medial_axis_distances_sorted = [distance[pt[0], pt[1]] for pt in longest_length[3]]
wormlength = 0 + np.arange(0, len(medial_axis_distances_sorted))

masks1 = medial_axis_distances_sorted
masks2 = medial_axis_distances_sorted
masks3 = medial_axis_distances_sorted

#Peaks - Peaks too small
peaks, properties = scipy.signal.find_peaks(masks3, prominence=35)

#Total variation - Not different enough
sum(abs(y - x) for x, y in zip(masks3[:-1], masks3[1:]))

#Autocorrelation - Not different enough
pd.Series(masks3).autocorr(lag=1)

#Quadratic fit
def fit_quadratic_and_calculate_mse(x, y):
    # Fit the quadratic polynomial
    coefs = np.polyfit(x, y, 2)
    p = np.poly1d(coefs)
    # Predict y values using the polynomial
    y_pred = p(x)
    # Calculate MSE
    mse = mean_squared_error(y, y_pred)
    return mse, p

mse1, p1 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks1)), masks1)
mse2, p2 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks2)), masks2)
mse3, p3 = fit_quadratic_and_calculate_mse(0 + np.arange(0, len(masks3)), masks3)


im1 = img_masks[0]['segmentation']
im2 = img_masks[1]['segmentation']
im3 = img_masks[2]['segmentation']

im1 = img_masks[thewormmask[0]]['segmentation']
im2 = img_masks[thewormmask[1]]['segmentation']
im3 = img_masks[thewormmask[2]]['segmentation']

img_array = np.uint8(real_worm) * 70 
img = Image.fromarray(img_array)
img.save('real_worm.png')

img_array = np.uint8(im2) * 100 
img = Image.fromarray(img_array)
img.save('im2.png')

img_array = np.uint8(im3) * 100 
img = Image.fromarray(img_array)
img.save('im3.png')
