###Extract features from eggmasks
import os
import numpy as np
import cv2

#Path to the directory where the original images are stored
source_dir = '/home/maxime/prg/phd/dev/eggs/mask_cutouts'

#Save list of egg masks from file
egg_masks = '/home/maxime/prg/phd/dev/eggs/egg_masks.txt'
with open(egg_masks, 'r') as f:
    egg_masks = f.read().splitlines()


###Extract features from egg masks
img = cv2.imread(os.path.join(source_dir, egg_masks[0]))
#Convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

##Area of the egg (count non-zero pixels)
area = np.count_nonzero(img)

##Perimeter of the egg (count non-zero pixels in the border)
contours, hierarchy = cv2.findContours(img, 1, 2)
perimeter = cv2.arcLength(contours[0],True)

##Elongation of the egg (ratio of the major and minor axes of the ellipse that has the same second-moments as the region)
#Find the contours
contours, hierarchy = cv2.findContours(img, 1, 2)
#Fit an ellipse to the contour
ellipse = cv2.fitEllipse(contours[0])
#Get the major and minor axes
major_axis = max(ellipse[1])
minor_axis = min(ellipse[1])
#Calculate the elongation
elongation = major_axis/minor_axis


###Extract features from all egg masks and save dict to pickle file
egg_features = {}
for egg_mask in egg_masks:
    img = cv2.imread(os.path.join(source_dir, egg_mask))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = np.count_nonzero(img)
    contours, hierarchy = cv2.findContours(img, 1, 2)
    perimeter = cv2.arcLength(contours[0],True)
    ellipse = cv2.fitEllipse(contours[0])
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    elongation = major_axis/minor_axis
    egg_features[egg_mask] = {'area': area, 'perimeter': perimeter, 'elongation': elongation}


#Save dict to csv file
import pandas as pd
df = pd.DataFrame(egg_features).T
#Name first column "file_name"
df.index.name = 'file_name'
df.to_csv('/home/maxime/prg/phd/dev/eggs/egg_features.csv')


#Save the features to a file
import pickle
with open('/home/maxime/prg/phd/dev/eggs/egg_features.pkl', 'wb') as f:
    pickle.dump(egg_features, f)