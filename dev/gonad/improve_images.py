import os
import glob

# Set the directory you want to start from
dir_path = 'dev/data_gonad'

# Open a random image from the folder with PIL
from PIL import Image   
import random
import numpy as np
img = Image.open(random.choice(glob.glob(os.path.join(dir_path, '*'))))




def normalize_I_mode_image(image):
    """
    Normalize an image in mode 'I' to the 0-255 range and convert it to mode 'L'.
    """
    image_array = np.array(image)
    min_val = image_array.min()
    max_val = image_array.max()
    normalized_array = ((image_array - min_val) / (max_val - min_val) * 150).astype(np.uint8)
    return Image.fromarray(normalized_array, 'L')



#Check the image mode
print(img.mode)
image1 = img.convert('L')
img2 = normalize_I_mode_image(img)

img2.save('img2.png')


#Normalize all images in folder and save them to new folder
# List all images in the first folder
savefolder = '/home/maxime/prg/phd/dev/gonad/enhanced_images/'

for img in glob.glob(os.path.join(dir_path, '*')):
    newimg = Image.open(img)
    newimg = normalize_I_mode_image(newimg)
    newimg.save(savefolder + "L_" + os.path.basename(img))
