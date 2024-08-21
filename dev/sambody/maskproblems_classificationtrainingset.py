import pickle
import numpy as np
from PIL import Image
import os

###Save the final worm mask images
with open("/home/maxime/prg/phd/dev/sambody/final_worm_mask.pkl", "rb") as file:
    allimages_finalwormmask = pickle.load(file)


for png in range(len(allimages_finalwormmask)):
    print(png)
    path = allimages_finalwormmask[png]['img_id']
    print(path)
    wmask_01 = allimages_finalwormmask[png]['final_wormmask']
    img_array = np.uint8(wmask_01) * 100 
    img = Image.fromarray(img_array)
    savename = "/home/maxime/prg/phd/dev/sambody/final_worm_masks/fm_" + path.split("/")[-1]
    img.save(savename)
    


###Save the original images with their corresponding final worm mask images for annotation
# Define the paths to your directories
first_folder_path = '/home/maxime/prg/phd/dev/data'
second_folder_path = '/home/maxime/prg/phd/dev/sambody/final_worm_masks'
output_folder_path = '/home/maxime/prg/phd/dev/sambody/maskproblems_combined'


def normalize_I_mode_image(image):
    """
    Normalize an image in mode 'I' to the 0-255 range and convert it to mode 'L'.
    """
    image_array = np.array(image)
    min_val = image_array.min()
    max_val = image_array.max()
    normalized_array = ((image_array - min_val) / (max_val - min_val) * 150).astype(np.uint8)
    return Image.fromarray(normalized_array, 'L')


# List all images in the first folder
first_folder_images = [f for f in os.listdir(first_folder_path) if f.endswith('.png')]

for image_name in first_folder_images:
    # Construct the path to the original image and its corresponding "fm_" image
    original_image_path = os.path.join(first_folder_path, image_name)
    modified_image_name = f"fm_{image_name}"
    modified_image_path = os.path.join(second_folder_path, modified_image_name)

    # Check if the corresponding "fm_" image exists
    if os.path.exists(modified_image_path):
        # Open the images
        image1 = Image.open(original_image_path)
        image2 = Image.open(modified_image_path)

        # Normalize and convert the 'I' mode image to 'L' if necessary
        if image1.mode == 'I':
            image1 = normalize_I_mode_image(image1)
        if image2.mode == 'I':
            image2 = normalize_I_mode_image(image2)

        # Ensure both images are now in 'L' mode, converting if necessary
        if image1.mode != 'L':
            image1 = image1.convert('L')
        if image2.mode != 'L':
            image2 = image2.convert('L')

        # Calculate dimensions for the combined image
        total_width = image1.width + image2.width
        max_height = max(image1.height, image2.height)

        # Create a new image with the appropriate dimensions
        new_image = Image.new('L', (total_width, max_height))

        # Paste the original images into the new image
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

        # Save the new image
        new_image_path = os.path.join(output_folder_path, f"combined_{image_name}")
        new_image.save(new_image_path)

        print(f"Created merged image: {new_image_path}")
    else:
        print(f"No corresponding image found for {image_name}")


###Extract the classification from the NDJSON file and copy the images to the corresponding folders
import os
import json
import shutil

ndjson_path = '/home/maxime/prg/phd/dev/sambody/Export v2 project - dev_maskproblems - 2_7_2024.ndjson'
original_folder_path = '/home/maxime/prg/phd/dev/sambody/final_worm_masks'
target_base_folder_path = '/home/maxime/prg/phd/dev/sambody/maskproblems_classification_training_wormmaskonly'

# Create target folder if it doesn't exist
if not os.path.exists(target_base_folder_path):
    os.makedirs(target_base_folder_path)

# Read and process the NDJSON file
with open(ndjson_path, 'r') as file:
    for line in file:
        # Parse the JSON data from the line
        data = json.loads(line)

        # Extract the external_id
        external_id = data['data_row']['external_id']

        # Navigate through the JSON to find the assigned class value
        classifications = data['projects']['clscgrl2k04sd07035gh25njo']['labels'][0]['annotations']['classifications']
        # Assuming there is only one classification per image
        assigned_class_value = classifications[0]['radio_answer']['value'] if classifications else None

        # If there's no classification, skip this image
        if not assigned_class_value:
            print(f"No classification found for {external_id}. Skipping.")
            continue

        # Remove the 'combined_' prefix from the filename
        new_filename = external_id.replace('combined', 'fm')

        # Create the classification folder if it doesn't exist
        classification_folder_path = os.path.join(target_base_folder_path, assigned_class_value)
        if not os.path.exists(classification_folder_path):
            os.makedirs(classification_folder_path)

        # Define the source and destination file paths
        source_file_path = os.path.join(original_folder_path, new_filename)
        destination_file_path = os.path.join(classification_folder_path, new_filename)

        # Copy the file from the original folder to the new folder
        shutil.copy2(source_file_path, destination_file_path)
        print(f'Copied {external_id} to {assigned_class_value}/{new_filename}')

print('Processing complete.')



###Split the images into training and validation sets
import os
import random
import shutil
from PIL import Image
import numpy as np
import cv2

# Define the base folder path
original_folder_path = "/home/maxime/prg/phd/dev/data"
base_folder_path = '/home/maxime/prg/phd/dev/sambody/maskproblems_classification_training_wormmaskonly'
save_folder_path = "/home/maxime/prg/phd/dev/sambody/maskproblems_classificiation_split"

# Define the training and validation folder paths
training_folder_path = os.path.join(save_folder_path, 'train')
validation_folder_path = os.path.join(save_folder_path, 'val')


# Create the training and validation folders if they don't exist
for folder_path in (training_folder_path, validation_folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Define the class folders
class_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
print(f'Class folders: {class_folders}')
for folder in [training_folder_path, validation_folder_path]:
    for subfolder in class_folders:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

# Define the training and validation split ratio
validation_split_ratio = 0.2

# Split the images into training and validation sets
for class_folder in class_folders:
    # Get the list of images in the class folder
    class_images = [f for f in os.listdir(os.path.join(base_folder_path, class_folder)) if f.endswith('.png')]

    # Calculate the number of images for the validation set
    num_validation_images = int(len(class_images) * validation_split_ratio)

    # Randomly select the images for the validation set
    validation_images = random.sample(class_images, num_validation_images)

    # Create a mask of the original images with the same name and save them to the validation folder
    for image in validation_images:
        source_path = os.path.join(base_folder_path, class_folder, image)
        destination_path = os.path.join(validation_folder_path, class_folder, image)
        original_image_path = os.path.join(original_folder_path, image.replace('fm_', ''))
        #Apply the mask to the original image
        mask = np.array(Image.open(source_path))
        mask[mask > 0] = 1
        original = Image.open(original_image_path)        
        cutout = Image.fromarray(original*mask)
        cutout.save(destination_path)
           
    #Make a copy of the images not used for validation to the training folder
    for image in class_images:
        if image not in validation_images:
            source_path = os.path.join(base_folder_path, class_folder, image)
            destination_path = os.path.join(training_folder_path, class_folder, image)
            original_image_path = os.path.join(original_folder_path, image.replace('fm_', ''))
            #Apply the mask to the original image
            mask = np.array(Image.open(source_path))
            mask[mask > 0] = 1
            original = Image.open(original_image_path)
            cutout = Image.fromarray(original*mask)
            cutout.save(destination_path)
            
            
             
### Splt the images into training and validation sets with only 2 classes
            
            
# Define the base folder path
original_folder_path = "/home/maxime/prg/phd/dev/data"
base_folder_path = '/home/maxime/prg/phd/dev/sambody/maskproblems_classification_training_wormmaskonly'
save_folder_path = "/home/maxime/prg/phd/dev/sambody/maskproblems_classfication_split_2classes"

# Define the training and validation folder paths
training_folder_path = os.path.join(save_folder_path, 'train')
validation_folder_path = os.path.join(save_folder_path, 'val')

# Create the training and validation folders if they don't exist
for folder_path in (training_folder_path, validation_folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Define the class folders
class_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
print(f'Class folders: {class_folders}')
train_classes = ["normal", "problematic"]
for folder in [training_folder_path, validation_folder_path]:
    for subfolder in train_classes:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)

# Define the training and validation split ratio
validation_split_ratio = 0.2

# Split the images into training and validation sets
for class_folder in class_folders:
    # Get the list of images in the class folder
    class_images = [f for f in os.listdir(os.path.join(base_folder_path, class_folder)) if f.endswith('.png')]

    # Calculate the number of images for the validation set
    num_validation_images = int(len(class_images) * validation_split_ratio)

    # Randomly select the images for the validation set
    validation_images = random.sample(class_images, num_validation_images)

    # Create a mask of the original images with the same name and save them to the validation folder
    for image in validation_images:
        source_path = os.path.join(base_folder_path, class_folder, image)
        #destination_path = os.path.join(validation_folder_path, class_folder, image)
        original_image_path = os.path.join(original_folder_path, image.replace('fm_', ''))
        #Apply the mask to the original image
        mask = np.array(Image.open(source_path))
        mask[mask > 0] = 1
        original = Image.open(original_image_path)        
        cutout = Image.fromarray(original*mask)
        if class_folder == "normal":
            destination_path = os.path.join(validation_folder_path, class_folder, image)
            cutout.save(destination_path)
        else:
            destination_path = os.path.join(validation_folder_path, "problematic", image)
            cutout.save(destination_path)
           
    #Make a copy of the images not used for validation to the training folder
    for image in class_images:
        if image not in validation_images:
            source_path = os.path.join(base_folder_path, class_folder, image)
            destination_path = os.path.join(training_folder_path, class_folder, image)
            original_image_path = os.path.join(original_folder_path, image.replace('fm_', ''))
            #Apply the mask to the original image
            mask = np.array(Image.open(source_path))
            mask[mask > 0] = 1
            original = Image.open(original_image_path)
            cutout = Image.fromarray(original*mask)
            if class_folder == "normal":
                destination_path = os.path.join(training_folder_path, class_folder, image)
                cutout.save(destination_path)
            else:
                destination_path = os.path.join(training_folder_path, "problematic", image)
                cutout.save(destination_path)
            


### Copy worm classes to two classes
import os
import random
import shutil
from PIL import Image
import numpy as np
import cv2

# Define the base folder path
base_folder_path = '/home/maxime/prg/phd/dev/sambody/maskproblems_classificiation_split'
save_folder_path = "/home/maxime/prg/phd/dev/sambody/maskproblems_classification_split_2classesmoved"

# Define the training and validation folder paths
training_folder_path = os.path.join(save_folder_path, 'train')
validation_folder_path = os.path.join(save_folder_path, 'val')


# Create the training and validation folders if they don't exist
for folder_path in (training_folder_path, validation_folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Define the class folders
class_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
print(f'Class folders: {class_folders}')
new_classes = ["normal", "problematic"]
for folder in [training_folder_path, validation_folder_path]:
    for subfolder in new_classes:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)


#Copy images from the "normal" folder in the "train" and "val" folders of the base_folder to the "normal" folder in the save_folder for both train and val folders
for folder in ["train", "val"]:
    for image in os.listdir(os.path.join(base_folder_path, folder, "normal")):
        source_path = os.path.join(base_folder_path, folder, "normal", image)
        destination_path = os.path.join(save_folder_path, folder, "normal", image)
        shutil.copy2(source_path, destination_path)
        

#Copy images from all other folders in the "problematic" folder of the "save_folder" for both train and val folders
for folder in ["train", "val"]:
    for classif in os.listdir(os.path.join(base_folder_path, folder)):
        if classif != "normal":
            for image in os.listdir(os.path.join(base_folder_path, folder, classif)):
                source_path = os.path.join(base_folder_path, folder, classif, image)
                destination_path = os.path.join(save_folder_path, folder, "problematic", image)
                shutil.copy2(source_path, destination_path)






img_array = np.uint8(cutout)
img = Image.fromarray(cutout)
img.save('cutout.png')