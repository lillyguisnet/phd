import json
import shutil
import os
from sklearn.model_selection import train_test_split

# Path to the directory where the original images are stored
source_dir = '/home/maxime/prg/phd/dev/eggs/mask_cutouts'

# Directories where the images will be copied
train_dir = '/home/maxime/prg/phd/dev/eggs/cutout_classification_split/train'
valid_dir = '/home/maxime/prg/phd/dev/eggs/cutout_classification_split/val'

# Make sure the directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Read the .njson file and extract the data
data = []
with open('/home/maxime/prg/phd/dev/eggs/Export v2 project - dev_eggs_classification - 3_1_2024.ndjson', 'r') as file:
    for line in file:
        entry = json.loads(line)
        project_labels = entry["projects"].values()
        for project in project_labels:
            for label in project["labels"]:
                classifications = label["annotations"]["classifications"]
                if classifications:  # Check if there are any classifications
                    for classification in classifications:
                        if classification["name"] == "cutout_classificiation":
                            data.append({
                                'external_id': entry["data_row"]["external_id"],
                                'classification': classification["radio_answer"]["value"]
                            })

# Filter out entries without a classification
filtered_data = [d for d in data if d['classification'] in ['egg', 'not_egg']]

###Save all "external_id" that have classification "egg" in a file
with open('/home/maxime/prg/phd/dev/eggs/egg_masks.txt', 'w') as f:
    for item in filtered_data:
        if item['classification'] == 'egg':
            f.write("%s\n" % item['external_id'])


# Split data into training and validation sets
classifications = [d['classification'] for d in filtered_data]
train, valid = train_test_split(filtered_data, test_size=0.2, stratify=classifications, random_state=42)

# Function to copy files to the respective directory
def copy_files(data_set, target_dir):
    for item in data_set:
        file_name = item['external_id']
        classification = item['classification']
        # Define the source and destination paths
        src = os.path.join(source_dir, file_name)
        dst = os.path.join(target_dir, classification, file_name)
        # Make sure the target subdirectory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # Copy the file
        shutil.copy(src, dst)

# Copy the files to the respective directories
copy_files(train, train_dir)
copy_files(valid, valid_dir)
