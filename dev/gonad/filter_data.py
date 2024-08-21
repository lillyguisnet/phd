import os
import glob

# Set the directory you want to start from
dir_path = 'dev/data_gonad'

# Use glob to match all files and then filter out the .png files
files_to_delete = [f for f in glob.glob(os.path.join(dir_path, '*')) if not f.endswith('.png')]

# Loop over the list of files and remove each one
for file_path in files_to_delete:
    os.remove(file_path)



# Remove all "a" versions of image files
for img in glob.glob(os.path.join(dir_path, '*')):
    rep = img.split('_')[-2]
    if "a" in rep:
        os.remove(img)
        print(f"Removed {img})")

#Count the number of files in the directory
num_files = len([f for f in glob.glob(os.path.join(dir_path, '*')) if os.path.isfile(f)])