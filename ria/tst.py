import cv2
import os
import glob

# Path to the folder containing images
vid_images = "/home/lilly/phd/ria/data_foranalysis/AG_WT/videotojpg/AG_WT-MMH17420250407_12"

# Get all jpg files and sort them numerically
image_files = sorted(
    glob.glob(os.path.join(vid_images, "*.jpg")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

# Read the first image to get the frame size
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(
    os.path.join(vid_images, "output_video.avi"),
    cv2.VideoWriter_fourcc(*'XVID'),
    5,  # FPS (change if needed)
    (width, height)
)

for img_file in image_files:
    img = cv2.imread(img_file)
    out.write(img)

out.release()
print("Video created successfully!")