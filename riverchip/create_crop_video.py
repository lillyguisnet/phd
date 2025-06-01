import cv2
import os
from tqdm import tqdm
import sys

def create_video_from_crops(crop_folder, output_video_path, fps=15):
    """Create a video from cropped images."""
    
    # Get all jpg files in the crop folder
    image_files = [f for f in os.listdir(crop_folder) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    if not image_files:
        print(f"No jpg files found in {crop_folder}")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(os.path.join(crop_folder, image_files[0]))
    height, width = first_image.shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(image_files)} images...")
    print(f"Video dimensions: {width}x{height}")
    print(f"Output: {output_video_path}")
    
    for image_file in tqdm(image_files, desc="Creating video"):
        # Read image
        image = cv2.imread(os.path.join(crop_folder, image_file))
        
        # Write frame to video
        out.write(image)
    
    # Release everything
    out.release()
    print(f"Video saved successfully!")

if __name__ == "__main__":
    # Look for the most recent crop folder
    base_dir = "/home/lilly/phd/riverchip/data_foranalysis/riacrop/"
    
    # Find crop folders
    crop_folders = [d for d in os.listdir(base_dir) if d.endswith("_crop") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not crop_folders:
        print("No crop folders found!")
        sys.exit(1)
    
    # Use the most recently modified crop folder
    crop_folders.sort(key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    crop_folder = os.path.join(base_dir, crop_folders[0])
    
    print(f"Using crop folder: {crop_folder}")
    
    # Output video path
    output_video = os.path.join(base_dir, f"{crop_folders[0]}_aligned.mp4")
    
    # Create video
    create_video_from_crops(crop_folder, output_video, fps=15) 