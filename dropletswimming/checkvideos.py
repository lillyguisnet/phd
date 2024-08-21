import cv2
import numpy as np

# Load the video
video_path = '/home/maxime/prg/phd/dropletswimming/data_original/ngm/a-02182022162408-0000.avi'  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Initialize a list to store the frames
frames = []

frame_index = 0  # Keep track of the frame index

while True:
    # Attempt to read the next frame
    ret, frame = cap.read()

    # Check if the frame was not read successfully
    if not ret:
        # Attempt to move to the next frame. This may help if the current frame is corrupt.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + 1)
           
        # Check if we have reached the end of the video
        if frame_index >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            # If we're at the end, exit the loop
            print(f"End of video {video_path}")
            break
        else:
            # Move to the next frame index and continue with the loop
            print(f"Error reading frame {frame_index}. Skipping...")
            frame_index += 1
            continue
    
    # If the frame is read successfully, add it to the list of frames
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray_frame)
    
    # Print a message indicating the frame was read successfully (optional)
    # print(f"Successfully read frame {frame_index}")
    
    # Move to the next frame index
    frame_index += 1

# Release the video capture object
cap.release()

import pickle  

#Save frames as a pickle file
with open("/home/maxime/prg/phd/dropletswimming/tst_frames.pkl", "wb") as f:
    pickle.dump(frames, f)
