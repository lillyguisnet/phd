import cv2
import numpy as np
import os
from glob import glob


def extract_segments_from_frames(frames_dir, segments, output_dir, prefix="segment", padding=10):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(glob(os.path.join(frames_dir, "*.jpg")))

    for frame_num, frame_path in enumerate(frame_files):
        frame = cv2.imread(frame_path)
        
        if frame_num in segments:
            for segment_id, mask in segments[frame_num].items():
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask.squeeze()
                
                # Ensure mask has the same dimensions as the frame
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

                # Convert boolean mask to uint8
                mask = (mask > 0).astype(np.uint8) * 255

                # Find bounding box of the segment
                y, x = np.where(mask > 0)
                if len(y) == 0 or len(x) == 0:  # Skip empty masks
                    continue
                top, bottom, left, right = y.min(), y.max(), x.min(), x.max()

                # Add padding
                top = max(0, top - padding)
                bottom = min(frame.shape[0], bottom + padding)
                left = max(0, left - padding)
                right = min(frame.shape[1], right + padding)

                # Crop the frame and mask
                cropped_frame = frame[top:bottom, left:right]
                cropped_mask = mask[top:bottom, left:right]

                # Apply the mask
                masked_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)

                # Save the masked frame
                output_path = os.path.join(output_dir, f"{prefix}_frame{frame_num}_segment{segment_id}.png")
                cv2.imwrite(output_path, masked_frame)


# Usage
frames_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/tstcropped"
output_dir = "/home/maxime/prg/phd/crawlingtracking/tstvideo_tojpg/wormsegmentsforclassification"
segments = video_segments
extract_segments_from_frames(frames_dir, segments, output_dir)