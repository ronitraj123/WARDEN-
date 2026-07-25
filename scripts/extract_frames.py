# scripts/extract_frames.py

import cv2
import os

VIDEO_PATH = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\un_data.MOV'
OUTPUT_DIR = 'datasets/new_unlabeled/images'
FRAME_INTERVAL = 15  # extract every Nth frame, tune this - see note below

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
saved_count = 0

print(f"Video FPS: {fps}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        filename = f"newvid_frame_{saved_count:04d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Extracted {saved_count} frames from {frame_count} total frames.")