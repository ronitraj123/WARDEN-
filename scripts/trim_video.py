# scripts/trim_video.py
import cv2

INPUT = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier.mp4'
OUTPUT = 'barrier_test_clip.mp4'
DURATION_SECONDS = 40

cap = cv2.VideoCapture(INPUT)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(DURATION_SECONDS * fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT, fourcc, fps, (w, h))

count = 0
while count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    count += 1

cap.release()
out.release()
print(f"Saved {count} frames ({count/fps:.1f}s) to {OUTPUT}")