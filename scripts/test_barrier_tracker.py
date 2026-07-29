import cv2
from scripts.barrier_state import detect_barrier_box, classify_barrier_state, BarrierStateTracker

VIDEO_PATH = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'  # or any clip showing a barrier closing
OUTPUT_VIDEO = 'barrier_tracker_test_output.mp4'
FRAME_SKIP = 2  # process every Nth frame to speed up testing; set to 1 for full accuracy

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: could not open video at {VIDEO_PATH}")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / FRAME_SKIP, (w, h))

tracker = BarrierStateTracker(window_size=5, min_agreement=3)

frame_idx = 0
transitions = []
last_state = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP == 0:
        box = detect_barrier_box(frame)
        raw_state = classify_barrier_state(box)
        smoothed_state = tracker.update(box)

        if smoothed_state != last_state:
            timestamp = frame_idx / fps
            transitions.append((frame_idx, timestamp, last_state, smoothed_state))
            last_state = smoothed_state

        # overlay for visual check
        color = {'open': (0, 255, 0), 'closing': (0, 255, 255), 'closed': (0, 0, 255), 'unknown': (128, 128, 128)}[smoothed_state]
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"raw: {raw_state} | smoothed: {smoothed_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)

    frame_idx += 1

cap.release()
out.release()

print(f"\nProcessed {frame_idx} frames.")
print("\nState transitions detected:")
for idx, t, from_s, to_s in transitions:
    print(f"  Frame {idx} ({t:.2f}s): {from_s} -> {to_s}")