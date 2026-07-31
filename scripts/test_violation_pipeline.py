import cv2
from scripts.violation_pipeline import ViolationPipeline

VIDEO_PATH = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'
OUTPUT_VIDEO = 'violation_pipeline_test_output.mp4'
MAX_DURATION_SECONDS = 45  # cover the closing + a bit of closed time from your earlier test
FRAME_SKIP = 2

OBJECT_MODEL_PATH = 'models/warden_objects/warden_objects_v2/best.pt'
BARRIER_MODEL_PATH = 'models/warden_barrier/weights/best.pt'

pipeline = ViolationPipeline(
    object_model_path=OBJECT_MODEL_PATH,
    barrier_model_path=BARRIER_MODEL_PATH,
)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: could not open video at {VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(MAX_DURATION_SECONDS * fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps / FRAME_SKIP, (w, h))

frame_idx = 0
violation_log = []

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= max_frames:
        break

    if frame_idx % FRAME_SKIP == 0:
        timestamp = frame_idx / fps
        result = pipeline.process_frame(frame, frame_idx=frame_idx, timestamp=timestamp)

        # log any violations
        for v in result['violations']:
            violation_log.append({
                'frame': frame_idx,
                'time': round(timestamp, 2),
                'class': v['class'],
                'conf': round(v['conf'], 2),
                'box': v['box'],
            })

        # visualize
        overlay = frame.copy()
        state_color = {'open': (0,255,0), 'closing': (0,255,255), 'closed': (0,0,255), 'unknown': (128,128,128)}[result['barrier_state']]
        cv2.putText(overlay, f"barrier: {result['barrier_state']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

        for d in result['detections']:
            x1, y1, x2, y2 = map(int, d['box'])
            color = (0, 0, 255) if d.get('in_roi') else (150, 150, 150)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{d['class']} {'VIOLATION' if d.get('in_roi') else ''}"
            cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(overlay)

    frame_idx += 1

cap.release()
out.release()

print(f"\nProcessed {frame_idx} frames ({frame_idx/fps:.1f}s).")
print(f"\nTotal violation detections logged: {len(violation_log)}")
print("\nSample violations (first 10):")
for v in violation_log[:10]:
    print(f"  Frame {v['frame']} ({v['time']}s): {v['class']} (conf={v['conf']}) box={v['box']}")