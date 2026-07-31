import cv2
import numpy as np
from scripts.violation_pipeline import ViolationPipeline
from scripts.violation_logger import ViolationEventLogger, export_events_to_csv
from config.roi_config import ROI_POLYGON

VIDEO_PATH = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'
OUTPUT_VIDEO = 'violation_pipeline_test_output.mp4'
OUTPUT_CSV = 'violation_events.csv'
MAX_DURATION_SECONDS = 45
FRAME_SKIP = 2

OBJECT_MODEL_PATH = 'models/warden_objects/warden_objects_v2/best.pt'
BARRIER_MODEL_PATH = 'models/warden_barrier/weights/best.pt'

pipeline = ViolationPipeline(
    object_model_path=OBJECT_MODEL_PATH,
    barrier_model_path=BARRIER_MODEL_PATH,
)
event_logger = ViolationEventLogger(timeout_frames=10)

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

while True:
    ret, frame = cap.read()
    if not ret or frame_idx >= max_frames:
        break

    if frame_idx % FRAME_SKIP == 0:
        timestamp = frame_idx / fps
        result = pipeline.process_frame(frame, frame_idx=frame_idx, timestamp=timestamp)

        event_logger.update(
            result['violations'], frame_idx, timestamp,
            result['barrier_state'], frame=frame
        )

        # --- visualization ---
        overlay = frame.copy()
        cv2.polylines(overlay, [np.array(ROI_POLYGON, dtype=np.int32)], True, (0, 255, 255), 3)

        state_color = {
            'open': (0, 255, 0), 'closing': (0, 255, 255),
            'closed': (0, 0, 255), 'unknown': (128, 128, 128),
        }[result['barrier_state']]
        cv2.putText(overlay, f"barrier: {result['barrier_state']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)

        for d in result['detections']:
            x1, y1, x2, y2 = map(int, d['box'])
            color = (0, 0, 255) if d.get('in_roi') else (150, 150, 150)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            tid = d.get('track_id', '?')
            label = f"{d['class']} #{tid} {'VIOLATION' if d.get('in_roi') else ''}"
            cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(overlay)

    frame_idx += 1

cap.release()
out.release()

final_events = event_logger.finalize()
export_events_to_csv(final_events, OUTPUT_CSV)

print(f"\nProcessed {frame_idx} frames ({frame_idx/fps:.1f}s).")
print(f"\nTotal deduplicated violation events: {len(final_events)}")
print("\nEvent summary:")
for e in final_events:
    print(f"  ID {e['violation_id']} | track #{e['track_id']} | {e['class']} | "
          f"{e['start_time']}s -> {e['end_time']}s (duration {e['duration']}s) | "
          f"barrier: {e['barrier_state']} | conf: {round(e['best_conf'], 2)}")