import cv2
import time
import numpy as np
from scripts.violation_pipeline import ViolationPipeline
from scripts.violation_logger import ViolationEventLogger, export_events_to_csv
from config.roi_config import ROI_POLYGON

SOURCE = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'  

CSV_PATH = 'violation_events.csv'
OBJECT_MODEL_PATH = 'models/warden_objects/warden_objects_v2/best.pt'
BARRIER_MODEL_PATH = 'models/warden_barrier/weights/best.pt'

pipeline = ViolationPipeline(object_model_path=OBJECT_MODEL_PATH, barrier_model_path=BARRIER_MODEL_PATH)
event_logger = ViolationEventLogger(timeout_frames=10)

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("ERROR: could not open source")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_delay = 1.0 / fps

frame_idx = 0
while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_idx / fps
    result = pipeline.process_frame(frame, frame_idx=frame_idx, timestamp=timestamp)
    event_logger.update(result['violations'], frame_idx, timestamp, result['barrier_state'], frame=frame)

    # --- overlay ---
    overlay = frame.copy()
    # No need to show polygon cv2.polylines(overlay, [np.array(ROI_POLYGON, dtype=np.int32)], True, (0, 255, 255), 3)
    state_color = {'open': (0,255,0), 'closing': (0,255,255), 'closed': (0,0,255), 'unknown': (128,128,128)}[result['barrier_state']]
    cv2.putText(overlay, f"barrier: {result['barrier_state']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
    cv2.putText(overlay, f"active violations: {len(result['violations'])}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    for d in result['detections']:
        x1, y1, x2, y2 = map(int, d['box'])
        color = (0, 0, 255) if d.get('in_roi') else (150, 150, 150)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} #{d.get('track_id','?')} {'VIOLATION' if d.get('in_roi') else ''}"
        cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- THIS is the live demo window ---
    cv2.imshow('WARDEN - Live Detection', overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

    # throttle to real playback speed so it looks/behaves like a live feed
    elapsed = time.time() - loop_start
    if elapsed < frame_delay:
        time.sleep(frame_delay - elapsed)

cap.release()
cv2.destroyAllWindows()

final_events = event_logger.finalize()
export_events_to_csv(final_events, CSV_PATH)
print(f"Session ended. {len(final_events)} violations logged to {CSV_PATH}")