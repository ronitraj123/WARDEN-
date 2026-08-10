import cv2
import time
import json
import numpy as np
from scripts.violation_pipeline import ViolationPipeline
from scripts.violation_logger import ViolationEventLogger, export_events_to_csv
from config.roi_config import ROI_POLYGON

SOURCE = r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\barrier_test_clip.mp4'

LIVE_FRAME_PATH = 'live_frame.jpg'
LIVE_STATUS_PATH = 'live_status.json'
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

    # --- draw overlay (same as test script) ---
    overlay = frame.copy()
    cv2.polylines(overlay, [np.array(ROI_POLYGON, dtype=np.int32)], True, (0, 255, 255), 3)
    state_color = {'open': (0,255,0), 'closing': (0,255,255), 'closed': (0,0,255), 'unknown': (128,128,128)}[result['barrier_state']]
    cv2.putText(overlay, f"barrier: {result['barrier_state']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2)
    for d in result['detections']:
        x1, y1, x2, y2 = map(int, d['box'])
        color = (0, 0, 255) if d.get('in_roi') else (150, 150, 150)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} #{d.get('track_id','?')} {'VIOLATION' if d.get('in_roi') else ''}"
        cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- write current frame + status for the dashboard to poll ---
    cv2.imwrite(LIVE_FRAME_PATH, overlay)
    with open(LIVE_STATUS_PATH, 'w') as f:
        json.dump({
            'frame_idx': frame_idx, 'timestamp': round(timestamp, 2),
            'barrier_state': result['barrier_state'],
            'active_violations': len(result['violations']),
        }, f)

    # --- periodically flush events so dashboard sees them without waiting for the whole video ---
    if frame_idx % 30 == 0:
        export_events_to_csv(event_logger.completed_events, CSV_PATH)

    frame_idx += 1

    # throttle to real playback speed
    elapsed = time.time() - loop_start
    if elapsed < frame_delay:
        time.sleep(frame_delay - elapsed)

cap.release()
final_events = event_logger.finalize()
export_events_to_csv(final_events, CSV_PATH)
print("Live processing finished.")