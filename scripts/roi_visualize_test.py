import cv2
import os
from ultralytics import YOLO
from config.roi_config import ROI_POLYGON
from scripts.roi_filter import filter_detections_by_roi
import numpy as np

MODEL_PATH = 'models/warden_objects/warden_objects_v2/best.pt'
TEST_IMAGES = [
    r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\objects\objects_v2\valid\images\frame_0037_jpg.rf.8a37344e6cab1c7aa190db6985ad4d52.jpg',
    r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\objects\objects_v2\valid\images\newvid_frame_0070_jpg.rf.cfdfc7c7c5549110dd7be18e783cec2d.jpg',
    r'C:\Users\ronit\OneDrive\Desktop\WARDEN-\datasets\objects\objects_v2\valid\images\frame_0189_jpg.rf.ac375ffa76e6f781318343e05deb0a1f.jpg',
]

model = YOLO(MODEL_PATH)  

for img_path in TEST_IMAGES:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read {img_path}")
        continue

    results = model.predict(frame, conf=0.4, iou=0.5, verbose=False)[0]
    detections = [
        {'class': model.names[int(b.cls[0])], 'box': b.xyxy[0].tolist(), 'conf': float(b.conf[0])}
        for b in results.boxes
    ]

    detections = filter_detections_by_roi(detections, ROI_POLYGON, frame.shape)

    overlay = frame.copy()
    cv2.polylines(overlay, [np.array(ROI_POLYGON, dtype=np.int32)], True, (0, 255, 255), 2)

    for d in detections:
        x1, y1, x2, y2 = map(int, d['box'])
        color = (0, 0, 255) if d['in_roi'] else (150, 150, 150)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{d['class']} {'IN' if d['in_roi'] else 'OUT'}"
        cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = f"roi_check_{os.path.basename(img_path)}"
    success = cv2.imwrite(out_path, overlay)
    if success:
        print(f"Saved: {out_path}")
    else:
        print(f"FAILED to save: {out_path}")