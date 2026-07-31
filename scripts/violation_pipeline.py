from ultralytics import YOLO
from scripts.barrier_state import detect_barrier_box, BarrierStateTracker
from scripts.roi_filter import filter_detections_by_roi
from config.roi_config import ROI_POLYGON
import numpy as np


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def resolve_rider_overlaps(detections, iou_threshold=0.3):
    """Collapse person+cycle/two-wheeler overlaps into a single vehicle entity.
    Keeps the vehicle's track_id (not the person's) since the vehicle is the
    more stable/persistent object to track across frames.
    """
    persons = [d for d in detections if d['class'] == 'person']
    vehicles = [d for d in detections if d['class'] in ('cycle', 'two wheeler')]
    resolved = [d for d in detections if d['class'] not in ('person', 'cycle', 'two wheeler')]

    used_persons = set()
    for v in vehicles:
        best_iou, best_idx = 0, None
        for i, p in enumerate(persons):
            if i in used_persons:
                continue
            iou = compute_iou(v['box'], p['box'])
            if iou > iou_threshold and iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx is not None:
            used_persons.add(best_idx)
        resolved.append(v)

    for i, p in enumerate(persons):
        if i not in used_persons:
            resolved.append(p)

    return resolved


class ViolationPipeline:
    def __init__(self, object_model_path, barrier_model_path, roi_polygon=ROI_POLYGON,
                 object_conf=0.4, object_iou=0.5, roi_overlap_threshold=0.3,
                 rider_iou_threshold=0.3, tracker_config='bytetrack.yaml'):
        self.object_model = YOLO(object_model_path)
        self.barrier_tracker = BarrierStateTracker(window_size=5, min_agreement=3)
        self.roi_polygon = roi_polygon
        self.object_conf = object_conf
        self.object_iou = object_iou
        self.roi_overlap_threshold = roi_overlap_threshold
        self.rider_iou_threshold = rider_iou_threshold
        self.tracker_config = tracker_config

        # barrier model is loaded inside scripts.barrier_state already

    def process_frame(self, frame, frame_idx=None, timestamp=None):
        # 1. barrier state (always evaluated, every frame, cheap single-class model)
        barrier_box = detect_barrier_box(frame)
        barrier_state = self.barrier_tracker.update(barrier_box)

        result = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'barrier_state': barrier_state,
            'barrier_box': barrier_box,
            'detections': [],
            'violations': [],
        }

        # gate: only run object detection + violation logic when barrier is closing or closed 
        if barrier_state not in ('closing', 'closed'):
            return result

        # object detection + tracking (persist=True maintains track IDs across successive calls on the same video stream)
        pred = self.object_model.track(
            frame, conf=self.object_conf, iou=self.object_iou,
            persist=True, tracker=self.tracker_config, verbose=False
        )[0]

        detections = []
        if pred.boxes.id is not None:
            for box, track_id in zip(pred.boxes, pred.boxes.id):
                detections.append({
                    'class': self.object_model.names[int(box.cls[0])],
                    'box': box.xyxy[0].tolist(),
                    'conf': float(box.conf[0]),
                    'track_id': int(track_id),
                })

        # 4. resolve rider/vehicle overlaps (avoid double-counting)
        detections = resolve_rider_overlaps(detections, self.rider_iou_threshold)

        # 5. ROI filtering
        detections = filter_detections_by_roi(
            detections, self.roi_polygon, frame.shape, self.roi_overlap_threshold
        )

        result['detections'] = detections
        result['violations'] = [d for d in detections if d['in_roi']]

        return result