import cv2
import numpy as np
from config.roi_config import ROI_POLYGON


def is_in_roi(box, roi_polygon, frame_shape, overlap_threshold=0.3):
    """
    box: [x1, y1, x2, y2]
    roi_polygon: list of (x, y) points
    frame_shape: (height, width) of the frame
    Returns True if the box overlaps the ROI polygon above threshold.
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [np.array(roi_polygon, dtype=np.int32)], 1)

    box_mask = np.zeros((h, w), dtype=np.uint8)
    box_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1

    intersection = np.logical_and(roi_mask, box_mask).sum()
    box_area = box_mask.sum()

    if box_area == 0:
        return False

    overlap_ratio = intersection / box_area
    return overlap_ratio >= overlap_threshold


def filter_detections_by_roi(detections, roi_polygon, frame_shape, overlap_threshold=0.3):
    """
    detections: list of dicts like {'class': ..., 'box': [x1,y1,x2,y2], 'conf': ...}
    Adds 'in_roi': True/False to each detection and returns the full list
    (so you can still visualize excluded detections, not just drop them silently).
    """
    for d in detections:
        d['in_roi'] = is_in_roi(d['box'], roi_polygon, frame_shape, overlap_threshold)
    return detections