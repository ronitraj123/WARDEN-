import csv
import os
import cv2


class ViolationEventLogger:
    """
    Deduplicates per-frame violation detections into discrete violation events
    using YOLO/ByteTrack track IDs. An event starts when a track_id first
    appears inside the ROI during a closing/closed barrier state, and ends
    once that track_id hasn't been seen for `timeout_frames` frames.
    """

    def __init__(self, timeout_frames=10, snapshot_dir='violation_snapshots'):
        self.active_events = {}
        self.timeout_frames = timeout_frames
        self.completed_events = []
        self._next_violation_id = 1
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def update(self, violations, frame_idx, timestamp, barrier_state, frame=None):
        """
        violations: list of dicts with 'track_id', 'class', 'box', 'conf'
        frame: optional raw frame (numpy array) - used to save a representative
               snapshot when a track's confidence improves
        """
        seen_ids = set()

        for v in violations:
            track_id = v['track_id']
            seen_ids.add(track_id)

            if track_id not in self.active_events:
                self.active_events[track_id] = {
                    'track_id': track_id,
                    'class': v['class'],
                    'barrier_state': barrier_state,
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'entry_frame': frame_idx,
                    'exit_frame': frame_idx,
                    'best_conf': v['conf'],
                    'best_box': v['box'],
                    'snapshot_path': None,
                }
                if frame is not None:
                    self._save_snapshot(self.active_events[track_id], frame, v['box'])
            else:
                event = self.active_events[track_id]
                event['end_time'] = timestamp
                event['exit_frame'] = frame_idx
                if v['conf'] > event['best_conf']:
                    event['best_conf'] = v['conf']
                    event['best_box'] = v['box']
                    event['class'] = v['class']
                    if frame is not None:
                        self._save_snapshot(event, frame, v['box'])

        # close out events not seen in this frame, after a timeout window
        for track_id in list(self.active_events.keys()):
            if track_id not in seen_ids:
                event = self.active_events[track_id]
                if frame_idx - event['exit_frame'] > self.timeout_frames:
                    self._finalize_event(event)
                    del self.active_events[track_id]

    def _save_snapshot(self, event, frame, box, padding_ratio=0.25, min_size=200):
        track_id = event['track_id']
        unique_key = f"{track_id}_{event['entry_frame']}"
        path = os.path.join(self.snapshot_dir, f"track_{unique_key}.jpg")

        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        box_w, box_h = x2 - x1, y2 - y1

    # add proportional padding around the detected box
        pad_x = box_w * padding_ratio
        pad_y = box_h * padding_ratio
        x1, y1 = x1 - pad_x, y1 - pad_y
        x2, y2 = x2 + pad_x, y2 + pad_y

    # enforce a minimum crop size so very small/tight boxes don't produce
    # a tiny, unusably small evidence image (widen symmetrically around center)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cur_w, cur_h = x2 - x1, y2 - y1
        if cur_w < min_size:
            x1, x2 = cx - min_size / 2, cx + min_size / 2
        if cur_h < min_size:
            y1, y2 = cy - min_size / 2, cy + min_size / 2

    # clamp to frame bounds
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            event['snapshot_path'] = path

    def _finalize_event(self, event):
        event['violation_id'] = self._next_violation_id
        event['duration'] = round(event['end_time'] - event['start_time'], 2)
        self._next_violation_id += 1
        self.completed_events.append(event)

    def finalize(self):
        """Call once at the end of the video to flush any still-active events."""
        for event in list(self.active_events.values()):
            self._finalize_event(event)
        self.active_events = {}
        return self.completed_events


def export_events_to_csv(events, output_path):
    fieldnames = [
        'violation_id', 'track_id', 'class', 'barrier_state',
        'start_time', 'end_time', 'duration',
        'best_conf', 'best_box', 'entry_frame', 'exit_frame', 'snapshot_path',
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            row = {k: e.get(k) for k in fieldnames}
            row['start_time'] = round(row['start_time'], 2)
            row['end_time'] = round(row['end_time'], 2)
            writer.writerow(row)