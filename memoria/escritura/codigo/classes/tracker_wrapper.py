from argparse import Namespace
from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore




class TrackerWrapper:
    FRAME_AGE = 60

    def __init__(self, frame_rate=30):
        self.args = Namespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=self.FRAME_AGE,
            match_thresh=0.8,
            fuse_score=True,
        )
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)

    class Detections:
        def __init__(self, boxes, confidences, class_ids):
            self.conf = confidences
            self.xywh = boxes
            self.cls = class_ids

    def track(self, detection_data, frame):
        detections = self.Detections(
            detection_data.xywh.numpy(),
            detection_data.conf.numpy(),
            detection_data.cls.numpy().astype(int),
        )
        return self.tracker.update(detections, frame)
