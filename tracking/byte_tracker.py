import numpy as np
import supervision as sv

from pipeline_context import Detection

from .base_tracker import BaseTracker


class ByteTrackTracker(BaseTracker):
    def __init__(self, config=None):
        config = config or {}
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.get("track_activation_threshold", 0.4),
            lost_track_buffer=config.get("lost_track_buffer", 50),
            minimum_matching_threshold=config.get("minimum_matching_threshold", 0.8),
            frame_rate=config.get("frame_rate", 30),
            minimum_consecutive_frames=config.get("minimum_consecutive_frames", 5),
        )

    def update(self, detections, frame):
        del frame

        if not detections:
            self.tracker.update_with_tensors(np.empty((0, 5), dtype=np.float32))
            return []

        tracked_detections = self.tracker.update_with_detections(
            sv.Detections(
                xyxy=np.asarray([detection.bbox for detection in detections], dtype=np.float32),
                confidence=np.asarray(
                    [detection.confidence for detection in detections],
                    dtype=np.float32,
                ),
                class_id=np.asarray(
                    [
                        detection.class_id if detection.class_id is not None else -1
                        for detection in detections
                    ],
                    dtype=np.int32,
                ),
            )
        )

        confidences = tracked_detections.confidence
        if confidences is None:
            confidences = np.zeros(len(tracked_detections.xyxy), dtype=np.float32)

        class_ids = tracked_detections.class_id
        if class_ids is None:
            class_ids = np.full(len(tracked_detections.xyxy), -1, dtype=np.int32)

        tracker_ids = tracked_detections.tracker_id
        if tracker_ids is None:
            tracker_ids = np.full(len(tracked_detections.xyxy), -1, dtype=np.int32)

        labels_by_class = {
            detection.class_id: detection.label
            for detection in detections
            if detection.class_id is not None
        }
        default_label = detections[0].label

        tracked = []
        for bbox, confidence, class_id, track_id in zip(
            tracked_detections.xyxy,
            confidences,
            class_ids,
            tracker_ids,
        ):
            resolved_class_id = int(class_id)
            if resolved_class_id < 0:
                resolved_class_id = None

            tracked.append(
                Detection(
                    label=labels_by_class.get(resolved_class_id, default_label),
                    confidence=float(confidence),
                    bbox=tuple(int(value) for value in bbox),
                    class_id=resolved_class_id,
                    track_id=None if int(track_id) < 0 else int(track_id),
                )
            )

        return tracked