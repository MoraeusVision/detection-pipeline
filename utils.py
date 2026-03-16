import logging
import cv2


class CleanupManager:
    def __init__(self):
        # Store cleanup functions
        self._tasks = []

    def add(self, func, *args, **kwargs):
        """
        Add a cleanup function to be called later.

        Args:
            func (callable): function to call
            *args, **kwargs: arguments for the function
        """
        self._tasks.append((func, args, kwargs))

    def run(self):
        """Run all registered cleanup tasks."""
        for func, args, kwargs in self._tasks:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.exception("Cleanup failed for %s: %s", func, e)
        self._tasks.clear()


class SaveManager:
    def __init__(self):
        self.out = None

    def handle_event(self, event, data):
        if event == "on_inference_result":
            frame = self._annotate_frame(data)
            if frame is None:
                return
            if data.is_static:
                self.save_image(frame)
            else:
                self.write_frame_to_video(frame)

    def save_image(self, frame):
        logging.info("Saving image..")
        cv2.imwrite("output.jpg", frame)

    def write_frame_to_video(self, frame, fps=30):
        if self.out is None:
            width = frame.shape[1]
            height = frame.shape[0]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
        self.out.write(frame)

    def save_video(self):
        logging.info("Saving video..")
        self.out.release()

    def _annotate_frame(self, data):
        """
        Annotade the image with bounding boxes, labels and tracking IDs.
        """
        if data is None or data.frame_context is None:
            logging.warning("Missing pipeline context")
            return None
        
        frame = data.frame_context.frame
        if frame is None:
            logging.warning("Missing frame")
            return None
        
        if not data.frame_context.detections:
            return frame.copy()

        annotated_frame = frame.copy()
        
        boxes = [detection.bbox for detection in data.frame_context.detections] or None
        labels = [
                self._format_detection_label(detection)
                for detection in data.frame_context.detections
            ] or None
        
        if boxes is not None:
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if labels is not None and index < len(labels):
                    text_origin = (x1, max(20, y1 - 10))
                    cv2.putText(
                        annotated_frame,
                        labels[index],
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
        return annotated_frame

    def _format_detection_label(self, detection):
        prefix = f"ID: {detection.track_id} " if detection.track_id is not None else ""
        return f"{prefix}{detection.label} {detection.confidence:.2f}"