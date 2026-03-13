import logging
import cv2
from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    @abstractmethod
    def handle_event():
        pass

class Visualizer(BaseVisualizer):
    def __init__(self, window_name: str="Detection", width: int=800, height: int=600):
        """
        Args:
            window_name (str): Name of the display window
            width (int): Width of the window
            height (int): Height of the window
        """
        self.window_name = window_name
        self.width = width
        self.height = height

        # playback control
        self.paused = False            # whether display is currently paused
        self.last_frame = None         # most recent frame shown (for pause)

        # Create a resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def handle_event(self, event, data):
        if event == "on_inference_result":
            boxes = [detection.bbox for detection in data.frame_context.detections] or None
            labels = [
                f"{detection.label} {detection.confidence:.2f}"
                for detection in data.frame_context.detections
            ] or None
            data.should_continue = self.show(
                frame=data.frame_context.frame,
                boxes=boxes,
                labels=labels,
                is_static=data.is_static,
            )

    def show(self, frame, boxes=None, labels=None, is_static=False):
        """
        Display a single frame with optional bounding boxes.

        Args:
            frame (np.ndarray): Image to display
            boxes (list of [x1, y1, x2, y2], optional): Bounding boxes to draw
        """
        # choose which frame to display; if we are paused, replay last_frame
        if self.paused and self.last_frame is not None and not is_static:
            display_frame = self.last_frame.copy()
        else:
            # Make a copy so original frame is not modified
            display_frame = frame.copy()
            # remember the frame so pausing can reuse it
            self.last_frame = display_frame

        # Draw bounding boxes if provided
        if boxes is not None:
            for index, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if labels is not None and index < len(labels):
                    text_origin = (x1, max(20, y1 - 10))
                    cv2.putText(
                        display_frame,
                        labels[index],
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # Show the frame in the window
        cv2.imshow(self.window_name, display_frame)

        if is_static:
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    return False

        # Determine how long to wait for a key press. If paused we block
        # indefinitely; otherwise we proceed with a short delay.
        if self.paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        # Toggle pause with spacebar
        if key == ord(' ') and not is_static:
            self.paused = not self.paused
            logging.info("Playback %s", "paused" if self.paused else "resumed")
            # after toggling pause we want to keep showing the same frame
            return True

        # Exit if q is pressed
        if key == ord('q'):
            return False

        return True
        
    def cleanup(self):
        """Close the display window."""
        logging.info("Closing windows..")
        cv2.destroyAllWindows()