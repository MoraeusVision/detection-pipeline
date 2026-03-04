
import os
from abc import ABC, abstractmethod
import cv2

# -----------------------------
# Base source interface
# -----------------------------
class BaseSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Returns a frame"""
        pass

# -----------------------------
# Image source
# -----------------------------
class ImageSource(BaseSource):
    def __init__(self, path):
        self.path = path
        self._consumed = False # This ensures an image is only processed once

    def get_frame(self):
        if self._consumed:
            return None
        
        frame = cv2.imread(self.path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {self.path}")
        self._consumed = True
        return frame

# -----------------------------
# Video source stub
# -----------------------------
class VideoSource(BaseSource):
    def __init__(self, path: str):
        """
        Args:
            path (str): Path to video file
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)

        # check that video opened correctly
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

    def get_frame(self):
        """
        Returns:
            frame (np.ndarray) or None if video has ended
        """
        ret, frame = self.cap.read()

        # if no frame is returned, video is finished
        if not ret:
            self.cap.release()
            return None

        return frame

# -----------------------------
# Stream source stub
# -----------------------------
class StreamSource(BaseSource):
    def __init__(self, url):
        self.url = url

    def get_frame(self):
        raise NotImplementedError("StreamSource not implemented yet")

# -----------------------------
# Source factory
# -----------------------------
class SourceFactory:
    @staticmethod
    def create(source_path):
        if source_path.startswith("rtsp://") or source_path.startswith("http://") or source_path.startswith("https://"):
            return StreamSource(source_path)
        elif os.path.isfile(source_path):
            ext = os.path.splitext(source_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                return ImageSource(source_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                return VideoSource(source_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        else:
            raise ValueError(f"Invalid source: {source_path}")