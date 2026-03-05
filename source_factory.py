
import os
from abc import ABC, abstractmethod
import cv2
from pathlib import Path

# -----------------------------
# Constants
# -----------------------------
MAX_SIZE_MB = 20 # Just to not starting to download hour long youtube videos

# -----------------------------
# Base source interface
# -----------------------------
class BaseSource(ABC):
    @abstractmethod
    def get_frame(self):
        """Returns a frame"""
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup"""
        pass

# -----------------------------
# Image source
# -----------------------------
class ImageSource(BaseSource):
    def __init__(self, path):
        self.path = path
        

    def get_frame(self):
        """
        Returns the image.
        """
        frame = cv2.imread(self.path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {self.path}")
        return frame
    
    def cleanup(self):
        """Nothing to cleanup if source is just an image"""
        pass

# -----------------------------
# Video source
# -----------------------------
class VideoSource(BaseSource):
    def __init__(self, path: str):
        """
        Args:
            path (str): Path to video file
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # check that video opened correctly
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

    def get_frame(self):
        """
        Returns a frame from the video.
        """
        ret, frame = self.cap.read()

        # if no frame is returned, video is finished
        if not ret:
            self.cap.release()
            return None

        return frame
    
    def cleanup(self):
        """Release the video."""
        if self.cap.isOpened():
            print("Closing video..")
            self.cap.release()

# -----------------------------
# Stream source
# -----------------------------
class StreamSource(BaseSource):
    def __init__(self, url, download_folder="temp", is_youtube=False):
        self.url = url
        self.is_youtube = is_youtube

        # If stream source is from Youtube
        if "youtube.com" in url or "youtu.be" in url:
            import yt_dlp
            self.is_youtube = True
            ydl_opts = {
                "format": "best[ext=mp4]",
                "outtmpl": str(Path(download_folder) / "video.%(ext)s"),
                "quiet": True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filesize_bytes = info.get("filesize") or info.get("filesize_approx")
                if filesize_bytes is None:
                    raise ValueError("Could not determine video file size.")
                
                filesize_mb = filesize_bytes / (1024 * 1024)
                if filesize_mb > MAX_SIZE_MB:
                    raise ValueError(f"The video exceeds the maximus size of {MAX_SIZE_MB}MB. Video is {int(filesize_mb)}MB.")
                
                self.downloaded_path = ydl.prepare_filename(info)

            self.cap = cv2.VideoCapture(self.downloaded_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) # Used for setting the visualization FPS
        else:
            self.cap = cv2.VideoCapture(url)
            
        if not self.cap.isOpened():
            raise ValueError(f"Could not open stream: {url}")

    def get_frame(self):
        """
        Returns a frame from the stream.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def cleanup(self):
        """Release the stream."""
        if self.cap.isOpened():
            self.cap.release()

        # Remove the downloaded youtube video
        if hasattr(self, "downloaded_path") and os.path.exists(self.downloaded_path):
            os.remove(self.downloaded_path)
    
# -----------------------------
# USB camera source
# -----------------------------
class USBCameraSource(BaseSource):
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            device_index (int): Camera index (0,1,...)
            width (int): Desired capture width
            height (int): Desired capture height
        """
        # Open camera
        self.cap = cv2.VideoCapture(device_index)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open USB camera at index {device_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """
        Returns a frame from the USB camera.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def cleanup(self):
        """Release the camera."""
        if self.cap.isOpened():
            print("Closing USB camera..")
            self.cap.release()

# -----------------------------
# Source factory
# -----------------------------
class SourceFactory:
    @staticmethod
    def create(source_path):
        # If source is a stream
        if source_path.startswith("rtsp://") or source_path.startswith("http://") or source_path.startswith("https://"):
            return StreamSource(source_path)
        
        # If source is an image or a video
        elif os.path.isfile(source_path):
            ext = os.path.splitext(source_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                return ImageSource(source_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                return VideoSource(source_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
        # If source is a USB camera
        elif source_path.isdigit():
                return USBCameraSource(int(source_path))
        else:
            raise ValueError(f"Invalid source: {source_path}")