from .base import BaseDetector


class RFDETRDetector(BaseDetector):
    def __init__(self, model_path):
        self.model_path = model_path

    def detect(self, frame):
        raise NotImplementedError()