from .base import BaseDetector
from pipeline_context import FrameContext, Detection
import time


class RFDETRDetector(BaseDetector):
    def __init__(self, model_path):
        self.model_path = model_path

    def detect(self, ctx: FrameContext) -> FrameContext:
        pass