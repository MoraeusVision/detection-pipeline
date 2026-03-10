from .base import BaseDetector
from pipeline_context import FrameContext


class RFDETRDetector(BaseDetector):
    def __init__(self, model_path=None, **kwargs):
        self.model_path = model_path
        self.params = kwargs

    def detect(self, ctx: FrameContext) -> FrameContext:
        return ctx