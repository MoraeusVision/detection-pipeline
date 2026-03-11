import time
from abc import ABC, abstractmethod

from pipeline_context import FrameContext, PipelineContext


class BasePipeline(ABC):
    def __init__(self, source, model, event_manager=None):
        self.source = source
        self.model = model
        self.event_manager = event_manager

    def run(self):
        is_static = self.source.is_static if self.source else False

        while True:
            frame = self.source.get_frame()
            if frame is None:
                break

            ctx = self.create_context_from_frame(frame, is_static)
            ctx = self.process_frame(ctx)

            self.notify("on_frame", ctx)
            if not ctx.should_continue:
                break

            if is_static:
                break

    def create_context_from_frame(self, frame, is_static):
        frame_context = FrameContext(frame=frame, timestamp=time.time())
        return PipelineContext(
            is_static=is_static,
            should_continue=True,
            frame_context=frame_context,
        )

    def notify(self, event_name, data=None):
        if self.event_manager is not None:
            self.event_manager.notify(event_name, data)

    @abstractmethod
    def process_frame(self, ctx):
        pass


class DetectionPipeline(BasePipeline):
    def process_frame(self, ctx):
        ctx.frame_context = self.model.detect(ctx.frame_context)
        return ctx
