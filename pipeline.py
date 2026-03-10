import time
from abc import ABC, abstractmethod

from pipeline_context import FrameContext


class BasePipeline(ABC):
    def __init__(self, source, event_manager=None, visualizer=None):
        self.source = source
        self.event_manager = event_manager
        self.visualizer = visualizer

    def run(self):
        ctx = None
        is_static = self.source.is_static if self.source else False

        while True:
            if self._should_fetch_next_frame(ctx):
                frame = self.source.get_frame()
                if frame is None:
                    break

                ctx = self.create_frame_context(frame)
                ctx = self.preprocess(ctx)
                ctx = self.process_frame(ctx)
                ctx = self.postprocess(ctx)
                self.publish(ctx)

                if is_static and self.visualizer is None:
                    break

            if ctx is None:
                break

            if not self.render(ctx, is_static=is_static):
                break

    def create_frame_context(self, frame):
        return FrameContext(frame=frame, timestamp=time.time())

    def preprocess(self, ctx: FrameContext) -> FrameContext:
        return ctx

    @abstractmethod
    def process_frame(self, ctx: FrameContext) -> FrameContext:
        raise NotImplementedError

    def postprocess(self, ctx: FrameContext) -> FrameContext:
        return ctx

    def publish(self, ctx: FrameContext) -> None:
        if self.event_manager is not None:
            self.event_manager.notify("on_frame", ctx)

    def render(self, ctx: FrameContext, is_static: bool) -> bool:
        if self.visualizer is None:
            return True

        return self.visualizer.show(
            frame=ctx.frame,
            boxes=[detection.bbox for detection in ctx.detections],
            is_image=is_static,
        )

    def _should_fetch_next_frame(self, ctx: FrameContext | None) -> bool:
        if ctx is None:
            return True

        if self.source.is_static:
            return False

        if self.visualizer is None:
            return True

        return not self.visualizer.paused


class DetectionPipeline(BasePipeline):
    def __init__(self, source, detector, event_manager=None, visualizer=None):
        super().__init__(source=source, event_manager=event_manager, visualizer=visualizer)
        self.detector = detector

    def process_frame(self, ctx: FrameContext) -> FrameContext:
        processed_ctx = self.detector.detect(ctx)
        if processed_ctx is None:
            raise ValueError("Detector strategy must return a FrameContext")
        return processed_ctx