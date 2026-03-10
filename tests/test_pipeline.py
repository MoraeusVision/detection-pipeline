from unittest.mock import MagicMock, patch

from pipeline import DetectionPipeline
from pipeline_context import Detection


class TestDetectionPipeline:
    @patch("pipeline.time.time", return_value=123.0)
    def test_run_processes_frames_with_detector_strategy(self, mock_time):
        source = MagicMock()
        source.is_static = False
        source.get_frame.side_effect = ["frame-1", "frame-2", None]

        detector = MagicMock()
        detector.detect.side_effect = lambda ctx: ctx

        event_manager = MagicMock()

        pipeline = DetectionPipeline(
            source=source,
            detector=detector,
            event_manager=event_manager,
        )

        pipeline.run()

        assert detector.detect.call_count == 2
        assert event_manager.notify.call_count == 2
        first_event_name, first_ctx = event_manager.notify.call_args_list[0][0]
        assert first_event_name == "on_frame"
        assert first_ctx.frame == "frame-1"
        assert first_ctx.timestamp == 123.0

    @patch("pipeline.time.time", return_value=123.0)
    def test_run_stops_after_single_static_frame_without_visualizer(self, mock_time):
        source = MagicMock()
        source.is_static = True
        source.get_frame.return_value = "frame-1"

        detector = MagicMock()
        detector.detect.side_effect = lambda ctx: ctx

        event_manager = MagicMock()

        pipeline = DetectionPipeline(
            source=source,
            detector=detector,
            event_manager=event_manager,
        )

        pipeline.run()

        source.get_frame.assert_called_once()
        detector.detect.assert_called_once()
        event_manager.notify.assert_called_once()

    @patch("pipeline.time.time", return_value=123.0)
    def test_render_passes_detection_boxes_to_visualizer(self, mock_time):
        source = MagicMock()
        source.is_static = True
        source.get_frame.return_value = "frame-1"

        detector = MagicMock()

        def add_detection(ctx):
            ctx.detections.append(
                Detection(label="drone", confidence=0.9, bbox=(1, 2, 3, 4))
            )
            return ctx

        detector.detect.side_effect = add_detection

        visualizer = MagicMock()
        visualizer.show.return_value = False

        pipeline = DetectionPipeline(
            source=source,
            detector=detector,
            visualizer=visualizer,
        )

        pipeline.run()

        visualizer.show.assert_called_once_with(
            frame="frame-1",
            boxes=[(1, 2, 3, 4)],
            is_image=True,
        )