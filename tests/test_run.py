from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run


class TestRun:
    @patch("run.DetectionPipeline")
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.DetectorFactory.create")
    @patch("run.SourceFactory.create")
    @patch("run.load_project_config")
    @patch("run.parse_arguments")
    def test_main_builds_pipeline_without_visualizer(
        self,
        mock_parse_arguments,
        mock_load_project_config,
        mock_source_create,
        mock_detector_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_pipeline_class,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(config="projects/demo/config.json")
        mock_load_project_config.return_value = SimpleNamespace(
            source="video.mp4",
            show=False,
            detector=SimpleNamespace(type="rfdetr", model_path="model.pth"),
        )

        mock_source = MagicMock()
        mock_source.is_static = False
        mock_source_create.return_value = mock_source
        mock_detector = MagicMock()
        mock_detector_create.return_value = mock_detector

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        run.main()

        mock_pipeline_class.assert_called_once_with(
            source=mock_source,
            detector=mock_detector,
            event_manager=mock_event_manager,
            visualizer=None,
        )
        mock_pipeline.run.assert_called_once()
        mock_cleanup.add.assert_called_once_with(mock_source.cleanup)
        mock_cleanup.run.assert_called_once()

    @patch("run.Visualizer")
    @patch("run.DetectionPipeline")
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.DetectorFactory.create")
    @patch("run.SourceFactory.create")
    @patch("run.load_project_config")
    @patch("run.parse_arguments")
    def test_main_builds_pipeline_with_visualizer(
        self,
        mock_parse_arguments,
        mock_load_project_config,
        mock_source_create,
        mock_detector_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_pipeline_class,
        mock_visualizer_class,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(config="projects/demo/config.json")
        mock_load_project_config.return_value = SimpleNamespace(
            source="image.jpg",
            show=True,
            detector=SimpleNamespace(type="rfdetr", model_path="model.pth"),
        )

        mock_source = MagicMock()
        mock_source.is_static = True
        mock_source_create.return_value = mock_source
        mock_detector = MagicMock()
        mock_detector_create.return_value = mock_detector

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer

        mock_pipeline = MagicMock()
        mock_pipeline_class.return_value = mock_pipeline

        run.main()

        mock_pipeline_class.assert_called_once_with(
            source=mock_source,
            detector=mock_detector,
            event_manager=mock_event_manager,
            visualizer=mock_visualizer,
        )
        mock_pipeline.run.assert_called_once()
        mock_cleanup.add.assert_any_call(mock_source.cleanup)
        mock_cleanup.add.assert_any_call(mock_visualizer.cleanup)
        mock_cleanup.run.assert_called_once()