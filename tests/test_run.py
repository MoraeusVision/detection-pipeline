from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run


class TestRun:
    @patch("run.time.time", return_value=123.0)
    @patch("run.CleanupManager")
    @patch("run.EventManager")
    @patch("run.SourceFactory.create")
    @patch("run.parse_arguments")
    def test_main_without_visualizer_does_not_register_none_observer(
        self,
        mock_parse_arguments,
        mock_source_create,
        mock_event_manager_class,
        mock_cleanup_manager_class,
        mock_time,
    ):
        mock_parse_arguments.return_value = SimpleNamespace(source="video.mp4", show=False)

        mock_source = MagicMock()
        mock_source.is_static = False
        mock_source.get_frame.side_effect = ["frame-1", None]
        mock_source_create.return_value = mock_source

        mock_event_manager = MagicMock()
        mock_event_manager_class.return_value = mock_event_manager

        mock_cleanup = MagicMock()
        mock_cleanup_manager_class.return_value = mock_cleanup

        run.main()

        mock_event_manager.register.assert_not_called()
        assert mock_event_manager.notify.call_count == 2
        mock_cleanup.add.assert_called_once_with(mock_source.cleanup)
        mock_cleanup.run.assert_called_once()