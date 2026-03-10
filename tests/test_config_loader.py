from pathlib import Path

from config_loader import load_project_config


class TestConfigLoader:
    def test_load_project_config_resolves_paths_relative_to_config(self, tmp_path: Path):
        project_dir = tmp_path / "project_a"
        project_dir.mkdir()
        (project_dir / "models").mkdir()

        config_path = project_dir / "config.json"
        config_path.write_text(
            """
{
  "source": "input.mp4",
  "show": true,
  "detector": {
    "type": "rfdetr",
    "model_path": "models/checkpoint_best_total.pth"
  }
}
""".strip(),
            encoding="utf-8",
        )

        config = load_project_config(config_path)

        assert config.show is True
        assert config.detector.type == "rfdetr"
        assert config.source == str((project_dir / "input.mp4").resolve())
        assert config.detector.model_path == str(
            (project_dir / "models" / "checkpoint_best_total.pth").resolve()
        )
        assert config.detector.params == {}

    def test_load_project_config_keeps_camera_and_stream_sources(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            """
{
  "source": "0",
  "detector": {
    "type": "rfdetr",
    "model_path": "model.pth"
  }
}
""".strip(),
            encoding="utf-8",
        )

        config = load_project_config(config_path)

        assert config.source == "0"

    def test_load_project_config_preserves_detector_params_and_optional_model_path(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            """
{
  "source": "0",
  "detector": {
    "type": "mediapipe",
    "params": {
      "min_detection_confidence": 0.5,
      "running_mode": "image"
    }
  }
}
""".strip(),
            encoding="utf-8",
        )

        config = load_project_config(config_path)

        assert config.detector.model_path is None
        assert config.detector.params == {
            "min_detection_confidence": 0.5,
            "running_mode": "image",
        }

    def test_load_project_config_rejects_non_object_detector_params(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            """
{
  "source": "0",
  "detector": {
    "type": "rfdetr",
    "model_path": "model.pth",
    "params": []
  }
}
""".strip(),
            encoding="utf-8",
        )

        try:
            load_project_config(config_path)
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert str(exc) == "detector.params must be an object"
