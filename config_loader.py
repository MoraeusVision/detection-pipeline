import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    # Describes the detector strategy to instantiate.
    type: str
    model_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectConfig:
    # Holds the normalized settings needed to run one project configuration.
    source: str
    show: bool
    detector: DetectorConfig
    config_path: Path


def load_project_config(config_path: str | Path) -> ProjectConfig:
    # Resolve the config path once so every relative path can be anchored to it.
    path = Path(config_path).expanduser().resolve()
    raw_config = json.loads(path.read_text(encoding="utf-8"))
    try:
        # Pull out the required fields early so invalid configs fail fast.
        detector = raw_config["detector"]
        detector_type = detector["type"]
        model_path = detector.get("model_path")
        detector_params = detector.get("params", {})
        source = raw_config["source"]
    except KeyError as exc:
        raise ValueError(f"Missing required config field: {exc.args[0]}") from exc

    if not isinstance(detector_params, dict):
        raise ValueError("detector.params must be an object")

    return ProjectConfig(
        source=_resolve_source_path(source, path.parent),
        show=bool(raw_config.get("show", False)),
        detector=DetectorConfig(
            type=detector_type,
            model_path=_resolve_file_path(model_path, path.parent) if model_path else None,
            params=detector_params,
        ),
        config_path=path,
    )


def _resolve_source_path(source: str, base_dir: Path) -> str:
    # Camera indices and stream URLs should stay untouched.
    if source.isdigit() or _is_stream_source(source):
        return source

    return _resolve_file_path(source, base_dir)


def _resolve_file_path(file_path: str, base_dir: Path) -> str:
    # Make relative project paths independent of the current working directory.
    path = Path(file_path)
    if path.is_absolute():
        return str(path)

    return str((base_dir / path).resolve())


def _is_stream_source(source: str) -> bool:
    return source.startswith(("rtsp://", "http://", "https://"))