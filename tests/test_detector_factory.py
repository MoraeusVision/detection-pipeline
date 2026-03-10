from types import SimpleNamespace

import pytest

from detectors.detector_factory import DetectorFactory
from detectors.rfdetr_detector import RFDETRDetector


class TestDetectorFactory:
    def test_create_builds_detector_from_config(self):
        detector = DetectorFactory.create(
            SimpleNamespace(
                type="rfdetr",
                model_path="models/checkpoint_best_total.pth",
                params={"device": "cpu"},
            )
        )

        assert isinstance(detector, RFDETRDetector)
        assert detector.model_path == "models/checkpoint_best_total.pth"
        assert detector.params == {"device": "cpu"}

    def test_create_rejects_unknown_detector_type(self):
        with pytest.raises(ValueError, match="Unsupported detector: unknown"):
            DetectorFactory.create(
                SimpleNamespace(type="unknown", model_path=None, params={})
            )