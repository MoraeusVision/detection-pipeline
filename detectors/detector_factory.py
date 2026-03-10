from .rfdetr_detector import RFDETRDetector


DETECTOR_REGISTRY = {
    "rfdetr": RFDETRDetector,
}


class DetectorFactory():
    @staticmethod
    def create(detector_config):
        detector_name = detector_config.type.lower()

        detector_class = DETECTOR_REGISTRY.get(detector_name)
        if detector_class is None:
            raise ValueError(f"Unsupported detector: {detector_name}")

        detector_kwargs = dict(detector_config.params)
        if detector_config.model_path is not None:
            detector_kwargs["model_path"] = detector_config.model_path

        return detector_class(**detector_kwargs)