from .rfdetr_detector import RFDETRDetector


class DetectorFactory():
    @staticmethod
    def create(detector_name, model_path):
        detector_name = detector_name.lower()

        if detector_name == "rfdetr":
            return RFDETRDetector(model_path)
        else:
            raise ValueError(f"Unsupported detector: {detector_name}")