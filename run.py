
import logging

from source_factory import SourceFactory
from detectors.detector_factory import DetectorFactory
from events import EventManager
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager
from config_loader import read_from_config
from pipeline import DetectionPipeline


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_arguments()
    config = read_from_config(args.config)
    
    source = SourceFactory.create(source_path=config["source"])  # Returns the source depending on media
    model = DetectorFactory.create(
        detector_name=config["detector"],
        model_path=config["model_path"],
        confidence_threshold=config.get("conf", 0.5),
    )
    visualizer = Visualizer() if config["show"] else None

    event_manager = EventManager()
    if visualizer:
        event_manager.register("on_inference_result", visualizer)

    pipeline = DetectionPipeline(source=source, model=model, event_manager=event_manager)
    
    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)

    try:
        pipeline.run()
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()