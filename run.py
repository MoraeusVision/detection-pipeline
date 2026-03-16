
import logging

from source_factory import SourceFactory
from detectors.detector_factory import DetectorFactory
from tracking.registry import TrackerFactory
from events import EventManager
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager, SaveManager
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
    tracker = None
    # Only create a tracker for non-static sources.
    if config.get("tracker") and not source.is_static:
        tracker = TrackerFactory.create(
            tracker_name=config["tracker"],
        )

    visualizer = Visualizer() if config["show"] else None
    saver = SaveManager() if config["save"] else None

    event_manager = EventManager()
    if visualizer:
        event_manager.register("on_inference_result", visualizer)
    if saver:
        event_manager.register("on_inference_result", saver)

    pipeline = DetectionPipeline(
        source=source,
        model=model,
        tracker=tracker,
        event_manager=event_manager,
    )
    
    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)
    if saver and not source.is_static:
        cleanup.add(saver.save_video)

    try:
        pipeline.run()
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()