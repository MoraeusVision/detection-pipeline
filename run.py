
import logging

from source_factory import SourceFactory
from detectors.detector_factory import DetectorFactory
from events import EventManager
from pipeline import DetectionPipeline
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_arguments()
    source = SourceFactory.create(source_path=args.source)  # Returns the source depending on media
    detector = DetectorFactory.create(detector_name=args.detector, model_path=args.model)
    visualizer = Visualizer() if args.show else None

    event_manager = EventManager()

    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)

    pipeline = DetectionPipeline(
        source=source,
        detector=detector,
        event_manager=event_manager,
        visualizer=visualizer,
    )

    try:
        pipeline.run()
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()