
import logging
import time

from source_factory import SourceFactory
from detectors.detector_factory import DetectorFactory
from events import EventManager
from pipeline_context import FrameContext
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager
from config_loader import read_from_config


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_arguments()
    config = read_from_config(args.config)
    
    source = SourceFactory.create(source_path=config["source"])  # Returns the source depending on media
    model = DetectorFactory.create(detector_name=config["detector"], model_path=config["model_path"])
    visualizer = Visualizer() if config["show"] else None

    event_manager = EventManager()
    if visualizer:
        event_manager.register("on_frame", visualizer)

    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)

    try:
        frame = None
        is_static = source.is_static if source else False

        while True:
            # Only fetch a new frame if we aren't paused or if we have no frame yet
            if frame is None or visualizer is None or not visualizer.paused:
                frame = source.get_frame()
                ctx = FrameContext(frame, time.time())

                event_manager.notify("on_frame", ctx)

                if frame is None:
                    break

                if visualizer is None and is_static:
                    break


            # Show frame
            if visualizer is not None:
                if not visualizer.show(frame=frame, is_image=is_static):
                    break
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()