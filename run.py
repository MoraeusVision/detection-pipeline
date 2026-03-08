
import logging

from source_factory import SourceFactory, ImageSource, VideoSource, StreamSource
from events import EventManager, FrameContext
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager
import time


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_arguments()
    source = SourceFactory.create(args.source)  # Returns the source depending on media
    visualizer = Visualizer() if args.show else None

    event_manager = EventManager()
    event_manager.register("on_frame", visualizer)

    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.cleanup)

    try:
        frame = None
        is_image = isinstance(source, ImageSource) if source else False

        while True:
            # Only fetch a new frame if we aren't paused or if we have no frame yet
            if frame is None or (visualizer and not visualizer.paused):
                frame = source.get_frame()
                ctx = FrameContext(frame, time.time())

                event_manager.notify("on_frame", ctx)

                if frame is None:
                    break


            # Show frame
            if visualizer is not None:
                if not visualizer.show(frame=frame, is_image=is_image):
                    break
    except Exception:
        logging.exception("Unhandled error in main loop")
    finally:
        # Ensure resources are cleaned up even on error
        cleanup.run()


if __name__ == "__main__":
    main()