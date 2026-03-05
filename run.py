
from source_factory import SourceFactory, ImageSource, VideoSource, StreamSource
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager


def main():
    args = parse_arguments()
    source = SourceFactory.create(args.source) # Returns the source depending on media
    visualizer = Visualizer() if args.show else None

    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    if source:
        cleanup.add(source.cleanup)
    if visualizer:
        cleanup.add(visualizer.close)    
    
    while True:
        frame = source.get_frame()
        if frame is None:
            break

        """
        Inference pipeline comes here
        """

        # Show frame
        if visualizer is not None:
            is_image = isinstance(source, ImageSource) # Keep the frame if just an image
            delay = 1
            if isinstance(source, VideoSource) or (isinstance(source, StreamSource) and source.is_youtube):
                delay = int(1000 / source.fps)
            if not visualizer.show(frame=frame, is_image=is_image, delay=delay):
                break

    # Clean up
    cleanup.run()


if __name__ == "__main__":
    main()