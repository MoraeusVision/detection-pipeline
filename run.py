
from source_factory import SourceFactory, ImageSource
from arguments import parse_arguments
from visualization import Visualizer
from utils import CleanupManager


def main():
    args = parse_arguments()
    source = SourceFactory.create(args.source) # Returns the source depending on media
    print(source)
    # Cleanup manager collects cleanup methods and run them in the end
    cleanup = CleanupManager()
    visualizer = Visualizer() if args.show else None
    if visualizer:
        cleanup.add(visualizer.close)
    
    while True:
        frame = source.get_frame()
        if frame is None:
            break

        # Show frame
        if visualizer is not None:
            is_image = isinstance(source, ImageSource)
            if not visualizer.show(frame=frame, is_image=is_image):
                break

    # Clean up
    cleanup.run()


if __name__ == "__main__":
    main()