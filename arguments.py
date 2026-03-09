import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run detection inference pipeline")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image, video file, or stream URL"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, visualizes the frames with bounding boxes"
    )
    
    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        choices=["yolo", "mediapipe", "rfdetr"],
        help="Detector type"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )
    
    return parser.parse_args()