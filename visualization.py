import cv2

class Visualizer:
    def __init__(self, window_name: str="Detection", width: int=800, height: int=600):
        """
        Args:
            window_name (str): Name of the display window
            width (int): Width of the window
            height (int): Height of the window
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        

        # Create a resizable window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def show(self, frame, boxes=None, is_image=False, delay: int = 1):
        """
        Display a single frame with optional bounding boxes.

        Args:
            frame (np.ndarray): Image to display
            boxes (list of [x1, y1, x2, y2], optional): Bounding boxes to draw
        """
        # Make a copy so original frame is not modified
        display_frame = frame.copy()

        # Draw bounding boxes if provided
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the frame in the window
        cv2.imshow(self.window_name, display_frame)
        
        if is_image:
            key = cv2.waitKey(0) & 0xFF # wait until key is pressed
        else:
            key = cv2.waitKey(delay) & 0xFF

        # Exit if q is pressed
        if key == ord('q'):
            return False

        return True
        
    def close(self):
        """Close the display window."""
        print("Closing windows..")
        cv2.destroyAllWindows()