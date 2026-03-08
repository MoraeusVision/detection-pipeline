class EventManager:
    def __init__(self):
        # Dictionary mapping event names to a list of observers
        # Example: {"on_frame": [Visualizer, Logger]}
        self._observers = {}

    def register(self, event_name, observer):
        """
        Register an observer for a specific event.

        Args:
            event_name (str): Name of the event (e.g., "on_frame")
            observer (object): Object implementing handle_event()
        """
        # Create a new list if this event has no observers yet
        if event_name not in self._observers:
            self._observers[event_name] = []

        # Add the observer to the event's observer list
        self._observers[event_name].append(observer)

    def notify(self, event_name, data=None):
        """
        Notify all observers subscribed to a specific event.

        Args:
            event_name (str): Event that occurred
            data (any): Optional data passed to observers
        """
        # If no observers are registered for this event, do nothing
        if event_name not in self._observers:
            return
        
        # Notify each observer by calling its handle_event method
        for observer in self._observers[event_name]:
            observer.handle_event(event_name, data)

    def get_observers(self, event_name):
        """
        Return all observers registered for a given event.

        Args:
            event_name (str): Event name

        Returns:
            list: List of observers (empty if none exist)
        """
        return self._observers.get(event_name, [])
    
    def get_observers_overview(self):
        """
        Return a simplified overview of all registered observers.

        Returns:
            dict: Mapping of event names to observer class names
                  Example:
                  {
                      "on_frame": ["Visualizer", "PersonDetector"],
                      "on_prediction": ["Logger"]
                  }
        """
        overview = {}

        # Iterate through all events and their observers
        for event, observers in self._observers.items():
            # Store only the class names for readability
            overview[event] = [type(o).__name__ for o in observers]

        return overview


class FrameContext:
    def __init__(self, frame, timestamp):
        """
        Context object passed through the pipeline for each frame.

        Args:
            frame (np.ndarray): Image/frame data
            timestamp (float): Time when the frame was captured/processed
        """
        self.frame = frame
        self.timestamp = timestamp