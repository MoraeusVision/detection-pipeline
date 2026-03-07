class EventManager:
    def __init__(self):
        self._observers = {}

    def register(self, event_name, observer):
        if event_name not in self._observers:
            self._observers[event_name] = []

        self._observers[event_name].append(observer)

    def notify(self, event_name, data=None):
        if event_name not in self._observers:
            return
        
        for observer in self._observers[event_name]:
            observer.handle_event(event_name, data)


class FrameContext:
    def __init__(self, frame, timestamp):
        self.frame = frame
        self.timestamp = timestamp