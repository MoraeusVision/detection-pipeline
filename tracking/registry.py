from .byte_tracker import ByteTrackTracker


class TrackerFactory:
	@staticmethod
	def create(tracker_name, config=None):
		if tracker_name is None:
			return None

		tracker_name = tracker_name.lower()
		if tracker_name == "bytetrack":
			return ByteTrackTracker(config=config)

		raise ValueError(f"Unsupported tracker: {tracker_name}")
