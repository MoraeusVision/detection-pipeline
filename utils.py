import logging


class CleanupManager:
    def __init__(self):
        # Store cleanup functions
        self._tasks = []

    def add(self, func, *args, **kwargs):
        """
        Add a cleanup function to be called later.

        Args:
            func (callable): function to call
            *args, **kwargs: arguments for the function
        """
        self._tasks.append((func, args, kwargs))

    def run(self):
        """Run all registered cleanup tasks."""
        for func, args, kwargs in self._tasks:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logging.exception("Cleanup failed for %s: %s", func, e)
        self._tasks.clear()