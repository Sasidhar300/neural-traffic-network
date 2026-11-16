# simulation_tracker.py
import threading

class SimulationTracker:
    """
    A thread-safe singleton class to track if a simulation is currently running
    and store its logs in memory.
    """
    _instance = None
    _lock = threading.Lock()
    _is_running = False
    _current_logs = ""
    _is_complete = False
    _just_started = False
    _is_error = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._is_running = False
                    cls._instance._current_logs = ""
                    cls._instance._is_complete = False
                    cls._instance._just_started = False
                    cls._instance._is_error = False
        return cls._instance

    def start_simulation(self) -> bool:
        """
        Attempts to mark the simulation as running and clears previous logs.
        """
        with self._lock:
            if self._is_running:
                return False  # Already running
            self._is_running = True
            self._is_complete = False
            self._just_started = True
            self._current_logs = ""
            self._is_error = False  # Reset error state
            return True

    def stop_simulation(self):
        """Marks the simulation as stopped."""
        with self._lock:
            self._is_running = False
            self._is_complete = True

    def is_running(self) -> bool:
        """Checks if a simulation is currently running."""
        with self._lock:
            return self._is_running

    def is_error(self) -> bool:
        """Checks if an error has occurred during the simulation."""
        with self._lock:
            return self._is_error

    def append_log(self, message: str):
        """Appends a log message to the current simulation's log storage."""
        if not message: # Avoid appending empty strings
            return
        with self._lock:
            # Ensure message ends with newline for consistency if needed later
            self._current_logs += message

    def append_error(self, message: str):
        """Appends a log message to the current simulation's log storage."""
        if not message: # Avoid appending empty strings
            return
        with self._lock:
            # Ensure message ends with newline for consistency if needed later
            self._current_logs += message
            self._is_error = True  # Mark as error

    def get_current_logs(self) -> str:
        """
        Gets all logs for the current Run
        """
        with self._lock:
            # Create a copy to avoid holding the lock while processing
            return self._current_logs

    def is_complete(self) -> bool:
        """Checks if the simulation has been marked as completed."""
        with self._lock:
            return self._is_complete

    def just_started(self) -> bool:
        """Checks if start_simulation() has been successfully called at least once."""
        with self._lock:
            if self._just_started:
                ret = True
                self._just_started = False
            else:
                ret = False
            return ret

    def reset(self):
        """Resets the simulation tracker to its initial state."""
        with self._lock:
            self._just_started = False

# --- Keep using the single instance export ---
simulation_tracker = SimulationTracker()
