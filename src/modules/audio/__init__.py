"""Audio Module for Student Use"""

import threading
import pyttsx3
from core.module import DogModule


class AudioModule(DogModule):
    """Audio Module class"""
    """
    High-level text-to-speech interface for students.

    ``AudioModule`` allows students to easily play audio or speech
    using simple text commands. It supports both **blocking** and
    **asynchronous** playback.

    Notes
    -----
    - Students do not need to manage threads or the underlying
      pyttsx3 engine.
    - Initialization is handled automatically during construction.
    """

    def __init__(self):
        """Initialize the AudioModule."""
        super().__init__("Audio")


    def initialize(self) -> None:
        """Initialize the text-to-speech engine."""
        if self._initialized:
            return
        
        self._engine = pyttsx3.init()
        self._initialized = True


    def play_audio(self, text: str, blocking=False) -> None:
        """
        Play audio from text.

        Parameters
        ----------
        text : str
            The text string to be spoken by the robot.
        blocking : bool, optional
            If True, the method blocks until the speech finishes.
            If False (default), playback occurs asynchronously in
            a background thread.
        """
        self._engine.say(text)
        
        if blocking:
            self._engine.runAndWait()
        else:
            threading.Thread(target=self._engine.runAndWait, daemon=True).start()


    def get_engine(self) -> pyttsx3.Engine:
        """
        Access the underlying pyttsx3 engine.

        Returns
        -------
        pyttsx3.Engine
            The initialized text-to-speech engine.

        Raises
        ------
        RuntimeError
            If the audio module has not been initialized yet.
        """
        if not self._initialized:
            raise RuntimeError("[Audio] Audio module must be initialized before accessing pyttsx3 engine")
        
        return self._engine


    def shutdown(self) -> None:
        """Shut down the audio module."""
        self._initialized = False


__all__ = ["AudioModule"]
