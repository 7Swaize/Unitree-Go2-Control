"""
Audio Module for User Use
=========================

This module provides simple text-to-speech functionality for the dog robot. 
It wraps the underlying pyttsx3 engine and handles thread-safe playback.

Users should not access or construct this class directly. 
Rather, they should access it through a :class:`~core.controller.Go2Controller` instance.
"""

import threading
import pyttsx3
from typing_extensions import override
from core.module import DogModule


class AudioModule(DogModule):
    """
    High-level text-to-speech interface for users. This is called internally,
    and should not be accessed directly by users.

    ``AudioModule`` allows users to easily play audio or speech
    using simple text commands. It supports both **blocking** and **asynchronous** playback.

    Notes
    -----
    - Users do not need to manage threads or the underlying pyttsx3 engine. Though it can be done via a call :meth:`AudioModule.get_engine`.
    """

    def __init__(self):
        super().__init__("Audio")

    @override
    def _initialize(self) -> None:
        """
        Initialize the text-to-speech engine.
        """
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
            If False (default), playback occurs asynchronously in a background thread.
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

        Notes
        -----
        This method is primarily for advanced use cases. Users typically
        do not need to call this directly. Unless they want to modify the 
        configuration of the text-to-speech
        """
        if not self._initialized:
            raise RuntimeError("[Audio] Audio module must be initialized before accessing pyttsx3 engine")
        
        return self._engine


    @override
    def _shutdown(self) -> None:
        """
        Shut down the audio module. This is handled automatically and shouldn't be called by users.

        Stops the text-to-speech engine and marks the module
        as uninitialized.
        """
        self._initialized = False