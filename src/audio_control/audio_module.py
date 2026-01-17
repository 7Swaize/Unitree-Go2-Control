"""
Audio Module for Student Use
============================

This module provides simple text-to-speech functionality for the dog robot. 
It wraps the underlying pyttsx3 engine and handles thread-safe playback.

Students should not access or construct this class directly. Rather, they should access it through the :class:`~src.core.unitree_control_core.UnitreeGo2Controller` instance.


Example
-------
>>> from src.core.unitree_control_core import UnitreeGo2Controller
>>>
>>> unitree_controller = UnitreeGo2Controller(use_sdk=False)
>>> unitree_controller.add_module(ModuleType.AUDIO)
>>> unitree_controller.audio.play_audio("Hello World") # Text-to-speech of the current string
"""

import threading
import pyttsx3
from src.core.base_module import DogModule


class AudioModule(DogModule):
    """
    High-level text-to-speech interface for students. This is called internally,
    and should not be called directly by students.

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
        """
        Initialize the AudioModule.

        The TTS engine is initialized automatically. No parameters
        are required.
        """
        super().__init__("Audio")


    def initialize(self) -> None:
        """
        Initialize the text-to-speech engine.

        This method is called automatically by the constructor. It
        ensures that the engine is only initialized once.
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
            If False (default), playback occurs asynchronously in
            a background thread.

        Examples
        --------
        >>> audio.play_audio("Hello!")                # asynchronous playback
        >>> audio.play_audio("Wait for me.", blocking=True)  # synchronous playback
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
        This method is primarily for advanced use cases. Students typically
        do not need to call this directly. Unless they want to modify the 
        configuration of the text-to-speech
        """
        if not self._initialized:
            raise RuntimeError("[Audio] Audio module must be initialized before accessing pyttsx3 engine")
        
        return self._engine


    def shutdown(self) -> None:
        """
        Shut down the audio module. This is handled automatically and shouldn't be called by students.

        Stops the text-to-speech engine and marks the module
        as uninitialized.
        """
        self._initialized = False