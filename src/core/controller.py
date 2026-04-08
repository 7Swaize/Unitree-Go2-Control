"""
Core Controller Module for User Use
===================================

This module provides the **primary public API** for controlling the Unitree Go2 robot.

Users should interact exclusively with :class:`Go2Controller`, which exposes
high-level functionality such as movement, video, audio, OCR, input, and LIDAR through properties.
"""

import threading
from typing import Callable, List, Dict

from modules.input import InputSignal
from core.module import DogModule
from core.registry import ModuleRegistry, ModuleType
from modules.audio import AudioModule
from modules.input import InputModule
from modules.movement import MovementModule
from modules.ocr import OCRModule
from modules.video import VideoModule
from modules.lidar import LIDARModule
from hardware.interfaces.hardware_interface import HardwareInterface
from hardware.native.native_hardware import NativeHardware
from hardware.virtual.virtual_hardware import VirtualHardware


# TTS: https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba
# Digging into Dog: https://www.darknavy.org/darknavy_insight/the_jailbroken_unitree_robot_dog

# Some very interesting turtorial with the G02: https://hackmd.io/@c12hQ00ySVi6JYIERU7bCg/ByAOr12qJg

# Update to JetPack 6.x on Dog's Jetson: https://theroboverse.com/unitree-go2-edu-jetpack-6-2-1-update/

# TETHERING + ROS2 SETUP: https://www.youtube.com/watch?v=Q_dqPLJDPms


class Go2Controller:
    """    
    Primary control interface for the Unitree Go2 robot.

    This class is the **main entry point** that users interact with.
    It manages hardware initialization, module lifecycles, safety checks, and shutdown.

    Modules are accessed via properties rather than direct instantiation.

    Notes
    -----
        - All hardware access is routed through this controller.
        - Modules creation, initialization, and shutdown is handled automatically.
        - Supports both SDK-backed hardware and a Mujoco simulation.

    See Also
    --------
    DogModule
    ModuleRegistry
    """

    def __init__(self, use_sdk: bool):
        """
        Create a new controller instance.

        Parameters
        ----------
        use_sdk : bool or None
            If ``True``, forces use of the Unitree SDK.
            If ``False``, forces simulation mode.

        Raises
        ------
        RuntimeError
            If hardware backend initialization fails.
        """
        self._use_sdk = use_sdk

        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._cleanup_callbacks: List[Callable[[], None]] = []

        self._hardware: HardwareInterface = (
            NativeHardware() if use_sdk else VirtualHardware()
        )
        self._hardware._initialize()

        self._modules: Dict[ModuleType, DogModule] = {}
        self._register_default_modules()
        self._initialize_input_bindings()

        print(f"[Controller] Initialized in {'SDK' if use_sdk else 'SIMULATION'} mode\n")


    def _initialize_input_bindings(self):
        if self._use_sdk:
            self.input.register_callback(
                InputSignal.BUTTON_A,
                lambda _: self._shutdown_event.set(),
                "emergency_stop"
            )


    def _register_default_modules(self):
        self.add_module(ModuleType.MOVEMENT, hardware=self._hardware)

        if self._use_sdk:
            self.add_module(ModuleType.INPUT, use_sdk=self._use_sdk)
            self.add_module(ModuleType.LIDAR, use_sdk=self._use_sdk)


    def add_module(self, module_type: ModuleType, **kwargs) -> None:
        """
        Add a module to the controller. Modules are initialized immediately upon addition.
        Manual module initialization is discouraged and should not be attempted. 

        Modules are identified using :class:`ModuleType` enums.

        Parameters
        ----------
        module_type : ModuleType
            The type of module to add.
        **kwargs
            Constructor arguments passed to the module implementation.

        Raises
        ------
        ValueError
            If the module type is not registered.
        ValueError
            If the module type depends on native hardware support, but the controller is not set up to provide it.
        """
        descriptor = ModuleRegistry.get_descriptor(module_type)
        if descriptor is None:
            raise ValueError(f"[Controller] Module type '{module_type.name}' is not registered")
        
        if descriptor._requires_sdk and not self._use_sdk:
            raise ValueError(f"[Controller] Module type '{module_type.name}' requires native hardware support")
        
        module: DogModule = descriptor._create_instance(**kwargs)
        self._modules[module_type] = module

        module._initialize()
    

    def has_module(self, module_type: ModuleType) -> bool:
        """
        Check whether a module is currently loaded.

        Returns
        -------
        bool
        """
        return module_type in self._modules
    
    
    def get_available_modules(self) -> list[ModuleType]:
        """
        List all module types available in the current mode.

        Returns
        -------
        list of ModuleType
        """
        return ModuleRegistry.get_list_available(self._use_sdk)
    

    @property
    def video(self) -> VideoModule:
        """
        Access the video capture module.

        Returns
        -------
        VideoModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.VIDEO)
        if not isinstance(module, VideoModule):
            raise RuntimeError("Video module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access Video module after shutdown has been requested")
        
        return module
    

    @property
    def movement(self) -> MovementModule:
        """
        Access the movement control module.

        Returns
        -------
        MovementModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.MOVEMENT)
        if not isinstance(module, MovementModule):
            raise RuntimeError("Movement module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access Movement module after shutdown has been requested")

        return module
    

    @property
    def ocr(self) -> OCRModule:
        """
        Access the ocr control module.

        Returns
        -------
        OCRModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.OCR)
        if not isinstance(module, OCRModule):
            raise RuntimeError("OCR module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access OCR module after shutdown has been requested")
        
        return module
    

    @property
    def audio(self) -> AudioModule:
        """
        Access the audio control module.

        Returns
        -------
        AudioModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.AUDIO)
        if not isinstance(module, AudioModule):
            raise RuntimeError("Audio module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access Audio module after shutdown has been requested")
        
        return module
    

    @property
    def input(self) -> InputModule:
        """
        Access the input control module.

        Returns
        -------
        InputModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.INPUT)
        if not isinstance(module, InputModule):
            raise RuntimeError("Input module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access Input module after shutdown has been requested")

        return module
    

    @property
    def lidar(self) -> LIDARModule:
        """
        Access the lidar control module.

        Returns
        -------
        LIDARModule

        Raises
        ------
        RuntimeError
            If the module is not loaded or shutdown has been requested.
        """
        module = self._modules.get(ModuleType.LIDAR)
        if not isinstance(module, LIDARModule):
            raise RuntimeError("LIDAR module not loaded")
        
        if self.is_shutdown_requested():
            raise RuntimeError("Cannot access LIDAR module after shutdown has been requested")
        
        return module
    

    def register_cleanup_callback(self, callback: Callable[[], None]):
        """
        Register a cleanup callback.

        Callbacks are executed during safe shutdown after modules are stopped but before hardware is released.

        Parameters
        ----------
        callback : callable
            Zero-argument function to execute during shutdown.
        """
        self._cleanup_callbacks.append(callback)


    def safe_shutdown(self):
        """
        Perform a coordinated and safe shutdown.

        This method:
            - Stops movement immediately
            - Signals shutdown to all subsystems
            - Shuts down all modules
            - Executes registered cleanup callbacks
            - Releases hardware resources

        Notes
        -----
        This method is **idempotent** and may be safely called multiple times.
        """
        print("\n[Controller] Starting safe shutdown...")
        self.movement.stop()
        
        with self._shutdown_lock:
            if not self._shutdown_event.is_set():
                self._shutdown_event.set()

            for module_type, module in self._modules.items():
                try:
                    module._shutdown()
                except Exception as e:
                    print(f"[Controller] Failed to shutdown {module_type.name}: {e}")
            
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[Controller] Cleanup callback failed: {e}")
            
            try:
                self._hardware._shutdown()
            except Exception as e:
                print(f"[Controller] Hardware shutdown failed: {e}")
            
            print("[Controller] Shutdown complete")


    def is_shutdown_requested(self) -> bool:
        """
        Check if shutdown has been requested
        
        Returns
        -------
        bool
        """
        return self._shutdown_event.is_set()
