"""
Core Controller Module for Student Use
======================================

This module provides the **primary public API** for controlling the Unitree Go2
robot.

Students interact exclusively with :class:`UnitreeGo2Controller`, which exposes
high-level functionality such as movement, video, audio, OCR, input, and LIDAR
through strongly-typed properties.

Design Goals
------------
    - Provide a **single entry point** for all robot capabilities
    - Hide hardware and SDK complexity behind safe abstractions
    - Support both **real hardware** and **simulation** transparently
    - Enforce clean startup and shutdown semantics

Usage Model
-----------
Students:
    - Create a controller
    - Access modules via properties (e.g., ``controller.movement``)
    - Write behavior/state logic on top of this API

Internal systems:
    - Manage hardware backends
    - Handle module registration and lifecycle
    - Enforce safety checks during shutdown

Example
-------
>>> import time
>>> from src.core.unitree_control_core import UnitreeGo2Controller
>>> 
>>> unitree_controller = UnitreeGo2Controller(use_sdk=True)
>>> unitree_controller.movement.stand_up()
>>> time.sleep(1)
>>> unitree_controller.safe_shutdown()
"""


from typing import Callable, List, Optional, Dict
import threading

from src.controller_input_control.input_signal import InputSignal
from src.core.base_module import DogModule
from src.core.module_registry import AudioModule, InputModule, ModuleRegistry, ModuleType, MovementModule, OCRModule, VideoModule
from src.core.hardware_control import HardwareInterface, SimulatedHardware, UnitreeSDKHardware
from src.lidar_control.decoder import LIDARModule


# TTS: https://medium.com/@vndee.huynh/build-your-own-voice-assistant-and-run-it-locally-whisper-ollama-bark-c80e6f815cba
# Digging into Dog: https://www.darknavy.org/darknavy_insight/the_jailbroken_unitree_robot_dog

# Github Repo Searcher: https://github.com/search?type=Code

# Some very interesting turtorial with the G02: https://hackmd.io/@c12hQ00ySVi6JYIERU7bCg/ByAOr12qJg

# Update to JetPack 6.x on Dog's Jetson: https://theroboverse.com/unitree-go2-edu-jetpack-6-2-1-update/


class UnitreeGo2Controller:
    """    
    Primary control interface for the Unitree Go2 robot.

    This class is the **main entry point** that students and users interact with.
    It manages hardware initialization, module lifecycles, safety checks, and
    shutdown coordination.

    Modules are accessed via typed properties rather than direct instantiation,
    ensuring compile-time safety and consistent behavior.

    Notes
    -----
        - All hardware access is routed through this controller.
        - Modules are created and initialized automatically.
        - Accessing modules after shutdown is prohibited.
        - Supports both SDK-backed hardware and simulation.

    See Also
    --------
    DogModule
    ModuleRegistry
    """

    def __init__(self, use_sdk: Optional[bool]):
        """
        Create a new controller instance.

        Parameters
        ----------
        use_sdk : bool or None
            If ``True``, forces use of the Unitree SDK.
            If ``False``, forces simulation mode.
            If ``None``, SDK availability is auto-detected.

        Raises
        ------
        RuntimeError
            If hardware initialization fails.
        """
        if use_sdk is None:
            use_sdk = self._detect_sdk()

        self.use_sdk = use_sdk

        self._shutdown_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._cleanup_callbacks: List[Callable[[], None]] = []

        self._hardware: HardwareInterface = (
            UnitreeSDKHardware() if use_sdk else SimulatedHardware()
        )
        self._hardware.initialize()

        self._modules: Dict[ModuleType, DogModule] = {}
        self._register_default_modules()
        self._initialize_input_bindings()

        print(f"[Controller] Initialized in {'SDK' if use_sdk else 'SIMULATION'} mode\n")


    def _detect_sdk(self) -> bool:
        try:
            import unitree_sdk2py
            return True
        
        except ImportError:
            return False
        

    def _initialize_input_bindings(self):
        if self.use_sdk:
            self.input.register_callback(
                InputSignal.BUTTON_A,
                lambda _: self._shutdown_event.set(),
                "emergency_stop"
            )


    def _register_default_modules(self):
        self.add_module(ModuleType.MOVEMENT, hardware=self._hardware)

        if self.use_sdk:
            self.add_module(ModuleType.INPUT, use_sdk=self.use_sdk)


    def add_module(self, module_type: ModuleType, **kwargs) -> None:
        """
        Add a module to the controller.

        Modules are identified using :class:`ModuleType` enums to ensure
        correctness and prevent invalid configurations.

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

        Notes
        -----
        - SDK-required modules are skipped automatically in simulation mode.
        - Modules are initialized immediately upon addition.

        Example
        -------
        >>> controller.add_module(ModuleType.VIDEO)
        """
        descriptor = ModuleRegistry.get_descriptor(module_type)
        if descriptor is None:
            raise ValueError(f"Module type {module_type} is not registered")
        
        if descriptor.requires_sdk and not self.use_sdk:
            print(f"[Controller] Warning: {module_type.name} requires SDK mode")
            return
        
        module: DogModule = descriptor.create_instance(**kwargs)
        self._modules[module_type] = module

        module.initialize()
    

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
        return ModuleRegistry.get_list_available(self.use_sdk)
    

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
        InputMOdule

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

        Callbacks are executed during safe shutdown after modules are stopped
        but before hardware is released.

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
                    module.shutdown()
                except Exception as e:
                    print(f"[Controller] Failed to shutdown {module_type.name}: {e}")
            
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[Controller] Cleanup callback failed: {e}")
            
            try:
                self._hardware.shutdown()
            except Exception as e:
                print(f"[Controller] Hardware shutdown failed: {e}")
            
            print("[Controller] Shutdown complete")


    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested"""
        return self._shutdown_event.is_set()

    