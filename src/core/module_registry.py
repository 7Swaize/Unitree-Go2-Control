from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Generic, Optional, Type, TypeVar

from src.core.base_module import DogModule
from src.audio_control.audio_module import AudioModule
from src.controller_input_control.input_module import InputModule
from src.lidar_control.lidar_module import LIDARModule
from src.movement.movement_module import MovementModule
from src.ocr_control.ocr_module import OCRModule
from src.video_control.video_module import VideoModule


T = TypeVar('T', bound=DogModule)

class ModuleType(Enum):
    """
    Enumeration of all supported default module categories.

    This enum is used by the module registry to identify and construct
    modules dynamically.
    """
    VIDEO = auto()
    MOVEMENT = auto()
    INPUT = auto()
    OCR = auto()
    AUDIO = auto()
    LIDAR = auto()


@dataclass
class ModuleDescriptor(Generic[T]):
    """
    Descriptor linking a module type to its implementation.

    A descriptor contains all metadata required to construct and
    display a module in the system.

    Attributes
    ----------
    module_type : ModuleType
        Enum identifying the module.
    module_class : Type[DogModule]
        Concrete implementation class.
    display_name : str
        Human-readable name.
    requires_sdk : bool, optional
        Whether this module requires SDK support.
    """
    module_type: ModuleType
    module_class: Type[T]
    display_name: str
    requires_sdk: bool = False
    
    def create_instance(self, *args, **kwargs) -> T:
        """
        Instantiate the module.

        Returns
        -------
        DogModule
            A new module instance.
        """
        return self.module_class(*args, **kwargs)
    

class ModuleRegistry:
    """
    Central registry for all available modules.

    The registry maps ``ModuleType`` values to their corresponding
    implementations and metadata. It is responsible for controlling
    which modules are available based on system configuration.

    Notes
    -----
    - This is an internal system component.
    - Modules are registered at startup.
    """

    _descriptors: Dict[ModuleType, ModuleDescriptor] = {}

    @classmethod
    def register(cls, descriptor: ModuleDescriptor) -> None:
        """
        Register a module descriptor.

        Parameters
        ----------
        descriptor : ModuleDescriptor
            Descriptor to register.
        """
        cls._descriptors[descriptor.module_type] = descriptor

    @classmethod
    def get_descriptor(cls, module_type: ModuleType) -> Optional[ModuleDescriptor]:
        """
        Retrieve the descriptor for a module type.

        Returns
        -------
        ModuleDescriptor or None
        """
        return cls._descriptors.get(module_type)

    @classmethod
    def get_class(cls, module_type: ModuleType) -> Optional[Type[DogModule]]:
        """
        Retrieve the implementation class for a module type.

        Returns
        -------
        Type[DogModule] or None
        """
        descriptor = cls._descriptors.get(module_type)
        return descriptor.module_class if descriptor else None

    @classmethod
    def is_registered(cls, module_type: ModuleType) -> bool:
        """
        Check whether a module type is registered.

        Returns
        -------
        bool
        """
        return module_type in cls._descriptors

    @classmethod
    def get_list_available(cls, sdk_enabled: bool = False) -> list[ModuleType]:
        """
        List available module types for the current configuration.

        Parameters
        ----------
        sdk_enabled : bool, optional
            Whether SDK-dependent modules should be included.

        Returns
        -------
        list of ModuleType
        """
        return [
            mt for mt, desc in cls._descriptors.items()
            if not desc.requires_sdk or sdk_enabled
        ]


def register_all_default_modules():
    """
    Register all default system modules.

    This function is called at import time to populate the module registry
    with all built-in modules.
    """
    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.VIDEO,
        VideoModule,
        "Video Capture",
        requires_sdk=False
    ))
    
    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.MOVEMENT,
        MovementModule,
        "Movement Control",
        requires_sdk=False
    ))
    
    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.OCR,
        OCRModule,
        "Optical Character Recognition",
        requires_sdk=False
    ))
    
    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.AUDIO,
        AudioModule,
        "Text-to-Speech",
        requires_sdk=False
    ))
    
    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.INPUT,
        InputModule,
        "Controller Input",
        requires_sdk=True  
    ))

    ModuleRegistry.register(ModuleDescriptor(
        ModuleType.LIDAR,
        LIDARModule,
        "LIDAR Capture",
        requires_sdk=True
    ))


register_all_default_modules()