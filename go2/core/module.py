from abc import ABC, abstractmethod
from typing import final

from ..hardware.interfaces.hardware_type import HardwareType


class DogModule(ABC):
    """
    Base class for all dog functionality modules.

    This class defines the common lifecycle shared by all internal modules (e.g., movement, video, etc.).
    Modules are not meant to be interacted with or instaniated directly. Users should only interact with initialized modules 
    via :class:`~core.controller.Go2Controller`.

    Notes
    -----
    - All modules must implement ``initialize`` and ``shutdown``.
    - Initialization is idempotent and guarded by ``_initialized``.
    - Modules are created and managed by the module registry.

    See Also
    --------
    ModuleRegistry
    """
    def __init__(self, name: str) -> None:
        """
        Parameters
        ----------
        name : str
            Human-readable module name.
        """
        self.name = name
        self._initialized = False

    @final 
    def _set_hardware_internal(self, hardware_type: HardwareType) -> None:
        """
        Intitializes the module to use a specific :class:`HardwareType`. This should not be called by users.
        """
        self._hardware_type = hardware_type

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the module.

        This method is called once during module registration to a :class:`~core.controller.Go2Controller` instance. This should not be called by users.
        """
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        """
        Cleanly shut down the module. This is handled automatically and shouldn't be called by users.

        This method is called during system shutdown and should release
        any resources acquired during initialization.
        """
        pass

    def is_initialized(self) -> bool:
        """
        Check whether the module has been initialized.

        Returns
        -------
        bool
            ``True`` if the module has completed initialization.
        """
        return self._initialized
