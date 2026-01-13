from abc import ABC, abstractmethod


class DogModule(ABC):
    """Base class for all dog functionality modules"""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the module"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Clean shutdown of the module"""
        pass

    def is_initialized(self) -> bool:
        return self._initialized