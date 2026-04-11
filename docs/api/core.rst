Go2 Controller
==============

This module provides the **primary public API** for controlling the Unitree Go2 robot.

Users should interact exclusively with :class:`~core.controller.Go2Controller`, which exposes
high-level functionality such as movement, video, audio, OCR, input, and LIDAR through properties.

.. automodule:: go2.core.controller
   :members:
   :undoc-members:
   :show-inheritance:


Hardware Support Types
======================

.. autoclass:: go2.hardware.hardware_type.HardwareType
   :members:
   :undoc-members:
   :show-inheritance:


Module Management
=================

.. autoclass:: go2.core.registry.ModuleType
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.core.registry.ModuleRegistry
   :members:
   :undoc-members:
   :show-inheritance:
