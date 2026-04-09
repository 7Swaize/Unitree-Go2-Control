Movement Module
===============

This module provides high-level movement controls for the dog robot.

Users should interact only with the ``MovementModule`` class to
command the robot. It wraps the underlying hardware interface and
performs safety checks to prevent unsafe motions.

Users should not access or construct this class directly.
Rather, they should access it through the :class:`~core.controller.Go2Controller` instance.

.. automodule:: go2.modules.movement.movement_module
   :members:
   :undoc-members:
   :show-inheritance:
