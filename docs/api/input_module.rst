Controller Input Module
=======================

Handles remote controller input for the robot. This module can only function on the Unitree-Go2, with a correctly paired remote controller. 

Users should not access or construct this class directly.
Rather, they should access it through the :class:`~core.controller.Go2Controller` instance.

This module allows users to:
    - Access the current controller state
    - Register callbacks for specific input signals (buttons, sticks, triggers)
    - Cleanly shutdown input resources (handled automatically)

Internally, it wraps the low-level Unitree SDK or controller parser, but users
interact only with the high-level :class:`~modules.input.input_module.InputModule` interface.


.. automodule:: src.modules.input.input_module
   :members:
   :undoc-members:
   :show-inheritance:
