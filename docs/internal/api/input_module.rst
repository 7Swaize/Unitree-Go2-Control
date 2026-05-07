Controller Input Module
=======================

Handles remote controller input for the robot.

This model is really only functional and practical when on native hardware, paired with the official Unitree Go2 Controller.
It doesn't do anything on virtual hardware.

This module allows users to:
    - Access the current controller state
    - Register callbacks for specific input signals (buttons, sticks, triggers)
    - Cleanly shutdown input resources (handled automatically)

Internally, it wraps the low-level Unitree SDK or controller parser, but users
interact only with the high-level :class:`~modules.input.input_module.InputModule` interface.


.. autoclass:: go2.modules.input.input_module.InputModule
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.input.input_signal.InputSignal
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.input.controller_state.ControllerState
   :members:
   :undoc-members:
   :show-inheritance:
