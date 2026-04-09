Public API Documentation
=========================

This section contains the public API documentation for users working with the Unitree Go2 robot.

Core
----

The entry point for all robot control.

.. toctree::
   :maxdepth: 2

   controller

Default Modules
---------------

These modules are **automatically available** upon instantiating the controller.

.. toctree::
   :maxdepth: 2

   movement_module
   input_module

Optional Modules
----------------

These modules must be **explicitly added** using ``controller.add_module()``.

.. toctree::
   :maxdepth: 2

   video_module
   audio_module
   lidar_module
   ocr_module

State Machine & Architecture
-----------------------------

For building behavior using a state machine pattern:

.. toctree::
   :maxdepth: 2

   robot_state