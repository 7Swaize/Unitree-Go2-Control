Public API Documentation
=========================

This section contains the public API documentation for students working with the Unitree Go2 robot.

Core
----

The entry point for all robot control:

.. toctree::
   :maxdepth: 2

   core

Default Modules
---------------

These modules are **automatically available** upon instantiating the controller.

.. toctree::
   :maxdepth: 2

   movement
   controller_input_control

Optional Modules
----------------

These modules must be **explicitly added** using ``controller.add_module()``.

.. toctree::
   :maxdepth: 2

   video_control
   audio_control
   lidar_control
   ocr_control
   aruko

State Machine & Architecture
-----------------------------

For building behavior using a state machine pattern:

.. toctree::
   :maxdepth: 2

   states