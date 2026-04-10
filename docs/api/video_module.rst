Video Module
============

This module provides a high-level interface for accessing camera frames
and streaming video from the robot or an attached webcam. Users should
interact only with :class:`~modules.video.video_module.VideoModule` and should not use lower-level
camera or streaming classes directly.

Users should not access or construct this class directly. Rather, they should access it through the :class:`~core.controller.Go2Controller` instance.

.. autoclass:: go2.modules.video.video_module.VideoModule
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.video.sources.camera_source_factory.CameraSourceFactory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.video.frame_result.FrameStatus
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.video.frame_result.FrameResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.video.frame_result.MultiFrameResult
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: go2.modules.video.streaming.stream_config.StreamConfig
   :members:
   :undoc-members:
   :show-inheritance:
