"""Screen capture and video source utilities.

Re-exports the existing source classes for use by the pipeline.
"""

from sources.screen import ScreenSource
from sources.camera import CameraSource
from sources.video import VideoSource

__all__ = ["ScreenSource", "CameraSource", "VideoSource"]
