"""Vision pipeline — webcam and tank surface capture with VLM descriptions.

Provides frame sources (WebcamCapture, SurfaceCapture), a VisionObserver
that calls the vision LLM, and a VisionBridge for game loop integration.
"""

from seaman_brain.vision.bridge import VisionBridge
from seaman_brain.vision.capture import SurfaceCapture, WebcamCapture
from seaman_brain.vision.observer import VisionObserver

__all__ = [
    "VisionBridge",
    "VisionObserver",
    "WebcamCapture",
    "SurfaceCapture",
]
