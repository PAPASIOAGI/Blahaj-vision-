"""
Object Detection System - Real-time Computer Vision
Entry point untuk aplikasi deteksi objek real-time menggunakan YOLOv8.
"""

from .detector import ObjectDetector
from .utils import draw_detections, calculate_fps, load_config

__all__ = ["ObjectDetector", "draw_detections", "calculate_fps", "load_config"]
__version__ = "1.0.0"