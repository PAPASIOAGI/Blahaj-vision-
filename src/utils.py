"""
Utility Functions: Helper untuk drawing, FPS calculation, config loading, dll.
"""

import cv2
import numpy as np
import yaml
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ===================== Config Loading =====================

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load konfigurasi dari file YAML.

    Args:
        config_path: Path ke file settings.yaml

    Returns:
        Dictionary konfigurasi
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration jika file tidak ditemukan."""
    return {
        "camera": {"device_index": 0, "width": 640, "height": 480, "fps": 30},
        "model": {"path": "models/yolov8n.pt", "task": "detect"},
        "inference": {
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "imgsz": 640,
            "max_det": 300,
            "half_precision": False,
            "device": "cpu"
        },
        "filter": {"enabled": True, "target_classes": []},
        "display": {
            "window_name": "Object Detection",
            "show_fps": True,
            "show_confidence": True,
            "box_thickness": 2,
            "font_scale": 0.6,
            "colors": {
                "person": [0, 255, 0],
                "cell phone": [255, 0, 0],
                "bottle": [0, 0, 255],
                "default": [255, 255, 0]
            }
        },
        "logging": {"level": "INFO", "save_detections": False, "log_file": "logs/detections.log"}
    }


# ===================== Drawing Functions =====================

def get_color_for_class(class_name: str, color_config: Dict[str, List[int]]) -> Tuple[int, int, int]:
    """Dapatkan warna BGR untuk kelas tertentu."""
    color = color_config.get(class_name.lower(), color_config.get("default", [255, 255, 0]))
    return tuple(color)  # BGR format untuk OpenCV


def draw_detections(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Gambar bounding box dan label pada frame.

    Args:
        frame: Input frame (BGR numpy array)
        detections: List hasil deteksi dari ObjectDetector.detect()
        config: Konfigurasi display

    Returns:
        Frame dengan anotasi (modified in-place, juga dikembalikan untuk chaining)
    """
    display_cfg = config.get("display", {})
    thickness = display_cfg.get("box_thickness", 2)
    font_scale = display_cfg.get("font_scale", 0.6)
    show_conf = display_cfg.get("show_confidence", True)
    colors = display_cfg.get("colors", {"default": [255, 255, 0]})

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cls_name = det["class_name"]

        # Dapatkan warna untuk kelas ini
        color = get_color_for_class(cls_name, colors)

        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Siapkan label text
        label = cls_name
        if show_conf:
            label += f" {conf:.2f}"

        # Hitung ukuran text untuk background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        # Gambar background label
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            color,
            -1  # Filled
        )

        # Gambar text label
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            thickness,
            cv2.LINE_AA
        )

    return frame


def draw_fps(frame: np.ndarray, fps: float, config: Dict[str, Any]) -> np.ndarray:
    """
    Gambar FPS counter di pojok frame.

    Args:
        frame: Input frame
        fps: Current FPS value
        config: Konfigurasi display

    Returns:
        Frame dengan FPS text
    """
    if not config.get("display", {}).get("show_fps", True):
        return frame

    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 0)  # Green

    # Position: top-left
    pos = (10, 30)

    # Background
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (pos[0] - 5, pos[1] - text_h - 5), (pos[0] + text_w + 5, pos[1] + baseline + 5), (0, 0, 0), -1)

    # Text
    cv2.putText(frame, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


# ===================== FPS Calculation =====================

class FPSCounter:
    """
    Class untuk menghitung FPS real-time menggunakan moving average.
    Lebih stabil dibanding simple frame counting.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: List[float] = []
        self.last_time: Optional[float] = None

    def update(self) -> float:
        """
        Update counter dan return current FPS.
        Call ini sekali per frame.
        """
        current_time = time.time()
        self.timestamps.append(current_time)

        # Keep only recent timestamps
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        # Calculate FPS
        if len(self.timestamps) >= 2:
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span > 0:
                fps = (len(self.timestamps) - 1) / time_span
            else:
                fps = 0.0
        else:
            fps = 0.0

        self.last_time = current_time
        return fps

    def get_fps(self) -> float:
        """Get current FPS tanpa update (read-only)."""
        if len(self.timestamps) >= 2:
            time_span = self.timestamps[-1] - self.timestamps[0]
            if time_span > 0:
                return (len(self.timestamps) - 1) / time_span
        return 0.0


def calculate_fps(prev_time: float, curr_time: float) -> float:
    """
    Simple FPS calculation antara dua timestamp.
    Deprecated: gunakan FPSCounter class untuk hasil lebih stabil.
    """
    elapsed = curr_time - prev_time
    return 1.0 / elapsed if elapsed > 0 else 0.0


# ===================== Camera Helpers =====================

def open_camera(config: Dict[str, Any]) -> cv2.VideoCapture:
    """
    Buka kamera dengan konfigurasi dari settings.

    Args:
        config: Konfigurasi kamera

    Returns:
        cv2.VideoCapture object
    """
    cam_cfg = config.get("camera", {})
    device_index = cam_cfg.get("device_index", 0)
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)
    fps = cam_cfg.get("fps", 30)

    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {device_index}")

    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

    # Verify actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    return cap


def release_camera(cap: cv2.VideoCapture) -> None:
    """Safely release camera resource."""
    if cap is not None:
        cap.release()
        logger.info("Camera released")


# ===================== Logging Setup =====================

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging berdasarkan konfigurasi."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_cfg.get("save_detections", False):
        log_file = log_cfg.get("log_file", "logs/detections.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


# ===================== Validation =====================

def validate_detection(det: Dict[str, Any]) -> bool:
    """Validasi format detection dictionary."""
    required_keys = ["bbox", "confidence", "class_id", "class_name"]
    if not all(k in det for k in required_keys):
        return False

    bbox = det["bbox"]
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return False

    if not (0.0 <= det["confidence"] <= 1.0):
        return False

    return True


def filter_detections_by_confidence(
    detections: List[Dict[str, Any]],
    min_conf: float
) -> List[Dict[str, Any]]:
    """Filter deteksi berdasarkan confidence threshold tambahan."""
    return [d for d in detections if d["confidence"] >= min_conf]