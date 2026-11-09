"""
Core AI Logic: Object Detector menggunakan YOLOv8 (Ultralytics).
Menangani load model, inference, dan post-processing (NMS).
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .utils import load_config

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Wrapper class untuk YOLOv8 object detection.
    Mengelola lifecycle model dan menyediakan interface inference yang bersih.
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Inisialisasi detector dengan konfigurasi.

        Args:
            config_path: Path ke file YAML konfigurasi
        """
        self.config = load_config(config_path)
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.target_class_ids: List[int] = []
        self._load_model()
        self._setup_class_filter()

    def _load_model(self) -> None:
        """Load model YOLO dari file weights."""
        model_path = self.config["model"]["path"]
        device = self.config["inference"]["device"]

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {device}")

        try:
            # Ultralytics akan auto-download jika file tidak ada (untuk model standard seperti yolov8n.pt)
            self.model = YOLO(model_path)
            self.model.to(device)

            # Ambil nama kelas dari model
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully. Classes: {len(self.class_names)}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _setup_class_filter(self) -> None:
        """Setup filter kelas berdasarkan konfigurasi."""
        filter_config = self.config.get("filter", {})
        if not filter_config.get("enabled", True):
            self.target_class_ids = list(self.class_names.keys())
            logger.info("Class filtering disabled - detecting all classes")
            return

        target_classes = filter_config.get("target_classes", [])
        if not target_classes:
            self.target_class_ids = list(self.class_names.keys())
            logger.info("No target classes specified - detecting all classes")
            return

        # Map nama kelas ke class ID
        name_to_id = {v.lower(): k for k, v in self.class_names.items()}
        self.target_class_ids = []

        for class_name in target_classes:
            class_name_lower = class_name.lower()
            if class_name_lower in name_to_id:
                self.target_class_ids.append(name_to_id[class_name_lower])
                logger.info(f"Filtering for class: {class_name} (ID: {name_to_id[class_name_lower]})")
            else:
                logger.warning(f"Class '{class_name}' not found in model classes. Available: {list(self.class_names.values())}")

        if not self.target_class_ids:
            logger.warning("No valid target classes found, falling back to all classes")
            self.target_class_ids = list(self.class_names.keys())

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Jalankan inference pada frame tunggal.

        Args:
            frame: Input frame sebagai numpy array (BGR format dari OpenCV)

        Returns:
            List of detection dictionaries dengan keys:
            - bbox: [x1, y1, x2, y2] (pixel coordinates)
            - confidence: float (0.0 - 1.0)
            - class_id: int
            - class_name: str
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        infer_config = self.config["inference"]

        # Run inference
        results = self.model(
            frame,
            conf=infer_config["conf_threshold"],
            iou=infer_config["iou_threshold"],
            imgsz=infer_config["imgsz"],
            max_det=infer_config["max_det"],
            half=infer_config["half_precision"],
            device=infer_config["device"],
            verbose=False,
            classes=self.target_class_ids if self.target_class_ids else None
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return detections

    def warmup(self, frame_shape: tuple = (640, 480, 3)) -> None:
        """
        Warmup model dengan dummy inference untuk menghindari latency spike pertama.
        """
        logger.info("Warming up model...")
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        _ = self.detect(dummy_frame)
        logger.info("Warmup complete")


def create_detector(config_path: str = "config/settings.yaml") -> ObjectDetector:
    """
    Factory function untuk membuat instance ObjectDetector.
    Berguna untuk dependency injection dan testing.
    """
    return ObjectDetector(config_path)