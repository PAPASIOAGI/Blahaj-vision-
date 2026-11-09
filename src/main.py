#!/usr/bin/env python3
"""
Main Entry Point: Object Detection Real-time Application.
Mengakses kamera, menjalankan loop inference, dan menampilkan GUI.
"""

import cv2
import signal
import sys
import logging
from pathlib import Path

from src.detector import ObjectDetector, create_detector
from src.utils import (
    load_config,
    setup_logging,
    open_camera,
    release_camera,
    draw_detections,
    draw_fps,
    FPSCounter,
)

logger = logging.getLogger(__name__)


class ObjectDetectionApp:
    """Main application class untuk object detection real-time."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = load_config(config_path)
        setup_logging(self.config)

        self.detector: ObjectDetector = None
        self.cap: cv2.VideoCapture = None
        self.fps_counter = FPSCounter(window_size=30)
        self.running = False

        # Setup signal handlers untuk graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def initialize(self) -> bool:
        """Inisialisasi semua komponen."""
        try:
            # Initialize detector
            logger.info("Initializing detector...")
            self.detector = create_detector(self.config)
            self.detector.warmup()

            # Open camera
            logger.info("Opening camera...")
            self.cap = open_camera(self.config)

            self.running = True
            logger.info("Initialization complete")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.cleanup()
            return False

    def run(self) -> None:
        """Main loop: capture -> detect -> display."""
        if not self.running:
            logger.error("App not initialized, call initialize() first")
            return

        window_name = self.config.get("display", {}).get("window_name", "Object Detection")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        logger.info("Starting detection loop... Press 'q' to quit")

        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                continue

            # Run detection
            detections = self.detector.detect(frame)

            # Draw detections
            frame = draw_detections(frame, detections, self.config)

            # Calculate and draw FPS
            fps = self.fps_counter.update()
            frame = draw_fps(frame, fps, self.config)

            # Display
            cv2.imshow(window_name, frame)

            # Handle key press (1ms delay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                logger.info("Quit key pressed")
                break
            elif key == ord('s'):  # 's' untuk screenshot
                self._save_screenshot(frame)

        self.cleanup()

    def _save_screenshot(self, frame) -> None:
        """Simpan screenshot frame saat ini."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")
        self.running = False

        if self.cap is not None:
            release_camera(self.cap)
            self.cap = None

        cv2.destroyAllWindows()
        logger.info("Cleanup complete")


def main():
    """Entry point function."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Object Detection with YOLOv8")
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Path to config file (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List available COCO classes and exit"
    )
    args = parser.parse_args()

    if args.list_classes:
        # Load model just to show classes
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("Available COCO classes:")
        for idx, name in model.names.items():
            print(f"  {idx}: {name}")
        return

    # Run application
    app = ObjectDetectionApp(args.config)

    if app.initialize():
        try:
            app.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.exception(f"Runtime error: {e}")
        finally:
            app.cleanup()
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()