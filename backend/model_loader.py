"""
model_loader.py
Loads YOLOv8 model and runs tumor detection inference.
Classes: pituitary, meningioma, glioma, notumor
"""

import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Tumor class labels (standard brain tumor dataset order)
TUMOR_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Severity descriptions
TUMOR_INFO = {
    "glioma": {
        "description": "Glioma originates from glial cells in the brain or spine.",
        "severity": "High",
        "color": "#FF4444"
    },
    "meningioma": {
        "description": "Meningioma grows from the meninges surrounding brain and spinal cord.",
        "severity": "Medium",
        "color": "#FF8C00"
    },
    "pituitary": {
        "description": "Pituitary tumor forms in the pituitary gland at the brain base.",
        "severity": "Medium",
        "color": "#FFD700"
    },
    "notumor": {
        "description": "No tumor detected in the MRI scan.",
        "severity": "None",
        "color": "#00C853"
    }
}


class TumorDetector:
    """YOLOv8-based brain tumor detector."""

    def __init__(self, model_path: Optional[str]):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load YOLOv8 model from .pt file."""
        if self.model_path is None or not Path(self.model_path).exists():
            logger.warning("⚠️  Model file not found. Running in DEMO mode with mock predictions.")
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info(f"✅ YOLOv8 model loaded: {self.model_path}")
        except ImportError:
            logger.error("❌ ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")

    def predict(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Run inference on MRI scan.
        Returns dict with class, confidence, and saves annotated image.
        """
        if self.model is None:
            return self._demo_predict(image_path, output_path)

        try:
            import cv2

            # Run YOLO inference
            results = self.model(image_path, save=False, conf=0.25)

            if not results or len(results) == 0:
                return self._fallback_result(image_path, output_path)

            result = results[0]

            # Get predicted class and confidence
            if result.boxes and len(result.boxes) > 0:
                # Object detection mode - take highest confidence box
                boxes = result.boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                best_idx = np.argmax(confidences)
                class_idx = classes[best_idx]
                confidence = float(confidences[best_idx])

                # Map class index to name
                if result.names:
                    class_name = result.names.get(class_idx, TUMOR_CLASSES[class_idx % len(TUMOR_CLASSES)])
                else:
                    class_name = TUMOR_CLASSES[class_idx % len(TUMOR_CLASSES)]

                # Save annotated image
                annotated = result.plot()
                cv2.imwrite(output_path, annotated)

            elif hasattr(result, 'probs') and result.probs is not None:
                # Classification mode
                probs = result.probs.data.cpu().numpy()
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])

                if result.names:
                    class_name = result.names.get(class_idx, TUMOR_CLASSES[class_idx % len(TUMOR_CLASSES)])
                else:
                    class_name = TUMOR_CLASSES[class_idx % len(TUMOR_CLASSES)]

                # For classification, draw label on original image
                img = cv2.imread(image_path)
                if img is not None:
                    self._draw_classification_label(img, class_name, confidence, output_path)

            else:
                return self._fallback_result(image_path, output_path)

            # Normalize class name
            class_name = class_name.lower().strip()
            if class_name not in TUMOR_CLASSES:
                class_name = "notumor"

            return {
                "class": class_name,
                "confidence": confidence,
                "info": TUMOR_INFO.get(class_name, TUMOR_INFO["notumor"]),
                "message": "Inference completed successfully"
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._fallback_result(image_path, output_path, error=str(e))

    def _draw_classification_label(self, img, class_name: str, confidence: float, output_path: str):
        """Draw classification result on image."""
        try:
            import cv2
            h, w = img.shape[:2]
            info = TUMOR_INFO.get(class_name, TUMOR_INFO["notumor"])

            # Choose color
            color_map = {
                "glioma": (0, 0, 255),
                "meningioma": (0, 140, 255),
                "pituitary": (0, 215, 255),
                "notumor": (0, 200, 83)
            }
            color = color_map.get(class_name, (255, 255, 255))

            # Draw semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

            # Draw text
            label = f"{class_name.upper()}  {confidence*100:.1f}%"
            cv2.putText(img, label, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

            cv2.imwrite(output_path, img)
        except Exception as e:
            logger.warning(f"Could not draw label: {e}")
            import shutil
            shutil.copy(image_path if hasattr(self, 'image_path') else output_path, output_path)

    def _fallback_result(self, image_path: str, output_path: str, error: str = "") -> Dict[str, Any]:
        """Return fallback when no detections found."""
        try:
            import shutil
            shutil.copy(image_path, output_path)
        except Exception:
            pass
        return {
            "class": "notumor",
            "confidence": 0.5,
            "info": TUMOR_INFO["notumor"],
            "message": f"No clear detection. {error}".strip()
        }

    def _demo_predict(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Demo mode: returns a mock prediction when model is not available.
        For testing UI without the actual .pt file.
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Cannot read image")

            # Mock prediction
            demo_classes = ["glioma", "meningioma", "pituitary", "notumor"]
            weights = [0.25, 0.25, 0.25, 0.25]
            class_name = random.choices(demo_classes, weights=weights)[0]
            confidence = random.uniform(0.70, 0.97)

            h, w = img.shape[:2]
            info = TUMOR_INFO[class_name]
            color_map = {
                "glioma": (0, 0, 255),
                "meningioma": (0, 140, 255),
                "pituitary": (0, 215, 255),
                "notumor": (0, 200, 83)
            }
            color = color_map[class_name]

            if class_name != "notumor":
                # Draw a mock bounding box in center
                cx, cy = w // 2, h // 2
                bw, bh = int(w * 0.35), int(h * 0.35)
                x1, y1 = cx - bw // 2, cy - bh // 2
                x2, y2 = cx + bw // 2, cy + bh // 2
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                label = f"[DEMO] {class_name} {confidence*100:.1f}%"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                label = f"[DEMO] No Tumor Detected {confidence*100:.1f}%"
                cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imwrite(output_path, img)

        except Exception as e:
            logger.error(f"Demo predict error: {e}")
            try:
                import shutil
                shutil.copy(image_path, output_path)
            except Exception:
                pass
            class_name = "notumor"
            confidence = 0.5

        return {
            "class": class_name,
            "confidence": confidence,
            "info": TUMOR_INFO[class_name],
            "message": "⚠️ DEMO MODE: Place best.pt in project root for real inference."
        }
