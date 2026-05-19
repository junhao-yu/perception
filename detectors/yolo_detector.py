import numpy as np
from ultralytics import YOLO
from .base import Detection, BaseDetector


class YOLODetector(BaseDetector):
    def __init__(self, weights_path: str, conf: float = 0.5, device: str = "cuda"):
        self.conf = conf
        self.model = YOLO(weights_path)
        self.model.to(device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model(image, conf=self.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id] if hasattr(self.model, 'names') else str(cls_id)
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                label=label,
                score=score,
            ))
        return detections

    def warmup(self, image: np.ndarray) -> None:
        self.detect(image)
