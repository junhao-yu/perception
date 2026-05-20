import numpy as np
from ultralytics import YOLOWorld
from .base import Detection, BaseDetector


class YOLOWorldDetector(BaseDetector):
    def __init__(self, weights_path: str, text_prompt: str,
                 conf: float = 0.3, device: str = "cuda"):
        self.conf = conf
        self.text_prompt = text_prompt
        self.classes = [c.strip() for c in text_prompt.split('.') if c.strip()]
        self.model = YOLOWorld(weights_path)
        self.model.to(device)
        self.model.set_classes(self.classes)

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(image, conf=self.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                label=label,
                score=score,
            ))
        return detections

    def warmup(self, image: np.ndarray) -> None:
        self.detect(image)
