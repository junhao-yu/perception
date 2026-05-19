import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from .base import Detection, BaseDetector


class GroundingDINODetector(BaseDetector):
    def __init__(self, model_path: str, text_prompt: str,
                 box_threshold: float = 0.3, text_threshold: float = 0.25,
                 device: str = "cuda"):
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device).eval()
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        H, W = image.shape[:2]
        pil_img = image[..., ::-1].copy()  # BGR → RGB for processor

        inputs = self.processor(images=pil_img, text=self.text_prompt, return_tensors="pt").to(self.device)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(H, W)]
        )[0]

        detections = []
        for i in range(len(results["boxes"])):
            x1, y1, x2, y2 = [int(v) for v in results["boxes"][i].tolist()]
            score = float(results["scores"][i])
            label = results["labels"][i]
            if isinstance(label, int):
                label = self.text_prompt.split(".")[label].strip().lstrip("a ")
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                label=label,
                score=score,
            ))
        return detections

    def warmup(self, image: np.ndarray) -> None:
        self.detect(image)
