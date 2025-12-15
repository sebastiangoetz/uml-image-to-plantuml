import os
from typing import Dict, Optional, List

from ultralytics import YOLO

DEBUG_PLOT = os.getenv("DEBUG_PLOT", "false").lower() in ["1", "true"]

class Control:
    attr: Dict

    def __init__(self, model_name: str):
        self.attr = {
            "model_path": model_name,
        }

class ControlModel:
    control: Control
    model: YOLO
    model_score_threshold: float
    from_name: str = "image"
    to_name: str = "image"
    model_score_threshold: float = 0.5
    type: str

    def __init__(self, model_name: str, model: YOLO, model_score_threshold: float):
        self.control = Control(model_name)
        self.model = model
        self.model_score_threshold = model_score_threshold

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis("off")
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the YOLO model.
        Args:
            path (str): Path to the file with media
        """
        raise NotImplementedError("This method should be overridden in derived classes")