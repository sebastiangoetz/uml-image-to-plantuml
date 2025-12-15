import re
import cv2
from pathlib import Path

import config

class OCRString:
    def __init__(self, image, score, x1, y1, x2, y2):
        self.image = image
        self.score = score
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.text = None

    def write_image(self):
        base_name = f"text_{self.text}.png" if self.text is not None else "text_unidentified.png"
        safe_name = self.to_safe_file_name(base_name)
        image_path = Path(config.PROCESSED_DIR) / safe_name

        print(image_path)
        cv2.imwrite(str(image_path), self.image)

    @staticmethod
    def to_safe_file_name(file_name) -> str:
        # simple Windows-style sanitation: replace invalid chars, strip trailing spaces/dots
        s = str(file_name)
        s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', s)
        s = s.rstrip(" .")
        return s or "untitled"

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"OCRString(text={self.text}, score={self.score}, x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"