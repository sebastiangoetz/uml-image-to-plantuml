import logging

from control_models.base import ControlModel
from typing import List, Dict

logger = logging.getLogger(__name__)

class PolygonLabelsModel(ControlModel):

    type = "PolygonLabels"

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        return self.create_polygons(results, path)

    def create_polygons(self, results, path):
        logger.debug(f"create_polygons: {self.from_name}")
        data = results[0].masks  # take masks from the first frame
        model_names = self.model.names
        regions = []

        if data is None:
            return regions

        for i in range(len(data)):
            score = float(results[0].boxes.conf[i])  # tensor => float
            points = (
                    data.xyn[i] * 100
            )  # get the polygon points for the current instance
            model_label = model_names[int(results[0].boxes.cls[i])]

            logger.debug(
                "----------------------\n"
                f"task id > {path}\n"
                f"type: {self.control}\n"
                f"polygon points > {points}\n"
                f"model label > {model_label}\n"
                f"score > {score}\n"
            )

            # bbox score is too low
            if score < self.model_score_threshold:
                continue

            # add new region with polygon
            region = {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "polygonlabels",
                "value": {
                    "polygonlabels": [model_label],
                    "points": points.tolist(),  # Converting the tensor to a list for JSON serialization
                    "closed": True,
                },
                "score": score,
            }
            regions.append(region)
        return regions
