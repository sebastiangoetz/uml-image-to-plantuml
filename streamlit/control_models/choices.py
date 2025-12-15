import logging
import numpy as np

from control_models.base import ControlModel
from typing import List, Dict

logger = logging.getLogger(__name__)

class ChoicesModel(ControlModel):

    type = "Choices"

    def predict_regions(self, path) -> List[Dict]:
        results = self.model.predict(path)
        self.debug_plot(results[0].plot())
        return self.create_choices(results, path)

    def create_choices(self, results, path):
        logger.debug(f"create_choices: {self.from_name}")
        mode = self.control.attr.get("choice", "single")
        data = results[0].probs.data.cpu().numpy()

        # single
        if mode in ["single", "single-radio"]:
            indexes = [
                i for i, name in self.model.names.items()
            ]
            data = data[indexes]
            model_names = [self.model.names[i] for i in indexes]
            # find the best choice
            index = np.argmax(data)
            probs = [data[index]]
            names = [model_names[index]]
        # multi
        else:
            # get indexes of data where data >= self.model_score_threshold
            indexes = np.where(data >= self.model_score_threshold)
            probs = data[indexes].tolist()
            names = [self.model.names[int(i)] for i in indexes[0]]

        if not probs:
            logger.debug("No choices found")
            return []

        score = np.mean(probs)
        logger.debug(
            "----------------------\n"
            f"task id > {path}\n"
            f"control: {self.control}\n"
            f"probs > {probs}\n"
            f"score > {score}\n"
            f"names > {names}\n"
        )

        if score < self.model_score_threshold:
            logger.debug(f"Score is too low for single choice: {names[0]} = {probs[0]}")
            return []

        # add new region with rectangle
        return [
            {
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "choices",
                "value": {"choices": names},
                "score": float(score),
            }
        ]
