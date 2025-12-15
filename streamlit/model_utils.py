from ultralytics import YOLO

from control_models.choices import ChoicesModel
from control_models.polygon_labels import PolygonLabelsModel
from control_models.rectangle_labels import RectangleLabelsModel

def create_static_models(model_size="m"):
    print("Loading static models...")
    static_models = {
        "class-detect.pt": create_rectangle_model(f"class-detect-{model_size}.pt", score_threshold=0.5),
        "text-detect.pt": create_rectangle_model(f"text-detect-{model_size}.pt", score_threshold=0.5),
        "relationship-seg.pt": create_polygon_model(f"relationship-seg-{model_size}.pt", score_threshold=0.5),
        "label-detect.pt": create_rectangle_model(f"label-detect-{model_size}.pt", score_threshold=0.5),
        "endpoint-detect.pt": create_rectangle_model(f"endpoint-detect-{model_size}.pt", score_threshold=0.3),
        "multiplicity-classify.pt": create_choices_model(f"multiplicity-classify-{model_size}.pt", score_threshold=0.5)
    }

    return static_models

def create_rectangle_model(name: str, score_threshold=0.0):
    return RectangleLabelsModel(name, YOLO("models/" + name), score_threshold)

def create_polygon_model(name: str, score_threshold=0.0):
    return PolygonLabelsModel(name, YOLO("models/" + name), score_threshold)

def create_choices_model(name: str, score_threshold=0.0):
    return ChoicesModel(name, YOLO("models/" + name), score_threshold)