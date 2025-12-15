import os
import random
import statistics
from pathlib import Path
from typing import List, Optional, Dict

import pulp

import cv2
import numpy as np
from shapely import Polygon
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from shapely.geometry import box, Point, LineString
from shapely.ops import unary_union
import requests
from xml.etree import ElementTree
from contextlib import contextmanager
import time
import base64

import predictor
import model_utils
import image_region_preparer
import relationship_evaluater
import plant_uml_encoder
import text_utils
import config
from data.ocr_string import OCRString
from data.uml_objects import UMLDiagram, UMLClass, UMLClassStereoType, UMLRelationship, UMLRelationshipType
from data.silent_ui import SilentUI


@contextmanager
def timed_spinner(ui, name, durations):
    with ui.spinner(name + "..."):
        start = time.time()
        yield
        durations.append((name, time.time() - start))

def run_uml_extraction_pipeline(image_path: str, model_size="m", ui=None, upload_file_name=None, skip_diagram_rendering=False):
    start_total_time = time.time()
    durations = []

    if upload_file_name is None:
        upload_file_name = os.path.basename(image_path)

    upload_file_name_without_extension = os.path.splitext(upload_file_name)[0]

    if ui is None:
        ui = SilentUI(plot_images=False)
    # Prepare directories and static models
    static_models = load_models(model_size)

    # Preprocess image
    with timed_spinner(ui, "Preprocessing Image", durations):
        preprocessed_image, image_width, image_height, preprocessed_image_path = preprocess_image(image_path)
        ui.markdown(f"### Preprocessed Image")
        ui.image(preprocessed_image_path, width=image_width)

    # Class prediction and image marking
    with timed_spinner(ui, "Detecting Classes", durations):
        predicted_class_regions, classes_image_path, class_boxes = predict_and_mark_classes(static_models, preprocessed_image_path, preprocessed_image)
        ui.markdown(f"### Classes")
        ui.image(classes_image_path, width=image_width)

    with timed_spinner(ui, "Detecting Class Texts", durations):
        collected_ocr_strings = []

        # Collect OCR boxes from class text
        classes_ocr_strings, class_text_image_path = collect_class_ocr_boxes(static_models, preprocessed_image,
                                                                             class_boxes, collected_ocr_strings)

        ui.markdown(f"### Class Text")
        ui.image(class_text_image_path, width=image_width)

    # Relationship prediction and image marking
    with timed_spinner(ui, "Segmenting Relationships", durations):
        predicted_relationship_regions, relationship_image_path, classes_and_relationship_image_path = predict_and_mark_relationships(static_models, classes_image_path, predicted_class_regions, preprocessed_image)
        ui.markdown(f"### Relationships")
        ui.image(classes_and_relationship_image_path, width=image_width)

    # Endpoint + crossing extraction
    with timed_spinner(ui, "Detecting Relationship Endpoints and End Shapes", durations):
        endpoints, crossings, end_shapes, endpoint_image_path = extract_endpoints_and_shapes(static_models, relationship_image_path,
                                                                                 image_width, image_height)
        ui.markdown(f"### Endpoints and End Shapes")
        ui.image(endpoint_image_path, width=image_width)

    # Class connection evaluation and skeleton graph
    with timed_spinner(ui, "Tracing Relationships", durations):
        class_connections, scale_factor, class_endpoints, relationship_paths = evaluate_class_connections(relationship_image_path,class_boxes, endpoints, crossings, end_shapes)
        ui.markdown(f"### Skeleton Graph")
        if Path(config.PROCESSED_DIR + "/graph_skeleton.png").exists():
            ui.image(config.PROCESSED_DIR + "/graph_skeleton.png", width=image_width)
        else:
            ui.image(relationship_image_path, width=image_width)

        connection_image_path = debug_visualize_connections(preprocessed_image, relationship_paths, class_connections)
        ui.markdown(f"### Connection Graph with Relationship Paths")
        ui.image(connection_image_path, width=image_width)

    # Multiplicity and relationship label prediction
    with timed_spinner(ui, "Detecting Relationship Labels and Multiplicities", durations):
        multiplicity_regions, multiplicity_boxes, label_regions, label_boxes = predict_multiplicities_and_relationship_labels(static_models, classes_and_relationship_image_path, image_width, image_height)

    # Relationship label (roles and relationship names) assignment
    with timed_spinner(ui, "Assigning Relationship Labels to Relationship Endpoints and Relationships", durations):
        label_clusters, label_cluster_boxes, role_cluster_endpoint_assignment, relationship_name_cluster_relationship_assignment = assign_roles_and_relationship_names(label_boxes, class_endpoints, relationship_paths, scale_factor)

        relationship_labels_image = preprocessed_image.copy()
        relationship_labels_image_path = config.PROCESSED_DIR + "/relationship_labels.png"
        relationship_labels_image = draw_assignments(relationship_labels_image, label_cluster_boxes, (0, 255, 0), class_endpoints, (0, 0, 255), role_cluster_endpoint_assignment)

        relationship_center_points = [
            (
                path.interpolate(path.length / 2).x,
                path.interpolate(path.length / 2).y
            )
            for path in relationship_paths
        ]  # (x,y)

        relationship_labels_image = draw_assignments(relationship_labels_image, label_cluster_boxes, (0, 128, 255), relationship_center_points, (0, 0, 255), relationship_name_cluster_relationship_assignment)
        cv2.imwrite(relationship_labels_image_path, relationship_labels_image)

        # Collect OCR boxes from roles and relationship names
        collect_and_assign_label_ocr_boxes(preprocessed_image, label_regions, label_clusters,
                                           role_cluster_endpoint_assignment,
                                           relationship_name_cluster_relationship_assignment, class_endpoints,
                                           class_connections, collected_ocr_strings)

        ui.markdown(f"### Relationship Labels")
        ui.image(relationship_labels_image_path, width=image_width)

    # Multiplicity assignment
    with timed_spinner(ui, "Assigning Multiplicities to Relationship Endpoints", durations):
        multiplicity_endpoint_assignment = assign_multiplicities(static_models, preprocessed_image, multiplicity_regions, multiplicity_boxes, class_endpoints, scale_factor, class_connections)

        multiplicity_image = preprocessed_image.copy()
        multiplicity_image_path = config.PROCESSED_DIR + "/multiplicities.png"
        multiplicity_image = draw_assignments(multiplicity_image, multiplicity_boxes, (0, 255, 0), class_endpoints, (0, 0, 255), multiplicity_endpoint_assignment)
        cv2.imwrite(multiplicity_image_path, multiplicity_image)

        ui.markdown(f"### Multiplicities")
        ui.image(multiplicity_image_path, width=image_width)

    with timed_spinner(ui, "Extracting Text from OCR Boxes", durations):
        # Extract text from OCR boxes in batch
        extract_text_from_ocr_boxes(collected_ocr_strings)

    with timed_spinner(ui, "Building PlantUML Code", durations):
        # UML diagram generation
        uml_class_diagram = generate_uml_diagram(class_boxes, classes_ocr_strings, class_connections)

        uml_code = uml_class_diagram.to_plantuml()
        with open(config.PROCESSED_DIR + "/model.puml", "w", encoding="utf-8") as f:
            f.write(uml_code)

        ui.markdown("### UML (PlantUML Code)")
        ui.code(uml_code, language="plantuml")

    if not skip_diagram_rendering:
        with timed_spinner(ui, "Rendering PlantUML Diagram", durations):
            html_code, svg_height, plantuml_svg_url, svg_text = render_uml_diagram(uml_code)

            ui.markdown("### UML Diagram")
            ui.components.v1.html(html_code, height=svg_height + 20)

            # Add download buttons

            ## Encode Code to base64
            b64_code = base64.b64encode(uml_code.encode("utf-8")).decode("utf-8")
            code_href = f'data:text/plain;base64,{b64_code}'

            ## Encode SVG (use previously fetched text) to base64
            b64_svg = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
            svg_href = f'data:image/svg+xml;base64,{b64_svg}'

            ## Fetch and encode PNG to base64
            plantuml_png_url = plantuml_svg_url.replace("/svg/", "/png/")
            response = requests.get(plantuml_png_url)
            png_content = response.content
            b64_png = base64.b64encode(png_content).decode("utf-8")
            png_href = f'data:image/png;base64,{b64_png}'

            ## Display both buttons side by side
            ui.markdown(f"""
            <div style="display:flex; gap:1em; margin-top:1em;">
                <a href="{code_href}" download="{upload_file_name_without_extension}.puml">
                    <button style="padding:0.5em 1em; font-size:1em;">📥 Download Code</button>
                </a>
                <a href="{svg_href}" download="{upload_file_name_without_extension}_plantuml.svg">
                    <button style="padding:0.5em 1em; font-size:1em;">📥 Download SVG</button>
                </a>
                <a href="{png_href}" download="{upload_file_name_without_extension}_plantuml.png">
                    <button style="padding:0.5em 1em; font-size:1em;">📥 Download PNG</button>
                </a>
            </div>
            """, unsafe_allow_html=True)

    total_duration = time.time() - start_total_time
    durations.append(("Total", total_duration))
    durations_chart = create_durations_chart(durations)

    ui.markdown("### Durations")
    ui.altair_chart(durations_chart)

    result = {
        "diagram_name": upload_file_name_without_extension,
        "number_of_detected_classes": len(class_boxes),
        "number_of_detected_relationships": len(class_connections),
        "number_of_detected_class_texts": sum(
                                         len(class_ocr_strings[text_type])
                                         for class_ocr_strings in classes_ocr_strings
                                         for text_type in class_ocr_strings
                                     ),
        "number_of_detected_labels": len(role_cluster_endpoint_assignment) + len(relationship_name_cluster_relationship_assignment),
        "number_of_detected_multiplicities": len(multiplicity_endpoint_assignment),
        "time": total_duration
    }

    return result

def load_models(model_size="m"):
    static_models = model_utils.create_static_models(model_size)

    return static_models

def load_image_and_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    return image, width, height

def preprocess_image(image_path):
    preprocessed_image_path = config.PROCESSED_DIR + "/preprocessed.png"
    predictor.preprocess(image_path, preprocessed_image_path)
    preprocessed_image = cv2.imread(preprocessed_image_path)
    height, width = preprocessed_image.shape[:2]
    return preprocessed_image, width, height, preprocessed_image_path

def predict_and_mark_classes(static_models, image_path, image):
    predicted_class_regions = predictor.predict_regions(static_models["class-detect.pt"], image_path, None, skip_preprocessing=True)

    classes_image_path = config.PROCESSED_DIR + "/classes.png"
    image_region_preparer.mark_classes(predicted_class_regions, image_path, classes_image_path)

    image_height, image_width = image.shape[:2]
    class_boxes = []
    for class_region in predicted_class_regions:
        label = class_region['value']['rectanglelabels'][0]
        if label == "Class":
            x = class_region['value']['x'] / 100 * image_width
            y = class_region['value']['y'] / 100 * image_height
            width = class_region['value']['width'] / 100 * image_width
            height = class_region['value']['height'] / 100 * image_height
            class_boxes.append([x, y, width, height])

    return predicted_class_regions, classes_image_path, class_boxes

def keep_only_polygons(input_image: np.ndarray, list_of_relative_shapely_polygons: list[Polygon]) -> np.ndarray:
    height, width = input_image.shape[:2]
    input_image = input_image.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    for relative_polygon in list_of_relative_shapely_polygons:
        absolute_polygon = scale(relative_polygon, xfact=width / 100.0, yfact=height / 100.0, origin=(0, 0))

        # Draw exterior
        exterior = np.array(absolute_polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [exterior], (255, 255, 255))

        # Remove holes
        for interior in absolute_polygon.interiors:
            hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [hole], (0, 0, 0))

    # Apply mask
    result = cv2.bitwise_and(input_image, input_image, mask=mask)

    # Change white pixels inside the polygon to light gray for contrast
    white_pixels = (result[:, :, 0] == 255) & (result[:, :, 1] == 255) & (result[:, :, 2] == 255) & (mask == 255)
    result[white_pixels] = [220, 220, 220]  # slightly gray

    # Set background to white where mask is not 255
    result[mask == 0] = [255, 255, 255]

    return result

def is_polygon_inside_any(polygon: Polygon, class_regions: List[Dict], min_ratio=0.95) -> bool:
    for class_region in class_regions:
        class_x1 = class_region['value']['x']
        class_y1 = class_region['value']['y']
        class_x2 = class_x1 + class_region['value']['width']
        class_y2 = class_y1 + class_region['value']['height']
        class_rect = box(class_x1, class_y1, class_x2, class_y2)
        intersection = polygon.intersection(class_rect)
        if intersection.is_empty:
            continue
        ratio = intersection.area / polygon.area
        if ratio >= min_ratio:
            print(f"Skipping relationship polygon, because it is to {min_ratio * 100} percent inside class")
            return True
    return False

def predict_and_mark_relationships(static_models, classes_image_path, class_regions, image):
    predicted_relationship_regions = predictor.predict_regions(static_models["relationship-seg.pt"], classes_image_path, None, skip_preprocessing=True)

    # filter out bad relationships that lay almost completely inside a class
    predicted_relationship_regions = [
        region for region in predicted_relationship_regions
        if not is_polygon_inside_any(region["value"]["polygon"], class_regions)
    ]

    classes_and_relationship_image_path = config.PROCESSED_DIR + "/classes_and_relationships.png"
    image_region_preparer.mark_relationships(predicted_relationship_regions, classes_image_path, classes_and_relationship_image_path)

    relative_shapely_polygon_list = [r["value"]["polygon"] for r in predicted_relationship_regions]
    relationship_image = keep_only_polygons(image, relative_shapely_polygon_list)
    relationship_image_path = config.PROCESSED_DIR + "/relationships.png"
    cv2.imwrite(relationship_image_path, relationship_image)

    return predicted_relationship_regions, relationship_image_path, classes_and_relationship_image_path

def extract_endpoints_and_shapes(static_models, relationship_image_path, image_width, image_height):
    predicted_endpoint_regions = predictor.predict_regions(static_models["endpoint-detect.pt"], relationship_image_path, None, skip_preprocessing=True)

    endpoint_image_path = config.PROCESSED_DIR + "/endpoints.png"
    image_region_preparer.mark_regions(list(filter(lambda r: r['value']['rectanglelabels'][0] != "Empty End", predicted_endpoint_regions)), relationship_image_path, endpoint_image_path, (0, 255, 0))
    image_region_preparer.mark_regions(list(filter(lambda r: r['value']['rectanglelabels'][0] == "Empty End", predicted_endpoint_regions)), endpoint_image_path, endpoint_image_path, (0, 0, 255))

    endpoints = []
    crossings = []
    end_shapes = []

    for region in predicted_endpoint_regions:
        label = region['value']['rectanglelabels'][0]
        score = region["score"]
        x = region['value']['x'] / 100 * image_width
        y = region['value']['y'] / 100 * image_height
        width = region['value']['width'] / 100 * image_width
        height = region['value']['height'] / 100 * image_height

        if label == "Empty End":
            endpoints.append((x + width / 2, y + height / 2))
        elif label == "Crossing":
            crossings.append([x, y, width, height])
        else:
            # we have to be sure for end shapes
            if score >= 0.5:
                end_shapes.append([x, y, width, height, label, score])

    return endpoints, crossings, end_shapes, endpoint_image_path

def evaluate_class_connections(relationship_image_path, class_boxes, endpoints, crossings, end_shapes):
    image = cv2.imread(relationship_image_path)

    widths = [box[2] for box in class_boxes]
    heights = [box[3] for box in class_boxes]
    median_width = statistics.median(widths)
    median_height = statistics.median(heights)
    scale_factor = (median_width + median_height) / 200

    class_connections = relationship_evaluater.evaluate(
        relationship_image=image,
        class_boxes=class_boxes,
        endpoints=endpoints,
        crossings=crossings,
        end_shapes=end_shapes,
        scale_factor=scale_factor,
    )

    class_endpoints = set()
    relationship_paths = []
    for (endpoint_a, endpoint_b), connection_parameters in class_connections.items():
        class_endpoints.add(endpoint_a)
        class_endpoints.add(endpoint_b)
        relationship_paths.append(LineString(connection_parameters["path"]))

        end_shape_a = connection_parameters["end_shape_0"]
        end_shape_b = connection_parameters["end_shape_1"]
        print(f"Connection found between {endpoint_a} ({end_shape_a}) and {endpoint_b} ({end_shape_b})")

    return class_connections, scale_factor, list(class_endpoints), relationship_paths

def debug_visualize_connections(image, relationship_paths, connections, node_color=(0, 0, 255), edge_color=(0, 255, 0), node_radius=2):
    image = image.copy()

    for idx, line in enumerate(relationship_paths):
        if line.is_empty or len(line.coords) < 2:
            continue

        coords = np.array(line.coords, dtype=np.int32)

        rng = random.Random(idx)
        color = tuple(rng.randint(0, 255) for _ in range(3))  # BGR

        # Draw each segment
        for i in range(len(coords) - 1):
            pt1 = tuple(coords[i])
            pt2 = tuple(coords[i + 1])
            cv2.line(image, pt1, pt2, color=color, thickness=2)

    for (endpoint_a, endpoint_b), _ in connections.items():
        pt1 = (int(endpoint_a[0]), int(endpoint_a[1]))
        pt2 = (int(endpoint_b[0]), int(endpoint_b[1]))

        # Draw edge line
        cv2.line(image, pt1, pt2, edge_color, thickness=1)

        # Draw nodes
        cv2.circle(image, pt1, radius=node_radius, color=node_color, thickness=-1)
        cv2.circle(image, pt2, radius=node_radius, color=node_color, thickness=-1)

    image_path = config.PROCESSED_DIR + "/graph_connection.png"
    cv2.imwrite(image_path, image)

    return image_path

def optimal_geometry_to_geometry_assignment(geometries_a: List[BaseGeometry], geometry_type_a: str,
                                            geometries_b: List[BaseGeometry], geometry_type_b: str,
                                            max_distance: float):
    num_geometries_a = len(geometries_a)
    num_geometries_b = len(geometries_b)

    valid_pairs = {}
    for i, geometry_a in enumerate(geometries_a):
        if geometry_a is None:
            # skip empty entries
            continue
        close_to_any = False
        min_dist = float("inf")
        for j, pt in enumerate(geometries_b):
            dist = geometry_a.distance(pt)
            if dist <= max_distance:
                valid_pairs[(i, j)] = dist
                close_to_any = True
            if dist < min_dist:
                min_dist = dist

        if not close_to_any:
            geometry_a_center_point = geometry_a.centroid
            print(f"Ignoring {geometry_type_a} {i} at {(geometry_a_center_point.x, geometry_a_center_point.y)}, because its not close to any {geometry_type_b} (all distances > {max_distance} px) with the closest {geometry_type_b} being {min_dist} px away")

    # Setup optimization
    prob = pulp.LpProblem("PolygonPointAssignment", pulp.LpMinimize)

    # Variables only for valid (i, j) pairs
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for (i, j) in valid_pairs
    }

    # Objective: minimize total distance
    prob += pulp.lpSum(x[i, j] * valid_pairs[i, j] for (i, j) in valid_pairs)

    # For each geometry_a, assign to exactly one geometry_b — only if any (i, j) is in valid_pairs
    for i in range(num_geometries_a):
        if any((i, j) in valid_pairs for j in range(num_geometries_b)):
            prob += pulp.lpSum(x[i, j] for j in range(num_geometries_b) if (i, j) in x) == 1

    # Each geometry_b assigned to at most one geometry_a
    for j in range(num_geometries_b):
        prob += pulp.lpSum(x[i, j] for i in range(num_geometries_a) if (i, j) in x) <= 1

    # Solve
    prob.solve()

    # Extract assignments
    assignments = [(i, j) for (i, j), var in x.items() if var.value() == 1]

    status = pulp.LpStatus[prob.status]
    print(f"Solver status: {status}")

    if status == "Infeasible":
        print("No feasible solution found. Falling back to greedy nearest-neighbor assignment.")

        # Step 1: Sort valid pairs by increasing distance
        sorted_pairs = sorted(valid_pairs.items(), key=lambda item: item[1])  # item = ((i, j), dist)

        assigned_a = set()
        assigned_b = set()
        assignments = []

        # Step 2: Greedy assignment (each a to at most one b, and vice versa)
        for (i, j), dist in sorted_pairs:
            if i not in assigned_a and j not in assigned_b:
                assignments.append((i, j))
                assigned_a.add(i)
                assigned_b.add(j)

    return assignments

def predict_multiplicities_and_relationship_labels(static_models, classes_and_relationship_image_path, image_width, image_height):
    predicted_label_regions = predictor.predict_regions(static_models["label-detect.pt"],
                                                        classes_and_relationship_image_path, None,
                                                        skip_preprocessing=True)
    multiplicity_regions = []
    multiplicity_boxes = []
    label_regions = []
    label_boxes = []
    for label_region in predicted_label_regions:
        label = label_region['value']['rectanglelabels'][0]
        x1 = label_region['value']['x'] / 100 * image_width
        y1 = label_region['value']['y'] / 100 * image_height
        x2 = x1 + label_region['value']['width'] / 100 * image_width
        y2 = y1 + label_region['value']['height'] / 100 * image_height
        if label == "Multiplicity":
            multiplicity_regions.append(label_region)
            multiplicity_boxes.append(box(x1, y1, x2, y2))
        elif label == "Relationship label":
            label_regions.append(label_region)
            label_boxes.append(box(x1, y1, x2, y2))

    return multiplicity_regions, multiplicity_boxes, label_regions, label_boxes

def assign_multiplicities(static_models, image, multiplicity_regions, multiplicity_boxes, class_endpoints, scale_factor, connections):
    image_height, image_width = image.shape[:2]

    shapely_endpoints = [Point(x, y) for x, y in class_endpoints]
    assignment = optimal_geometry_to_geometry_assignment(multiplicity_boxes, "multiplicity", shapely_endpoints, "endpoint", max_distance=round(scale_factor * 40, 3))

    for multiplicity_index, class_endpoint_index in assignment:
        multiplicity_region = multiplicity_regions[multiplicity_index]

        multiplicity_x1 = int(multiplicity_region["value"]["x"] / 100 * image_width)
        multiplicity_y1 = int(multiplicity_region["value"]["y"] / 100 * image_height)
        multiplicity_x2 = int(multiplicity_x1 + multiplicity_region["value"]["width"] / 100 * image_width)
        multiplicity_y2 = int(multiplicity_y1 + multiplicity_region["value"]["height"] / 100 * image_height)
        multiplicity_image = image[multiplicity_y1:multiplicity_y2, multiplicity_x1:multiplicity_x2]
        multiplicity_image_path = config.PROCESSED_DIR + f"/multiplicity_{multiplicity_index}.png"
        cv2.imwrite(multiplicity_image_path, multiplicity_image)

        predicted_multiplicities_results = predictor.predict_regions(static_models["multiplicity-classify.pt"], multiplicity_image_path, None, skip_preprocessing=True)

        if len(predicted_multiplicities_results) > 0:
            predicted_multiplicities = predicted_multiplicities_results[0]["value"]["choices"]
            multiplicity = predicted_multiplicities[0].replace("STAR", "*")
            if multiplicity == "Struck":
                print(f"Ignoring multiplicity {multiplicity_index} at ({multiplicity_x1}, {multiplicity_y1}) because it was classified as \"Struck\".")
                continue
        else:
            # multiplicity = "U"
            print(f"Ignoring multiplicity {multiplicity_index} at ({multiplicity_x1}, {multiplicity_y1}) because it couldn't be classified.")
            continue

        endpoint = class_endpoints[class_endpoint_index]

        for (endpoint_a, endpoint_b), connection_parameters in connections.items():
            if endpoint == endpoint_a:
                connection_parameters["multiplicity_0"] = multiplicity
                print(f"Assigned multiplicity {multiplicity_index} with value \"{multiplicity}\" at ({multiplicity_x1}, {multiplicity_y1}) to endpoint {class_endpoint_index} at ({endpoint[0]}, {endpoint[1]}).")
                break

            if endpoint == endpoint_b:
                connection_parameters["multiplicity_1"] = multiplicity
                print(f"Assigned multiplicity {multiplicity_index} with value \"{multiplicity}\" at ({multiplicity_x1}, {multiplicity_y1}) to endpoint {class_endpoint_index} at ({endpoint[0]}, {endpoint[1]}).")
                break

    return assignment

def debug_multiplicities(image, assignments, multiplicity_boxes, class_endpoints):
    multiplicities_image = image.copy()
    multiplicities_image_path = config.PROCESSED_DIR + "/multiplicities.png"
    for m_idx, ep_idx in assignments:
        multiplicity_box = multiplicity_boxes[m_idx]  # shapely box
        class_endpoint = class_endpoints[ep_idx]  # (x,y)

        # Get rectangle corners from shapely box
        minx, miny, maxx, maxy = multiplicity_box.bounds
        top_left = (int(minx), int(miny))
        bottom_right = (int(maxx), int(maxy))

        # Draw green rectangle
        overlay = multiplicities_image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), thickness=-1)

        # Blend the filled rectangle onto the original image
        multiplicities_image = cv2.addWeighted(overlay, 0.3, multiplicities_image, 1 - 0.3, 0)

        # Draw red circle at endpoint
        ep_x, ep_y = map(int, class_endpoint)
        cv2.circle(multiplicities_image, (ep_x, ep_y), radius=4, color=(0, 0, 255), thickness=-1)

        # Compute center of the box
        center_x = int((minx + maxx) / 2)
        center_y = int((miny + maxy) / 2)

        # Draw dark blue line from center to endpoint
        cv2.line(multiplicities_image, (center_x, center_y), (ep_x, ep_y), color=(128, 0, 0), thickness=2)
    cv2.imwrite(multiplicities_image_path, multiplicities_image)

    return multiplicities_image_path

# cluster role labels
def cluster_text_boxes(boxes, overlap_thresh=0.4, gap_thresh=0.6, aligned_thresh=0.4):
    horizontal = []
    vertical = []

    for idx, b in enumerate(boxes):
        x, y = b.bounds[0], b.bounds[1]
        w = b.bounds[2] - b.bounds[0]
        h = b.bounds[3] - b.bounds[1]
        if w >= h:
            horizontal.append((idx, x, y, w, h))
        else:
            vertical.append((idx, x, y, w, h))

    clusters = []

    def cluster_boxes(sorted_boxes, direction='vertical'):
        active_set = []
        clusters_local = []
        box_to_cluster = {}

        for idx, x, y, w, h in sorted_boxes:
            matched = False
            matched_cluster = None

            for a_idx, xa, ya, wa, ha in active_set:
                if direction == 'vertical':
                    aligned = abs((x + w / 2) - (xa + wa / 2)) < max(w, wa) * aligned_thresh
                    vertical_gap = y - (ya + ha)
                    close = -1 * (max(h, ha) * overlap_thresh) <= vertical_gap < max(h, ha) * gap_thresh
                else:
                    aligned = abs((y + h / 2) - (ya + ha / 2)) < max(h, ha) * aligned_thresh
                    horizontal_gap = x - (xa + wa)
                    close = -1 * (max(w, wa) * overlap_thresh) <= horizontal_gap < max(w, wa) * gap_thresh

                if aligned and close:
                    matched = True
                    matched_cluster = box_to_cluster[a_idx]
                    break

            if matched:
                matched_cluster.append(idx)
                box_to_cluster[idx] = matched_cluster
            else:
                # remove outdated actives
                new_active_set = []
                for a_idx, xa, ya, wa, ha in active_set:
                    if direction == 'vertical':
                        still_close = abs(y - ya) < ha * (1 + gap_thresh)
                    else:
                        still_close = abs(x - xa) < wa * (1 + gap_thresh)
                    if still_close:
                        new_active_set.append((a_idx, xa, ya, wa, ha))

                active_set = new_active_set
                # create new cluster
                new_cluster = [idx]
                clusters_local.append(new_cluster)
                box_to_cluster[idx] = new_cluster

            active_set.append((idx, x, y, w, h))

        return clusters_local

    vertical_sorted = sorted(vertical, key=lambda b: (b[1], b[2]))  # x, then y
    horizontal_sorted = sorted(horizontal, key=lambda b: (b[2], b[1]))  # y, then x

    clusters.extend(cluster_boxes(vertical_sorted, direction='horizontal'))
    clusters.extend(cluster_boxes(horizontal_sorted, direction='vertical'))

    # build bounding boxes for each cluster
    cluster_rects = []
    for cluster in clusters:
        clustered_boxes = [boxes[i] for i in cluster]
        union = unary_union(clustered_boxes).bounds
        x, y, x2, y2 = union
        cluster_rects.append(box(x, y, x2, y2))

    return clusters, cluster_rects

def assign_roles_and_relationship_names(label_boxes, class_endpoints, relationship_paths, scale_factor):
    label_clusters, label_cluster_boxes = cluster_text_boxes(label_boxes)

    shapely_endpoints = [Point(x, y) for x, y in class_endpoints]
    role_cluster_endpoint_assignment = optimal_geometry_to_geometry_assignment(label_cluster_boxes, "role",
                                                                               shapely_endpoints, "endpoints",
                                                                               max_distance=round(scale_factor * 30, 3))

    # build remaining_label_clusters from the unassigned clusters
    remaining_label_clusters = []
    remaining_label_cluster_boxes = []

    for i in range(len(label_clusters)):
        # check if cluster is not assigned yet
        if not any(assignment[0] == i for assignment in role_cluster_endpoint_assignment):
            remaining_label_clusters.append(label_clusters[i])
            remaining_label_cluster_boxes.append(label_cluster_boxes[i])
        else:
            # Add None to keep the indexes the same
            remaining_label_clusters.append(None)
            remaining_label_cluster_boxes.append(None)

    relationship_name_cluster_relationship_assignment = optimal_geometry_to_geometry_assignment(remaining_label_cluster_boxes, "relationship name",
                                                                                            relationship_paths, "relationship",
                                                                                            max_distance=round(scale_factor * 20, 3))

    return label_clusters, label_cluster_boxes, role_cluster_endpoint_assignment, relationship_name_cluster_relationship_assignment

def draw_assignments(image, box_list, box_color, point_list, point_color, box_point_assignment):
    for box_index, box in enumerate(box_list):

        assigned_point = [p_idx for b_idx, p_idx in box_point_assignment if b_idx == box_index]
        if assigned_point:
            point_index = assigned_point[0]
            point = point_list[point_index]
            chosen_box_color = box_color
        else:
            point = None
            chosen_box_color = (100, 100, 100)

        # Get rectangle corners from shapely box
        minx, miny, maxx, maxy = box.bounds
        top_left = (int(minx), int(miny))
        bottom_right = (int(maxx), int(maxy))

        # Draw rectangle
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, chosen_box_color, thickness=-1)

        # Blend the filled rectangle onto the original image
        image = cv2.addWeighted(overlay, 0.3, image, 1 - 0.3, 0)

        if point is not None:
            # Draw circle at point
            p_x, p_y = map(int, point)
            cv2.circle(image, (p_x, p_y), radius=4, color=point_color, thickness=-1)

            # Compute center of the box
            center_x = int((minx + maxx) / 2)
            center_y = int((miny + maxy) / 2)

            # Draw dark blue line from center to endpoint
            cv2.line(image, (center_x, center_y), (p_x, p_y), color=(128, 0, 0), thickness=2)

    return image

def extract_ocr_strings_from_cluster(image, image_width, image_height, label_regions, cluster_indexes):
    ocr_strings = []
    for index in cluster_indexes:
        region = label_regions[index]
        confidence_score = region['score']
        if confidence_score < 0.5:
            continue

        x1 = int(region["value"]["x"] / 100 * image_width)
        y1 = int(region["value"]["y"] / 100 * image_height)
        x2 = int(x1 + region["value"]["width"] / 100 * image_width)
        y2 = int(y1 + region["value"]["height"] / 100 * image_height)
        cropped_image = image[y1:y2, x1:x2]

        ocr_strings.append(OCRString(cropped_image, confidence_score, x1, y1, x2, y2))
    return ocr_strings

def collect_and_assign_label_ocr_boxes(image, label_regions, label_clusters, role_cluster_endpoint_assignment, relationship_name_cluster_relationship_assignment, class_endpoints, connections, collected_ocr_strings):
    image_height, image_width = image.shape[:2]

    for role_cluster_index, class_endpoint_index in role_cluster_endpoint_assignment:
        role_indexes = label_clusters[role_cluster_index]
        role_cluster_ocr_strings = extract_ocr_strings_from_cluster(image, image_width, image_height, label_regions, role_indexes)

        endpoint = class_endpoints[class_endpoint_index]
        for (endpoint_a, endpoint_b), connection_parameters in connections.items():
            if endpoint == endpoint_a:
                connection_parameters["role_ocr_strings_0"] = role_cluster_ocr_strings
                break
            if endpoint == endpoint_b:
                connection_parameters["role_ocr_strings_1"] = role_cluster_ocr_strings
                break

        collected_ocr_strings.extend(role_cluster_ocr_strings)

    for relationship_name_cluster_index, relationship_index in relationship_name_cluster_relationship_assignment:
        relationship_name_indexes = label_clusters[relationship_name_cluster_index]
        relationship_name_cluster_ocr_strings = extract_ocr_strings_from_cluster(image, image_width, image_height, label_regions, relationship_name_indexes)

        for idx, (_, connection_parameters) in enumerate(connections.items()):
            if relationship_index == idx:
                connection_parameters["relationship_name_ocr_strings"] = relationship_name_cluster_ocr_strings
                break

        collected_ocr_strings.extend(relationship_name_cluster_ocr_strings)

def collect_class_ocr_boxes(static_models, image, class_boxes, collected_ocr_strings):
    classes_ocr_strings = []
    class_ocr_string_count = 0

    for i, class_box in enumerate(class_boxes):
        class_x1 = int(class_box[0])
        class_y1 = int(class_box[1])
        class_x2 = class_x1 + int(class_box[2])
        class_y2 = class_y1 + int(class_box[3])

        class_image = image[class_y1:class_y2, class_x1:class_x2]
        class_image_height, class_image_width = class_image.shape[:2]
        class_image_path = config.PROCESSED_DIR + f"/class_{i}.png"
        cv2.imwrite(class_image_path, class_image)

        text_regions = predictor.predict_regions(static_models["text-detect.pt"], class_image_path, None, skip_preprocessing=True)
        text_regions = sorted(text_regions, key=lambda r: r['value']['y'])

        class_ocr_strings = {}

        first_ocr_string = None
        first_ocr_string_collection = None

        for j, text_region in enumerate(text_regions):
            print(f"Reading text ({j + 1}/{len(text_regions)}) from class ({i + 1}/{len(class_boxes)})...")

            text_confidence_score = text_region['score']
            if text_confidence_score < 0.5:
                continue

            label = text_region['value']['rectanglelabels'][0]
            text_x1 = int(text_region['value']['x'] / 100 * class_image_width)
            text_y1 = int(text_region['value']['y'] / 100 * class_image_height)
            text_x2 = int(text_x1 + text_region['value']['width'] / 100 * class_image_width)
            text_y2 = int(text_y1 + text_region['value']['height'] / 100 * class_image_height)

            text_image = class_image[text_y1:text_y2, text_x1:text_x2]
            if text_image.size == 0:
                continue  # skip empty crops

            pil_image = Image.fromarray(text_image)
            ocr_string = OCRString(pil_image, text_confidence_score, text_x1, text_y1, text_x2, text_y2)

            if label == "Stereotype":
                ocr_string_type = "stereotype"
            elif label == "Name":
                ocr_string_type = "name"
            elif label == "Property":
                ocr_string_type = "properties"
            else:
                # currently predictions with labels "Package" and "Other" are not used
                continue

            if ocr_string_type not in class_ocr_strings:
                class_ocr_strings[ocr_string_type] = []
            class_ocr_strings[ocr_string_type].append(ocr_string)

            if first_ocr_string is None:
                first_ocr_string = ocr_string
            if first_ocr_string_collection is None:
                first_ocr_string_collection = ocr_string_type

            collected_ocr_strings.append(ocr_string)
            class_ocr_string_count += 1

        # do some post-processing, to check if class has at least 1 name (if we have detected at least 1 text)
        if not "name" in class_ocr_strings and first_ocr_string is not None:
            # we take the first element as class name
            class_ocr_strings["name"] = [first_ocr_string]
            class_ocr_strings[first_ocr_string_collection].remove(first_ocr_string)

        classes_ocr_strings.append(class_ocr_strings)

    print(f"Found a total of {class_ocr_string_count} ocr strings in classes")

    # write class text image for visualization

    class_text_image = image.copy()
    class_text_image_path = config.PROCESSED_DIR + "/class_text.png"

    colors = {
        "stereotype": (0, 0, 255),
        "name": (0, 255, 0),
        "properties": (255, 0, 0),
    }

    # For each class
    for i, class_ocr_strings in enumerate(classes_ocr_strings):
        # For each type
        for ocr_string_type, ocr_strings in class_ocr_strings.items():
            # For each string
            for ocr_string in ocr_strings:
                class_offset_x = int(class_boxes[i][0])
                class_offset_y = int(class_boxes[i][1])

                text_x1 = class_offset_x + ocr_string.x1
                text_y1 = class_offset_y + ocr_string.y1
                text_x2 = class_offset_x + ocr_string.x2
                text_y2 = class_offset_y + ocr_string.y2

                top_left = (int(text_x1), int(text_y1))
                bottom_right = (int(text_x2), int(text_y2))

                # Draw green rectangle
                overlay = class_text_image.copy()
                cv2.rectangle(overlay, top_left, bottom_right, colors[ocr_string_type], thickness=-1)

                # Blend the filled rectangle onto the original image
                class_text_image = cv2.addWeighted(overlay, 0.3, class_text_image, 1 - 0.3, 0)

    cv2.imwrite(class_text_image_path, class_text_image)

    return classes_ocr_strings, class_text_image_path

def extract_text_from_ocr_boxes(ocr_strings):
    # Load model and processor
    print("Loading OCR model...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1", use_fast=True)
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    # Optional: move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    batch_size = 5
    print(f"Reading out {len(ocr_strings)} OCR strings in batches of {batch_size}...")
    for batch in chunks(ocr_strings, batch_size):
        batch_images = [ocr.image for ocr in batch]
        inputs = processor(images=batch_images, return_tensors="pt").to(device)

        with torch.no_grad():  # disables gradients, speeds things up
            generated_ids = trocr_model.generate(inputs.pixel_values)

        decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for ocr, text in zip(batch, decoded_texts):
            ocr.text = text
            ocr.write_image()

def generate_uml_diagram(class_boxes, classes_ocr_strings, class_connections):
    uml = UMLDiagram()

    for box, class_ocr in zip(class_boxes, classes_ocr_strings):
        if "name" in class_ocr:
            name = " ".join(s.text for s in class_ocr["name"])
        else :
            print(f"No class name was found for class at {(box[0], box[1])}. Putting \"Unknown\".")
            name = "Unknown"

        stereotype = None
        if "stereotype" in class_ocr and len(class_ocr["stereotype"]) > 0:
            stereotype = sorted(class_ocr["stereotype"], key=lambda s: s.score)[0].text

        attributes = []
        methods = []
        mode = "attributes"

        for s in class_ocr.get("properties", []):
            text = s.text
            if len(text) == 0:
                # Skip empty texts
                continue

            text = text_utils.correct_similarities_to_parenthesis(text)
            if mode == "methods" or text_utils.is_method(text):
                mode = "methods"
                methods.append(text)
            else:
                attributes.append(text)

        uml_class = UMLClass(
            stereotype=UMLClassStereoType.find_closest_match(stereotype),
            name=name,
            attributes=attributes,
            methods=methods
        )
        uml.add_class(uml_class)

    for (endpoint_a, endpoint_b), connection_parameters in class_connections.items():
        class_a_index = connection_parameters["class_index_0"]
        class_b_index = connection_parameters["class_index_1"]
        end_shape_label_a, end_shape_score_a = connection_parameters["end_shape_0"]
        end_shape_label_b, end_shape_score_b = connection_parameters["end_shape_1"]
        multiplicity_a = connection_parameters.get("multiplicity_0")
        multiplicity_b = connection_parameters.get("multiplicity_1")
        role_a = " ".join(s.text for s in connection_parameters.get("role_ocr_strings_0") or [])
        role_b = " ".join(s.text for s in connection_parameters.get("role_ocr_strings_1") or [])

        relationship_name = " ".join(s.text for s in connection_parameters.get("relationship_name_ocr_strings") or [])
        if end_shape_label_a != "Empty End" and end_shape_label_b != "Empty End":
            # Both have end shapes

            # Check if both are arrows -> convert both to empty ends
            if  end_shape_label_a == "Arrow End" and end_shape_label_b == "Arrow End":
                end_shape_label_a = "Empty End"
                end_shape_label_b = "Empty End"
            # Check if one is an arrow: if yes -> convert to empty end
            elif end_shape_label_a == "Arrow End" and end_shape_label_b != "Arrow End":
                end_shape_label_a = "Empty End"
            elif end_shape_label_b == "Arrow End" and end_shape_label_a != "Arrow End":
                end_shape_label_b = "Empty End"
            else:
                # Keep only the end shape with the higher score
                if end_shape_score_a > end_shape_score_b:
                    print(f"Relationship ({endpoint_a} to {endpoint_b})  has 2 end shapes, ignoring {end_shape_score_b} as its score ({end_shape_score_a}) is lower then the score of the {end_shape_label_b} ({end_shape_score_b}).)")
                    end_shape_label_b = "Empty End"
                else:
                    print(f"Relationship ({endpoint_a} to {endpoint_b})  has 2 end shapes, ignoring {end_shape_score_a} as its score ({end_shape_score_b}) is lower then the score of the {end_shape_label_a} ({end_shape_score_a}).)")
                    end_shape_label_a = "Empty End"

        if end_shape_label_a != "Empty End" and end_shape_label_b == "Empty End":
            # Swap
            class_a_index, class_b_index = class_b_index, class_a_index
            end_shape_label_a, end_shape_label_b = end_shape_label_b, end_shape_label_a
            multiplicity_a, multiplicity_b = multiplicity_b, multiplicity_a
            role_a, role_b = role_b, role_a

        uml_source_class = uml.get_class(class_a_index)
        uml_target_class = uml.get_class(class_b_index)
        uml_relationship_type = UMLRelationshipType.from_end_shape(end_shape_label_b)

        uml.add_relationship(UMLRelationship(uml_source_class, uml_target_class, uml_relationship_type, role_a, multiplicity_a, role_b, multiplicity_b, relationship_name))

    return uml

def render_uml_diagram(uml_code):
    encoded = plant_uml_encoder.encode_plantuml(uml_code)
    plantuml_svg_url = f"http://www.plantuml.com/plantuml/svg/{encoded}"
    print(f"Requesting rendered SVG from {plantuml_svg_url}")
    # Download SVG content
    response = requests.get(plantuml_svg_url)
    svg_text = response.text
    # Parse width/height or viewBox
    root = ElementTree.fromstring(svg_text)
    viewbox = root.attrib.get("viewBox")
    if viewbox:
        _, _, svg_width, svg_height = map(float, viewbox.strip().split())
    else:
        svg_width = float(root.attrib.get("width", 1000))
        svg_height = float(root.attrib.get("height", 300))
    # Show it in Streamlit
    html_code = f"""
                <object data="{plantuml_svg_url}" type="image/svg+xml"
                        width="{svg_width}px"
                        height="{svg_height}px"
                        style="display:block;"></object>
                """
    return html_code, svg_height, plantuml_svg_url, svg_text

def create_durations_chart(durations):
    import pandas as pd
    import altair as alt

    df = pd.DataFrame(durations, columns=["Step", "Time (s)"])
    df["Step"] = pd.Categorical(df["Step"], categories=[s[0] for s in durations], ordered=True)
    df["Is Total"] = df["Step"] == "Total"

    chart = alt.Chart(df).mark_bar(size=20).encode(
        x=alt.X("Time (s):Q", title="Time (s)"),
        y=alt.Y("Step:N", sort=None, title=None),
        color=alt.condition(
            alt.datum["Is Total"],
            alt.value("#E74C3C"),  # red for "Total"
            alt.value("#1f77b4")  # default blue
        ),
        tooltip=["Step", "Time (s)"]
    ).properties(
        width=700,
        height=30 * len(df)
    ).configure_axis(
        labelLimit=300  # allow longer text in y-axis
    ).configure_view(
        stroke=None
    )

    # write to file
    with open(config.PROCESSED_DIR + "/durations.txt", "w", encoding="utf-8") as f:
        f.write(df.drop(columns="Is Total").to_string(index=False))

    return chart