import os
import logging

from control_models.base import ControlModel

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import tempfile
import cv2
import numpy as np
from shapely.geometry import Polygon
from math import sqrt
import copy

logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

def calculate_global_median_color(image, num_samples=1000):
    h, w, _ = image.shape
    total_pixels = h * w

    # Flatten the image to a 2D array of pixels (N, 3)
    flat_pixels = image.reshape(-1, 3)

    # Randomly choose pixel indices without replacement
    if num_samples > total_pixels:
        num_samples = total_pixels
    indices = np.random.choice(total_pixels, size=num_samples, replace=False)

    # Sampled pixel values
    sampled_pixels = flat_pixels[indices]

    # Calculate median for each channel
    median_b = int(np.median(sampled_pixels[:, 0]))
    median_g = int(np.median(sampled_pixels[:, 1]))
    median_r = int(np.median(sampled_pixels[:, 2]))

    return median_b, median_g, median_r

def is_dark_mode(image):
    (median_r, median_g, median_b) = calculate_global_median_color(image)
    median_color_brightness = (median_r + median_g + median_b) / 3
    return median_color_brightness < 128

def preprocess(source_image_path, target_image_path=None):
    base, ext = os.path.splitext(os.path.basename(source_image_path))
    image = cv2.imread(source_image_path, cv2.IMREAD_UNCHANGED)

    # Preprocessing
    logger.info(f"Preprocessing to {source_image_path}")
    try:
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # Check if image has an alpha channel
        if image.shape[2] == 4:
            # Convert image to float for proper blending
            b, g, r, a = cv2.split(image.astype(float))

            # Normalize alpha to range 0..1
            alpha = a / 255.0

            # Create white background
            white = np.ones_like(alpha) * 255.0

            # Composite each channel over white
            b = b * alpha + white * (1 - alpha)
            g = g * alpha + white * (1 - alpha)
            r = r * alpha + white * (1 - alpha)

            # Stack and convert back to uint8
            image = cv2.merge([b, g, r]).astype(np.uint8)

        if is_dark_mode(image):
            logger.info("Dark mode detected. Inverting image...")
            image = cv2.bitwise_not(image)

        # Add white border
        border_pixels = 10
        image = cv2.copyMakeBorder(
            image,
            top=border_pixels,
            bottom=border_pixels,
            left=border_pixels,
            right=border_pixels,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]  # white
        )

        # Save the preprocessed image to a temporary path
        if target_image_path is not None:
            with open(target_image_path, "w", encoding="utf-8"):
                cv2.imwrite(target_image_path, image)
                return target_image_path
        else:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f"{base}_preprocessed{ext}") as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, image)
                return temp_path
    except Exception as e:
        raise Exception(f"Error during preprocessing.", e)

def calculate_intersection_area(region1, region2):
    x1 = region1['value']['x']
    y1 = region1['value']['y']
    width1 = region1['value']['width']
    height1 = region1['value']['height']

    x2 = region2['value']['x']
    y2 = region2['value']['y']
    width2 = region2['value']['width']
    height2 = region2['value']['height']

    # Calculate overlap
    x_overlap = max(0.0, min(x1 + width1, x2 + width2) - max(x1, x2))
    y_overlap = max(0.0, min(y1 + height1, y2 + height2) - max(y1, y2))

    return x_overlap * y_overlap

# Sweep Line Algorithm
def clear_overlapping_regions(regions: List[Dict], overlap_threshold: float = 0.3):
    # Sort boxes by x-coordinate
    regions = sorted(regions, key=lambda r: r['value']['x'])

    activeMap = {}

    for region in regions[:]:
        label = region['value']['rectanglelabels'][0]
        if label in ["Arrow End", "Crossing", "Empty Diamond End", "Filled Diamond End", "Triangle End"]:
            label = "End Shape"
        if label in ["Stereotype", "Name", "Property", "Package", "Other"]:
            label = "Class Text"
        if label in ["Class", "Struck Class", "Note"]:
            label = "UML Rectangle"
        active = activeMap.get(label, [])

        x = region['value']['x']
        y = region['value']['y']
        width = region['value']['width']
        height = region['value']['height']
        area = width * height
        new_active = []

        region_removed = False

        for active_region in active:
            # If current region is already removed, keep all following active_regions and continue
            if region_removed:
                new_active.append(active_region)
                continue

            # Remove boxes that are completely to the left of the current box
            if active_region['value']['x'] + active_region['value']['width'] < x:
                continue

            active_region_removed = False

            # Check overlap with active boxes
            intersection = calculate_intersection_area(region, active_region)
            if intersection > 0:
                active_area = active_region['value']['width'] * active_region['value']['height']
                overlap_ratio = intersection / min(area, active_area)
                if overlap_ratio > overlap_threshold:
                    if region['score'] < active_region['score']:
                        regions.remove(region)
                        region_removed = True
                    else:
                        regions.remove(active_region)
                        active_region_removed = True
            if not active_region_removed:
                new_active.append(active_region)

        if not region_removed:
            new_active.append(region)

        activeMap[label] = new_active

    return regions

def clean_points(points):
    # Remove consecutive duplicate points
    cleaned = [points[0]]
    for p in points[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    return cleaned

def points_to_clean_polygon(points):
    try:
        polygon = Polygon(clean_points(points))
        polygon = polygon.buffer(0)
        return polygon
    except Exception as e:
        print(f"Warning: could not clean polygon: {e}")
        return Polygon()

def polygon_overlap_relative(poly_a, poly_b):
    inter = poly_a.intersection(poly_b).area
    return inter / (min(poly_a.area, poly_b.area) + 1e-6)

def polygon_to_points_safe(polygon) -> List[List[float]]:
    # Always take the largest polygon if MultiPolygon
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda p: p.area)
        logger.warning("MultiPolygon will cause problems")
    if not polygon.is_empty:
        return [[float(x), float(y)] for x, y in polygon.simplify(0.03, preserve_topology=True).exterior.coords[:-1]]
    raise ValueError(f"Polygon is empty: {polygon}")

def calculate_stretch_distance(polygon_points):
    if len(polygon_points) < 3:
        return 0.0
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    for p in polygon_points:
        if p[0] < min_x:
            min_x = p[0]
        if p[0] > max_x:
            max_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
        if p[1] > max_y:
            max_y = p[1]

    return sqrt((min_x - max_x) ** 2 + (min_y - max_y) ** 2)

def save_polygon_to_file(polygon, filename, folder='polygon_images', color='blue'):
    import matplotlib.pyplot as plt
    from pathlib import Path

    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    fig, ax = plt.subplots()

    if polygon.geom_type == 'Polygon':
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color)
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color)
    else:
        print(f"Unsupported geometry type: {polygon.geom_type}")
        return

    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert Y axis so polygon is right side up
    plt.axis('off')  # turn off axes

    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # close figure to avoid memory leak
    print(f"Saved plotted polygon to: {Path(filepath).absolute()}")

def find(parent, i):
    # Path compression
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, i, j):
    root_i = find(parent, i)
    root_j = find(parent, j)
    if root_i != root_j:
        parent[root_j] = root_i  # merge j into i

def extract_loops(points: List[List[float]]) -> List[List[List[float]]]:
    def find_first_loop(seq: List[List[float]]) -> Tuple[int, int] | None:
        seen = {}
        for idx, p in enumerate(seq):
            key = tuple(p)  # exact match, no rounding
            if key in seen:
                return seen[key], idx
            seen[key] = idx
        return None

    loops = [points]  # start with the outermost loop

    loop_idx = 0
    while loop_idx < len(loops):
        loop = loops[loop_idx]
        changed = True
        while changed:
            changed = False
            result = find_first_loop(loop)
            if result:
                start, end = result
                subloop = loop[start:end + 1]
                if len(subloop) == len(loop):
                    break
                # remove subloop from parent
                loop = loop[:start + 1] + loop[end + 1:]
                loops[loop_idx] = loop
                loops.append(subloop)
                changed = True
        loop_idx += 1

    return loops

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

def plot_shapely_polygon(polygon: BaseGeometry, title="Untitled", color='blue', alpha=0.5, show=True):
    if polygon.is_empty:
        print("Empty geometry.")
        return

    fig, ax = plt.subplots()
    ax.set_title(title)

    def _plot_single_poly(poly):
        x, y = poly.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color='white', alpha=1.0)

    if polygon.geom_type == 'Polygon':
        _plot_single_poly(polygon)
    elif polygon.geom_type == 'MultiPolygon':
        for poly in polygon.geoms:
            _plot_single_poly(poly)
    else:
        raise TypeError(f"Unsupported geometry type: {polygon.geom_type}")

    ax.set_aspect('equal', adjustable='datalim')
    plt.axis('on')
    if show:
        plt.show()

def merge_relationships(regions: List[Dict], merge_threshold: float = 0.1) -> List[Dict]:
    if len(regions) == 0:
        return []
    print("Merging regions...")
    merged = []

    # Precompute cleaned polygons
    cleaned_polygons = []
    polygon_regions = []

    for region in regions:
        loops = extract_loops(region['value']['points'])
        for j, loop in enumerate(loops):
            if len(loop) < 4:
                continue

            part_polygon = points_to_clean_polygon(loop)

            if part_polygon.geom_type == "GeometryCollection":
                # skip bad polygons
                continue

            if part_polygon.area > 0.1:
                cleaned_polygons.append(part_polygon)
                separated_region = copy.deepcopy(region)
                # if separated_region['id'] == "910,0 960x960 2":
                    # save_polygon_to_file(part_polygon, f"{separated_region['id']}_s{j}" + ".png")
                separated_region["id"] = f"{separated_region['id']}_s{j}"
                polygon_regions.append(separated_region)

    # Handle MultiPolygon separation
    i = 0
    while i < len(cleaned_polygons):
        polygon = cleaned_polygons[i]
        region = polygon_regions[i]
        if polygon.geom_type == 'MultiPolygon':
            # Replace current item with first part, append rest
            geoms = list(polygon.geoms)
            cleaned_polygons[i] = geoms[0]
            for j, child in enumerate(geoms[1:], start=1):
                if child.geom_type == "GeometryCollection":
                    # skip bad polygons
                    continue
                cleaned_polygons.append(child)
                new_region = copy.deepcopy(region)
                new_region["id"] = f"{region['id']}_p{j}"
                polygon_regions.append(new_region)
        i += 1

    parent = {}
    # Initialize each region to be its own parent
    for i in range(len(cleaned_polygons)):
        parent[i] = i

    for i in range(len(cleaned_polygons)):
        region_a = polygon_regions[i]
        polygon_a = cleaned_polygons[i]

        for j in range(i + 1, len(cleaned_polygons)):
            region_b = polygon_regions[j]
            polygon_b = cleaned_polygons[j]

            overlap = polygon_overlap_relative(polygon_a, polygon_b)

            if overlap >= merge_threshold:
                print(f"Overlap of {region_a['id']} and {region_b['id']} is {overlap}")
                union(parent, i, j)

    clusters = defaultdict(list)

    for i in range(len(cleaned_polygons)):
        root = find(parent, i)
        clusters[root].append(i)

    connected_regions = list(clusters.values())

    for connected_region_indexes in connected_regions:
        for region_index in connected_region_indexes:
            print(polygon_regions[region_index]['id'])
        print("--------")

    for connected_region_indexes in connected_regions:
        current_polygon = cleaned_polygons[connected_region_indexes[0]]
        first_region = polygon_regions[connected_region_indexes[0]]
        current_score = first_region['score']

        for region_index in connected_region_indexes[1:]:
            next_polygon = cleaned_polygons[region_index]
            if current_polygon.union(next_polygon).geom_type == "GeometryCollection":
                # skip bad unions
                continue
            current_score = (current_score * current_polygon.area + polygon_regions[region_index]['score'] * next_polygon.area) / (current_polygon.area + next_polygon.area)
            current_polygon = current_polygon.union(next_polygon)

        result_polygon_points = polygon_to_points_safe(current_polygon)

        stretch_distance = calculate_stretch_distance(result_polygon_points)
        if stretch_distance > 2:
            result_region = first_region
            result_region['value']['points'] = result_polygon_points
            result_region['value']['polygon'] = current_polygon
            result_region['score'] = current_score
            merged.append(result_region)

    return merged

def stitch_predictions(predictions: List[Dict], image_width, image_height, remove_edge_predictions=False, set_tile_position_as_id=False) -> List[Dict]:
    stitched_regions = []
    for pred in predictions:
        region_index = 1
        for region in pred['result']:
            regionType = region["type"]
            if regionType == "rectanglelabels":

                # Calculate if the predictions were right at a tile border
                border_pixels = 10
                is_left_tile_border_prediction = region['value']['x'] * pred['tile_width'] / 100 < border_pixels
                is_right_tile_border_prediction = (region['value']['x'] + region['value']['width']) * pred['tile_width'] / 100 > pred['tile_width'] - border_pixels
                is_top_tile_border_prediction = region['value']['y'] * pred['tile_height'] / 100 < border_pixels
                is_bottom_tile_border_prediction = (region['value']['y'] + region['value']['height']) * pred['tile_height'] / 100 > pred['tile_height'] - border_pixels

                # Adjust relative values x, y, width and height to the total image

                # First scale
                region['value']['x'] *= pred['tile_width'] / image_width
                region['value']['y'] *= pred['tile_height'] / image_height
                region['value']['width'] *= pred['tile_width'] / image_width
                region['value']['height'] *= pred['tile_height'] / image_height

                # Second add offset (for x and y)
                region['value']['x'] += pred['offset_px_x'] / image_width * 100
                region['value']['y'] += pred['offset_px_y'] / image_height * 100

                if set_tile_position_as_id:
                    # Set Id
                    region['id'] = f"{pred['offset_px_x']},{pred['offset_px_y']} {pred['tile_width']}x{pred['tile_height']} {region_index}"

                is_left_image_border_prediction = region['value']['x'] * image_width / 100 < border_pixels
                is_right_image_border_prediction = (region['value']['x'] + region['value']['width']) * image_width / 100 > image_width - border_pixels
                is_top_image_border_prediction = region['value']['y'] * image_height / 100 < border_pixels
                is_bottom_image_border_prediction = (region['value']['y'] + region['value']['height']) * pred['tile_height'] / 100 > image_height - border_pixels

                if not remove_edge_predictions or (not ((is_left_tile_border_prediction and not is_left_image_border_prediction) or (is_right_tile_border_prediction and not is_right_image_border_prediction) or (is_top_tile_border_prediction and not is_top_image_border_prediction) or (is_bottom_tile_border_prediction and not is_bottom_image_border_prediction))):
                    stitched_regions.append(region)

            elif regionType == "polygonlabels":
                # Extract point coordinates
                points = region['value']['points']

                # Adjust relative values of each point to the total image
                for p in points:
                    # First scale
                    p[0] *= pred['tile_width'] / image_width
                    p[1] *= pred['tile_height'] / image_height

                    # Second add offset
                    p[0] += pred['offset_px_x'] / image_width * 100
                    p[1] += pred['offset_px_y'] / image_height * 100

                # Optionally set id
                if True:
                    region['id'] = f"{pred['offset_px_x']},{pred['offset_px_y']} {pred['tile_width']}x{pred['tile_height']} {region_index}"

                stitched_regions.append(region)
            else:
                raise ValueError(f"Type '{regionType}' is not supported")

            region_index += 1

    filtered_regions = []
    filtered_regions += clear_overlapping_regions(filter(lambda r: r["type"] == "rectanglelabels", stitched_regions))
    filtered_regions += merge_relationships(list(filter(lambda r: r["type"] == "polygonlabels", stitched_regions)))
    filtered_regions = sorted(filtered_regions, key=lambda r: r['score'], reverse=True)
    return filtered_regions

def tile_image(image, overlap, tile_size):
    height, width = image.shape[:2]
    step_h = int(tile_size * (1 - overlap))
    step_w = int(tile_size * (1 - overlap))

    tiles = []
    seen_positions = set()

    for y in range(0, height, step_h):
        for x in range(0, width, step_w):
            new_x1 = x
            new_x2 = x + tile_size
            new_y1 = y
            new_y2 = y + tile_size

            if new_x2 > width:
                new_x1 = width - tile_size
                new_x2 = width
            if new_y2 > height:
                new_y1 = height - tile_size
                new_y2 = height

            if new_x1 < 0:
                new_x1 = 0
            if new_x2 > width:
                new_x2 = width
            if new_y1 < 0:
                new_y1 = 0
            if new_y2 > height:
                new_y2 = height

            # Only add tile if this position was not added before
            pos_key = (new_x1, new_y1)
            if pos_key not in seen_positions:
                tile = image[new_y1:new_y2, new_x1:new_x2]
                if np.all(tile == 255):
                    continue
                tiles.append((tile, new_x1, new_y1))
                seen_positions.add(pos_key)

    return tiles

def predict_regions(model: ControlModel, image_path: str, static_models: Optional[Dict[str, ControlModel]], set_tile_position_as_id=False, skip_preprocessing=False, min_score=0.0):
    model_name = os.path.basename(model.control.attr["model_path"])
    max_predict_size = model.model.model.args["imgsz"]

    if not skip_preprocessing and model_name != "endpoint-detect.pt" and model_name != "multiplicity-classify.pt":
        image_path = preprocess(image_path)

    if model_name.startswith("class-detect"):
        return predict_classes(model, image_path, max_predict_size, set_tile_position_as_id, min_score)
    elif model_name.startswith("text-detect"):
        return predict_class_text(model, image_path, min_score)
    elif model_name.startswith("relationship-seg"):
        return predict_relationships(model, image_path, static_models, max_predict_size, set_tile_position_as_id, skip_preprocessing, min_score)
    elif model_name.startswith("endpoint-detect"):
        return predict_endpoints(model, image_path, max_predict_size, set_tile_position_as_id, min_score)
    elif model_name.startswith("label-detect"):
        return predict_relationship_labels(model, image_path, static_models, max_predict_size, set_tile_position_as_id, skip_preprocessing, min_score)
    elif model_name.startswith("multiplicity-classify"):
        return predict_multiplicities(model, image_path)
    else:
        raise Exception(f"Model name '{model_name}' is not supported")

def predict_classes(model, path, max_predict_size, set_tile_position_as_id, min_score=0.0):
    image = cv2.imread(path)
    image_height, image_width = image.shape[:2]
    base, ext = os.path.splitext(os.path.basename(path))

    task_predictions = []

    # Always do the scaled version of the image
    scaled_image_result = model.predict_regions(path)
    task_predictions.append(
        {'result': scaled_image_result, 'tile_width': image_width, 'tile_height': image_height,
         'offset_px_x': 0, 'offset_px_y': 0})
    # If one side is larger than the MAX_PREDICT_SIZE we do multiple scale levels
    if max(image_width, image_height) > max_predict_size:
        logger.info(f"Tiling image: {path}")
        current_tile_size = int(max(image_width, image_height) / 2)
        finish = False
        while True:
            # Check if the current_tile_size is now even lower than the MAX_PREDICT_SIZE
            if current_tile_size < max_predict_size:
                # If it is nearly half the MAX_PREDICT_SIZE we can stop instantly, because the last current_tile_size almost did this size
                if current_tile_size < max_predict_size * 0.8:
                    break

                # Else this iteration will be the last with exactly the MAX_PREDICT_SIZE as current_tile_size
                current_tile_size = max_predict_size
                finish = True

            tiles = tile_image(image, overlap=0.5, tile_size=current_tile_size)

            for tile, abs_x, abs_y in tiles:
                rel_x =  abs_x / image_width * 100
                rel_y =  abs_y / image_height * 100
                logger.info(f"Processing tile: Relative Position: {rel_x},{rel_y} - Absolute Position: {abs_x},{abs_y} - Absolute Size: {current_tile_size}x{current_tile_size}")
                # Save the preprocessed image to a temporary path
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f"{base}_tile_{current_tile_size}x{current_tile_size}_{int(rel_x)}_{int(rel_y)}{ext}") as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, tile)
                    tile_result = model.predict_regions(temp_path)
                    tile_height, tile_width = tile.shape[:2]
                    task_predictions.append(
                        {'result': tile_result, 'tile_width': tile_width, 'tile_height': tile_height,
                         'offset_px_x': abs_x, 'offset_px_y': abs_y})

            current_tile_size = int(current_tile_size / 2)
            if finish:
                break

    predictions = stitch_predictions(task_predictions, image_width, image_height, remove_edge_predictions=True, set_tile_position_as_id=set_tile_position_as_id)
    return [pred for pred in predictions if pred["score"] >= min_score]

def predict_class_text(model, path, min_score=0.0):
    predictions = model.predict_regions(path)
    predictions = clear_overlapping_regions(predictions, overlap_threshold=0.6)
    return [pred for pred in predictions if pred["score"] >= min_score]

def predict_relationships(model: ControlModel, path, static_models: Dict[str, ControlModel], max_model_size, set_tile_position_as_id, skip_preprocessing=False, min_score=0.0):
    if not skip_preprocessing:
        import image_region_preparer

        # prepare classes
        base, ext = os.path.splitext(path)
        processed_path = f"{base}_with_classes_marked{ext}"
        predicted_classes = predict_classes(static_models["class-detect.pt"], path, max_model_size, set_tile_position_as_id, min_score=0.5)
        image_region_preparer.mark_classes(predicted_classes, path, processed_path)
        path = processed_path

    image = cv2.imread(path)
    image_height, image_width = image.shape[:2]
    base, ext = os.path.splitext(os.path.basename(path))

    task_predictions = []

    max_predict_size = int(max_model_size / 2)

    # If one side is larger than the MAX_PREDICT_SIZE we do multiple scale levels
    if max(image_width, image_height) > max_predict_size:
        logger.info(f"Tiling image: {path}")
        current_tile_size = int(max(image_width, image_height))
        finish = False
        while True:
            # Check if the current_tile_size is now even lower than the MAX_PREDICT_SIZE
            if current_tile_size < max_predict_size:
                # If it is nearly half the MAX_PREDICT_SIZE we can stop instantly, because the last current_tile_size almost did this size
                if current_tile_size < max_predict_size * 0.8:
                    break

                # Else this iteration will be the last with exactly the MAX_PREDICT_SIZE as current_tile_size
                current_tile_size = max_predict_size
                finish = True

            overlap = 0.5 if current_tile_size > max_model_size else 0.0
            tiles = tile_image(image, overlap=overlap, tile_size=current_tile_size)

            for tile, abs_x, abs_y in tiles:
                rel_x = abs_x / image_width * 100
                rel_y = abs_y / image_height * 100
                logger.info(
                    f"Processing tile: Relative Position: {rel_x},{rel_y} - Absolute Position: {abs_x},{abs_y} - Absolute Size: {current_tile_size}x{current_tile_size}")
                # Save the preprocessed image to a temporary path
                with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                 suffix=f"{base}_tile_{current_tile_size}x{current_tile_size}_{int(rel_x)}_{int(rel_y)}{ext}") as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, tile)
                    tile_result = model.predict_regions(temp_path)
                    tile_height, tile_width = tile.shape[:2]
                    task_predictions.append(
                        {'result': tile_result, 'tile_width': tile_width, 'tile_height': tile_height,
                         'offset_px_x': abs_x, 'offset_px_y': abs_y})

            current_tile_size = int(current_tile_size / 2)
            if finish:
                break
    else:
        prediction_result = model.predict_regions(path)
        task_predictions.append(
            {'result': prediction_result, 'tile_width': image_width, 'tile_height': image_height,
             'offset_px_x': 0, 'offset_px_y': 0})

    predictions = stitch_predictions(task_predictions, image_width, image_height, remove_edge_predictions=False, set_tile_position_as_id=set_tile_position_as_id)
    return [pred for pred in predictions if pred["score"] >= min_score]

def predict_relationship_labels(model, path, static_models, max_predict_size, set_tile_position_as_id, skip_preprocessing=False, min_score=0.0):
    import image_region_preparer

    if not skip_preprocessing:
        # prepare classes
        base, ext = os.path.splitext(path)
        preprocessed_path = f"{base}_with_classes_and_relationships_marked{ext}"
        predicted_classes = predict_classes(static_models["class-detect.pt"], path, max_predict_size, set_tile_position_as_id, min_score=0.5)
        image_region_preparer.mark_classes(predicted_classes, path, preprocessed_path)
        # prepare relationships
        predicted_relationships = predict_relationships(static_models["relationship-seg.pt"], preprocessed_path, static_models, max_predict_size, set_tile_position_as_id, skip_preprocessing=True, min_score=0.5)
        image_region_preparer.mark_relationships(predicted_relationships, preprocessed_path, preprocessed_path)
        path = preprocessed_path

    image = cv2.imread(path)
    image_height, image_width = image.shape[:2]

    # If one side is larger than the max_predict_size we tile the image
    if max(image_width, image_height) > max_predict_size:
        task_predictions = []
        base, ext = os.path.splitext(os.path.basename(path))

        logger.info(f"Tiling image: {path}")
        tile_size = max_predict_size

        tiles = tile_image(image, overlap=0.5, tile_size=tile_size)

        for tile, abs_x, abs_y in tiles:
            rel_x = abs_x / image_width * 100
            rel_y = abs_y / image_height * 100
            logger.info(
                f"Processing tile: Relative Position: {rel_x},{rel_y} - Absolute Position: {abs_x},{abs_y} - Absolute Size: {tile_size}x{tile_size}")
            # Save the preprocessed image to a temporary path
            with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                             suffix=f"{base}_tile_{tile_size}x{tile_size}_{int(rel_x)}_{int(rel_y)}{ext}") as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, tile)
                tile_result = model.predict_regions(temp_path)
                tile_height, tile_width = tile.shape[:2]
                task_predictions.append(
                    {'result': tile_result, 'tile_width': tile_width, 'tile_height': tile_height,
                     'offset_px_x': abs_x, 'offset_px_y': abs_y})

        predictions = stitch_predictions(task_predictions, image_width, image_height, remove_edge_predictions=True, set_tile_position_as_id=set_tile_position_as_id)
    else:
        predictions = model.predict_regions(path)
    return [pred for pred in predictions if pred["score"] >= min_score]

def predict_endpoints(model, path, max_model_size, set_tile_position_as_id, min_score=0.0):
    image = cv2.imread(path)
    image_height, image_width = image.shape[:2]
    base, ext = os.path.splitext(os.path.basename(path))

    task_predictions = []

    max_predict_size = int(max_model_size / 2)

    # If one side is larger than the MAX_PREDICT_SIZE we do multiple scale levels
    if max(image_width, image_height) > max_predict_size:
        logger.info(f"Tiling image: {path}")
        current_tile_size = int(max(image_width, image_height) / 2)
        finish = False
        while True:
            # Check if the current_tile_size is now even lower than the MAX_PREDICT_SIZE
            if current_tile_size < max_predict_size:
                # If it is nearly half the MAX_PREDICT_SIZE we can stop instantly, because the last current_tile_size almost did this size
                if current_tile_size < max_predict_size * 0.8:
                    break

                # Else this iteration will be the last with exactly the MAX_PREDICT_SIZE as current_tile_size
                current_tile_size = max_predict_size
                finish = True

            overlap = 0.5 if current_tile_size > max_model_size else 0.0
            tiles = tile_image(image, overlap=overlap, tile_size=current_tile_size)

            for tile, abs_x, abs_y in tiles:
                rel_x = abs_x / image_width * 100
                rel_y = abs_y / image_height * 100
                logger.info(
                    f"Processing tile: Relative Position: {rel_x},{rel_y} - Absolute Position: {abs_x},{abs_y} - Absolute Size: {current_tile_size}x{current_tile_size}")
                # Save the preprocessed image to a temporary path
                with tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                 suffix=f"{base}_tile_{current_tile_size}x{current_tile_size}_{int(rel_x)}_{int(rel_y)}{ext}") as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, tile)
                    tile_result = model.predict_regions(temp_path)
                    tile_height, tile_width = tile.shape[:2]
                    task_predictions.append(
                        {'result': tile_result, 'tile_width': tile_width, 'tile_height': tile_height,
                         'offset_px_x': abs_x, 'offset_px_y': abs_y})

            current_tile_size = int(current_tile_size / 2)
            if finish:
                break
    else:
        prediction_result = model.predict_regions(path)
        task_predictions.append(
            {'result': prediction_result, 'tile_width': image_width, 'tile_height': image_height,
             'offset_px_x': 0, 'offset_px_y': 0})

    predictions = stitch_predictions(task_predictions, image_width, image_height, remove_edge_predictions=False,
                                     set_tile_position_as_id=set_tile_position_as_id)
    return [pred for pred in predictions if pred["score"] >= min_score]

def predict_multiplicities(model, path):
    return model.predict_regions(path)

def get_bounding_box_relative(region):
    return (region['value']['x'], region['value']['y'], region['value']['width'], region['value']['height'])