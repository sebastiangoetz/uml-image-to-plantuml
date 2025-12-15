import shutil

import cv2
import numpy as np
import predictor
from enum import Enum
from shapely.affinity import scale

class PrepareType(Enum):
    MARK = "mark"
    MARK_WITH_BORDER = "mark_with_border"
    REMOVE = "remove"

def calculate_global_median_color(image, sample_step=50):
    # Extract samples using fixed step size
    sampled_pixels = image[::sample_step, ::sample_step].reshape(-1, 3)

    # Calculate the median for each channel
    median_b = int(np.median(sampled_pixels[:, 0]))
    median_g = int(np.median(sampled_pixels[:, 1]))
    median_r = int(np.median(sampled_pixels[:, 2]))

    return median_b, median_g, median_r

def overwrite_regions(regions, source_path, target_path, color, type: PrepareType, cut_pixels=3, cut_percent=0.00):
    try:
        image = cv2.imread(source_path)
        if image is None:
            print(f"Could not read image: {source_path}")
            return

        height, width = image.shape[:2]

        for region in regions:
            region_type = region["type"]

            if region_type == "rectanglelabels":
                box = predictor.get_bounding_box_relative(region)
                # Convert relative coordinates to absolute pixel values
                x = int((box[0] + cut_percent * box[2]) / 100 * width) + cut_pixels
                y = int((box[1] + cut_percent * box[3]) / 100 * height) + cut_pixels
                w = int((box[2] * (1 - 2 * cut_percent)) / 100 * width) - 2 * cut_pixels
                h = int((box[3] * (1 - 2 * cut_percent)) / 100 * height) - 2 * cut_pixels

                # Ensure the coordinates are within image boundaries
                x_start = max(x, 0)
                y_start = max(y, 0)
                x_end = min(x + w, width)
                y_end = min(y + h, height)

                if x_end < x_start or y_end < y_start:
                    # rectangle size is negative (because of cut) - > skip
                    continue

                if type == PrepareType.MARK or type == PrepareType.MARK_WITH_BORDER:
                    fill_alpha = 0.3
                    border_alpha = 0.6

                    overlay = image.copy()
                    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), color, thickness=-1)

                    # Blend the filled rectangle onto the original image
                    image = cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0)

                    if type == PrepareType.MARK_WITH_BORDER:
                        # Step 2: Draw the border with more opacity
                        border_overlay = image.copy()
                        cv2.rectangle(border_overlay, (x_start, y_start), (x_end, y_end), color, thickness=2)

                        # Blend the border onto the image
                        image = cv2.addWeighted(border_overlay, border_alpha, image, 1 - border_alpha, 0)
                elif type == PrepareType.REMOVE:
                    image[y_start:y_end, x_start:x_end] = color
                else:
                    raise ValueError(f"Type {type} is not supported.")

            elif region_type == "polygonlabels":
                if "polygon" in region["value"]:
                    shapely_polygon = region["value"]["polygon"]
                    shapely_polygon = scale(shapely_polygon, xfact=width / 100.0, yfact=height / 100.0, origin=(0, 0))

                    if type == PrepareType.MARK or type == PrepareType.MARK_WITH_BORDER:
                        fill_alpha = 0.3
                        border_alpha = 0.6
                        overlay = image.copy()

                        # Fill exterior
                        exterior = np.array(shapely_polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [exterior], color)

                        # Subtract holes
                        for interior in shapely_polygon.interiors:
                            hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(overlay, [hole], (255, 255, 255))  # white hole

                        image = cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0)

                        if type == PrepareType.MARK_WITH_BORDER:
                            border_overlay = image.copy()
                            cv2.polylines(border_overlay, [exterior], isClosed=True, color=color, thickness=2)
                            for interior in shapely_polygon.interiors:
                                hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(border_overlay, [hole], isClosed=True, color=(0, 0, 0), thickness=2)
                            image = cv2.addWeighted(border_overlay, border_alpha, image, 1 - border_alpha, 0)

                    elif type == PrepareType.REMOVE:
                        cv2.fillPoly(image,[np.array(shapely_polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))], color)
                        for interior in shapely_polygon.interiors:
                            hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(image, [hole], (255, 255, 255))  # erase hole

                    else:
                        raise ValueError(f"Type {type} is not supported.")
                else:
                    # Get points in relative coordinates
                    points_rel = region['value']['points']

                    # Convert to absolute pixel coordinates
                    points_abs = []

                    for x_percent, y_percent in points_rel:
                        x_abs = int(x_percent / 100 * width)
                        y_abs = int(y_percent / 100 * height)
                        points_abs.append([x_abs, y_abs])

                    pts = np.array(points_abs, dtype=np.int32).reshape((-1, 1, 2))
                    if type == PrepareType.MARK or type == PrepareType.MARK_WITH_BORDER:
                        fill_alpha = 0.3
                        border_alpha = 0.6
                        overlay = image.copy()
                        cv2.fillPoly(overlay, [pts], color)
                        image = cv2.addWeighted(overlay, fill_alpha, image, 1 - fill_alpha, 0)
                        if type == PrepareType.MARK_WITH_BORDER:
                            border_overlay = image.copy()
                            cv2.polylines(border_overlay, [pts], isClosed=True, color=color, thickness=2)
                            image = cv2.addWeighted(border_overlay, border_alpha, image, 1 - border_alpha, 0)

                    elif type == PrepareType.REMOVE:
                        cv2.fillPoly(image, [pts], color)
                    else:
                        raise ValueError(f"Type {type} is not supported.")
            else:
                raise ValueError(f"Region type {region_type} is not supported.")

        # Save the modified image
        cv2.imwrite(target_path, image)
        print(f"Wrote image with overwritten regions to {target_path}")

    except Exception as e:
        print(f"Error processing {source_path}: {e}")

def mark_classes(predicted_regions, source_file_path, target_file_path):
    class_regions = filter(lambda r: r['value']['rectanglelabels'][0] == "Class", predicted_regions)
    struck_class_regions = filter(lambda r: r['value']['rectanglelabels'][0] == "Struck Class", predicted_regions)
    mark_regions(class_regions, source_file_path, target_file_path, (0, 255, 0))
    mark_regions(struck_class_regions, target_file_path, target_file_path, (0, 0, 255))

def mark_relationships(predicted_regions, source_file_path, target_file_path):
    color = (255, 0, 0)
    mark_regions(predicted_regions, source_file_path, target_file_path, color)

def mark_regions(predicted_regions, source_file_path, target_file_path, color):
    overwrite_regions(predicted_regions, source_file_path, target_file_path, color, PrepareType.MARK)

def mark_classes_with_border(predicted_regions, source_file_path, target_file_path):
    color = (0, 255, 0)
    mark_regions_with_border(predicted_regions, source_file_path, target_file_path, color)

def mark_relationships_with_border(predicted_regions, source_file_path, target_file_path):
    color = (255, 0, 0)
    mark_regions_with_border(predicted_regions, source_file_path, target_file_path, color)

def mark_regions_with_border(predicted_regions, source_file_path, target_file_path, color):
    overwrite_regions(predicted_regions, source_file_path, target_file_path, color, PrepareType.MARK_WITH_BORDER)

def remove_regions(predicted_regions, source_file_path, target_file_path, cut_pixels=0):
    image = cv2.imread(source_file_path)
    color = calculate_global_median_color(image, sample_step=50)
    overwrite_regions(predicted_regions, source_file_path, target_file_path, color, PrepareType.REMOVE, cut_pixels=cut_pixels)

def main():
    import os
    import sys
    from pathlib import Path
    from model_utils import create_static_models

    if len(sys.argv) < 5:  # sys.argv[0] is the script name
        print("Usage: python image_region_preparer.py input_folder_path output_folder_path prepare_type region_type1 region_type2 ...")
        sys.exit(1)

    input_folder_path = sys.argv[1]
    output_folder_path = sys.argv[2]
    prepare_type = PrepareType(sys.argv[3])
    region_types = sys.argv[4:]

    if not Path(input_folder_path).exists():
        print(f"Input folder '{Path(input_folder_path).absolute()}' doesn't exist")
        sys.exit()

    Path(output_folder_path).mkdir(parents=True, exist_ok=True)

    print("Starting model for class detection...")

    static_models = create_static_models()

    file_names = list(os.listdir(input_folder_path))
    total_files = len(file_names)

    for i, file_name in enumerate(file_names, start=1):
        print(f"{i}/{total_files} {file_name}")

        source_file_path = os.path.join(input_folder_path, file_name)
        target_file_path = os.path.join(output_folder_path, file_name)

        if source_file_path != target_file_path:
            shutil.copy(source_file_path, target_file_path)

        if "classes" in region_types:
            predicted_regions = predictor.predict_regions(static_models["class-detect.pt"], source_file_path, static_models, False, min_score=0.5)

            class_regions = filter(lambda r: r['value']['rectanglelabels'][0] == "Class", predicted_regions)
            struck_class_regions = filter(lambda r: r['value']['rectanglelabels'][0] == "Struck Class", predicted_regions)

            if prepare_type == PrepareType.MARK:
                mark_regions(class_regions, target_file_path, target_file_path, color=(0, 255, 0))
                mark_regions(struck_class_regions, target_file_path, target_file_path, color=(0, 0, 255))
            elif prepare_type == PrepareType.MARK_WITH_BORDER:
                mark_regions_with_border(class_regions, target_file_path, target_file_path, color=(0, 255, 0))
                mark_regions_with_border(struck_class_regions, target_file_path, target_file_path, color=(0, 0, 255))
            elif prepare_type == PrepareType.REMOVE:
                remove_regions(class_regions, target_file_path, target_file_path)
                remove_regions(struck_class_regions, target_file_path, target_file_path)
            else:
                raise ValueError(f"Type {prepare_type} is not supported.")

        if "relationships" in region_types:
            predicted_regions = predictor.predict_regions(static_models["relationship-seg.pt"], source_file_path, static_models, False, min_score=0.3)

            if prepare_type == PrepareType.MARK:
                mark_regions(predicted_regions, target_file_path, target_file_path, color=(255, 0, 0))
            elif prepare_type == PrepareType.MARK_WITH_BORDER:
                mark_regions_with_border(predicted_regions, target_file_path, target_file_path, color=(255, 0, 0))
            elif prepare_type == PrepareType.REMOVE:
                remove_regions(predicted_regions, target_file_path, target_file_path)
            else:
                raise ValueError(f"Type {prepare_type} is not supported.")

if __name__ == "__main__":
    main()