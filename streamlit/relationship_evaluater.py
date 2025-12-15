import math
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from skimage.morphology import skeletonize

import config

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
        return Polygon()  # return empty polygon if all fails

def fix_small_overlaps(boxes):
    boxes = boxes.copy()
    n = len(boxes)

    for i in range(n):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = boxes[j]

            # Compute box bounds
            xi2, yi2 = xi + wi, yi + hi
            xj2, yj2 = xj + wj, yj + hj

            # Compute overlap
            ox1 = max(xi, xj)
            oy1 = max(yi, yj)
            ox2 = min(xi2, xj2)
            oy2 = min(yi2, yj2)

            if ox1 < ox2 and oy1 < oy2:
                ow = ox2 - ox1
                oh = oy2 - oy1

                if oh < ow:
                    # Vertical overlap: split above/below
                    cut = oh / 2
                    if yi < yj:
                        # i is above
                        hi = max(1, hi - cut)
                        yj = yj + cut
                        hj = max(1, hj - cut)
                    else:
                        # j is above
                        hj = max(1, hj - cut)
                        yi = yi + cut
                        hi = max(1, hi - cut)
                else:
                    # Horizontal overlap: split left/right
                    cut = ow / 2
                    if xi < xj:
                        # i is left of j
                        wi = max(1, wi - cut)
                        xj = xj + cut
                        wj = max(1, wj - cut)
                    else:
                        # j is left of i
                        wj = max(1, wj - cut)
                        xi = xi + cut
                        wi = max(1, wi - cut)

                # Update boxes
                boxes[i] = (xi, yi, wi, hi)
                boxes[j] = (xj, yj, wj, hj)

    return boxes

def is_crossing(node):
    return node.startswith("cross_") and node.endswith("_center")

def is_crossing_connection(node):
    return node.startswith("cross_") and not node.endswith("_center")

def is_endpoint(node):
    return node.startswith("ep_")

def skeletonize_image(image, scale_factor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)  # make lines white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(2 * scale_factor), int(2 * scale_factor)))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    skeleton = skeletonize(closed // 255)  # normalize to [0, 1]
    return skeleton.astype(np.uint8)

def walk_skeleton(skeleton, start_points, stop_conditions, forbidden=None):
    visited = set()
    stack = list(start_points)
    parent = {}  # maps child -> parent

    while stack:
        current = stack.pop(0)
        if current in visited or current == forbidden:
            continue
        visited.add(current)

        stop_result = stop_conditions(current)
        if stop_result is not None:
            # Reconstruct the true path (from start to current)
            true_path = [current]
            while current in parent:
                current = parent[current]
                true_path.append(current)
            true_path.reverse()
            return stop_result, true_path

        y, x = current
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)
                if (
                    0 <= ny < skeleton.shape[0]
                    and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] == 1
                    and neighbor not in visited
                    and neighbor != forbidden
                ):
                    stack.append(neighbor)
                    parent[neighbor] = current

    return None, []

def skeleton_to_graph(skeleton, endpoints, crossings):
    import networkx as nx
    import numpy as np

    G = nx.Graph()
    edge_paths = {}
    skeleton_points = np.argwhere(skeleton == 1)

    connection_points = {}
    crossing_centers = {}
    border_to_center = {}
    outside_neighbors = {}
    endpoint_index_to_node = {}

    crossings = fix_small_overlaps(crossings)

    # Step 1: Add crossing centers and border points, collect outside neighbors
    for i, (x, y, w, h) in enumerate(crossings):
        x, y, w, h = map(int, (x, y, w, h))
        center = (x + w / 2, y + h / 2)
        center_node = f"cross_{i}_center"
        G.add_node(center_node, pos=center)
        crossing_centers[center_node] = np.array(center)

        sub_skeleton = skeleton[y:y + h, x:x + w]
        ys, xs = np.where(sub_skeleton == 1)

        for j, (sx, sy) in enumerate(zip(xs, ys)):
            if sx == 0 or sx == w - 1 or sy == 0 or sy == h - 1:
                abs_x, abs_y = x + sx, y + sy
                border_node = f"cross_{i}_pt_{j}"
                G.add_node(border_node, pos=(abs_x, abs_y))
                G.add_edge(border_node, center_node)
                edge_paths[(border_node, center_node)] = [(abs_x, abs_y), center]

                key = (abs_y, abs_x)
                connection_points[key] = border_node
                border_to_center[border_node] = center_node

                # collect outside neighbors
                neighbors = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx_, ny_ = abs_x + dx, abs_y + dy
                        if (
                            0 <= ny_ < skeleton.shape[0]
                            and 0 <= nx_ < skeleton.shape[1]
                            and skeleton[ny_, nx_] == 1
                            and (nx_ < x or nx_ >= x + w or ny_ < y or ny_ >= y + h)
                        ):
                            neighbors.append((ny_, nx_))  # (y, x)
                outside_neighbors[key] = neighbors

    # Step 2: Map endpoints to skeleton
    endpoint_pixels = {}
    for i, ep in enumerate(endpoints):
        ep = np.array(ep)
        ep_yx = tuple(ep[::-1])
        distances = np.linalg.norm(skeleton_points - ep_yx, axis=1)
        closest_index = np.argmin(distances)
        closest_point = tuple(skeleton_points[closest_index])
        closest_distance = distances[closest_index]

        endpoint_node = f"ep_{i}"
        G.add_node(endpoint_node, pos=tuple(ep))
        endpoint_pixels[closest_point] = endpoint_node
        endpoint_index_to_node[i] = endpoint_node

        connection_points[closest_point] = endpoint_node

        print(f"Endpoint {tuple(ep)} → Closest skeleton pixel {closest_point[::-1]} | Distance: {closest_distance:.2f}")

    for start_pt, start_node in connection_points.items():
        origin_id = start_node

        def stop_condition_crossing(pt):
            if pt in endpoint_pixels and endpoint_pixels[pt] != origin_id:
                return endpoint_pixels[pt]
            elif pt in connection_points:
                stop_node = connection_points[pt]
                if stop_node != origin_id and border_to_center[stop_node] != border_to_center[origin_id]:
                    return stop_node
            return None

        def stop_condition_endpoint(pt):
            if pt in endpoint_pixels and endpoint_pixels[pt] != origin_id:
                return endpoint_pixels[pt]
            elif pt in connection_points:
                stop_node = connection_points[pt]
                if stop_node != origin_id:
                    return stop_node
            return None

        if is_crossing_connection(start_node):
            result, path = walk_skeleton(
                skeleton,
                start_points=outside_neighbors.get(start_pt, []),
                stop_conditions=stop_condition_crossing,
                forbidden=start_pt
            )
        elif is_endpoint(start_node):
            result, path = walk_skeleton(
                skeleton,
                start_points=[start_pt],
                stop_conditions=stop_condition_endpoint
            )
        else:
            raise ValueError("Point is not an endpoint and not a crossing")

        if result:
            converted_path = [(x, y) for y, x in path]

            if not G.has_edge(result, start_node) or converted_path < edge_paths[(result, start_node)]:
                if G.has_edge(result, start_node):
                    G.remove_edge(result, start_node)
                    del edge_paths[(result, start_node)]
                G.add_edge(start_node, result)
                edge_paths[(start_node, result)] = converted_path
                print(f"Connected {start_node} to {result} via {len(path)} steps")

        else:
            print(f"{start_node} leads to deadend")

    return G, edge_paths, endpoint_index_to_node

def predict_connections_from_skeleton(image, endpoints, crossings, class_boxes, end_shapes, scale_factor):
    skeleton = skeletonize_image(image, scale_factor)
    G, edge_paths, endpoint_index_to_node = skeleton_to_graph(skeleton, endpoints, crossings)
    visualize_graph(G, skeleton, class_boxes=class_boxes, end_shapes=end_shapes)

    def direction(a, b):
        a = np.array(a)
        b = np.array(b)
        v = b - a
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else np.zeros_like(v)

    def angle_between(v1, v2):
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(dot)

    connection_map = {}

    def dfs(start_idx, current, path, visited, deviations):
        if is_endpoint(current) and current != path[0]:
            end_pos = G.nodes[current]['pos']
            total_div = round(sum(deviations), 2)

            # Construct full skeleton path
            full_path = []
            for i in range(len(path) - 1):
                if (path[i], path[i + 1]) in edge_paths:
                    full_path.extend(edge_paths[(path[i], path[i + 1])])
                elif (path[i + 1], path[i]) in edge_paths:
                    full_path.extend(edge_paths[(path[i + 1], path[i])][::-1])

            print(f"Reached endpoint at ({float(end_pos[0]):.2f}, {float(end_pos[1]):.2f}) with deviations: {[float(round(d, 2)) for d in deviations]}")
            connection_map[start_idx].append({
                "endpoint_index": int(current[len("ep_"):]),
                "total_div": total_div,
                "path": full_path
            })
            return

        visited.add(current)

        for neighbor in G.neighbors(current):
            if neighbor in visited:
                continue

            new_divs = deviations.copy()

            if len(path) >= 2 and is_crossing(current):
                incoming_pos = G.nodes[path[-2]]['pos']
                center_pos = G.nodes[current]['pos']
                outgoing_pos = G.nodes[neighbor]['pos']

                incoming_dir = direction(incoming_pos, center_pos)
                outgoing_dir = direction(center_pos, outgoing_pos)

                if np.linalg.norm(incoming_dir) > 0 and np.linalg.norm(outgoing_dir) > 0:
                    angle = angle_between(incoming_dir, outgoing_dir)
                    angle_degrees = np.degrees(angle)
                    new_divs.append(angle_degrees)

            dfs(start_idx, neighbor, path + [neighbor], visited.copy(), new_divs)

    for idx, start_node in endpoint_index_to_node.items():
        x, y = G.nodes[start_node]['pos']
        print(f"\nTracing from endpoint ({float(x)}, {float(y)})...")
        connection_map[idx] = []
        dfs(idx, start_node, [start_node], set(), [])

    # Normalize deviations to probabilities
    for idx, connections in connection_map.items():
        if not connections:
            continue
        inverse_scores = [1 / (connection["total_div"] + 1e-6) ** 2 for connection in connections]
        total = sum(inverse_scores)
        for i in range(len(connections)):
            probability = (inverse_scores[i] / total) * 100
            connections[i]["probability"] = probability
        connections.sort(key=lambda c: -c["probability"])  # sort by probability descending

    return connection_map

def visualize_graph(graph, skeleton_image, class_boxes=None, end_shapes=None,node_color=(0, 0, 255), edge_color=(0, 255, 0)):
    vis_img = skeleton_image.copy()
    if vis_img.max() <= 1:
        vis_img = (vis_img * 255).astype(np.uint8)

    if vis_img.ndim == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    # Draw class boxes (green transparent)
    if class_boxes:
        for x, y, w, h in class_boxes:
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), -1)
            vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)

    # Draw end shapes (orange transparent) and label
    if end_shapes:
        for x, y, w, h, label, score in end_shapes:
            overlay = vis_img.copy()
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (0, 165, 255), -1)
            vis_img = cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0)

            cx, cy = int(x + w / 2), int(y + h / 2)

            text = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            margin = 2  # pixels

            # Try different font scales until it fits
            max_width = w - 2 * margin
            max_height = h - 2 * margin

            font_scale = 1.0
            while font_scale > 0.25:
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                if text_width <= max_width and text_height <= max_height:
                    break
                font_scale -= 0.05  # Reduce and try again

            # Center the text
            text_x = cx - text_width // 2
            text_y = cy + text_height // 2

            cv2.putText(vis_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # Draw edges
    for u, v, data in graph.edges(data=True):
        pos_u = graph.nodes[u].get('pos')
        pos_v = graph.nodes[v].get('pos')
        if pos_u is None or pos_v is None:
            continue
        x1, y1 = map(int, pos_u)
        x2, y2 = map(int, pos_v)
        cv2.line(vis_img, (x1, y1), (x2, y2), edge_color, 1)

    # Draw nodes
    for node, attr in graph.nodes(data=True):
        pos = attr.get('pos')
        if pos is not None:
            x, y = map(int, pos)
            if 0 <= y < vis_img.shape[0] and 0 <= x < vis_img.shape[1]:
                vis_img[y, x] = node_color  # Set the pixel

    output_path = config.PROCESSED_DIR + "/graph_skeleton.png"
    cv2.imwrite(output_path, vis_img)

def point_to_box_distance(px, py, box):
    x_min = box[0]
    y_min = box[1]
    x_max = x_min + box[2]
    y_max = y_min + box[3]

    # Compute dx: 0 if point inside horizontally, else distance to closest edge
    dx = max(x_min - px, 0, px - x_max)
    dy = max(y_min - py, 0, py - y_max)

    return (dx**2 + dy**2) ** 0.5

def point_to_point_distance(px, py, ox, oy):
    return ((px - ox)**2 + (py - oy)**2)**0.5

def filter_endpoints(class_boxes, endpoints: List[Tuple[float, float]], scale_factor: float):
    endpoints = endpoints.copy()
    i = 0

    # Remove duplicate detections of the same endpoint
    i = 0
    while i < len(endpoints):
        px, py = endpoints[i]
        j = i + 1
        while j < len(endpoints):
            qx, qy = endpoints[j]
            dist = math.hypot(px - qx, py - qy)
            min_differentiation_distance = round(5 * scale_factor, 3)
            if dist < min_differentiation_distance:
                dist_i = min(point_to_box_distance(px, py, box) for box in class_boxes)
                dist_j = min(point_to_box_distance(qx, qy, box) for box in class_boxes)
                # Keep the one closer to a class
                if dist_i <= dist_j:
                    removed = endpoints[j]
                    kept = endpoints[i]
                    endpoints.pop(j)
                    print(
                        f"Ignoring endpoint {removed}, because it is too close (distance < {min_differentiation_distance} px) to endpoint {kept} which is closer to a class. Distance is {dist:.3f} px.")
                else:
                    removed = endpoints[i]
                    kept = endpoints[j]
                    endpoints.pop(i)
                    print(
                        f"Ignoring endpoint {removed}, because it is too close (distance < {min_differentiation_distance} px) to endpoint {kept} which is closer to a class. Distance is {dist:.3f} px.")
                    i -= 1  # step back, since current was removed
                    break  # restart inner loop from new i
            else:
                j += 1
        i += 1

    # Apply remaining filters
    filtered = []
    belonging_class = []
    endpoint_gap_connections = set()
    original_to_filtered_index = {}  # maps index in `endpoints` to index in `filtered`

    for i, point in enumerate(endpoints):
        px, py = point
        distances_to_classes = [point_to_box_distance(px, py, box) for box in class_boxes]
        min_class_dist = min(distances_to_classes)
        closest_class_index = distances_to_classes.index(min_class_dist) if distances_to_classes else None

        other_points = endpoints[:i] + endpoints[i + 1:]
        distances_to_points = [point_to_point_distance(px, py, ox, oy) for ox, oy in other_points]
        min_endpoint_dist = min(distances_to_points)
        closest_in_others = distances_to_points.index(min_endpoint_dist) if distances_to_points else None

        # Convert closest_in_others back to global index in `endpoints`
        if closest_in_others is not None:
            if closest_in_others >= i:
                closest_endpoint_index_in_endpoints = closest_in_others + 1
            else:
                closest_endpoint_index_in_endpoints = closest_in_others
        else:
            continue

        max_distance_class = round(10 * scale_factor, 3)
        max_distance_other_point = round(15 * scale_factor, 3)

        if min_class_dist < max_distance_class:
            if i not in original_to_filtered_index:
                filtered_index = len(filtered)
                filtered.append(point)
                belonging_class.append(closest_class_index)
                original_to_filtered_index[i] = filtered_index
            else:
                # overwrite class
                filtered_index = original_to_filtered_index[i]
                belonging_class[filtered_index] = closest_class_index
            continue

        if min_endpoint_dist < max_distance_other_point:
            if i not in original_to_filtered_index:
                # Add current point
                filtered_index = len(filtered)
                filtered.append(point)
                belonging_class.append(None)
                original_to_filtered_index[i] = filtered_index

            if closest_endpoint_index_in_endpoints not in original_to_filtered_index:
                # Add the closest point
                filtered_index = len(filtered)
                filtered.append(endpoints[closest_endpoint_index_in_endpoints])
                belonging_class.append(None)
                original_to_filtered_index[closest_endpoint_index_in_endpoints] = filtered_index

            # Map both endpoints to filtered space
            filtered_index = original_to_filtered_index[i]
            other_filtered_index = original_to_filtered_index[closest_endpoint_index_in_endpoints]

            if (filtered_index, other_filtered_index) not in endpoint_gap_connections and (other_filtered_index, filtered_index) not in endpoint_gap_connections:
                endpoint_gap_connections.add((filtered_index, other_filtered_index))
            continue

        print(f"Ignoring endpoint {point}, because it is not close to a class (distance < {max_distance_class} px) or another endpoint (distance < {max_distance_other_point} px). Closest class is {min_class_dist} px and closest endpoint is {min_endpoint_dist} px away.")
    return filtered, belonging_class, endpoint_gap_connections

def compute_end_shape_mapping(endpoints: List[Tuple[float, float]], end_shapes: List[List], scale_factor: float) -> List[Tuple[str, float]]:
    endpoint_labels = {pt: ("Empty End", 1.0) for pt in endpoints}

    for shape in end_shapes:
        label = shape[4]
        score = shape[5]

        # Find the closest point to the shape
        distances = [(pt, point_to_box_distance(pt[0], pt[1], shape)) for pt in endpoints]
        closest_point, min_dist = min(distances, key=lambda x: x[1])

        max_distance = round(10 * scale_factor, 3)
        if min_dist > max_distance:
            x_center = shape[0] + shape[2] / 2
            y_center = shape[1] + shape[3] / 2
            print(f"Ignoring end shape at {(x_center, y_center)} with label \"{label}\" because it is not close (distance < {max_distance} px) to an endpoint. Closest endpoint is {min_dist} px away.")
            continue  # ignore this shape

        max_dist = 1.2 * min_dist
        for pt, dist in distances:
            if dist <= max_dist:
                endpoint_labels[pt] = (label, score)

    return [endpoint_labels[pt] for pt in endpoints]

def find_connection(endpoint, class_connections):
    return next((((a, b), 1) if a == endpoint else ((a, b), 0) for a, b in class_connections.keys() if endpoint in (a, b)), (None, None))

def evaluate(relationship_image, class_boxes, endpoints, crossings, end_shapes, scale_factor: float):
    class_connections = {}
    ignored_connections = []

    min_path_length = round(30 * scale_factor, 3)

    if len(endpoints) > 2:
        endpoints, belonging_classes, endpoint_gap_connections = filter_endpoints(class_boxes, endpoints, scale_factor)
        end_shapes_mapping = compute_end_shape_mapping(endpoints, end_shapes, scale_factor)
        connections_map = predict_connections_from_skeleton(relationship_image, endpoints, crossings, class_boxes, end_shapes, scale_factor)

        for endpoint_a_index, connections in connections_map.items():
            endpoint_a = endpoints[endpoint_a_index]
            end_shape_a = end_shapes_mapping[endpoint_a_index]
            class_a_index = belonging_classes[endpoint_a_index]

            if len(connections) == 0:
                print(f"Ignoring endpoint {endpoint_a}, because no connection to another endpoint was found.")
                continue

            current_best_index = 0

            while True:
                if current_best_index == len(connections):
                    print(f"Ignoring endpoint {endpoint_a}, because no 2-class connection and no long enough same-class connection to another endpoint was found.")
                    best_connection = None
                    break

                best_connection = connections[current_best_index]
                endpoint_b_index = best_connection["endpoint_index"]
                path = best_connection["path"]
                endpoint_b = endpoints[endpoint_b_index]
                end_shape_b = end_shapes_mapping[endpoint_b_index]
                class_b_index = belonging_classes[endpoint_b_index]

                if class_a_index != class_b_index or len(path) > min_path_length:
                    break

                # only print once for each ignored connection
                if (endpoint_a, endpoint_b) not in ignored_connections and (endpoint_b, endpoint_a) not in ignored_connections:
                    print(f"Ignoring same class connection between {endpoint_a} and {endpoint_b}, because path is too short (length < {min_path_length} px). Path is {len(path)} px long.")
                    ignored_connections.append((endpoint_a, endpoint_b))

                current_best_index += 1

            if best_connection is None:
                continue

            if (endpoint_a, endpoint_b) in class_connections or (endpoint_b, endpoint_a) in class_connections:
                continue

            class_connections[(endpoint_a, endpoint_b)] = {
                "class_index_0": class_a_index,
                "class_index_1": class_b_index,
                "end_shape_0": end_shape_a,
                "end_shape_1": end_shape_b,
                "path": path
            }

        # Initial mapping of endpoint indices to connection pairs
        endpoint_to_pairs = defaultdict(set)
        for endpoint_pair, _ in class_connections.items():
            endpoint_a, endpoint_b = endpoint_pair
            endpoint_a_index = endpoints.index(endpoint_a)
            endpoint_b_index = endpoints.index(endpoint_b)

            endpoint_to_pairs[endpoint_a_index].add(endpoint_pair)
            endpoint_to_pairs[endpoint_b_index].add(endpoint_pair)

        # Keep restarting until no changes are made
        while True:
            restart_needed = False

            for endpoint_index, pairs in list(endpoint_to_pairs.items()):
                if len(pairs) == 2:
                    endpoint = endpoints[endpoint_index]
                    this_class_index = belonging_classes[endpoint_index]

                    pair_a, pair_b = pairs

                    if pair_a not in class_connections or pair_b not in class_connections:
                        restart_needed = True
                        break  # Exit inner loop to restart

                    other_index_a = 1 if endpoint == pair_a[0] else 0
                    other_endpoint_a = pair_a[other_index_a]
                    connection_parameters_a = class_connections[pair_a]
                    other_class_index_a = connection_parameters_a[f"class_index_{other_index_a}"]
                    other_end_shape_a = connection_parameters_a[f"end_shape_{other_index_a}"]
                    path_a = connection_parameters_a["path"] if other_index_a == 1 else connection_parameters_a["path"][::-1]

                    other_index_b = 1 if endpoint == pair_b[0] else 0
                    other_endpoint_b = pair_b[other_index_b]
                    connection_parameters_b = class_connections[pair_b]
                    other_class_index_b = connection_parameters_b[f"class_index_{other_index_b}"]
                    other_end_shape_b = connection_parameters_b[f"end_shape_{other_index_b}"]
                    path_b = connection_parameters_b["path"] if other_index_b == 1 else connection_parameters_b["path"][::-1]

                    if this_class_index == other_class_index_a or this_class_index == other_class_index_b:
                        print(f"Removing endpoint {endpoint} because it is a duplicate class connection point (center).")
                        del class_connections[pair_a]
                        del class_connections[pair_b]

                        if this_class_index == other_class_index_a and other_end_shape_a[0] == "Empty End" and end_shapes_mapping[endpoint_index][0] != "Empty End":
                            # take end shape of gap (most of the time they are very close)
                            other_end_shape_a = end_shapes_mapping[this_class_index]
                            print(f"Transferring end shape {other_end_shape_a} from removed endpoint to new connection.")
                        elif this_class_index == other_class_index_b and other_end_shape_b[0] == "Empty End" and end_shapes_mapping[endpoint_index][0] != "Empty End":
                            # take end shape of gap (most of the time they are very close)
                            other_end_shape_b = end_shapes_mapping[this_class_index]
                            print(f"Transferring end shape {other_end_shape_b} from removed endpoint to new connection.")

                        class_connections[(other_endpoint_a, other_endpoint_b)] = {
                            "class_index_0": other_class_index_a,
                            "class_index_1": other_class_index_b,
                            "end_shape_0": other_end_shape_a,
                            "end_shape_1": other_end_shape_b,
                            "path": path_a[::-1] + path_b
                        }
                    elif other_class_index_a == other_class_index_b:
                        print(f"Removing endpoint {other_endpoint_b} because it is a duplicate class connection point (edge).")
                        del class_connections[pair_b]
                        if other_end_shape_a[0] == "Empty End" and other_end_shape_b[0] != "Empty End":
                            # take end shape of gap (most of the time they are very close)
                            connection_parameters_a[f"end_shape_{other_index_a}"] = other_end_shape_b
                            print(f"Transferring end shape {other_end_shape_b} from removed endpoint to new connection.")

            if restart_needed:
                # Rebuild endpoint_to_pairs from scratch
                endpoint_to_pairs = defaultdict(set)
                for endpoint_pair, _ in class_connections.items():
                    endpoint_a, endpoint_b = endpoint_pair
                    endpoint_a_index = endpoints.index(endpoint_a)
                    endpoint_b_index = endpoints.index(endpoint_b)

                    endpoint_to_pairs[endpoint_a_index].add(endpoint_pair)
                    endpoint_to_pairs[endpoint_b_index].add(endpoint_pair)
                continue  # Restart main while-loop

            break  # Exit loop if no changes

        for endpoint_gap_a_index, endpoint_gap_b_index in endpoint_gap_connections:
            pair_a, other_index_a = find_connection(endpoints[endpoint_gap_a_index], class_connections)
            pair_b, other_index_b = find_connection(endpoints[endpoint_gap_b_index], class_connections)

            if pair_a == pair_b:
                continue

            print(f"Removing endpoints {endpoints[endpoint_gap_a_index]} and {endpoints[endpoint_gap_b_index]} because they are a gap connection.")

            if pair_a is not None:
                # Collect values of other_endpoint_a
                other_endpoint_a = pair_a[other_index_a]
                connection_parameters_a = class_connections[pair_a]
                other_class_index_a = connection_parameters_a[f"class_index_{other_index_a}"]
                other_end_shape_a = connection_parameters_a[f"end_shape_{other_index_a}"]
                if other_end_shape_a[0] == "Empty End" and end_shapes_mapping[endpoint_gap_a_index][0] != "Empty End":
                    # take end shape of gap (most of the time they are very close)
                    other_end_shape_a = end_shapes_mapping[endpoint_gap_a_index]
                    print(f"Transferring end shape {other_end_shape_a} from removed endpoint to new connection.")
                path_a = connection_parameters_a["path"] if other_index_a == 0 else connection_parameters_a["path"][::-1]
                del class_connections[pair_a]
            else:
                # Use endpoint_a as other_endpoint_a
                other_endpoint_a = endpoints[endpoint_gap_a_index]
                other_class_index_a = belonging_classes[endpoint_gap_a_index]
                other_end_shape_a = end_shapes_mapping[endpoint_gap_a_index]
                path_a = [other_endpoint_a]

            if pair_b is not None:
                # Collect values of other_endpoint_b
                other_endpoint_b = pair_b[other_index_b]
                connection_parameters_b = class_connections[pair_b]
                other_class_index_b = connection_parameters_b[f"class_index_{other_index_b}"]
                other_end_shape_b = connection_parameters_b[f"end_shape_{other_index_b}"]
                if other_end_shape_b[0] == "Empty End" and end_shapes_mapping[endpoint_gap_b_index][0] != "Empty End":
                    # take end shape of gap (most of the time they are very close)
                    other_end_shape_b = end_shapes_mapping[endpoint_gap_b_index]
                    print(f"Transferring end shape {other_end_shape_b} from removed endpoint to new connection.")
                path_b = connection_parameters_b["path"] if other_index_b == 1 else connection_parameters_b["path"][::-1]
                del class_connections[pair_b]
            else:
                # Use endpoint_b as other_endpoint_b
                other_endpoint_b = endpoints[endpoint_gap_b_index]
                other_class_index_b = belonging_classes[endpoint_gap_b_index]
                other_end_shape_b = end_shapes_mapping[endpoint_gap_b_index]
                path_b = [other_endpoint_b]

            class_connections[(other_endpoint_a, other_endpoint_b)] = {
                "class_index_0": other_class_index_a,
                "class_index_1": other_class_index_b,
                "end_shape_0": other_end_shape_a,
                "end_shape_1": other_end_shape_b,
                "path": path_a + path_b
            }

        # filter out every connection that is a non-class connection, is a self connection
        class_connections = {
            (endpoint_a, endpoint_b): value
            for (endpoint_a, endpoint_b), value in class_connections.items()
            if value["class_index_0"] is not None and value["class_index_1"] is not None # filter out non-class connections
               and endpoint_a != endpoint_b # filter out self connections
               and not (value["class_index_0"] == value["class_index_1"] and len(value["path"]) < min_path_length)
        }

    return class_connections