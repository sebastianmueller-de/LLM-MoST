import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from utils.ego_lanelet_utils import characterize_ego_lanelet
from collections import deque
import random

# For now is only concerned with traffic lights and signs
# Example return value: lane_dict = {"speed_limit": 13.888888889, "stop": 1, "traffic_light": 1}
def get_ego_lane_from_xml(xml_string: str) -> dict[str, float]:
    root = ET.fromstring(xml_string)
    ego_lane = root.find("data").find("egoLane")

    # Construct a dictionary of traffic signs
    lane_dict = {}
    traffic_signs = ego_lane.find("trafficSigns")
    traffic_lights = ego_lane.find("trafficLights")
    for element in traffic_signs:
        if element.tag == "speed_limit" or "required_speed":
            lane_dict[element.tag] = float(element.text)
        else:
            lane_dict[element.tag] = 1 # If a sign does not have an additional value, it is stored in the dict with a value of 1 (symbolizing its presence)
    if traffic_lights.find("trafficLight") is not None:
        lane_dict["traffic_light"] = 1

    return lane_dict

def get_obstacles_from_xml(xml_string: str) -> dict[str, int]:
    obstacles = {}
    root = ET.fromstring(xml_string)
    data = root.find("data")

    for obstacle in data.find("obstacles"):
        obs_type = obstacle.tag
        obs_number = obstacle.text
        obstacles[obs_type] = int(obs_number)

    print("Obstacles as extracted from the data object: ", obstacles)

    return obstacles

def get_velocity_from_xml(xml_string: str) -> float:
    root = ET.fromstring(xml_string)
    data = root.find("data")

    for velo in data.findall("initialVelocity"):
        print("Velocity as from the data object: ", float(velo.text))
        return float(velo.text)

    return -1.0

def location_from_name(filename:str):
    # Remove any prefix like C- and the file extension
    name = filename.replace("C-", "").replace(".xml", "")
    # Regex to match COUNTRY_SCENE_... pattern
    match = re.match(r"(?P<country_code>[A-Z]{3})_(?P<scene>[^-_]+)-\d+", name)
    if match:
        return {
            "country_code": match.group("country_code"),
            "specifier": match.group("scene")
        }
    else:
        return None  # Or raise an exception/log a warning

def add_metadata_to_xml(file_path_str: str) -> tuple[str, str, dict, dict]:
    file_path = Path(file_path_str)
    original_scenario = file_path.name

    if not original_scenario.endswith('.xml') or len(original_scenario) < 7:
        print(f"Invalid scenario file: {original_scenario}")
        return original_scenario, original_scenario, {}, {}

    pred_value = original_scenario[-7]
    if pred_value not in {"S", "T", "I"}:
        print(f"No dynamic obstacles in scenario: {original_scenario}")
        return original_scenario, original_scenario, {}, {}

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Location, treated as metadata for the chroma object
        location = location_from_name(original_scenario)
        if location is None:
            location = {}
        print("Location from file name: ", location)

        # Tags, treated as metadata for the chroma object
        scenario_tags = root.find("scenarioTags")
        #tags = [tag.tag for tag in scenario_tags] if scenario_tags is not None else []
        tags = {item.tag:1 for item in scenario_tags}
        print("Tags: ", tags)

        data = ET.Element("data")
        ET.SubElement(data, "pred").text = pred_value

        # Obstacles
        obstacles = {}
        for dynamic_obstacle in root.findall("dynamicObstacle"):
            obs_type = dynamic_obstacle.find("type").text
            if obs_type not in obstacles.keys():
                obstacles[obs_type] = 1
            else:
                obstacles[obs_type] += 1
        for static_obstacle in root.findall("staticObstacle"):
            obs_type = static_obstacle.find("type").text
            if obs_type not in obstacles.keys():
                obstacles[obs_type] = 0
            else:
                obstacles[obs_type] += 1
        print("All the obstacles within the XML file: ", obstacles)
        obs_element = ET.SubElement(data, "obstacles")
        for k, v in obstacles.items():
            ET.SubElement(obs_element, k).text = str(v)
        print("Obstacles element only:", ET.tostring(obs_element, encoding="unicode"))

        # Planning problem (we only allow scenarios with a planning problem in them)
        pp = root.find("planningProblem")
        if pp is None:
            print(f"No planning problem found in: {original_scenario}")
            return original_scenario, original_scenario, {}, {}
        # Initial velocity
        velocity_node = pp.find("initialState/velocity")
        if velocity_node is not None:
            value = velocity_node.find("exact").text
            ET.SubElement(data, "initialVelocity").text = value
            print(f"---\nScenario added with init velocity: {value}\n---")

        # Goal types
        goal_types_el = ET.SubElement(data, "goalTypes")
        for goal in pp.find("goalState") or []:
            ET.SubElement(goal_types_el, goal.tag)

        goals_count = len(pp.findall("goalState"))
        ET.SubElement(data, "goalsPerPlanningProblem").text = str(goals_count)

        # Traffic signs and lights in relation to ego lanelet
        ego_lane = characterize_ego_lanelet(ET.tostring(root, encoding="unicode"))
        data.append(ego_lane)

        # Insert metadata at top
        root.insert(0, data)
        print(ET.tostring(data, encoding="unicode"))
        return ET.tostring(root, encoding="unicode"), original_scenario, tags, location

    except ET.ParseError:
        os.remove(file_path)
        print(f"XML parse error — deleted: {file_path}")
        return "ERROR", file_path.name, {}, {}
    except Exception as e:
        print(f"Unexpected error in {file_path}: {e}")
        return "ERROR", file_path.name, {}, {}

# Takes in a string as a file path and returns an ElementTree
def load_xml_from_file_path(file_path: str):
    return ET.parse(file_path)

# Takes in an ElementTree and returns it as a string representation
def xml_tree_to_string(tree:ET.ElementTree):
    return ET.tostring(tree.getroot(), encoding="unicode")

import xml.etree.ElementTree as ET

def extract_and_validate_routes_xml(llm_output: str) -> str:
    """
    Extracts the first <routes>...</routes> section from the LLM output.
    If </think> is present, only considers sections after it.
    Validates that it's well-formed XML.

    Raises:
        ValueError: if <routes> section is not found.
        ET.ParseError: if the extracted XML is not valid.

    Returns:
        str: Cleaned and validated <routes> XML string.
    """
    think_end_tag = "</think>"
    start_tag = "<routes"
    end_tag = "</routes>"

    # Check if </think> exists
    think_end_idx = llm_output.find(think_end_tag)
    if think_end_idx != -1:
        # Search only after </think>
        search_start = think_end_idx + len(think_end_tag)
    else:
        # Search from beginning
        search_start = 0

    start_idx = llm_output.find(start_tag, search_start)
    end_idx = llm_output.find(end_tag, start_idx)

    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"<routes> section not found in LLM output:\n\n{llm_output}")

    # Include end tag in result
    end_idx += len(end_tag)
    routes_content = llm_output[start_idx:end_idx]

    # Try parsing to check validity
    try:
        ET.fromstring(routes_content)
    except ET.ParseError as e:
        raise ET.ParseError(
            f"Extracted <routes> section is not valid XML:\n\n{routes_content}\n\nError: {e}"
        )

    return routes_content

# Checks if a given scenario needs to have exit trajectories generated ("scenario repair"), before it can be modified run with Frenetix (the motion planner breaks, if disappearing vehicles are contained in the SUMO-converted files)
# Inputs are the paths to the SUMO .net.xml file and the .vehicles.rou.xml file (including the suffix)
def requires_trajectory_generation(net_file_path_str: str, routes_file_path_str: str) -> tuple[bool, list[str], list[str]]:
    net_tree = load_xml_from_file_path(net_file_path_str)
    route_tree = load_xml_from_file_path(routes_file_path_str)

    net_root = net_tree.getroot()
    route_root = route_tree.getroot()

    connected_from_edges = set()
    connected_to_edges = set()

    for connection in net_root.iter('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        if from_edge and not from_edge.startswith(":"):
            connected_from_edges.add(from_edge)
        if to_edge and not to_edge.startswith(":"):
            connected_to_edges.add(to_edge)

    exit_edges = connected_to_edges - connected_from_edges
    exit_edges_list = list(exit_edges)

    print(f"🧭 Identified {len(exit_edges)} exit edge(s) in the network.")
    if len(exit_edges) < 5:
        print(f"   👉 Exit edge IDs: {sorted(exit_edges)}")

    failing_vehicles = []

    for vehicle in route_root.iter('vehicle'):
        veh_id = vehicle.get('id', 'UNKNOWN_ID')
        route_elem = vehicle.find('route')

        if route_elem is None or not route_elem.get('edges'):
            print(f"⚠️ Vehicle '{veh_id}' has no usable <route> definition.")
            failing_vehicles.append(veh_id)
            continue

        edges = route_elem.get('edges').strip().split()
        if not edges:
            print(f"⚠️ Vehicle '{veh_id}' has no edges in its route.")
            failing_vehicles.append(veh_id)
            continue

        last_edge = edges[-1]
        if last_edge not in exit_edges:
            print(f"🚫 Vehicle '{veh_id}' ends on edge '{last_edge}', which is NOT an exit edge.")
            failing_vehicles.append(veh_id)

    if failing_vehicles:
        print(f"\n🚨 {len(failing_vehicles)} vehicle(s) do not end on a valid exit edge:")
        for vid in failing_vehicles:
            print(f"   • {vid}")
        print("\n❌ Route validation failed. These vehicles need trajectory extension.")
        return True, failing_vehicles, exit_edges_list

    print("✅ All vehicles end on valid exit edges. No repair needed.")
    return False, [], exit_edges_list

def generate_exit_trajectories_without_llm(net_file_path_str: str, routes_file_path_str: str, failing_vehicles: list[str], exit_edges: list[str]):
    net_tree = load_xml_from_file_path(net_file_path_str)
    route_tree = load_xml_from_file_path(routes_file_path_str)

    net_root = net_tree.getroot()
    route_root = route_tree.getroot()

    # Step 1: Build the graph of edge connectivity
    edge_graph = {}
    for connection in net_root.iter('connection'):
        from_edge = connection.get('from')
        to_edge = connection.get('to')
        if from_edge and to_edge and not from_edge.startswith(":") and not to_edge.startswith(":"):
            edge_graph.setdefault(from_edge, set()).add(to_edge)

    exit_edge_set = set(exit_edges)
    modified_count = 0

    # Step 2: For each failing vehicle, extend the route
    for vehicle in route_root.iter('vehicle'):
        veh_id = vehicle.get('id', 'UNKNOWN_ID')
        if veh_id not in failing_vehicles:
            continue

        route_elem = vehicle.find('route')
        if route_elem is None or not route_elem.get('edges'):
            print(f"⚠️ Skipping vehicle '{veh_id}' due to missing or empty route.")
            continue

        edges = route_elem.get('edges').strip().split()
        if not edges:
            continue

        last_edge = edges[-1]

        # Step 3: Find a path from last_edge to an exit edge using BFS
        path = bfs_find_exit_path(edge_graph, last_edge, exit_edge_set)

        if path:
            edges.extend(path[1:])  # Avoid repeating the last_edge
            route_elem.set('edges', ' '.join(edges))
            print(f"🛠️ Extended route for vehicle '{veh_id}' by {len(path) - 1} edges ➝ exits at '{path[-1]}'.")
            modified_count += 1
        else:
            print(f"🚨 No exit path found from edge '{last_edge}' for vehicle '{veh_id}'.")

    print(f"\n✅ Route repair completed. {modified_count} vehicle(s) had their routes extended.")

    route_tree.write(routes_file_path_str, encoding='utf-8', xml_declaration=True)

    return route_tree


def bfs_find_exit_path(edge_graph: dict[str, set[str]], start_edge: str, exit_edges: set[str]) -> list[str] | None:
    visited = set()
    queue = deque([[start_edge]])

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current in exit_edges:
            return path

        if current in visited:
            continue

        visited.add(current)

        for neighbor in edge_graph.get(current, []):
            if neighbor not in visited:
                queue.append(path + [neighbor])

    return None


def update_benchmark_id(file_path_str: str, new_id: str) -> str:
    # Parse the XML file
    tree = ET.parse(file_path_str)
    root = tree.getroot()

    # Update the benchmarkID attribute (usually on the root <commonRoad> tag)
    if 'benchmarkID' in root.attrib:
        old_id = root.attrib['benchmarkID']
        root.attrib['benchmarkID'] = new_id
        print(f"Updated benchmarkID from {old_id} to {new_id}")
    else:
        print("No benchmarkID found in root element. Adding new one.")
        root.attrib['benchmarkID'] = new_id

    # Write the updated XML back to the file
    tree.write(file_path_str, encoding='UTF-8', xml_declaration=True)

    return file_path_str

# Loads in an XML file, adds the desired traffic lights, and returns a string ready to be written to a file
# Expects the signs to come in as a list of ids

def add_traffic_lights(file_path_str: str, target_path_str: str, new_traffic_lights: list[str]) -> str:
    # Load XML
    tree = ET.parse(file_path_str)
    root = tree.getroot()

    # Basic traffic light template (with placeholder ID)
    basic_light_config = """
    <trafficLight id="TEMPLATE_ID">
        <cycle>
            <cycleElement>
                <duration>400</duration>
                <color>green</color>
            </cycleElement>
            <cycleElement>
                <duration>30</duration>
                <color>yellow</color>
            </cycleElement>
            <cycleElement>
                <duration>570</duration>
                <color>red</color>
            </cycleElement>
        </cycle>
        <direction>all</direction>
        <active>true</active>
    </trafficLight>
    """

    # Loop through all given lanelet IDs
    for lanelet_id in new_traffic_lights:
        # Find lanelet element by attribute id
        lanelet = root.find(f".//lanelet[@id='{lanelet_id}']")
        if lanelet is None:
            continue  # Skip if not found

        # Create a unique 5-digit ID for the traffic light
        light_id = str(random.randint(10000, 99999))

        # Add trafficLightRef inside the lanelet
        trafficLightRef = ET.SubElement(lanelet, "trafficLightRef")
        trafficLightRef.set("ref", light_id)

        # Create trafficLight element at the root level
        trafficLight_xml = basic_light_config.replace("TEMPLATE_ID", light_id)
        trafficLight_element = ET.fromstring(trafficLight_xml)

        # Append to root
        root.append(trafficLight_element)

    # Save new file
    tree.write(target_path_str, encoding='UTF-8', xml_declaration=True)

    return ET.tostring(root, encoding='unicode')

#base_net_file = "/home/avsaw1/sebastian/ChaBot7/Scenarios/DEU_Damme-17_1_T-1/Original/DEU_Damme-17_1_T-1.net.xml"
#base_route_file = "/home/avsaw1/sebastian/ChaBot7/Scenarios/DEU_Damme-17_1_T-1/Original/DEU_Damme-17_1_T-1.vehicles.rou.xml"
#requires_repair, broken_vehicles, exit_edges = requires_trajectory_generation(base_net_file, base_route_file)
#generate_exit_trajectories_without_llm(base_net_file, base_route_file, broken_vehicles, exit_edges)

