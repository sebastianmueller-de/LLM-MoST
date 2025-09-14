# This code contains helper functions for dealing with the road network in relation to the ego vehicle's position

from pathlib import Path
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point

# TODO: Fix turning priorities nomenclature if desired
traffic_sign_dict = {
    "102": "right_before_left_rule",
    "205": "yield",
    "206": "stop",
    "301": "right_of_way",
    "306": "priority_road",
    # Turning priority roads are currently classified into three categories based on the direction for which the ego lane has priority; this is enough for the DB search (the internal logic corresponding to the size of the intersection remains) --> but is it enough for modification?
    "1002-10": "turning_priority_left",
    "1002-11": "turning_priority_wait",
    "1002-12": "turning_priority_left",
    "1002-13": "turning_priority_left",
    "1002-14": "turning_priority_wait",
    "1002-20": "turning_priority_right",
    "1002-21": "turning_priority_wait",
    "1002-22": "turning_priority_right",
    "1002-23": "turning_priority_right",
    "1002-24": "turning_priority_wait",
    "274": "speed_limit",
    "275": "required_speed",
    "276": "no_overtaking",
    "720": "green_arrow_sign",
    "310": "town_sign",
    "260": "ban_on_motorcycles_and_multi_lane_vehicles",
    "272": "no_u_turn"
}

# Simple function to load in an XML file and transform it into a string
def load_xml(file_path_str:str):
    file_path = Path(file_path_str)
    xml_str = file_path.read_text()
    return xml_str

def lanelets_to_polygons(xml_string:str):
    root = ET.fromstring(xml_string)
    polygon_dict = dict()

    for lanelet in root.findall('lanelet'):
        lanelet_id = lanelet.attrib['id']

        left_bound_points = []
        right_bound_points = []

        for element in lanelet.findall('leftBound'):
            for point in element.findall('point'):
                left_bound_points.append([float(point.find("x").text), float(point.find("y").text)])

        for element in lanelet.findall('rightBound'):
            for point in element.findall('point'):
                right_bound_points.append([float(point.find("x").text), float(point.find("y").text)])

        if not left_bound_points or not right_bound_points:
            continue  # Skip this lanelet if any boundary is missing

        # Reverse the right bound to ensure the correct clockwise or counter-clockwise order
        right_bound_points.reverse()

        # Combine leftBound and reversed rightBound to form a loop
        polygon_points = left_bound_points + right_bound_points

        # Create a Shapely Polygon from the points
        try:
            polygon = Polygon(polygon_points)
            # Ensure the polygon is valid (may raise ValueError for incomplete or degenerate shapes)
            if not polygon.is_valid:
                raise ValueError(f"Polygon is invalid for lanelet {lanelet_id}")

            # Add the polygon to the dictionary with the corresponding ID
            polygon_dict[lanelet_id] = polygon
        except (ValueError, Exception) as e:
            print(f"Error creating a polygon for lanelet {lanelet_id}: {e}")

    # Return the dictionary of lanelet ID to Polygon mappings
    return polygon_dict

# Function to check lanelets for a single point
def check_point_in_polygons(x, y, polygons):

    current_point = Point(x, y)
    for polygon_id, polygon in polygons.items():
        if polygon.contains(current_point):
            return polygon_id
    return None

# Uses Polygon approach to determine the lanelet belonging to the ego
def find_ego_lanelet(xml_string:str):
    root = ET.fromstring(xml_string)

    ego_point = [
        float(root.find("planningProblem").find("initialState").find("position").find("point").find("x").text),
        float(root.find("planningProblem").find("initialState").find("position").find("point").find("y").text)
    ]

    polygons = lanelets_to_polygons(xml_string)

    ego_lanelet_id = check_point_in_polygons(ego_point[0], ego_point[1], polygons)

    return ego_lanelet_id

# Construct and return egoLanelet as en ET.Element
# Considers traffic lights and traffic signs on the ego lanelet currently
# Also returns information on the adjacent lanes left and right
def characterize_ego_lanelet(xml_string:str) -> ET.Element:
    ego_lanelet_id = find_ego_lanelet(xml_string)
    root = ET.fromstring(xml_string)

    new_element = ET.Element("egoLane")
    new_element.attrib['id'] = ego_lanelet_id

    ego_lanelet = None

    for lanelet in root.findall('lanelet'):
        if lanelet.attrib['id'] == ego_lanelet_id:
            ego_lanelet = lanelet
            break
    if ego_lanelet is None:
        raise ValueError(f"Ego lanelet with ID {ego_lanelet_id} not found")

    # Traffic signs
    new_traffic_signs = ET.SubElement(new_element, "trafficSigns")
    # Create lookup dictionary for traffic signs by ID
    traffic_signs = {sign.attrib['id']: sign for sign in root.findall('trafficSign')}
    # Iterate over trafficSignRef elements in ego_lanelet
    for traffic_sign_ref in ego_lanelet.findall('trafficSignRef'):
        ref_id = traffic_sign_ref.attrib['ref']
        traffic_sign = traffic_signs.get(ref_id)

        if traffic_sign is None:
            raise ValueError(
                f"The traffic sign with ID {ref_id} referenced in the planning problem is not present in the file.")

        # Figure out which type a given traffic sign is
        ts_type = traffic_sign.find("trafficSignElement").find("trafficSignID").text
        ts_type = traffic_sign_dict.get(ts_type)

        # If the sign is a speed limit or required speed sign, add its value as a sub-element
        if ts_type == "speed_limit" or ts_type == "required_speed":
            value = traffic_sign.find("trafficSignElement").find("additionalValue").text
            element = ET.SubElement(new_traffic_signs, ts_type)
            element.attrib['id'] = ref_id
            element.text = value
        # If the sign is a different category, ignore the value sub-element
        else:
            ET.SubElement(new_traffic_signs, ts_type).attrib['id'] = ref_id

    # Traffic lights
    new_traffic_lights = ET.SubElement(new_element, "trafficLights")
    for traffic_light_ref in ego_lanelet.findall('trafficLightRef'):
        ref_id = traffic_light_ref.attrib['ref']
        ET.SubElement(new_traffic_lights, "trafficLight").attrib['ref'] = ref_id

    # LEFT LANES
    # adjacent_left = ego_lanelet.find("adjacentLeft")
    # left_lanes = ET.SubElement(new_element, "leftLanes")
    # while adjacent_left is not None:
    #     al = ET.SubElement(left_lanes, "adjacentLeft")
    #     al.attrib["id"] = adjacent_left.attrib["ref"]
    #     al.attrib["drivingDir"] = adjacent_left.attrib["drivingDir"]

    #     next_adjacent = None
    #     for lanelet in root.findall('lanelet'):
    #         if lanelet.attrib.get('id') == adjacent_left.attrib['ref']:
    #             next_adjacent = lanelet.find("adjacentLeft")
      #           break
     #    adjacent_left = next_adjacent

    # RIGHT LANES
    # adjacent_right = ego_lanelet.find("adjacentRight")
    # right_lanes = ET.SubElement(new_element, "rightLanes")
    # while adjacent_right is not None:
    #   ar = ET.SubElement(right_lanes, "lane")
      #   ar.attrib["id"] = adjacent_right.attrib["ref"]
        # ar.attrib["drivingDir"] = adjacent_right.attrib["drivingDir"]

        # next_adjacent = None
        # for lanelet in root.findall('lanelet'):
          #   if lanelet.attrib.get('id') == adjacent_right.attrib['ref']:
            #     next_adjacent = lanelet.find("adjacentRight")
              #   break
        # adjacent_right = next_adjacent

    return new_element

# Creates a summary of start and end edges of dynamic obstacles based on the CR file
def dynamic_summary_from_cr(file_path_str: str) -> str:
    xml = load_xml(file_path_str)
    polygons = lanelets_to_polygons(xml)

    root = ET.fromstring(xml)
    dynamic_obstacles = root.findall("dynamicObstacle")

    summary_lines = []

    for dynamic_obstacle in dynamic_obstacles:
        obstacle_id = dynamic_obstacle.attrib["id"]

        # Starting point and lanelet
        starting_point = [
            float(dynamic_obstacle.find("initialState").find("position").find("point").find("x").text),
            float(dynamic_obstacle.find("initialState").find("position").find("point").find("y").text)
        ]
        starting_lanelet = check_point_in_polygons(starting_point[0], starting_point[1], polygons)

        # End point and lanelet
        trajectory = dynamic_obstacle.find("trajectory")
        final_state = trajectory.findall("state")[-1]
        end_point = [
            float(final_state.find("position").find("point").find("x").text),
            float(final_state.find("position").find("point").find("y").text)
        ]
        end_lanelet = check_point_in_polygons(end_point[0], end_point[1], polygons)

        summary_lines.append(f"Vehicle {obstacle_id}: {starting_lanelet} -> {end_lanelet}")

    return "\n".join(summary_lines)

#print(dynamic_summary_from_cr("/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_60_T-1/Original/ZAM_Tjunction-1_60_T-1.cr.xml"))