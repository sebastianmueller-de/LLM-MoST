import os
from typing import List, Union, Dict
from PIL import Image

from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import imageio.v3 as iio
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import State, CustomState
from commonroad.planning.planning_problem import PlanningProblem
from cr_scenario_handler.utils.agent_status import AgentStatus
# commonroad_dc
from commonroad_dc import pycrcc
from commonroad.planning.planning_problem import PlanningProblemSet

from commonroad.scenario.scenario import Scenario
from commonroad.visualization.draw_params import MPDrawParams, DynamicObstacleParams, ShapeParams
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle

from cr_scenario_handler.utils.configuration import Configuration

from wale_net_lite.visualization import draw_uncertain_predictions
from commonroad.common.file_reader import CommonRoadFileReader
import os
from omegaconf import OmegaConf
from cr_scenario_handler.utils.configuration import SimConfiguration
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
import json
from matplotlib.animation import FuncAnimation

# Note: for now, this file expects the frenetix venv to be active

def get_plot_limits(scenario: Scenario, frame_count):
    """
    The plot limits track the center of the ego vehicle.
    """

    def flatten(list_to_flat):
        return [item for sublist in list_to_flat for item in sublist]

    center_vertices = np.array(
        flatten(
            [lanelet.center_vertices for lanelet in scenario.lanelet_network.lanelets]
        )
    )

    min_coords = np.min(center_vertices, axis=0)
    max_coords = np.max(center_vertices, axis=0)
    dict_plot_limits = [
        [min_coords[0], max_coords[0], min_coords[1], max_coords[1]]
    ] * frame_count

    return dict_plot_limits

def make_plots():
    plt.rcParams['figure.max_open_warning'] = 50

    # File path to scenario
    file_path = "/home/avsaw1/sebastian/ChaBot7/data/raw_scenarios/DEU_Lengede-30_1_T-39.xml"

    # Read in scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    # Output directory for images
    plot_dir = "/home/avsaw1/sebastian/ChaBot7/commonroad_interface/test_plots"
    os.makedirs(plot_dir, exist_ok=True)

    print(get_plot_limits(scenario, 1))

    # Plot the scenario for 40 timesteps
    for i in range(0, 40):
        plt.figure(figsize=(20, 10))  # Consistent with the ego-free renderer default

        # Create renderer and draw parameters
        rnd = MPRenderer()
        rnd.draw_params = MPDrawParams()
        rnd.draw_params.time_begin = i

        # Apply visual settings (hardcoded here; config-dependent ones are skipped)
        # Static obstacles
        rnd.draw_params.static_obstacle.show_label = True
        rnd.draw_params.static_obstacle.occupancy.shape.facecolor = "#a30000"
        rnd.draw_params.static_obstacle.occupancy.shape.edgecolor = "#756f61"

        # Dynamic obstacles
        rnd.draw_params.dynamic_obstacle.time_begin = i
        rnd.draw_params.dynamic_obstacle.draw_icon = True
        rnd.draw_params.dynamic_obstacle.show_label = True
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

        # Plot scenario and planning problems
        scenario.draw(rnd, draw_params=rnd.draw_params)
        planning_problem_set.draw(rnd)

        rnd.render()

        # Save each timestep as an image
        plt.axis('off')
        plt.savefig(f"{plot_dir}/scenario_timestep_{i:03d}.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

def make_gif(duration: float = 0.1):
    """ Create an animated GIF from saved image files.

    Images are assumed to be saved as <log_path>/plots/<scenario_id>_<timestep>.png
    Does not check the plotting configuration in order to simplify independent
    configurations on agent and simulation level. This has to be done by the caller.

    :param scenario: CommonRoad scenario object.
    :param time_steps: List or range of time steps to include in the GIF
    :param log_path: Base path containing the plots folder with the input images
    :param duration: Duration of the individual frames (default: 0.1s)
    """

    scenario_name = "DEU_Lengede-30_1_T-39"
    log_path = "/home/avsaw1/sebastian/ChaBot7/commonroad_interface/test_plots"
    time_steps = range(40)

    print("Generating GIF")
    images = []
    filenames = []

    # directory, where single images are outputted (see visualize_planner_at_timestep())
    path_images = Path(log_path)

    for step in time_steps:
        im_path = os.path.join(path_images, "scenario_timestep_" + f"{step:03d}.png")
        filenames.append(im_path)

    img_width = 0
    img_height = 0

    for filename in filenames:
        img = Image.open(filename)
        width, height = img.size

        if not img_width:
            img_width = width
            img_height = height

        # Calculate the area to crop
        left = 0
        top = 0
        right = img_width if width != img_width else width
        bottom = img_height if height != img_height else height

        # Crop the image
        img_cropped = img.crop((left, top, right, bottom))

        # Convert the PIL image to an array for imageio
        img_array = np.array(img_cropped)
        images.append(img_array)

    iio.imwrite(os.path.join(log_path, scenario_name + ".gif"),
                images, duration=duration, loop=0)

# This should be able to create a nice, non-choppy GIF of the scenario
# TODO: align visual components with Frenetix aesthetic / integrate yaml configuration for setting parameters
def create_video(
        scenario: Scenario,
        output_folder: str,
        planning_problem_set: PlanningProblemSet = None,
        suffix: str = "",
        file_type: str = "mp4",
) -> str:
    """
    Create video for a simulated scenario and the list of ego vehicles.

    :param scenario: Final commonroad scenario
    :param output_folder: path to output folder
    :param planning_problem_set: possibility to plot a Commonroad planning problem
    :param trajectory_pred: list of one or more ego vehicles or their trajectory predictions
    :param follow_ego: focus video on the ego vehicle(s)
    :param suffix: possibility to add suffix to file name
    :param file_type: mp4 or gif files supported
    :return:
    """
    assert file_type in ("mp4", "gif")

    #config = ConfigurationBuilder.build_sim_configuration(
        # TODO: this is why benchmark ID needs to be adjusted in modified files --> leads to the scenario_id here (presumably)
    #    scenario_name=scenario.scenario_id,
    #    scenario_folder="/home/avsaw1/sebastian/BalancedDB/",
    #    root_path="/Users/sebastian/PycharmProjects/PythonProject/ChatBot7/Frenetix-Motion-Planner/configurations",
    #    module="simulation"  # or "my_custom_module"
    #)

    # add short padding to create a short break before the loop (1sec)
    frame_count_padding = int(1 / scenario.dt)
    frame_count = (
            max([obstacle.prediction.final_time_step for obstacle in scenario.obstacles])
            + frame_count_padding
    )

    dict_plot_limits = get_plot_limits(scenario, frame_count)

    interval = (
            1000 * scenario.dt
    )  # delay between frames in milliseconds, 1 second * dt to get actual time in ms

    # Modify dpi to increase resolution, originally dpi = 150
    dpi = 300
    figsize = (5, 5)
    draw_params = MPDrawParams()
    draw_params.axis_visible = False

    # Modify draw_params to show label (and potentially other information)
    draw_params.lanelet_network.show_label = True
    draw_params.dynamic_obstacle.show_label = True
    draw_params.dynamic_obstacle.draw_icon = True
    draw_params.dynamic_obstacle.draw_shape = True

    # TODO: compare this back to Frenetix visualization module
    # draw_params.dynamic_obstacle.time_begin = timestep
    draw_params.dynamic_obstacle.draw_icon = True
    draw_params.dynamic_obstacle.show_label = True
    draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
    draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

    rnd = MPRenderer(figsize=figsize, draw_params=draw_params)
    rnd.ax.axes.get_xaxis().set_visible(False)
    (ln,) = plt.plot([], [], animated=True)

    def init_plot():
        plt.cla()
        if planning_problem_set is not None:
            planning_problem_set.draw(rnd)

        draw_params = MPDrawParams()

        # Modify draw_params to show label (and potentially other information)
        draw_params.lanelet_network.show_label = True
        draw_params.dynamic_obstacle.show_label = True
        draw_params.dynamic_obstacle.draw_icon = True
        draw_params.dynamic_obstacle.draw_shape = True

        # TODO: compare this back to Frenetix visualization module
        # draw_params.dynamic_obstacle.time_begin = timestep
        draw_params.dynamic_obstacle.draw_icon = True
        draw_params.dynamic_obstacle.show_label = True
        draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"


        draw_params.time_begin = 0
        draw_params.time_end = 0
        scenario.draw(renderer=rnd, draw_params=draw_params)
        rnd.plot_limits = dict_plot_limits[0]
        rnd.render()
        plt.draw()
        rnd.f.tight_layout()
        return (ln,)

    def animate_plot(frame):
        rnd.clear(keep_static_artists=False)
        if planning_problem_set is not None:
            planning_problem_set.draw(rnd)

        draw_params = MPDrawParams()

        # Modify draw_params to show label (and potentially other information)
        draw_params.lanelet_network.show_label = True
        draw_params.dynamic_obstacle.show_label = True
        draw_params.dynamic_obstacle.draw_icon = True
        draw_params.dynamic_obstacle.draw_shape = True

        # TODO: compare this back to Frenetix visualization module
        # draw_params.dynamic_obstacle.time_begin = timestep
        draw_params.dynamic_obstacle.draw_icon = True
        draw_params.dynamic_obstacle.show_label = True
        draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#E37222"
        draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#003359"

        draw_params.time_begin = frame
        draw_params.time_end = frame
        rnd.draw_list(scenario.dynamic_obstacles, draw_params=draw_params)

        draw_params = MPDrawParams()

        draw_params.time_begin = 0
        draw_params.time_end = 0
        scenario.lanelet_network.draw(renderer=rnd, draw_params=draw_params)

        rnd.plot_limits = dict_plot_limits[frame]
        rnd.f.tight_layout()
        rnd.render()
        return (ln,)

    anim = FuncAnimation(
        rnd.f,
        animate_plot,
        frames=frame_count,
        init_func=init_plot,
        blit=True,
        interval=interval,
    )

    file_name = str(scenario.scenario_id) + suffix + os.extsep + file_type
    anim.save(os.path.join(output_folder, file_name), dpi=dpi, writer="ffmpeg")
    plt.close(rnd.f)

    # TODO: remove, if coordinate stuff removed
    # === Store metadata for coordinate mapping ===
    metadata = {
        "dpi": dpi,
        "figsize": figsize,
        "frame_count": frame_count,
        "plot_limits": dict_plot_limits[0],
    }
    json_name = str(scenario.scenario_id) + suffix + ".json"
    json_path = os.path.join(output_folder, json_name)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return file_name

make_plots()

# File path to scenario
file_path = "/home/avsaw1/sebastian/ChaBot7/data/raw_scenarios/ZAM_Tjunction-1_321_T-1.xml"
# Read in scenario and planning problem set
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
create_video(scenario=scenario, planning_problem_set=planning_problem_set, output_folder="/home/avsaw1/sebastian/ChaBot7/commonroad_interface/test_plots", file_type="gif")