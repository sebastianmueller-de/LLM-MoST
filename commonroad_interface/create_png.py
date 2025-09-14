import os

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.mp_renderer import MPDrawParams
import numpy as np
from pathlib import Path

def visualize_png(file_path: str, target_name: str, output_folder: str) -> str:
    """
    Visualizes a CommonRoad scenario and planning problem, then saves the figure as a PNG.

    Parameters:
        file_path (str): Path to the CommonRoad XML scenario file.
        target_name (str): Desired base name of the output image file (without extension).
        output_folder (str): Folder where the output image will be saved.

    Returns:
        str: Path to the saved PNG image.
    """
    print(f"\n📄 Loading scenario file: {file_path}")

    try:
        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
        print(f"✅ Scenario and planning problems loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load scenario: {e}")
        raise

    # Set up the plot
    print("🖌️  Initializing renderer and draw parameters...")
    plt.figure(figsize=(50, 20))  # Originally 25, 10
    rnd = MPRenderer()
    draw_params = MPDrawParams()

    # Customize what to display for dynamic obstacles
    draw_params.dynamic_obstacle.show_label = True
    draw_params.dynamic_obstacle.draw_icon = True
    draw_params.dynamic_obstacle.draw_shape = True
    print("🎨 Draw parameters set for dynamic obstacles")

    try:
        print("🧭 Drawing scenario and planning problem set...")
        scenario.draw(rnd, draw_params=draw_params)
        planning_problem_set.draw(rnd)
        rnd.render(show=True)
        print("✅ Rendering complete")
    except Exception as e:
        print(f"❌ Failed to render scenario: {e}")
        raise

    # Save the figure
    try:
        save_path = Path(output_folder) / f"{target_name}.png"
        plt.savefig(save_path)
        print(f"💾 Saved visualization to: {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"❌ Failed to save visualization: {e}")
        raise


# visualize("/home/avsaw1/sebastian/ChaBot7/Scenarios/USA_US101-23_1_T-1/Original/USA_US101-23_1_T-1.xml", "USA_US101-23_1_T-1", "/home/avsaw1/sebastian/ChaBot7/Scenarios/USA_US101-23_1_T-1/Original/")