import os
from crdesigner.map_conversion.map_conversion_interface import commonroad_to_sumo
import os
from lxml import etree

from commonroad.scenario.scenario import Tag
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader

import sys
import os
import pickle
import argparse
from pathlib import Path

# Add the commonroad-scenario-designer to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "commonroad-scenario-designer"))

try:
    from crdesigner.map_conversion.sumo_map.config import SumoConfig
    from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry

    print("✅ CommonRoad Scenario Designer modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing CommonRoad Scenario Designer: {e}")
    print("💡 Make sure to activate the conda environment first:")
    print("   conda activate scenariodesigner")
    print("💡 Also check that CommonRoad Scenario Designer is installed in the parent directory")
    sys.exit(1)


def create_simulation_config(scenario_name: str,
                             simulation_steps: int = 300,
                             dt: float = 0.1,
                             delta_steps: int = 2,
                             country_id: SupportedTrafficSignCountry = SupportedTrafficSignCountry.ZAMUNDA,
                             with_sumo_gui: bool = False,
                             output_dir: str = None) -> Path:
    """
    Create simulation_config.p file for CommonRoad Interactive Scenarios

    Args:
        scenario_name: Name of the scenario
        simulation_steps: Number of simulation steps (default: 300)
        dt: Simulation time step in seconds (default: 0.1)
        delta_steps: SUMO sub-steps per simulation step (default: 2)
        country_id: Traffic sign country (default: ZAMUNDA)
        with_sumo_gui: Whether to use SUMO GUI (default: False)
        output_dir: Output directory (default: current directory)

    Returns:
        Path to created simulation_config.p file
    """

    print(f"🔧 Creating simulation config for: {scenario_name}")

    # Create SumoConfig object with proper parameters
    sumo_config = SumoConfig.from_scenario_name(scenario_name)

    # Set simulation parameters manually
    sumo_config.simulation_steps = simulation_steps
    sumo_config.dt = dt
    sumo_config.delta_steps = delta_steps
    sumo_config.with_sumo_gui = True  # Enable GUI mode by default
    sumo_config.country_id = country_id
    sumo_config.presimulation_steps = 0
    sumo_config.highway_mode = True
    # Remove lateral_resolution to avoid conflicts with SUMO config file
    # The SUMO config file will handle this setting

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        sumo_config.output_path = str(output_path)
        sumo_config.scenario_path = str(output_path)
    else:
        output_path = Path(".")

    output_path.mkdir(parents=True, exist_ok=True)
    config_file = output_path / "simulation_config.p"

    # Save configuration as pickle file
    try:
        with open(config_file, "wb") as f:
            pickle.dump(sumo_config, f)

        print(f"✅ Created simulation config: {config_file}")
        print(f"   📊 Scenario: {sumo_config.scenario_name}")
        print(f"   🎮 Simulation steps: {sumo_config.simulation_steps}")
        print(f"   ⏱️  Time step: {sumo_config.dt}s")
        print(f"   🚦 Delta steps: {sumo_config.delta_steps}")
        print(f"   🌍 Country: {sumo_config.country_id}")
        print(f"   🖥️  GUI mode: {sumo_config.with_sumo_gui}")

        return config_file

    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return None


# The following code takes in a CR scenario file and creates all required SUMO files except for the .p file
# Currently from_trajectories = True in the crdesigner package of the cr37 conda environment

import os
from pathlib import Path
from commonroad.scenario.scenario import Scenario  # Example import, adjust if needed
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader


def convert_cr_to_sumo_scenario(scenario_path: str, target_name: str, collection_folder: str):
    """
    Converts a CommonRoad scenario to a SUMO scenario, including the simulation config (.p file),
    and stores the output in a subdirectory named after the scenario (without .xml).

    Parameters:
        scenario_path (str): Full path to the .xml scenario file. Example format: /home/avsaw1/sebastian/BalancedDB/DEU_Damme-17_1_T-1.xml
        target_name (str): Resulting name of the conversion. Example format: DEU_Damme-17_1_T-1_M4444 (when converting a non-modified, raw scenario, this will simply be the file name without suffix)
        collection_folder (str): Path to the folder where the converted scenario should be stored.

    With the above example, the converted files would be stored in collection_folder/target_name
    """
    print(f"🔄 Starting conversion of scenario: {scenario_path}")
    # TODO: add check so that a .cr.xml file can also be processed as usual --> should work now

    # Extract the scenario name without the .xml extension
    # scenario_name = Path(scenario_path).stem.removesuffix(".xml")
    # print(f"📛 Scenario name: {scenario_name}")

    # Create the output subdirectory
    # output_folder = os.path.join(collection_folder, scenario_name)
    # os.makedirs(output_folder, exist_ok=True)
    # print(f"📁 Created output folder: {output_folder}")

    # TODO: rm if rm target_name
    output_folder = collection_folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Created output folder: {output_folder}")

    # Copy the original .xml file to the new folder with a .cr.xml suffix
    # TODO: move this part maybe elsewhere and use it for CR modification; also: after simulation, replace this file with the simulated result (maybe?)
    # cr_xml_path = os.path.join(output_folder, scenario_name + ".cr.xml")
    # print(f"📄 Copying scenario file to: {cr_xml_path}")
    # with open(scenario_path, "r") as src, open(cr_xml_path, "w") as dst:
    #     dst.write(src.read())
    # print(f"✅ Scenario file copied and renamed to .cr.xml")

    # TODO: rm if rm target_name
    cr_xml_path = os.path.join(output_folder, target_name + ".cr.xml")
    print(f"📄 Copying scenario file to: {cr_xml_path}")
    with open(scenario_path, "r") as src, open(cr_xml_path, "w") as dst:
        dst.write(src.read())
    print(f"✅ Scenario file copied and renamed to .cr.xml")

    # Load the scenario using CommonRoadFileReader
    print(f"📥 Loading scenario from: {cr_xml_path}")
    scenario, planning_problem = CommonRoadFileReader(cr_xml_path).open()
    print(f"✅ Scenario and planning problems loaded")

    # Convert to SUMO format
    print(f"🔁 Converting scenario to SUMO format...")
    commonroad_to_sumo(cr_xml_path, cr_xml_path)
    print(f"✅ Scenario converted to SUMO")

    # Create the simulation config including the .p file
    print(f"🛠️  Generating simulation configuration file...")
    config_path = create_simulation_config(target_name, output_dir=output_folder)
    if config_path:
        print(f"✅ Simulation config created at: {config_path}")
    else:
        print(f"❌ Failed to create simulation config.")

# TODO: may need to be adjusted to above method and target name
def convert_all_cr_scenarios_in_folder(input_folder: str, output_folder: str):
    """
    Converts all CommonRoad .xml scenarios in a given input folder to SUMO format.
    Each scenario will be processed individually, and any failures will be logged without stopping the entire process.

    Parameters:
        input_folder (str): Folder containing CommonRoad .xml scenario files.
        output_folder (str): Destination folder to store converted scenarios.
    """
    print(f"📁 Starting batch conversion of scenarios from: {input_folder}")
    print(f"📤 Output folder for converted scenarios: {output_folder}")

    # Make sure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Counter for summary
    total = 0
    success = 0
    failed = []

    # Iterate over all XML files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".xml"):
            total += 1
            scenario_path = os.path.join(input_folder, file)
            print(f"\n🔄 [{total}] Converting: {file}")
            target_name = Path(scenario_path).name.removesuffix(".xml")

            try:
                convert_cr_to_sumo_scenario(scenario_path, target_name, output_folder)
                success += 1
                print(f"✅ Successfully converted: {file}")
            except Exception as e:
                print(f"❌ Failed to convert: {file}")
                print(f"   🧨 Error: {e}")
                failed.append(file)

    # Summary
    print("\n📊 Conversion Summary:")
    print(f"   ✅ Successful conversions: {success}/{total}")
    if failed:
        print(f"   ❌ Failed scenarios ({len(failed)}):")
        for f in failed:
            print(f"      - {f}")
    else:
        print("   🎉 All scenarios converted successfully!")

# convert_cr_to_sumo_scenario("/home/avsaw1/sebastian/BalancedDB/DEU_Damme-17_1_T-1.xml", "DEU_Damme-17_1_T-1_M4444", "/home/avsaw1/sebastian/manipulate_trajectory/converted_scenarios")
# convert_all_cr_scenarios_in_folder("/home/avsaw1/sebastian/ChatBot6/data/raw_scenarios", "/home/avsaw1/sebastian/ChatBot6/data/run_scenarios")
