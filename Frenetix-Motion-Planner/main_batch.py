import argparse
import os
import sys
import csv
import shutil
from datetime import datetime
import traceback
import concurrent.futures
from cr_scenario_handler.simulation.simulation import Simulation
from cr_scenario_handler.utils.configuration_builder import ConfigurationBuilder
from cr_scenario_handler.utils.general import get_scenario_list
from pathlib import Path


def run_simulation_wrapper(scenario_info):
    scenario_file, scenario_folder, mod_path, logs_path, use_cpp = scenario_info
    start_simulation(scenario_file, scenario_folder, mod_path, logs_path, use_cpp, start_multiagent=False)


def start_simulation(scenario_name, scenario_folder, mod_path, logs_path, use_cpp, start_multiagent=False, count=0):
    log_path = os.path.join(logs_path, scenario_name)

    # For batch structure, construct the path to the XML file in the Original subfolder
    if "Scenarios" in scenario_folder:
        # Path structure: Scenarios_batch/scenario_name/Original/scenario_name.xml
        xml_path = os.path.join(scenario_folder, scenario_name, "Original", scenario_name + ".xml")
        if not os.path.exists(xml_path):
            print(f"Warning: XML file not found at {xml_path}")
            return
    else:
        # Fallback to flat structure
        xml_path = os.path.join(scenario_folder, scenario_name + ".xml")

    # Create a custom configuration that points to the correct XML file
    config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
    config_sim.simulation.use_multiagent = start_multiagent

    # Override the scenario path for batch structure
    if "Scenarios" in scenario_folder:
        config_sim.simulation.scenario_path = xml_path

    config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name)
    config_planner.debug.use_cpp = use_cpp

    simulation = None

    try:
        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()

    except Exception as e:
        try:
            simulation.close_processes()
        except:
            pass
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open(os.path.join(logs_path, 'log_failures.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            writer.writerow(["Scenario Name: " + log_path.split("/")[-1] + "\n" +
                             "Error time: " + str(current_time) + "\n" +
                             "In Scenario Timestep: " + current_timestep + "\n" +
                             "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n"])
            print(error_traceback)

def start_simulation_single(scenario_name, scenario_folder, mod_path, logs_path, use_cpp, start_multiagent=False, count=0):
    log_path = os.path.join(logs_path, scenario_name)
    config_sim = ConfigurationBuilder.build_sim_configuration(scenario_name, scenario_folder, mod_path)
    config_sim.simulation.use_multiagent = start_multiagent

    config_planner = ConfigurationBuilder.build_frenetplanner_configuration(scenario_name)
    config_planner.debug.use_cpp = use_cpp

    simulation = None

    try:
        simulation = Simulation(config_sim, config_planner)
        simulation.run_simulation()

    except Exception as e:
        try:
            simulation.close_processes()
        except:
            pass
        error_traceback = traceback.format_exc()  # This gets the entire error traceback
        with open(os.path.join(logs_path, 'log_failures.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            current_time = datetime.now().strftime('%H:%M:%S')
            # Check if simulation is not None before trying to access current_timestep
            current_timestep = str(simulation.global_timestep) if simulation else "N/A"
            writer.writerow(["Scenario Name: " + log_path.split("/")[-1] + "\n" +
                             "Error time: " + str(current_time) + "\n" +
                             "In Scenario Timestep: " + current_timestep + "\n" +
                             "CODE ERROR: " + str(e) + error_traceback + "\n\n\n\n"])
            print(error_traceback)

def main():
    if sys.platform == "darwin":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    mod_path = os.path.dirname(os.path.abspath(__file__))
    logs_path = os.path.join(mod_path, "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="If Single Mode: Path to the XML file in question. If Batch Mode: Path to the csv file with the selected scenario names")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="run in batch mode (default: single mode)",
    )
    args = parser.parse_args()
    batch = args.batch

    input_path = Path(args.input_file).resolve()

    # *************************
    # Set Python or C++ Planner
    # *************************
    use_cpp = True

    # *********************************************************
    # BATCH EVALUATION: Use Scenarios_batch folder structure
    # *********************************************************
    evaluation_pipeline = batch

    # **********************************************************************
    # Configuration for batch structure
    # **********************************************************************
    # Use batch structure (Scenarios_batch)
    # You can change this path to wherever your Scenarios_batch folder is located
    scenario_folder = "/home/avsaw1/sebastian/ChaBot7/Scenarios"  # Absolute path to your scenarios
    
    # Alternative: If you want to keep it relative to the script location, uncomment this line:
    # scenario_folder = os.path.join(mod_path, "Scenarios_batch")
    
    # Use specific scenario list from CSV file
    use_specific_scenario_list = batch
    # scenario_list_csv_path = os.path.join(mod_path, "scenario_batch_list.csv")
    
    # Fallback scenario name (not used when evaluation_pipeline=True)
    scenario_name = "CHN_Beijing-7_7_T-1"  # do not add .xml format to the name
    
    # Get list of scenarios to process
    if batch and use_specific_scenario_list and os.path.exists(input_path):
        # Use CSV file to specify which scenarios to run
        scenario_files = get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, input_path, True)
        print(f"Loaded {len(scenario_files)} scenarios from CSV file: {input_path}")
    elif batch:
        # Auto-detect all scenarios in batch folders
        scenario_files = get_scenario_list(scenario_name, scenario_folder, evaluation_pipeline, None, False)
        print(f"Auto-detected {len(scenario_files)} scenarios in batch folders")

    # Check if scenario folder exists
    if not os.path.exists(scenario_folder):
        print(f"ERROR: Scenario folder not found at: {scenario_folder}")
        print("Please update the 'scenario_folder' variable in main_batch.py to point to your Scenarios_batch folder")
        print("Current path:", scenario_folder)
        sys.exit(1)
    
    # Check if we have scenarios to process
    if batch and not scenario_files:
        print(f"ERROR: No scenarios found in folder: {scenario_folder}")
        print("Please check your CSV file and folder structure")
        sys.exit(1)

    # ***************************************************
    # Delete former logs & Create new score overview file
    # ***************************************************
    delete_former_logs = batch
    if delete_former_logs:
        shutil.rmtree(logs_path, ignore_errors=True)
    os.makedirs(logs_path, exist_ok=True)
    if not os.path.exists(os.path.join(logs_path, "score_overview.csv")):
        with open(os.path.join(logs_path, "score_overview.csv"), 'a') as file:
            line = "scenario;agent;timestep;status;message;result;collision_type;colliding_object_id\n"
            file.write(line)

    if evaluation_pipeline:
        num_workers = 12  # or any number you choose based on your resources and requirements
        print(f"Starting batch evaluation with {num_workers} workers...")
        print(f"Scenarios to process: {scenario_files[:5]}{'...' if len(scenario_files) > 5 else ''}")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a list of tuples that will be passed to start_simulation_wrapper
            scenario_info_list = [(scenario_file, scenario_folder, mod_path, logs_path, use_cpp)
                                  for scenario_file in scenario_files]
            results = executor.map(run_simulation_wrapper, scenario_info_list)

    else:
        # If not in evaluation_pipeline mode, just run one scenario
        scenario_folder = str(input_path.parent)
        scenario_name = input_path.stem
        scenario_files = [scenario_name]
        print(f"Running single scenario: {scenario_files[0]}")
        start_simulation_single(scenario_files[0], scenario_folder, mod_path, logs_path, use_cpp)


if __name__ == '__main__':
    main()
