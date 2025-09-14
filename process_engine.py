import json
import os.path
import threading
import queue
from pathlib import Path

from db_wrapper import ScenarioDBWrapper
from llm_wrapper import CommercialLLMWrapper, OllamaLLMWrapper
from utils.xml_parsing_utils import requires_trajectory_generation, generate_exit_trajectories_without_llm, update_benchmark_id, add_traffic_lights
from config import SessionConfig

from utils.ego_lanelet_utils import dynamic_summary_from_cr
from utils.format_handling_utils import generate_modified_scenario_name, extract_clean_yaml, build_batch_options
from utils.cost_handling_utils import get_min_cost, format_analysis
from utils.data_utils import make_combined_bar_plot, make_batch_summary_string

import subprocess

# Class running the search loop in an extra thread
class ProcessEngine:
    def __init__(self):
        # LLM setup
        self.db = ScenarioDBWrapper(persist_dir="chroma")
        # self.llm = LLMWrapper(scenario_db=self.db)
        self.llm = None

        # Debug log
        # Provides abstract and specifically formatted information
        # Mainly important for the query part
        self.debug_log = "Debugger Active. Any part of the dialogue can be skipped by pressing enter without adding any text.\n###################"
        self.query_counter = 1

        # Uses the literal console output for closer analysis and to not clutter the debug log
        self.console_log = ""

        # Search setup
        self.extracted_location_json = {}
        self.after_location_ids = []
        self.extracted_tags = []
        self.after_tags_ids = []
        self.after_tags_docs = []
        self.extracted_road_net_json = {}
        self.after_road_net_ids = []
        self.after_road_net_docs = []
        self.extracted_obstacles_json = {}
        self.after_obstacles_ids = []
        self.after_obstacles_docs = []
        self.extracted_velocity_json = {}
        self.after_velocity_ids = []
        self.after_velocity_docs = []

        self.modifications = ["Base Scenario - No Modification"]

        # Keeping track of the current top results for the visualization output
        self.current_step = 0
        self.current_results = []

        # The final result at the end of the query part
        self.result = ""
        self.file_path = ""

        # The GIFs corresponding to the selected scenario
        self.gif_displays = []

        # The GIFs of the scenarios run through the motion planner
        self.planner_displays = []

        # Keeping track of run simulations
        self.simulations = ""
        self.batch_simulations = ""

        self.plot = None

        # Setup motion planners here
        self.planners = {
            "FRENETIX": {
                # Path to the venv where the planner is set up
                "venv_path": str(Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()),
                # Path to the main.py method used for single simulation
                "script_path": str(Path("Frenetix-Motion-Planner/main_batch.py").resolve()),
                # Path to a directory ending in /logs
                # The scenario-bound information should be stored in a folder like ".../logs/<scenario_name>/"
                # The same goes for the GIF - it should be stored in ".../logs/<scenario_name>/<scenario_name>.gif"
                "log_dir": str(Path("Frenetix-Motion-Planner/logs").resolve()),
                # If cost functions of a defined style (see Frenetix JSON cost logs) are supported, enable this (y = yes, n = no)
                "cost_support": "y",
                # If any types of weights can be adjusted via a YAML file, put it here
                "weights_file": str(Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()),
            },
            "MP-RBFN": {
                "venv_path": str(Path("RBFN-Motion-Primitives/venv/bin/activate").resolve()),
                "script_path": str(Path("RBFN-Motion-Primitives/scripts/run_cr_simulation_batch.py").resolve()),
                "log_dir": str(Path("RBFN-Motion-Primitives/logs").resolve()),
                "cost_support": "n",
                "weights_file": ""
            }
        }
        self.selected_planner = "FRENETIX"

        # Threading setup
        self.input_queue = queue.Queue()
        self.result_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set_llm(self, llm_parameters: dict):
        if self.llm is None and llm_parameters["mode"] == "Commercial":
            self.llm = CommercialLLMWrapper(scenario_db=self.db, model=llm_parameters["model"], api_key=llm_parameters["key"])
        elif self.llm is None and llm_parameters["mode"] == "Ollama":
            self.llm = OllamaLLMWrapper(scenario_db=self.db, model=llm_parameters["ollama_model"], base_url=llm_parameters["ollama_url"])

    def _run(self):
        """
        Background loop that waits for (step, input) pairs
        and calls the corresponding handler.
        """
        handlers = {
            "1": self._handle_location,
            "2": self._handle_tags,
            "3": self._handle_road_net,
            "4": self._handle_obstacles,
            "5": self._handle_velocity
        }
        # self.db = ScenarioDBWrapper(persist_dir="chroma")
        # self.llm = LLMWrapper(
        #     scenario_db=self.db,
        # )

        while self.running:
            try:
                step, user_input = self.input_queue.get(timeout=1)
                step = str(step)
                if step == "-1":
                    if len(self.after_velocity_ids) > 0:
                        self.result = self.after_velocity_ids
                        self.file_path = self.db.get_file_path(self.after_velocity_ids[0])
                    break
                if step in handlers:
                    handlers[step](user_input)
            except queue.Empty:
                continue

        self._finalize()

    # TODO: still need to decide how to handle requests for specific cities (e.g. Schwetzingen, Munich)
    def _handle_location(self, input:str):
        print(f"[Processor] Running Step 1 with input: {input}")

        # Check if LLM call necessary
        if len(input) == 0:
            self.debug_log += "\nSkipping location.\n###################"
            return

        self.extracted_location_json = self.llm.extract_location(input)
        # LLM call was necessary, but user still might have supplied unspecific information
        if not self.extracted_location_json:
            self.debug_log += "\nEither the user desires no specific location or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_location_ids = self.db.find_best_location(self.extracted_location_json)
        # End condition here --> if location not found, abandon immediately
        # Important piece of info: Right now, the immediate early fail of location is not implemented

        if len(self.after_location_ids) == 0:
            self.debug_log += "\nThe desired location is not available in our data base. It is recommended to run a new search to receive a scenario relevant to the query. To restart the search immediately, please press the restart button.\n###################"
        else:
            self.debug_log += f"\nLocation processed successfully. Extracted JSON object: {self.extracted_location_json}\nRemaining number of candidates: {len(self.after_location_ids)}\n###################"

            self.current_step = 1 # We use an approach of setting the current step to a number instead of +1, since steps may be skipped
            self.current_results = self.after_location_ids

    def _handle_tags(self, input:str):
        self.extracted_tags = self.llm.extract_tags(input)

        # Note that tags can handle the case of being supplied an empty list of location ids --> this means location will be ignored for the remainder of the run
        self.after_tags_ids, self.after_tags_docs = self.db.find_best_tags(self.after_location_ids, self.extracted_tags)

        if len(input) == 0: # This means the user  did not supply any tags and the LLM made a random choice
            self.debug_log += f"\nTags processed successfully. Tags were chosen by the LLM, since none were supplied: {self.extracted_tags}\nRemaining number of candidates: {len(self.after_tags_ids)}\n###################"
        else:
            self.debug_log += f"\nTags processed successfully. Extracted tags: {self.extracted_tags}\nRemaining number of candidates: {len(self.after_tags_ids)}\n###################"

            self.current_step = 2
            self.current_results = self.after_tags_ids

    def _handle_road_net(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping road network.\n##################"
            self.after_road_net_ids, self.after_road_net_docs = self.after_tags_ids, self.after_tags_docs

            self.current_step = 3
            self.current_results = self.after_road_net_ids

            return

        self.extracted_road_net_json = self.llm.extract_road_net(input)
        if not self.extracted_road_net_json:
            self.after_road_net_ids, self.after_road_net_docs = self.after_tags_ids, self.after_tags_docs
            self.debug_log += "\nEither the user desires no specific road network or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_road_net_ids, self.after_road_net_docs = self.db.find_best_road_net(self.after_tags_ids, self.after_tags_docs, self.extracted_road_net_json)

        if len(self.after_road_net_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired road network. A relaxed search will be run. The current lookup failed with the following extracted parameters: {self.extracted_road_net_json}\nPlease press the restart button to start a new query.\n###################"

            # If no results were found, run a search with relaxed parameters
            # self.run_relaxation()
        else:
            self.debug_log += f"\nRoad network processed successfully. Extracted JSON object: {self.extracted_road_net_json}\nRemaining number of candidates: {len(self.after_road_net_ids)}\n###################"

            self.current_step = 3
            self.current_results = self.after_road_net_ids

    def _handle_obstacles(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping obstacles.\n##################"
            self.after_obstacles_ids, self.after_obstacles_docs = self.after_road_net_ids, self.after_road_net_docs

            self.current_step = 4
            self.current_results = self.after_obstacles_ids

            return

        self.extracted_obstacles_json = self.llm.extract_obstacles(input)
        if not self.extracted_obstacles_json:
            self.after_obstacles_ids, self.after_obstacles_docs = self.after_road_net_ids, self.after_road_net_docs
            self.debug_log += "\nEither the user desires no specific obstacles or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_obstacles_ids, self.after_obstacles_docs = self.db.find_best_obstacles(self.after_road_net_ids, self.after_road_net_docs, self.extracted_obstacles_json)

        if len(self.after_obstacles_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired obstacles. A relaxed search will be run. The current lookup failed with the following extracted parameters: {self.extracted_obstacles_json}\nPlease press the restart button to start a new query.\n###################"

            # If no results were found, run a search with relaxed parameters
            # self.run_relaxation()
        else:
            self.debug_log += f"\nObstacles processed successfully. Extracted JSON object: {self.extracted_obstacles_json}\nRemaining number of candidates: {len(self.after_obstacles_ids)}\n###################"

            self.current_step = 4
            self.current_results = self.after_obstacles_ids

    def _handle_velocity(self, input:str):
        if len(input) == 0:
            self.debug_log += "\nSkipping velocity.\n###############"
            self.after_velocity_ids, self.after_velocity_docs = self.after_obstacles_ids, self.after_obstacles_docs

            self.current_step = 5
            self.current_results = self.after_velocity_ids

            return

        self.extracted_velocity_json = self.llm.extract_velocity(input)
        if not self.extracted_velocity_json:
            self.after_velocity_ids, self.after_velocity_docs = self.after_obstacles_ids, self.after_obstacles_docs
            self.debug_log += "\nEither the user desires no specific velocity or the LLM was unable to extract a description. Skipping this step.\n###################"
            return

        self.after_velocity_ids, self.after_velocity_docs = self.db.find_best_velocity(self.after_obstacles_ids, self.after_obstacles_docs, self.extracted_velocity_json)

        if len(self.after_velocity_ids) == 0:
            self.debug_log += f"\nThere are no results in our data base after specifying the desired velocity. A relaxed search will be run. The current lookup failed with the following extracted parameters: {self.extracted_velocity_json}\nPlease press the restart button to start a new query.\n###################"

            # If no results were found, run a search with relaxed parameters
            # self.run_relaxation()
        else:
            self.debug_log += f"\nVelocity processed successfully. Extracted JSON object: {self.extracted_velocity_json}\nRemaining number of candidates: {len(self.after_velocity_ids)}\n###################"

            self.current_step = 5
            self.current_results = self.after_velocity_ids

    def handle_input(self, step:int, user_input:str):
        """
        Called by UI logic to provide input for a specific step.
        """

        # If we are in modification mode, different behavior will be desired
        if step >= 6:
            return

        self.input_queue.put((step, user_input))

    # Returns the final result at the end as well as the corresponding file path
    def end_and_wait(self):
        """
        Signals the processor that input is complete and waits for result.
        """
        self.input_queue.put((-1, None))
        self.result_ready.wait() # Waits until processing is done
        return self.result, self.file_path

    def _finalize(self):
        print("[Processor] Finalizing result...")
        self.result_ready.set()

    def get_debug_log(self):
        return self.debug_log[-3000:]  # Only return last N chars

    # TODO: for handling file paths everywhere, try to make sure they work on all OS's
    # Returns a list of file paths as strings; does not allow possibility for image db outside of current working directory
    # Returns the step after which the images were created
    def get_images(self):
        output = []

        for result in self.current_results:
            new_name = result.removesuffix(".xml").removesuffix(".cr")
            image_path = Path(f"Scenarios/{new_name}/Original/{new_name}.png").resolve()

            if not image_path.exists():
                self.debug_log += f"\nCould not find image. Rendering started.\n###################"
                proc = subprocess.Popen(
                    [
                        "conda", "run", "-n", "cr37",
                        "python", "commonroad_interface/create_png_subprocess.py",
                        "--input-file", f"Scenarios/{new_name}/Original/{result}",
                        "--target-name", new_name,
                        "--output-folder", f"Scenarios/{new_name}/Original",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True
                )

                # Read output line by line as it arrives
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')  # print live to console
                    self.console_log += stdout_line  # append to debug_log live

                # Also read stderr line by line (optional, or you can merge it into stdout)
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')  # print live to console
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                return_code = proc.wait()

                if return_code != 0:
                    raise RuntimeError("PNG creation failed.")

                self.debug_log += f"\nPNG creation complete.\n###################"

            output.append(str(image_path))

        return output, self.current_step

    def get_selected_images(self, selected_step: int):
        output = []

        # Helper function for processing image IDs
        def process_image_ids(image_ids):
            for result in image_ids:
                new_name = result.removesuffix(".xml").removesuffix(".cr")
                image_path = Path(f"Scenarios/{new_name}/Original/{new_name}.png").resolve()

                if not image_path.exists():
                    self.debug_log += f"\nCould not find image. Rendering started.\n###################"
                    proc = subprocess.Popen(
                        [
                            "conda", "run", "-n", "cr37",
                            "python", "commonroad_interface/create_png_subprocess.py",
                            "--input-file", f"Scenarios/{new_name}/Original/{result}",
                            "--target-name", new_name,
                            "--output-folder", f"Scenarios/{new_name}/Original",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )

                    # Read output line by line as it arrives
                    for stdout_line in proc.stdout:
                        print(stdout_line, end='')  # print live to console
                        self.console_log += stdout_line

                    # Also read stderr line by line (optional, or you can merge it into stdout)
                    for stderr_line in proc.stderr:
                        print(stderr_line, end='')  # print live to console
                        self.console_log += stderr_line

                    proc.stdout.close()
                    proc.stderr.close()

                    return_code = proc.wait()

                    if return_code != 0:
                        raise RuntimeError("PNG creation failed.")

                    self.debug_log += f"\nPNG creation complete.\n###################"

                output.append(str(image_path))

        if selected_step == 1:
            process_image_ids(self.after_location_ids)
            return output, 1
        elif selected_step == 2:
            process_image_ids(self.after_tags_ids)
            return output, 2
        elif selected_step == 3:
            process_image_ids(self.after_road_net_ids)
            return output, 3
        elif selected_step == 4:
            process_image_ids(self.after_obstacles_ids)
            return output, 4
        elif selected_step == 5:
            process_image_ids(self.after_obstacles_ids)
            return output, 5
        else:
            return self.gif_displays, 7

    def select_initial_gif(self, selected_candidate: str) -> list[str]:
        """
        Selects or generates a GIF for the given scenario candidate.

        The function assumes a predefined folder structure where the scenario is located under:
            Scenarios/<ScenarioName>/Original/<ScenarioName>.xml

        If the corresponding GIF does not already exist, the function will call the
        Frenetix motion planner (via a subprocess) to generate it. The function also
        adds console and debug log information and summarizes vehicle start/end edges
        if a CommonRoad .cr file (.cr.xml) is present.

        Parameters:
        ----------
        selected_candidate : str
            Path to the selected scenario file (either .xml or .png). This is used to
            extract the base scenario name.

        Returns:
        -------
        list[str] or None
            Returns a list with the file path to the rendered GIF (as a string), or
            None if an error occurred during GIF generation.
        """

        file_name = Path(selected_candidate).name.removesuffix(".png").removesuffix(".xml")
        folder_path = Path("Scenarios").resolve() / file_name
        original_path = Path(folder_path) / "Original"
        gif_path = (Path(original_path) / file_name).with_suffix(".gif")

        print("GIF should be at", gif_path)
        self.console_log += f"\nGIF should be at {gif_path}"

        if not gif_path.exists():
            scenario_xml = original_path / f"{file_name}.xml"
            self.debug_log += "\nGIF does not exist and is being rendered now.\n###################"

            # Path to the virtual environment activation script
            venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()

            # Path to the Frenetix rendering script
            script_path = Path("Frenetix-Motion-Planner/main_without_ego.py").resolve()

            # Bash command to activate the virtual environment and run the GIF creation script
            command = f"source {venv_python} && python {script_path} --input-file {str(scenario_xml)} --output-dir {str(original_path)}"

            try:
                # Launch subprocess with line-buffered stdout/stderr
                # This renders the GIF based on a Frenetix simulation (without an ego vehicle)
                proc = subprocess.Popen(
                    ["bash", "-c", command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Stream stdout live into console log
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')
                    self.console_log += stdout_line

                # Stream stderr live into console log
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                return_code = proc.wait()

                if return_code != 0:
                    # Subprocess failed — append error and return fallback value
                    error_msg = f"\n❌ Frenetix subprocess exited with code {return_code}.\nCheck input XML or dependencies.\n"
                    self.console_log += error_msg
                    return None

                self.debug_log += "\nGIF was successfully rendered.\n###################"

            except Exception as e:
                # Unexpected exception — log and return fallback value
                error_msg = f"\n❌ Exception during GIF rendering subprocess: {str(e)}\n"
                self.console_log += error_msg
                return None

        else:
            print("GIF already exists.")

        # Try to summarize vehicle dynamics from the CommonRoad .cr.xml file if available
        cr_xml_path = original_path / f"{file_name}.cr.xml"
        if os.path.exists(cr_xml_path):
            summary = dynamic_summary_from_cr(str(cr_xml_path))
        else:
            summary = dynamic_summary_from_cr(str(original_path / f"{file_name}.xml"))

        self.debug_log += (
            "\nHere is a summary of the start and end edges of the dynamic vehicles in this simulation:"
            f"\n{summary}\n##################"
        )

        self.gif_displays = [str(gif_path)]
        self.console_log += f"\nGIF is at {gif_path}"

        return self.gif_displays

    def _handle_modification(self, user_input: str, scenario: str, mod_type: str):
        """
        Modifies the provided CommonRoad scenario based on the user's prompt using a language model.
        The method:
          1. Converts the scenario to SUMO format (if not already available).
          2. Optionally repairs broken vehicle routes.
          3. Prompts an LLM to modify the scenario's route file based on user input.
          4. Simulates the modified scenario.
          5. Generates a visual GIF of the new scenario.
          6. Updates internal state to track the new GIF and returns it.

        Parameters:
            user_input (str): Instruction or modification request for the LLM.
            scenario (str): Path to the original scenario GIF file.

        Returns:
            tuple: (List of paths to all GIFs including the new one, Index of the new GIF)
                   If an error occurs, returns (None, 0)
        """
        self.debug_log += "\n\nMODIFICATION STARTED\n"

        scenario_pre = Path(scenario).resolve().name.removesuffix(".gif").removesuffix(".png").removesuffix(
            "_without_ego")
        full_path = str(Path(scenario).resolve().parent)

        repaired_config = Path(full_path) / f"{scenario_pre}_repaired.vehicles.rou.xml"
        regular_config = Path(full_path) / f"{scenario_pre}.vehicles.rou.xml"

        if repaired_config.exists():
            self.debug_log += f"A repaired SUMO configuration of base scenario exists at {full_path}\n###################"
            used_base_config = Path(full_path) / f"{scenario_pre}_repaired"
        elif regular_config.exists():
            self.debug_log += f"\nA SUMO configuration of base scenario exists at {full_path}/SUMO\n###################"
            used_base_config = Path(full_path) / f"{scenario_pre}"
        else:
            used_base_config = Path(full_path) / f"{scenario_pre}"
            self.debug_log += f"\nNo SUMO configuration for the base scenario exists. Conversion process started.\n###################"

            try:
                proc = subprocess.Popen(
                    [
                        "conda", "run", "-n", "cr37",
                        "python", "commonroad_interface/convert_to_sumo_subprocess.py",
                        "--input-file", f"{full_path}/{scenario_pre}.xml",
                        "--target-name", scenario_pre,
                        "--sumo-root", f"{full_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                for stdout_line in proc.stdout:
                    print(stdout_line, end='')
                    self.console_log += stdout_line
                for stderr_line in proc.stderr:
                    print(stderr_line, end='')
                    self.console_log += stderr_line

                proc.stdout.close()
                proc.stderr.close()

                if proc.wait() != 0:
                    raise RuntimeError("SUMO Conversion subprocess failed.")
            except Exception as e:
                self.debug_log += f"\nError during SUMO conversion subprocess: {e}\n"
                return None, 0

            self.debug_log += "\nSUMO Conversion process of base scenario finished successfully.\n###################"

        base_net_file = f"{used_base_config}.net.xml"
        base_route_file = f"{used_base_config}.vehicles.rou.xml"

        requires_repair, broken_vehicles, exit_edges = requires_trajectory_generation(base_net_file,
                                                                                      base_route_file)
        if requires_repair and SessionConfig.get_use_repair_module():
            self.console_log += "\n🔧The scenario contains broken trajectories. Repair will be run.🔧\n"
            generate_exit_trajectories_without_llm(base_net_file, base_route_file, broken_vehicles, exit_edges)
        elif requires_repair:
            self.console_log += "\n🔧The scenario contains broken trajectories. Running a repair is recommended.🔧\n"
            self.debug_log += "\nWARNING: Broken trajectories may cause motion planner issues.\n###################"

        scenario_post = generate_modified_scenario_name(scenario_pre, mod_type)

        if Path(full_path).name == "Original":
            folder_path = Path(full_path).parent / "Modified" / scenario_post
        else:
            folder_path = Path(full_path).parent / scenario_post
        os.makedirs(folder_path, exist_ok=True)

        self.debug_log += "\nThe folder for the modified SUMO files has been created. Conversion process started.\n###################"

        try:
            proc = subprocess.Popen(
                [
                    "conda", "run", "-n", "cr37",
                    "python", "commonroad_interface/convert_to_sumo_subprocess.py",
                    "--input-file", f"{used_base_config}.xml",
                    "--target-name", scenario_post,
                    "--sumo-root", f"{folder_path}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.debug_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.debug_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("SUMO Conversion subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during modified SUMO conversion: {e}\n"
            return None, 0

        self.debug_log += "\nSUMO Conversion process terminated successfully.\n###################"
        self.debug_log += "\nThe LLM is now being prompted.\n###################"

        try:
            modified_rou_file = self.llm.modify(
                net_file_path=base_net_file,
                rou_file_path=base_route_file,
                user_prompt=user_input
            )
        except Exception as e:
            self.debug_log += f"\nError during LLM modification: {e}\n"
            return None, 0

        self.debug_log += "\nThe LLM has responded.\n###################"

        try:
            with open(f"{folder_path}/{scenario_post}.vehicles.rou.xml", "w") as dst:
                dst.write(modified_rou_file)
        except Exception as e:
            self.debug_log += f"\nError writing modified route file: {e}\n"
            return None, 0

        self.debug_log += "\nRoute file has been modified.\n###################"
        self.debug_log += "\nRunning simulation of newly created scenario.\n###################"

        try:
            proc = subprocess.Popen(
                [
                    "conda", "run", "-n", "cr37",
                    "python", "commonroad_interface/simulate_sumo_subprocess.py",
                    "--folder_path", str(folder_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("Simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during simulation subprocess: {e}\n"
            return None, 0

        new_file = update_benchmark_id(str(folder_path / f"{scenario_post}.xml"), scenario_post)

        self.debug_log += "\nSimulation finished.\n###################"
        self.debug_log += "\nGIF creation started.\n###################"

        venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()
        script_path = Path("Frenetix-Motion-Planner/main_without_ego.py").resolve()
        command = f"source {venv_python} && python {script_path} --input-file {new_file} --output-dir {folder_path}"

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("Frenetix simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during GIF creation subprocess: {e}\n"
            return None, 0

        self.debug_log += "\nGIF creation finished.\n##################"

        gif_path = folder_path / (scenario_post + ".gif")
        self.gif_displays.append(str(gif_path))

        self.debug_log += "\n\nMODIFICATION FINISHED\n"

        summary = dynamic_summary_from_cr(new_file)
        self.debug_log += f"\nHere is a summary of the start and end edges of the dynamic vehicles in this simulation:\n{summary}\n##################"

        # Construct list of modifications to keep track of changes
        # Indices align with gif list, so it is always clear which scenario had which modification
        self.modifications.append(f"Modified from: {scenario} - Modification: {user_input}")

        return self.gif_displays, len(self.gif_displays) - 1

    def _handle_cost_param_change(self, user_input: str):
        # Store the old, unedited version for safety purposes
        cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
        old_cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve().parent / "cost_old.yaml"
        if not os.path.exists(old_cost_file_path):
            with open(str(cost_file_path), "r") as f:
                cost_file = f.read()
            with open(str(old_cost_file_path), "w") as dst:
                dst.write(cost_file)

        new_cost_file = self.llm.update_frenetix_cost(user_input, str(cost_file_path))

        # TODO: this check does not work entirely and is maybe more complicated than necessary
        # if not check_yaml_structure(cost_file, new_cost_file):
            # return False, new_cost_file

        cleaned_cost_file = extract_clean_yaml(new_cost_file)

        if cleaned_cost_file is None:
            return False, new_cost_file

        with open(str(cost_file_path), "w") as dst:
            dst.write(cleaned_cost_file)

        return True, cleaned_cost_file

    def _handle_traffic_light_modification(self, user_input: str, scenario: str):
        traffic_lights = self.llm.modify_traffic_lights(user_input)

        scenario_pre = Path(scenario).resolve().name.removesuffix(".gif").removesuffix(".png").removesuffix(
            "_without_ego")
        full_path = str(Path(scenario).resolve().parent)

        # TODO: update mod_type to support traffic lights as well ("T" --> for now is treated as a trajectory modification)
        scenario_post = generate_modified_scenario_name(scenario_pre, "T")

        # TODO: 1) create new folder 2) create temporary? file 3) start conversion

        # Folder path for the modified scenario
        if Path(full_path).name == "Original":
            folder_path = Path(full_path).parent / "Modified" / scenario_post
        else:
            folder_path = Path(full_path).parent / scenario_post
        os.makedirs(folder_path, exist_ok=True)

        # Temporary file which will be modified used as the input for the SUMO conversion
        temp_file = folder_path / f"{scenario_post}.xml"

        # Adds light to file, but does not show up in frenetix -> we might need the position parameter? Easy fix: use edge point of target lanelet as position
        add_traffic_lights(file_path_str=f"{full_path}/{scenario_pre}.xml", target_path_str=str(temp_file), new_traffic_lights=traffic_lights["traffic_lights"])

        return

    def _execute_batch_simulation(self, batch_action: dict):

        def generate_csv_based_on_ids(ids: list[str]) -> str:
            csv_string = ""
            for _id in ids:
                _id = Path(_id).name.removesuffix(".gif").removesuffix(".png").removesuffix(".xml")
                csv_string += f"{_id}\n"
            with open("scenario_batch_list.csv", "w") as f:
                f.write(csv_string)
            return str(Path("scenario_batch_list.csv").resolve())

        if batch_action is None or "fail" in batch_action.keys():
            return
        # User has chosen a base preset for evaluation
        elif "base" in batch_action.keys():
            scenarios_csv = f"batch_{batch_action['base']}_scenarios.csv"
            scenarios_csv_path = str(Path(f"Frenetix-Motion-Planner/batch_presets/{scenarios_csv}").resolve())
            batch_type = f"Base {batch_action['base']} Scenarios"
        # User has chosen to evaluate based on a step from the query process
        elif "query" in batch_action.keys():
            step = batch_action["query"]
            batch_type = f"Query Step {step}"
            if step == "1":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_location_ids)
            elif step == "2":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_tags_ids)
            elif step == "3":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_road_net_ids)
            elif step == "4":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_obstacles_ids)
            elif step == "5":
                scenarios_csv_path = generate_csv_based_on_ids(self.after_velocity_ids)
            else:
                raise Exception("Query approach, but no step could be matched.")

        else:
            raise Exception("Neither fail, nor base, nor query recognized.")

        # Absolute path to the virtual environment's Python interpreter
        venv_path = self.planners[self.selected_planner]["venv_path"]
        # Path to the main script
        script_path = self.planners[self.selected_planner]["script_path"]
        command = f"source {venv_path} && python {script_path} --input-file {scenarios_csv_path} --batch"

        try:
            proc = subprocess.Popen(
                ["bash", "-c", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for stdout_line in proc.stdout:
                print(stdout_line, end='')
                self.console_log += stdout_line
            for stderr_line in proc.stderr:
                print(stderr_line, end='')
                self.console_log += stderr_line

            proc.stdout.close()
            proc.stderr.close()

            if proc.wait() != 0:
                raise RuntimeError("Frenetix batch simulation subprocess failed.")
        except Exception as e:
            self.debug_log += f"\nError during batch simulation subprocess: {e}\n"
            return None, 0

        # If weights file does not exist, provide a fictional path
        if not self.planners[self.selected_planner]["weights_file"]:
            weights_file = "non_existent"
        else:
            weights_file = self.planners[self.selected_planner]["weights_file"]
        self.plot = make_combined_bar_plot(logs_path=Path(self.planners[self.selected_planner]["log_dir"]), weights_path=Path(weights_file).resolve())
        self.batch_simulations += make_batch_summary_string(logs_path=Path(self.planners[self.selected_planner]["log_dir"]),weights_path=Path(weights_file).resolve(), batch_type=batch_type) + "\n"

        return

    # This function first uses the LLM to decide which action the user wants to take. Based on the user's input, the LLM will transform the received information into JSON objects (or similar structures), which will call the corresponding method. For now, the only allowed action is the modification.
    def react_to_request(self, user_input: str, gifs: [str], index: int):

        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")

        action_json, raw_response = self.llm.find_desired_reaction(user_input)
        if "fail" in action_json.keys():
            return gifs, index, action_json["fail"]

        if "vehicle_mod" in action_json.keys():
            mod_type = action_json["vehicle_mod"]
            gifs, index = self._handle_modification(user_input, gifs[index], mod_type)

        elif "param_mod" in action_json.keys():
            param_success, output = self._handle_cost_param_change(user_input)
            if param_success:
                return gifs, index, f"The parameters were updated successfully. Here are the new parameters:\n```yaml\n{output}\n```"
            else:
                return gifs, index, output

        elif "param_qa" in action_json.keys():
            cost_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
            with open(str(cost_file_path), "r") as f:
                output = f.read()

            answer = self.llm.param_qa(user_input, output)

            return gifs, index, f"{answer}\n```yaml\n{output}\n```"

        elif "qa" in action_json.keys():
            return gifs, index, action_json["qa"]

        elif "analysis" in action_json.keys():
            analysis = self.llm.analyze_simulation(user_input, self.simulations)
            return gifs, index, analysis

        elif "batch_qa" in action_json.keys():
            batch_options = build_batch_options(loc=self.extracted_location_json, loc_ids=self.after_location_ids, tags=self.extracted_tags, tags_ids=self.after_tags_ids, road=self.extracted_road_net_json, road_ids=self.after_road_net_ids, obs=self.extracted_obstacles_json, obs_ids=self.after_obstacles_ids, vel=self.extracted_velocity_json, vel_ids=self.after_velocity_ids)
            response = self.llm.batch_run_qa(user_input, batch_options)
            return gifs, index, response

        elif "batch_simulation" in action_json.keys():
            batch_action, response = self.llm.batch_simulation(user_input)
            self._execute_batch_simulation(batch_action=batch_action)
            return gifs, index, response

        elif "batch_analysis" in action_json.keys():
            response = self.llm.analyze_batch(user_prompt=user_input, batch_simulations=self.batch_simulations, log_path_str=self.planners[self.selected_planner]["log_dir"] + "/score_overview.csv")
            return gifs, index, response

        return gifs, index, ""

    # A function that takes in the list of gifs as well as the index to determine the selected candidate and start a simulation with the Frenetix motion planner
    # Updates the list for the planner output; returns this list and the index pointing towards the latest element
    # TODO: define where and how the resulting simulation GIF + logs will be stored; for now they end up in the default folder of the Frenetix module
    def run_with_frenetix(self, gifs: [str], index: int):
        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")

        scenario_gif_path = gifs[index]
        scenario = Path(scenario_gif_path).name.removesuffix(".gif").removesuffix("_without_ego") # Consider adding .removesuffix(".xml")
        scenario_xml_path = str(Path(scenario_gif_path).parent) + "/" + scenario + ".xml"

        self.debug_log += "\n\nFRENETIX SIMULATION STARTED\n"
        self.debug_log += f"\nThe provided GIF was: {scenario_gif_path}\n"
        self.debug_log += f"\nThe extracted scenario is: {scenario_xml_path}\n"

        # Absolute path to the virtual environment's Python interpreter
        venv_python = Path("Frenetix-Motion-Planner/venv/bin/activate").resolve()

        # Path to the Frenetix script
        script_path = Path("Frenetix-Motion-Planner/main.py").resolve()

        # Shell command: activate venv, then run script
        command = f"source {venv_python} && python {script_path} --input-file {str(scenario_xml_path)}"

        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Read output line by line as it arrives
        for stdout_line in proc.stdout:
            print(stdout_line, end='')  # print live to console
            self.console_log += stdout_line  # append to debug_log live

        # Also read stderr line by line (optional, or you can merge it into stdout)
        for stderr_line in proc.stderr:
            print(stderr_line, end='')  # print live to console
            self.console_log += stderr_line

        proc.stdout.close()
        proc.stderr.close()

        return_code = proc.wait()

        final_output = "\n".join(self.console_log.strip().splitlines()[-12:])

        if return_code != 0:
            raise RuntimeError("Frenetix simulation subprocess failed.")

        self.debug_log += "\nFRENETIX SIMULATION FINISHED\n"

        # TODO: adjust this, once a proper storage/naming scheme has been implemented
        log_dir = Path("Frenetix-Motion-Planner/logs").resolve() / scenario

        cost_dir = log_dir / "cost"
        cost = get_min_cost(str(cost_dir))

        weight_file_path = Path("Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml").resolve()
        with open(str(weight_file_path), "r") as f:
            weights = f.read()

        # TODO: add modification field to format_analysis
        if index < len(self.modifications):
            modification = self.modifications[index]
        else:
            modification = self.modifications[0]
        self.simulations += format_analysis(scenario_name=scenario, cost=cost, weights=weights, final_output=final_output)

        gif_files = list(log_dir.glob("*.gif"))

        if not gif_files:
            raise FileNotFoundError(f"No GIF found in {log_dir}")

        gif_path = str(gif_files[0])

        self.planner_displays.append(gif_path)

        return self.planner_displays, len(self.planner_displays) - 1

    def run_with_planner(self, gifs: [str], index: int):
        if len(gifs) == 0 or len(gifs) < index + 1:
            raise ValueError(
                f"Provided list of GIFs was empty. (Or possibly the index does not match: {len(gifs)} GIFs, index = {index})")
        scenario_gif_path = gifs[index]
        scenario = Path(scenario_gif_path).name.removesuffix(".gif").removesuffix(
            "_without_ego")  # Consider adding .removesuffix(".xml")
        scenario_xml_path = str(Path(scenario_gif_path).parent) + "/" + scenario + ".xml"

        self.debug_log += "\n\nMOTION PLANNER SIMULATION STARTED\n"
        self.debug_log += f"\nThe provided GIF was: {scenario_gif_path}\n"
        self.debug_log += f"\nThe extracted scenario is: {scenario_xml_path}\n"

        # Absolute path to the virtual environment's Python interpreter
        venv_path = self.planners[self.selected_planner]["venv_path"]
        # Path to the main script
        script_path = self.planners[self.selected_planner]["script_path"]

        # Shell command: activate venv, then run script
        command = f"source {venv_path} && python {script_path} --input-file {str(scenario_xml_path)}"

        proc = subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Read output line by line as it arrives
        for stdout_line in proc.stdout:
            print(stdout_line, end='')  # print live to console
            self.console_log += stdout_line  # append to debug_log live

        # Also read stderr line by line (optional, or you can merge it into stdout)
        for stderr_line in proc.stderr:
            print(stderr_line, end='')  # print live to console
            self.console_log += stderr_line

        proc.stdout.close()
        proc.stderr.close()

        return_code = proc.wait()

        final_output = "\n".join(self.console_log.strip().splitlines()[-12:])

        if return_code != 0:
            raise RuntimeError("Frenetix simulation subprocess failed.")

        self.debug_log += "\nFRENETIX SIMULATION FINISHED\n"

        log_dir = Path(self.planners[self.selected_planner]["log_dir"]) / scenario

        cost = -1.0
        weights = "No weights supported."
        if self.planners[self.selected_planner]["cost_support"] == "y":
            cost_dir = log_dir / "cost"
            cost = get_min_cost(str(cost_dir))
        if self.planners[self.selected_planner]["weights_file"]:
            with open(self.planners[self.selected_planner]["weights_file"], "r") as f:
                weights = f.read()

        # TODO: add simulator field for cross-planner comparison to analysis part
        if index < len(self.modifications):
            modification = self.modifications[index]
        else:
            modification = self.modifications[0]
        self.simulations += format_analysis(scenario_name=scenario, cost=cost, weights=weights,
                                            final_output=final_output, mod_info=modification)

        # TODO: this requires a structure
        gif_files = list(log_dir.glob("*.gif"))

        if not gif_files:
            raise FileNotFoundError(f"No GIF found in {log_dir}")

        gif_path = str(gif_files[0])

        self.planner_displays.append(gif_path)

        return self.planner_displays, len(self.planner_displays) - 1

    # Loads the JSON file containing dimensions
    # Assumed to be in the same folder as the image/GIF
    def load_plot_dimensions_json(self, image_path:str) -> dict:
        folder_path = Path(image_path).parent
        file_path = folder_path / f"{Path(image_path).stem}.json"
        with open(file_path) as f:
            metadata = json.load(f)
        return metadata

    def set_planner(self, planner: str):
        self.selected_planner = planner

    # Restart everything in order but keep the previous log
    def restart(self):
        # self.end_and_wait()
        self.query_counter += 1
        self.debug_log += f"\nSuccessfully reset the DB search engine. This is query number {self.query_counter} of this session.\n###################"

        self.db = ScenarioDBWrapper("chroma")
        self.llm = None

        self.extracted_location_json = {}
        self.after_location_ids = []
        self.extracted_tags = []
        self.after_tags_ids = []
        self.after_tags_docs = []
        self.extracted_road_net_json = {}
        self.after_road_net_ids = []
        self.after_road_net_docs = []
        self.extracted_obstacles_json = {}
        self.after_obstacles_ids = []
        self.after_obstacles_docs = []
        self.extracted_velocity_json = {}
        self.after_velocity_ids = []
        self.after_velocity_docs = []

        # Uses the literal console output for closer analysis and to not clutter the debug log
        self.console_log = ""

        self.current_step = 0
        self.current_results = []

        self.result = ""
        self.file_path = ""

        # The GIFs corresponding to the selected scenario
        self.gif_displays = []

        # The GIFs of the scenarios run through the motion planner
        self.planner_displays = []

        self.input_queue = queue.Queue()
        self.result_ready = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

#engine = ProcessEngine()
# print(engine.select_initial_gif("/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_321_T-1/ZAM_Tjunction-1_321_T-1.png"))
# print(engine._handle_modification("Car 349 should move to 21933", "/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_321_T-1/Modified/ZAM_Tjunction-1_321_T-1_M6838/ZAM_Tjunction-1_321_T-1_M6838.gif"))
#engine.run_with_frenetix(["/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_207_T-1/Original/ZAM_Tjunction-1_207_T-1.gif"], 0)
#print(engine.simulations)
#engine.run_with_frenetix(["/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_147_T-1/Modified/ZAM_Tjunction-1_147_T-1_BP4860/ZAM_Tjunction-1_147_T-1_BP4860.xml"], 0)
#print(engine.simulations)
# print(engine._handle_cost_param_change("make all parameters 1.0"))

#print(engine.react_to_request("Please remove all vehicles but one from the simulation.", ["/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Zip-1_32_T-1/Modified/ZAM_Zip-1_32_T-1_B8503/ZAM_Zip-1_32_T-1_B8503.gif"], 0))
#print(engine.react_to_request("Set all motion planner parameters to 1.0", ["/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Zip-1_32_T-1/Modified/ZAM_Zip-1_32_T-1_B8503/ZAM_Zip-1_32_T-1_B8503.gif"], 0))
#print(engine.react_to_request("Add a vehicle to lane 24.", ["/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Zip-1_32_T-1/Modified/ZAM_Zip-1_32_T-1_B8503/ZAM_Zip-1_32_T-1_B8503.gif"], 0)

# engine._handle_traffic_light_modification("Add a traffic light to 50201", "/home/avsaw1/sebastian/ChaBot7/Scenarios/ZAM_Tjunction-1_147_T-1/Original/ZAM_Tjunction-1_147_T-1.gif")

# engine.run_with_planner(["/home/avsaw1/sebastian/ChaBot7/Scenarios/GRC_NeaSmyrni-98_1_T-9/Original/GRC_NeaSmyrni-98_1_T-9.gif"], 0)