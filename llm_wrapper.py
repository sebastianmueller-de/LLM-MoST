from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from regex import regex
from transformers import BertTokenizerFast
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

from db_wrapper import ScenarioDBWrapper
import re

from utils.xml_parsing_utils import load_xml_from_file_path, xml_tree_to_string, extract_and_validate_routes_xml
from utils.format_handling_utils import separate_text_and_leading_json

class LLMWrapper:
    # Add global config parameter to this, so that LLM is chosen via subclass
    def __init__(self, scenario_db:ScenarioDBWrapper):
        self.scenario_db = scenario_db

        self.llm = None

        self.memory = None

        self.net_file_seen = False

    # Starts LLM with an initial prompt, providing it with context and the overall task
    def start_llm(self):
        with open("extraction_prompt_files/starter_prompt.txt", "r", encoding="utf-8") as f:
            starter_prompt = f.read()
        response = self.prompt_llm(human_message="Do you feel ready to help me in generating an accurate and helpful scenario description?", system_message=starter_prompt, context="")

    # TODO: remove this later after testing
    def start_testing(self):
        with open("extraction_prompt_files/starter_prompt_modification_test.txt.txt", "r", encoding="utf-8") as f:
            starter_prompt = f.read()
        response = self.prompt_llm(human_message="Do you feel ready helping me modify a scenario?s", system_message=starter_prompt, context="")

    # Contacts the LLM and returns the answer
    def prompt_llm(self, human_message:str, system_message:str, context):
        conversation_history = self.memory.buffer

        long_string = human_message + system_message + conversation_history

        """It should realistically not happen that many tokens are in a message. However, if this is the case, memory is excluded from the prompt. The value of 125000 is a good choice for most state of the art, current LLM models. However, this (as well as the tokenizer model) can easily be adapted to support different models/pricing considerations."""

        if self.check_token_size(long_string) > 125000:
            print("\n---\nMore than 125000 tokens in use. Memory was excluded from prompt.\n---\n")
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                HumanMessage(content=human_message),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                SystemMessage(content=f"Summary of conversation so far:\n{conversation_history}"),
                HumanMessage(content=human_message),
            ])
        formatted_prompt = prompt.format()

        response = self.llm.invoke(formatted_prompt)
        self.memory.save_context({"input": human_message}, {"output": response.content})

        return response

    def extract_tags(self, query:str) -> list[str]:
        with open("extraction_prompt_files/tag_list.txt", "r", encoding="utf-8") as f:
            tag_list = f.read()

        # Tags prompt needs work
        system_message = f"""In this step, the user will provide you with a block of text, from which you have to extract certain keywords. Return them as a ranked list seperated by line breaks. The most relevant keywords should be on top. Do not include any keywords for which you have no good reason to include them. Try to return between 1 and 5 of them. If the user does not specify any keywords, choose simulated as the single keyword and return it. Never return empty output. The format should look something like this:\n\n keyword1\nkeyword2\nkeyword3\n...\n\nHere is a list of the possible keywords, containing the individual words as well as a description and some example phrases:\n{tag_list} 
            """

        response = self.prompt_llm(human_message=query, system_message=system_message, context="")
        print("Extracted keywords from prompt:\n", response.content)

        lines = response.content.strip().splitlines()
        # Clean each line
        keywords = []
        for line in lines:
            cleaned = re.sub(r"^[\-\*\d\.\)\s]*", "", line).strip()  # remove bullets, numbers, whitespace
            transformed = cleaned.lower().replace(" ", "_")  # normalize formatting
            if transformed and transformed in self.scenario_db.tag_values:  # skip empty or irrelevant entries
                keywords.append(transformed)

        print("Cleaned keywords:\n", keywords)

        return keywords

    # TODO: add extra format checks here (LLM output should be well-formed)
    def get_json_from_output(self, response:str) -> dict:
        # Use recursive pattern to extract the outermost {...} block
        match = regex.search(r"\{(?:[^{}]|(?R))*\}", response)
        if match:
            json_str = match.group(0)
            print("Extracted JSON:", json_str)
            # Optional: parse to dict
            import json
            data = json.loads(json_str)
        else:
            print("No JSON object found.")
            data = {}

        return data

    # Return traffic signs and lights as a dict from JSON
    def extract_road_net(self, query:str) -> dict:
        with open("extraction_prompt_files/road_net_prompt.txt", "r", encoding="utf-8") as f:
            traffic_sign_list = f.read()
        system_message = f"{traffic_sign_list}"
        response = self.prompt_llm(human_message=query, system_message=system_message, context="")
        traffic_signs = self.get_json_from_output(response.content)

        print(response.content)

        return traffic_signs # Returns a dictionary of the resp. sign and its value

    def extract_obstacles(self, query:str) -> dict:
        with open("extraction_prompt_files/obstacle_all_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        system_message = f"{prompt}"
        response = self.prompt_llm(human_message=query, system_message=system_message, context="")
        obstacles = self.get_json_from_output(response.content)

        # TODO: make sure that obstacles are from list and not hallucinated

        print("Obstacles raw: ", response.content)
        print("Obstacles after formatting: ", obstacles)

        return obstacles

    def extract_velocity(self, query:str) -> dict:
        with open("extraction_prompt_files/initial_velocity_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        system_message = f"{prompt}"
        response = self.prompt_llm(human_message=query, system_message=system_message, context="")
        initial_velocity = self.get_json_from_output(response.content)

        print("Init velocity raw: ", response.content)
        print("Init velocity after formatting: ", initial_velocity)

        return initial_velocity

    def extract_location(self, query:str) -> dict:
        with open("extraction_prompt_files/location_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        system_message = f"{prompt}"
        response = self.prompt_llm(human_message=query, system_message=system_message, context="")
        location = self.get_json_from_output(response.content)

        # Sanity check - this step might be prone to confusion for certain LLMs
        if "location" in location.keys():
            location = location["location"]
        if "country_codes" in location.keys():
            new_location = {"country_code": location["country_codes"]}
            location = new_location
        if "country_code" in location and isinstance(location["country_code"], list):
            location["country_code"] = location["country_code"][0]

        print("Location after formatting: ", location)

        return location

    def check_token_size(self, token_string:str) -> int:
        tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        tokens = tokenizer.tokenize(token_string)
        num_toks = len(tokens)
        print(num_toks)
        return num_toks

    def extract_relaxation(self, json_big:str) -> dict:
        with open("extraction_prompt_files/relaxation_prompt.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
        system_message = f"{prompt}"
        response = self.prompt_llm(human_message=json_big, system_message=system_message, context="")
        relaxed_json = self.get_json_from_output(response.content)

        return relaxed_json

    # Asks the LLM to find the desired action corresponding to user input
    # For now simply finds out which type of modification is desired (type is needed before SUMO conversion can start for naming scheme)
    # Can be extended as general exit point for all kinds of LLM-based functions
    def find_desired_reaction(self, user_prompt: str, modification_type_prompt_path: str = "modification_prompt_files/modification_type_prompt.txt") -> tuple[dict, str]:

        with open(modification_type_prompt_path, "r", encoding="utf-8") as f:
            modification_type_prompt = f.read()

        response = self.prompt_llm(human_message=user_prompt, system_message=modification_type_prompt, context="")

        action_json = self.get_json_from_output(response.content)

        return action_json, response.content

    def modify(
        self,
        net_file_path: str,
        rou_file_path: str,
        user_prompt: str,
        net_prompt_template_path: str = "modification_prompt_files/net_file_prompt.txt",
        rou_prompt_template_path: str = "modification_prompt_files/manipulation_prompt.txt"
    ) -> str:
        """
        Modifies a SUMO route file based on a user instruction using LLM guidance.

        Parameters:
            net_file_path (str): Path to the SUMO network file (.net.xml).
            rou_file_path (str): Path to the SUMO route file (.rou.xml).
            user_prompt (str): The user's instruction, e.g., "Move vehicle X to Y".
            net_prompt_template_path (str): Path to the system prompt template for the network file.
            rou_prompt_template_path (str): Path to the system prompt template for the route file.

        Returns:
            str: The modified content from the LLM based on the route file and prompt.
        """

        if not self.net_file_seen:
            # === 1. Load and format the SUMO net file ===
            net_tree = load_xml_from_file_path(net_file_path)
            net_xml_string = xml_tree_to_string(net_tree)

            # Load the system prompt for the net file
            with open(net_prompt_template_path, "r", encoding="utf-8") as f:
                net_system_prompt = f.read()

            # Combine the system prompt with the net XML content
            full_net_prompt = net_system_prompt + net_xml_string

            # TODO: remove this later - serves to test direct modification for removal, since smaller open source LLMs seemingly prioritize human commands so much over system prompt that net file analysis causes inability to modify route file
            # Use this for testing different approach -> maybe come to conclusion that open source models should be treated differently?
            #net_response = self.prompt_llm(
            #    human_message="Analyze the net file, please. You will implement my modification later.",
            #    system_message=full_net_prompt,
            #    context=""
            #)

            # Get LLM response using the network file and user prompt
            net_response = self.prompt_llm(
                human_message=user_prompt,
                system_message=full_net_prompt,
                context=""
            )

            print("=== LLM Response to Net File ===")
            print(net_response.content)

            self.net_file_seen = True

        # === 2. Load and format the SUMO route file ===
        rou_tree = load_xml_from_file_path(rou_file_path)
        rou_xml_string = xml_tree_to_string(rou_tree)

        # Load the system prompt for the route file
        with open(rou_prompt_template_path, "r", encoding="utf-8") as f:
            rou_system_prompt = f.read()

        # Combine the system prompt with the route XML content
        full_rou_prompt = rou_system_prompt + rou_xml_string

        # Get LLM response using the route file and user prompt
        rou_response = self.prompt_llm(
            human_message=user_prompt,
            system_message=full_rou_prompt,
            context=""
        )

        content = rou_response.content

        # Validate LLM output before processing it further
        cleaned_xml = ""
        try:
            cleaned_xml = extract_and_validate_routes_xml(content)
        except Exception as e:
            print(f"Error: {e}")

        print("=== LLM Response to Route File ===")
        print(rou_response.content)

        return cleaned_xml

    # Generates exit trajectories for each vehicle
    # Could be done algorithmically, but understanding of the map is advantageous for the LLM and the modification process anyway
    def generate_full_trajectories(
            self,
            net_file_path: str,
            rou_file_path: str,
            net_prompt_template_path: str = "modification_prompt_files/generate_routes_net_analysis_prompt.txt",
            generate_routes_prompt_template_path: str = "modification_prompt_files/generate_routes_prompt.txt",
    ) -> str:

        # === 1. Load and format the SUMO net file ===
        net_tree = load_xml_from_file_path(net_file_path)
        net_xml_string = xml_tree_to_string(net_tree)

        # Load the system prompt for the net file
        with open(net_prompt_template_path, "r", encoding="utf-8") as f:
            net_system_prompt = f.read()

        # Combine the system prompt with the net XML content
        full_net_prompt = net_system_prompt + net_xml_string

        # Get LLM response using the network file and user prompt
        net_response = self.prompt_llm(
            human_message="Please create trajectories, so that each vehicle leaves the simulation.",
            system_message=full_net_prompt,
            context=""
        )

        print("=== LLM Response to Net File ===")
        print(net_response.content)

        # === 2. Load and format the SUMO route file ===
        rou_tree = load_xml_from_file_path(rou_file_path)
        rou_xml_string = xml_tree_to_string(rou_tree)

        # Load the system prompt for the route file
        with open(generate_routes_prompt_template_path, "r", encoding="utf-8") as f:
            rou_system_prompt = f.read()

        # Combine the system prompt with the route XML content
        full_rou_prompt = rou_system_prompt + rou_xml_string

        # Get LLM response using the route file and user prompt
        rou_response = self.prompt_llm(
            human_message="Please create trajectories, so that each vehicle leaves the simulation.",
            system_message=full_rou_prompt,
            context=""
        )

        content = rou_response.content

        # Validate LLM output before processing it further
        cleaned_xml = ""
        try:
            cleaned_xml = extract_and_validate_routes_xml(content)
        except Exception as e:
            print(f"Error: {e}")

        print("=== LLM Response to Route File ===")
        print(rou_response.content)

        return cleaned_xml

    def update_frenetix_cost(self, user_prompt: str, cost_yaml_file_path: str = "Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml", cost_yaml_prompt_file: str = "frenetix_parameter_prompt_files/cost_yaml_prompt.txt"):
        with open(cost_yaml_prompt_file, "r", encoding="utf-8") as f:
            cost_yaml_prompt = f.read()

        with open(cost_yaml_file_path, "r", encoding="utf-8") as f:
            cost_yaml_file = f.read()

        system_message = cost_yaml_prompt + cost_yaml_file

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Frenetix Cost File ===")
        print(response.content)

        return response.content

    # Returns a JSON object containing the ids for lanelets which should have traffic lights
    def modify_traffic_lights(self, user_prompt: str, traffic_light_modification_file_path: str = "modification_prompt_files/traffic_light_modification_prompt.txt") -> dict:
        with open(traffic_light_modification_file_path, "r", encoding="utf-8") as f:
            traffic_light_prompt = f.read()

        response = self.prompt_llm(human_message=user_prompt, system_message=traffic_light_prompt, context="")

        print("=== LLM Response to Frenetix Cost File ===")
        print(response.content)

        return self.get_json_from_output(response.content)

    def analyze_simulation(self, user_prompt: str, simulations: str, simulation_analysis_file_path: str = "frenetix_parameter_prompt_files/simulation_analysis_prompt.txt") -> str:
        with open(simulation_analysis_file_path, "r", encoding="utf-8") as f:
            simulation_analysis_prompt = f.read()

        system_message = simulation_analysis_prompt + simulations

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Analysis Request ===")
        print(response.content)

        return response.content

    def analyze_batch(self, user_prompt: str, batch_simulations: str, batch_analysis_file_path: str = "frenetix_parameter_prompt_files/batch_analysis_prompt.txt", log_path_str: str = "Frenetix-Motion-Planner/logs/score_overview.csv") -> str:
        with open(batch_analysis_file_path, "r", encoding="utf-8") as f:
            batch_analysis_prompt = f.read()
        system_message = batch_analysis_prompt + batch_simulations + f"\n\nDo not include this text in the results reproduction section. Here is the <folder_link>: {log_path_str}"

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Batch Analysis Request ===")
        print(response.content)

        return response.content

    def param_qa(self, user_prompt: str, cost_yaml_file: str, param_qa_file_path: str = "frenetix_parameter_prompt_files/param_qa_prompt.txt") -> str:
        with open(param_qa_file_path, "r", encoding="utf-8") as f:
            param_qa_prompt = f.read()

        system_message = param_qa_prompt + cost_yaml_file

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Parameter QA Request ===")
        print(response.content)

        return response.content

    def batch_run_qa(self, user_prompt: str, batch_options: str, batch_run_prompt_file_path: str = "frenetix_parameter_prompt_files/batch_run_prompt.txt") -> str:
        with open(batch_run_prompt_file_path, "r", encoding="utf-8") as f:
            batch_run_prompt = f.read()

        system_message = batch_run_prompt + batch_options

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Batch Run Options Request ===")
        print(response.content)

        return response.content

    def batch_simulation(self, user_prompt: str, batch_simulation_file_path: str = "frenetix_parameter_prompt_files/batch_simulation_prompt.txt") -> tuple:
        with open(batch_simulation_file_path, "r", encoding="utf-8") as f:
            batch_simulation_prompt = f.read()

        system_message = batch_simulation_prompt

        response = self.prompt_llm(human_message=user_prompt, system_message=system_message, context="")

        print("=== LLM Response to Batch Simulation Request ===")
        print(response.content)

        json, response_string = separate_text_and_leading_json(response.content)

        return json, response_string

# Subclassing for different behavior

class CommercialLLMWrapper(LLMWrapper):
    def __init__(self, scenario_db: ScenarioDBWrapper, model, api_key):
        super().__init__(scenario_db)
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key
        )
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=100000)
        self.start_llm()

class OllamaLLMWrapper(LLMWrapper):
    def __init__(self, scenario_db: ScenarioDBWrapper, model, base_url):
        super().__init__(scenario_db)
        self.llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url
        )
        self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=100000)
        self.start_llm()
