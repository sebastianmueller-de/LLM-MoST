import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from utils.xml_parsing_utils import *
import os
from pathlib import Path

class ScenarioDBWrapper:
    def __init__(self, persist_dir: str, embedding_function: str | None = None):
        self.persist_dir = persist_dir

        # embed with sentence transformers; remember: eventually embedding method should be changeable via parameter for experiments
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction()
        # using Chroma raw (instead of LangChain wrapper), since granular filter control is desired
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)

        self.collection = self.chroma_client.get_or_create_collection(
            name="scenarios",
            embedding_function=self.embedding_function,
            metadata={
                "description": "Scenario Dataset",
            }
        )

        self.tag_values = [
            "comfort",
            "critical",
            "emergency_breaking",
            "evasive",
            "highway",
            "illegal_cut_in",
            "intersection",
            "interstate",
            "lane_change",
            "lane_following",
            "merging_lanes",
            "multi_lane",
            "no_oncoming_traffic",
            "oncoming_traffic",
            "parallel_lanes",
            "race_track",
            "roundabout",
            "rural",
            "simulated",
            "single_lane",
            "slip_road",
            "speed_limit",
            "traffic_jam",
            "turn_left",
            "turn_right",
            "two_lane",
            "urban"
        ]

    # Store a single scenario to the DB
    # Adds a scenario to chroma, as well as to the folder directory
    def save_scenario(self, file_path_str: str) -> bool:
        scenario_content, scenario_name, tags, location = add_metadata_to_xml(file_path_str)

        # This means something went wrong while processing
        if scenario_content == "ERROR": return False

        # Store tags, location, and the file path as metadata for quick and easy access
        mapping = tags | location | {"file_path": file_path_str}
        self.collection.add(ids=scenario_name,
                            documents=scenario_content,
                            metadatas=[mapping])

        print("Successfully saved scenario {}".format(scenario_name))

        file_name = Path(file_path_str).stem.removesuffix(".xml").removesuffix(".cr")

        scenario_path = (Path("Scenarios") / file_name).resolve()

        if not os.path.exists(scenario_path):
            os.makedirs(scenario_path / "Original")
            target_file_path = (scenario_path / "Original" / f"{file_name}.xml")
            with target_file_path.open("w") as f:
                # Nor sure if this modified file will work with CR --> seems to work just fine
                f.write(scenario_content)
        else:
            print(f"Scenario {scenario_name} already in DB.")

        return True

    # Just storage, but for whole folder of scenarios
    def save_folder(self, folder_path_str: str):
        folder = Path(folder_path_str).resolve()
        files = os.listdir(folder)

        successfully_processed = 0

        for file in files:
            full_path = folder / file
            if self.save_scenario(str(full_path)):
                successfully_processed += 1

        print(f"Successfully saved {successfully_processed} scenarios out of {len(files)}")

    # Find the file path of a scenario, given its id (which is the name of the xml file, including the ending)
    def get_file_path(self, id_:str) -> str:
        meta = self.collection.get(ids=[id_], include=["metadatas"])
        file_path = meta.get("metadatas")[0].get("file_path")

        if file_path is None:
            print("File could not be found. Make sure it is still in its original location. This does not refer to the Chroma DB, since the file is needed in its original XML forma.")
            return "File Not Found"

        return file_path

    def find_best_location(self, location:dict) -> list[str]:
        location = [{k: {"$eq": v}} for k, v in location.items()]

        if len(location) == 1:
            where = location[0]
        else:
            where = {"$and": location}

        query_result = self.collection.get(where=where, limit=400, include=["metadatas"])  # Limit here is an arbitrary value. Returning more than 400 scenarios, however, does not seem useful.

        filtered_ids = query_result["ids"]

        return filtered_ids

    # TODO: Tags are immediately looped, until a result is found. This needs to change if we focus more on interactive scenarios, since these (possibly?) tend to be sparsely annotated.
    def find_best_tags(self, after_location_ids:list[str], tags:list[str]):
        filtered_ids = []
        filtered_docs = []

        eq_tags = [{tag: {"$eq": 1}} for tag in tags]

        iter_counter = 0
        while len(filtered_ids) == 0:

            # If the user provides no tags, no results will be returned
            if len(eq_tags) == 0:
                print("---\nNo fitting scenario exists for the specified location. Tags were iteratively removed until only 1 left. Should inform the user and restart the search process in the interface.py file. Use the fact that filtered_ids is empty for this behavior.\n---")
                break

            if len(eq_tags) == 1:
                where = eq_tags[0]
            else:
                where = {"$and": eq_tags}

            if len(after_location_ids) == 0: # If this list is empty, we do not consider location, because the user skipped it
                query_result = self.collection.get(where=where, limit = 400, include=["documents"])
            else:
                query_result = self.collection.get(ids=after_location_ids, where=where, limit=400, include=["documents"])

            filtered_ids = query_result["ids"]
            filtered_docs = query_result["documents"]

            eq_tags.pop() # Here for counting when 0 tags are left

            iter_counter += 1

        # Returns ids (= names) and documents of found results
        return filtered_ids, filtered_docs

    def find_best_road_net(self, after_tags_ids:list[str], after_tags_docs:list[str], road_net_from_user:dict):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_tags_ids, after_tags_docs):
            from_xml = get_ego_lane_from_xml(doc)
            add_to_filtered = True

            # to_be_added = {} # Stores all the traffic signs/lights that should be added to the XML file

            for k, v in road_net_from_user.items():
                if k not in from_xml:
                    if 0 in v:
                        continue
                    else: # TODO: enter modification call for adding new traffic sign

                        # Do not forget to update from_xml and the lists after receiving the modified file
                        # to_be_added.add({k: v})

                        add_to_filtered = False
                        break

                if not self.in_range(from_xml[k], v): # TODO: enter modification call for changing value of existing sign
                    add_to_filtered = False
                    break

            # if not add_to_filtered:
                # modified_id, modified_doc, debug_info = add_traffic_sign_or_light(id_, doc, to_be_added)
                # filtered_ids.append(modified_id)
                # filtered_docs.append(modified_doc)

            if add_to_filtered:
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents

    # Helper for range matches
    # value is an exact value, bounds is a list with one or two elements
    def in_range(self, value, bounds):
        if len(bounds) == 1:
            return value == bounds[0]
        elif len(bounds) == 2:
            return bounds[0] <= value <= bounds[1]
        else:
            raise ValueError("Bounds must be a list of 1 or 2 elements.")

    def find_best_obstacles(self, after_l1_ids: list[str], after_l1_docs: list[str], obs_from_user: dict[str, dict]):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_l1_ids, after_l1_docs): # If 0 stop signs are specified in the step before and then 0 pedestrians later, this loop is unreachable
            from_xml = get_obstacles_from_xml(xml_string=doc)
            add_to_filtered = True

            for k, v in obs_from_user.items():
                if k not in from_xml:
                    if 0 in v: continue # Example: if the user wants 0 cars, the value is 0. But if a scenario contains no cars, the count will not be 0 - there simply will not be a car-object present. This is why the document can be added to the results, even though an obstacle key desired by the user is not present in the XML file.
                    else:
                        add_to_filtered = False
                        break
                if not self.in_range(from_xml[k], obs_from_user[k]):
                    add_to_filtered = False
                    break

            if add_to_filtered:
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents

    def find_best_velocity(self, after_obstacles_ids: list[str], after_obstacles_docs: list[str], initial_velocity_from_user: dict[str, list[float]]):
        filtered_ids = []
        filtered_documents = []

        for id_, doc in zip(after_obstacles_ids, after_obstacles_docs):
            from_xml = get_velocity_from_xml(xml_string=doc)

            if from_xml == -1.0: continue
            if self.in_range(from_xml, initial_velocity_from_user.get("initial_velocity")):
                filtered_ids.append(id_)
                filtered_documents.append(doc)

        return filtered_ids, filtered_documents


# db = ScenarioDBWrapper("chroma")
# db.save_scenario("/home/avsaw1/sebastian/BalancedDB/DEU_Lengede-12_1_T-2.xml")
# db.save_folder("/home/avsaw1/sebastian/BalancedDB3")