import os
import json

# Reads in all cost logs, keeps track of the minima, and returns the sum over these
# This cost function describes the trajectory actually used by the motion planner, since at every time step it chooses the trajectory with the lowest cost for that step
# Note: the returned cost is only good for relative comparisons - currently, not every single time step is logged (this would lead to unnecessarily large storage use)
# Instead, the cost can be used for relative comparisons, when changing a parameter of the motion planner, or an aspect of the scenario
def get_min_cost(folder_path) -> float:
    min_sum = 0.0
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    # Extract minimum cost in this file
                    costs = (traj_data["cost"] for traj_data in data.values())
                    min_cost = min(costs, default=None)
                    min_sum += min_cost
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"Skipping {file_name}: {e}")
    return min_sum

# Create the context that is used by the LLM
def format_analysis(scenario_name: str, cost: float, weights: str, final_output: str, mod_info: str) -> str:
    return f"""\n
=== Simulation Result ===
Scenario: {scenario_name}
Modification Info: {mod_info}
Cost: {cost:.2f}

[Cost Weights]
{weights.strip()}

[Final Output from Motion Planner]
{final_output.strip()}

------------------------
""".strip("\n")