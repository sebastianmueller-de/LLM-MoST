from pathlib import Path
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt

def make_bar_plot_success_failure():
    frenetix_path = Path(__file__).parent.parent / "Frenetix-Motion-Planner"
    os.makedirs(frenetix_path / "logs", exist_ok=True)
    score_path = frenetix_path / "logs" / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")

    # Count scenarios by result type
    counts = df["result"].value_counts()

    # Make bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)

    ax.set_title("Scenario Results Overview")
    ax.set_xlabel("Result Type")
    ax.set_ylabel("Number of Scenarios")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    return

def make_bar_plot_failure_analysis():
    frenetix_path = Path(__file__).parent.parent / "Frenetix-Motion-Planner"
    os.makedirs(frenetix_path / "logs", exist_ok=True)
    score_path = frenetix_path / "logs" / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")

    # Filter only failures
    df_failures = df[df["result"] == "Failed"]

    # Count by failure message
    counts = df_failures["message"].value_counts()

    # Make bar plot
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="tomato")

    ax.set_title("Failure Analysis by Message")
    ax.set_xlabel("Failure Reason (message)")
    ax.set_ylabel("Number of Scenarios")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()

import yaml
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

def make_combined_bar_plot(logs_path: Path, weights_path: Path):
    score_path = logs_path / "score_overview.csv"

    if not score_path.exists():
        raise Exception("No score file exists.")

    print(f"Reading scores from: {score_path}")

    # Load CSV (semicolon separated)
    df = pd.read_csv(score_path, sep=";")

    # --- Create figure with 3 subplots (vertical) ---
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(3, 1, 1)  # top plot
    ax2 = fig.add_subplot(3, 1, 2)  # middle plot
    ax3 = fig.add_subplot(3, 1, 3)  # bottom axes for table

    # --- Plot 1: Success / Failure overview ---
    if "result" in df.columns and not df["result"].dropna().empty:
        counts_overview = df["result"].value_counts()
        counts_overview.plot(kind="bar", ax=ax1)
        ax1.set_title("Scenario Results Overview")
        ax1.set_xlabel("Result Type")
        ax1.set_ylabel("Number of Scenarios")
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        ax1.set_title("Scenario Results Overview (no data)")
        ax1.text(0.5, 0.5, "No results available", ha="center", va="center")
        ax1.axis("off")

    # --- Plot 2: Failure analysis ---
    if "result" in df.columns and "message" in df.columns:

        # TODO: need to define proper scheme for the desired .csv format; this is just an emergency solution
        # Define what counts as "success"
        success_labels = {"Success", "success", "successful"}
        # Mark everything else as a failure
        df_failures = df[~df["result"].isin(success_labels)]

        if not df_failures.empty:
            counts_failures = df_failures["message"].value_counts()
            if not counts_failures.empty:
                counts_failures.plot(kind="bar", ax=ax2, color="tomato")
                ax2.set_title("Failure Analysis by Message")
                ax2.set_xlabel("Failure Reason (message)")
                ax2.set_ylabel("Number of Scenarios")
                ax2.grid(axis="y", linestyle="--", alpha=0.7)
                ax2.tick_params(axis='x', rotation=30, labelsize=8)
            else:
                ax2.set_title("Failure Analysis by Message (no failures)")
                ax2.text(0.5, 0.5, "No failure messages available", ha="center", va="center")
                ax2.axis("off")
        else:
            ax2.set_title("Failure Analysis by Message (no failures)")
            ax2.text(0.5, 0.5, "No failed scenarios found", ha="center", va="center")
            ax2.axis("off")
    else:
        ax2.set_title("Failure Analysis by Message (missing data)")
        ax2.text(0.5, 0.5, "Required columns missing", ha="center", va="center")
        ax2.axis("off")

    # --- Plot 3: table of cost_weights ---
    ax3.axis("off")  # hide axes
    if weights_path.exists():
        with open(weights_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        cost_weights = yaml_data.get("cost_weights", {})
        if cost_weights:
            # Split long column names into multiple lines (optional)
            col_labels = [label.replace("_", "_\n") for label in cost_weights.keys()]
            row_values = [list(cost_weights.values())]

            # Add table in ax3
            table = ax3.table(cellText=row_values,
                              colLabels=col_labels,
                              loc="center",
                              cellLoc="center",
                              colLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 4)  # increase vertical spacing for line breaks

    plt.tight_layout()
    plt.savefig(logs_path / "batch_stats.png", format="png", bbox_inches="tight")
    # plt.show()
    return plt

PRESETS = {
    "Comfort": {
        "acceleration": 0.2, "jerk": 0.2, "lateral_jerk": 0.8, "longitudinal_jerk": 0.8,
        "orientation_offset": 0.0, "path_length": 0.0, "lane_center_offset": 0.5,
        "velocity_offset": 0.3, "velocity": 0.0, "distance_to_reference_path": 1.0,
        "distance_to_obstacles": 0.3, "prediction": 0.2, "responsibility": 0.1,
    },
    "Balanced": {
        "acceleration": 0.3, "jerk": 0.3, "lateral_jerk": 0.5, "longitudinal_jerk": 0.5,
        "orientation_offset": 0.1, "path_length": 0.2, "lane_center_offset": 0.7,
        "velocity_offset": 0.6, "velocity": 0.2, "distance_to_reference_path": 2.0,
        "distance_to_obstacles": 0.8, "prediction": 0.3, "responsibility": 0.2,
    },
    "Efficiency-Sporty": {
        "acceleration": 0.2, "jerk": 0.15, "lateral_jerk": 0.25, "longitudinal_jerk": 0.25,
        "orientation_offset": 0.2, "path_length": 0.6, "lane_center_offset": 0.6,
        "velocity_offset": 1.0, "velocity": 0.4, "distance_to_reference_path": 2.0,
        "distance_to_obstacles": 0.6, "prediction": 0.2, "responsibility": 0.1,
    },
    "Safety-Conservative": {
        "acceleration": 0.2, "jerk": 0.2, "lateral_jerk": 0.4, "longitudinal_jerk": 0.4,
        "orientation_offset": 0.0, "path_length": 0.0, "lane_center_offset": 1.0,
        "velocity_offset": 0.3, "velocity": 0.0, "distance_to_reference_path": 5.0,
        "distance_to_obstacles": 2.0, "prediction": 0.5, "responsibility": 0.5,
    }
}

def compare_to_templates(weights_path: Path) -> str:
    """
    Compare a cost weights YAML file against predefined presets.
    Returns the preset name if exact match is found, otherwise 'Custom'.
    """

    # Load YAML
    with open(weights_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Extract cost_weights section
    cost_weights = yaml_data.get("cost_weights", {})
    if not cost_weights:
        return "Custom"

    # Normalize to floats for safe comparison
    normalized = {k: float(v) for k, v in cost_weights.items() if v is not None}

    # Compare with each preset
    for preset_name, preset_values in PRESETS.items():
        # Must have same keys and same values (exact match)
        if set(normalized.keys()) == set(preset_values.keys()):
            all_match = all(
                float(normalized[k]) == float(preset_values[k])
                for k in preset_values.keys()
            )
            if all_match:
                return preset_name

    # No exact match found
    return "Custom"

def make_batch_summary_string(logs_path: Path, weights_path: Path, batch_type: str):
    score_path = logs_path / "score_overview.csv"

    if not score_path.exists():
        raise FileNotFoundError(f"No score file exists at {score_path}")

    # Determine cost model
    if not weights_path.exists():
        cost_model = "Not Supported"
    else:
        cost_model = compare_to_templates(weights_path=weights_path)

    print(f"Reading scores from: {score_path}")
    df = pd.read_csv(score_path, sep=";")

    # --- Success counting ---
    success_labels = {"Success", "success", "successful"}
    is_success = df["result"].isin(success_labels)
    total = len(df)
    successes = int(is_success.sum())
    failures = total - successes

    # --- Average timestep analysis ---
    avg_time_all = df["timestep"].mean()
    avg_time_success = df.loc[is_success, "timestep"].mean() if successes > 0 else None
    avg_time_failure = df.loc[~is_success, "timestep"].mean() if failures > 0 else None

    avg_time_str = (
        f"Average time until outcome:\n"
        f"    - Overall: {avg_time_all:.1f} timesteps\n"
    )
    if avg_time_success is not None:
        avg_time_str += f"    - Success: {avg_time_success:.1f} timesteps\n"
    if avg_time_failure is not None:
        avg_time_str += f"    - Failure: {avg_time_failure:.1f} timesteps\n"

    # --- Failure analysis ---
    failure_lines = []
    if failures > 0 and "message" in df.columns:
        df_failures = df.loc[~is_success]
        message_counts = df_failures["message"].value_counts()
        for msg, count in message_counts.items():
            failure_lines.append(f"    - {msg}: {count} out of {failures}")
    else:
        failure_lines.append("    - No failures")

    # --- Final summary string ---
    summary = (
        f"---------\nBatch Simulation\n"
        f"Batch Selection: {batch_type}\n"
        f"Cost Model: {cost_model}\n"
        f"Success: {successes} out of {total}\n"
        f"{avg_time_str}"
        f"Failure Analysis (Messages):\n" +
        "\n".join(failure_lines)
    )

    return summary


# make_combined_bar_plot(logs_path=Path("/home/avsaw1/sebastian/ChaBot7/RBFN-Motion-Primitives/logs/score_overview.csv"), weights_path=Path("").resolve())
# make_combined_bar_plot(logs_path=Path("/home/avsaw1/sebastian/ChaBot7/RBFN-Motion-Primitives/logs").resolve(), weights_path=Path("none"))
# print(make_batch_summary_string(Path("/home/avsaw1/sebastian/ChaBot7/Frenetix-Motion-Planner/logs"), Path("/home/avsaw1/sebastian/ChaBot7/Frenetix-Motion-Planner/configurations/frenetix_motion_planner/cost.yaml"), "Base 100 Scenarios"))