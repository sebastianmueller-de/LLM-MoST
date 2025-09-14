# LLM-MoST
Code coming soon.

---

## Installing the Overall Module
1. Clone the Git repository.
2. Create a venv (will be referred to here as "chatbot venv") and activate it.
3. Run:
   ```bash 
   pip install -r requirements.txt

---

## Installing Necessary Submodules

For certain steps of the process, other submodules from the **CommonRoad framework** are required.
Due to package conflicts—both in required version imports and in slightly differing script implementations—these submodules must be executed in their own virtual environments, and are called as subprocesses.

This section lists the necessary steps to install the required environments.

---

### 1. SUMO Conversion Package

* Follow the instructions from the [CommonRoad Interactive Scenarios repository](https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios) to install the SUMO conversion conda environment.
* The environment must be named **exactly** as in the instructions (`cr37`), since it will be called by name when a required subprocess is started.
* The storage location of the environment and files is irrelevant. What matters is that the **LLM-MoST** program can access it via a command like:

  ```bash
  conda run -n cr37 python <script_name> <--optional_inputs>
  ```

After that, a small patch needs to be implemented. A file in the installed conda package needs to be changed. It is located at 

```bash
$CONDA_PREFIX/lib/pythonX.Y/site-packages/sumocr/interface/id_mapper.py
```

(where $CONDA_PREFIX points to the active Conda environment, and X.Y depends on your Python version, e.g. 3.7).

To fix it, run:

```bash
conda activate cr37
cd $CONDA_PREFIX/lib/pythonX.Y/site-packages/sumocr/interface
patch -p1 < /absolute/path/to/LLM-MoST/id_mapper_patch.diff
```

---

### 2. Frenetix Motion Planner & Visualization Package

*(Required for base functionality, even if the motion planner is not used for evaluation.)*

#### Dependencies

Ensure the following dependencies are installed on your system for the C++ implementation:

* **Eigen3**

  ```bash
  sudo apt-get install libeigen3-dev
  ```
* **Boost**

  ```bash
  sudo apt-get install libboost-all-dev
  ```
* **OpenMP**

  ```bash
  sudo apt-get install libomp-dev
  ```
* **Python 3.11 (full & dev)**

  ```bash
  sudo apt-get install python3.11-full
  sudo apt-get install python3.11-dev
  ```

#### Installation Steps

1. Navigate to the `LLM-MoST/Frenetix-Motion-Planner` directory.
2. Create a virtual environment:

   ```bash
   python3.11 -m venv venv
   ```
3. Activate and install the package:

   ```bash
   source venv/bin/activate
   pip install .
   ```

---

### 3. RBFN-Motion-Primitives
Coming soon. 

---

## Building the Data Base

To add scenario to Chroma and the internal directory used for storing XML files and GIFs, use the add_scenarios.py script. Where it says db.save_folder(""), simply put in the absolute path to your folder of CommonRoad .xml files in the quotes. Then, run the script (in the chatbot venv).

---

## Adding LLMs

Use a .env file of the following style:

DEFAULT_API_KEY=

DEFAULT_API_MODEL=gemini-2.5-flash


DEFAULT_OLLAMA_MODEL=qwen3:30b

DEFAULT_OLLAMA_URL=


DEFAULT_MODE=commercial

## Starting the program
To start the program, activate the chatbot venv, navigate to the LLM-MoST/ directory, and run "python interface.py"