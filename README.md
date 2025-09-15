# 🚗 LLM-MoST

---

## 📦 Installing the Overall Module

1. Clone the Git repository.
2. Create a virtual environment (referred to here as **chatbot venv**) and activate it.
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
   
---

## 🔧 Installing Necessary Submodules

Some steps require additional submodules from the **CommonRoad framework**.
Due to package conflicts—both in version requirements and script implementations—these submodules must be installed in **separate virtual environments** and executed as subprocesses.

The following sections describe how to set up each required environment.

---

### 1️⃣ SUMO Conversion Package

* Follow the instructions from the [CommonRoad Interactive Scenarios repository](https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios) to install the SUMO conversion conda environment.
* The environment must be named **exactly** as in the instructions (`cr37`), since it will be called by name when subprocesses are started.
* The location of the environment/files is irrelevant. What matters is that **LLM-MoST** can access it with a command like:

  ```bash
  conda run -n cr37 python <script_name> <--optional_inputs>
  ```

#### 🔧 Patch Required

After installation, a small patch must be applied to the installed package.

Locate the file at:

```bash
$CONDA_PREFIX/lib/pythonX.Y/site-packages/sumocr/interface/id_mapper.py
```

(where `$CONDA_PREFIX` points to the active conda environment, and `X.Y` corresponds to your Python version, e.g. 3.7).

Apply the patch:

```bash
conda activate cr37
cd $CONDA_PREFIX/lib/pythonX.Y/site-packages/sumocr/interface
patch -p1 < /absolute/path/to/LLM-MoST/id_mapper_patch.diff
```

---

### 2️⃣ Frenetix Motion Planner & Visualization Package

*(Required for base functionality, even if the motion planner is not used for evaluation.)*

#### 📋 Dependencies

Ensure the following are installed for the C++ implementation:

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

#### ⚙️ Installation Steps

1. Navigate to:

   ```bash
   cd LLM-MoST/Frenetix-Motion-Planner
   ```
2. Create a Python 3.11 virtual environment:

   ```bash
   python3.11 -m venv venv
   ```
3. Activate the environment and install:

   ```bash
   source venv/bin/activate
   pip install .
   ```

---

### 3️⃣ RBFN-Motion-Primitives

Coming soon.

---

## 🗄️ Building the Database

To add scenarios to **Chroma** and the internal storage (for XML files and GIFs), use the `add_scenarios.py` script.

* In the script, modify:

  ```python
  db.save_folder("")
  ```

  by inserting the absolute path to your folder of CommonRoad `.xml` files.

* Run the script in the **chatbot venv**.

⚠️ Note: The repository includes an **empty database**.
To enable pre-created batch simulations, download [these scenarios](https://drive.google.com/file/d/1KplwGZeh6XW3YnrK2Ch9136GDM5UDfe9/view?usp=share_link), unzip the file, and add them to the database as described above.

---

## 🤖 Adding LLMs

Configure your API keys and models in a `.env` file:

```env
DEFAULT_API_KEY=

DEFAULT_API_MODEL=gemini-2.5-flash

DEFAULT_OLLAMA_MODEL=qwen3:30b

DEFAULT_OLLAMA_URL=

DEFAULT_MODE=commercial
```

---

## ▶️ Starting the Program

1. Activate the **chatbot venv**.
2. Navigate to the project root:

   ```bash
   cd LLM-MoST/
   ```
3. Run the interface:

   ```bash
   python interface.py
   ```

