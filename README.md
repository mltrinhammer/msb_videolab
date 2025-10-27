# msb_videolab
This repo contains all the scripts needed to run various algorithms for data cleaning, data alignment and other for Professor Uwe Altmann's lab at the Medical School Berlin.
Each folder contain either a Python or an R script (or both) to execute the process that is described in the README.md file of that repo.

## To start
Make sure you have downloaded git (choose all default suggestions) and (optionally) VS Code.
Remember to install a Python environment if you plan to execute the Python files.

Below are platform-specific quick-start instructions for Windows (PowerShell) and macOS (bash / zsh). Choose the section matching your environment.

### Windows (PowerShell)
- Create a virtual environment (uses the system `python`):

	`python -m venv msb`

- Activate the virtual environment:

	`.\msb\Scripts\activate`

- Install required Python packages:

	`pip install pandas numpy openpyxl pyyaml`

### macOS (bash / zsh)

- Create a virtual environment (use `python3` if your system maps `python` to Python 2):

	`python3 -m venv msb`

- Activate the virtual environment:

	`source msb/bin/activate`

- Install required Python packages (use `pip` from the activated venv; `pip` will point to the correct interpreter):

	`pip install pandas numpy openpyxl pyyaml`

Notes:
- On macOS it's common to use `python3` and `pip3`. 

### R (both Windows and macOS)
For the R scripts used in some folders, install the required packages from within R or RStudio:

`install.packages(c("readxl", "data.table", "zoo", "optparse"))`

If you want, you can also create and use an R project in RStudio which manages library paths per-project.

