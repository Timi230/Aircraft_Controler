# Aircraft Dynamics Simulation

This repository contains a Python implementation for simulating and analyzing aircraft dynamics, including equilibrium point determination, short-period mode, phugoid mode analysis, and LQR-based controller synthesis.


Co-worker : Tristan MONSELLIER / Titouan MILLET

## Prerequisites

To use this project, you need to have:

1. **Miniconda Environment**: Ensure Miniconda is installed on your system. You can download it from [Miniconda&#39;s website](https://docs.conda.io/en/latest/miniconda.html).
2. Python version == 3.11.

## Installation Steps

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Timi230/Aircraft_Controler.git
   ```
2. Create and activate a conda environment:

   ```bash
   conda create -n aircraft_env python=3.11 -y
   conda activate aircraft_env
   ```
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Navigate to the root directory of the project.
2. Execute the `main.py` file to run the simulation and analyses:

   ```bash
   python main.py
   ```
3. The program will compute:

   - Equilibrium points.
   - Short-period and phugoid modes.
   - Step response visualizations.
   - Controller synthesis using LQR.
4. Outputs, including plots and calculations, will be displayed in the terminal and pop-up windows.

## Documentation

A detailed report of the analysis and methodology is available [here](https://github.com/Timi230/Aircraft_Controler/blob/main/Au511_report_MILLET_MONSELLIER.pdf). It includes theoretical explanations, simulation results, and controller design discussions.

## Notes

- Ensure all dependencies in `requirements.txt` are correctly installed before running the program.
- Modify parameters in `data.py` to customize simulations for different scenarios.
