# SPE Application: Poisson Process Lecture Materials

This repository contains lecture materials and code for a hands-on introduction to the Poisson process within Software Performance Engineering (SPE). The goal is to bridge theory and practice by generating Poisson workloads, validating their statistical properties, and connecting them to M/M/1 queueing theory predictions.

## Repository Contents
- `poisson_spe_lecture.ipynb`: Interactive Jupyter notebook used during the lecture. It
  - introduces Poisson process fundamentals via transition probabilities
  - demonstrates simulation and visualization of arrivals and inter-arrival times
  - validates exponential behavior and fits using statistical tests
  - explores the link to M/M/1 formulas (utilization, response time, queue length)
- `poisson_plots.py`: Reusable plotting and analysis utilities for the notebook (distribution plots, simulation comparisons, inter-arrival analyses, etc.).
- `outline_poisson_lecture.md`: Lecture outline, learning objectives, and program.
- `requirements.txt`: Minimal Python dependencies to run the notebook and utilities.

## Objectives
- Implement a Poisson workload generator.
- Validate statistical properties (Poisson counts, exponential inter-arrivals).
- Compare simulation results with M/M/1 theoretical predictions.

## Quick Start
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd SPE_Application_1
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Register a dedicated Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name spe-lecture --display-name "Python (spe-lecture)"
   ```
5. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
   In Jupyter/VS Code, select the kernel "Python (spe-lecture)" if you created it.

## Usage
- Run cells in `poisson_spe_lecture.ipynb` sequentially.
- The notebook imports utilities from `poisson_plots.py` for plotting and validation.
- Adjust parameters like `lambda_rate`, time window `tau`, and experiment sizes to explore different regimes and compare with M/M/1 predictions.

## Notes
- Avoid installing packages directly inside the notebook with `!pip install` for reproducibility; prefer the virtual environment + `requirements.txt` approach.
- If you prefer a modern workflow with lock files, consider using Poetry or uv and a `pyproject.toml`.

## License
See `LICENSE` for license information.
