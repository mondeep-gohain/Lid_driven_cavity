# Lid_driven_cavity
# ME 605 (CFD) Project 4: Lid-Driven Cavity Flow Solver

This repository contains the Python code for Project 4 of the **ME 605 | Computational Fluid Dynamics** course. The script (`lid_driven_cavity_flow_mg_sb.py`) implements the **SIMPLE (Semi-Implicit Method for Pressure Linked Equations)** algorithm to solve the 2D, incompressible, steady-state Navier-Stokes equations for a lid-driven cavity flow.

## 📋 Problem Setup

The simulation is configured to model the classic lid-driven cavity problem with the following parameters:

* **Problem:** 2D Lid-Driven Cavity
* **Reynolds Number (Re):** 100
* **Domain:** 1x1 square domain
* **Algorithm:** SIMPLE on a staggered grid
* **Under-relaxation:** 0.8 for velocity and pressure
* **Convergence Tolerance:** 1e-7

## ✨ Features

* **SIMPLE Algorithm:** Solves for the `u` and `v` velocity fields and the pressure (`p`) field iteratively.
* **Grid Independence Study:** The solver is designed to run on multiple grid resolutions to check for grid independence. The default sizes are `[21x21, 51x51, 81x81]`.
* **Staggered to Collocated Mapping:** Computes the solution on a staggered grid (for stability) and then interpolates the final results to a collocated grid for easy visualization.
* **Progress Monitoring:** Prints the current iteration, error, and elapsed time to the console to monitor convergence, as noted in the code comments.
* **Visualization:** Automatically generates and displays several plots to analyze the results.

## 🚀 How to Use

### 1. Prerequisites

You must have Python and the following libraries installed:
* `numpy`
* `matplotlib`

You can install them using pip:
```bash
pip install numpy matplotlib
