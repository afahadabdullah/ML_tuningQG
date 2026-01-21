# ML_tuningQG: High-Res vs Low-Res QG Turbulence

## Overview

This framework compares high-resolution (256x256) and low-resolution (64x64) quasi-geostrophic (QG) simulations to identify subgrid parameterization needs. The goal is to tune low-res parameters so it behaves like high-res, enabling ML-based optimization.

[![QG Simulation Demo](https://img.youtube.com/vi/pmDJhbrb-0E/0.jpg)](https://www.youtube.com/shorts/pmDJhbrb-0E)
*Click to watch a demo of the simulation.*

## Directory Structure

- **`src/`**: Core Python models and scripts.
- **`notebooks/`**: Jupyter notebooks for exploration and analysis.
- **`figures/`**: Generated plots and results.
- **`docs/`**: Experimental plans and additional documentation.

## Quick Start

1.  **Installation**:
    Ensure you have Python installed with `numpy`, `scipy`, `matplotlib`, and `xarray`.

2.  **Running the Comparison**:
    Navigate to the `src` directory and run the main comparison script:

    ```bash
    cd src
    python main_comparison.py
    ```

    This will:
    - Run a **High-Resolution (256x256)** simulation as ground truth.
    - Run a **Low-Resolution (64x64)** simulation with default parameters.
    - Compare statistics and generate plots in the `figures/` directory (or locally, depending on script config).
    - Save results to pickle files.

## The Resolution Gap

- **High-Resolution (256x256)**: Grid spacing ~5.9 km. Resolves mesoscale eddies natively. Accurate but expensive.
- **Low-Resolution (64x64)**: Grid spacing ~23.4 km. Missing eddies < 50 km. Needs subgrid parameterization. 16x faster.

## Tunable Subgrid Parameters

Located in `config_lowres['subgrid_params']` in `src/main_comparison.py`:

1.  **`viscosity_scale`**: Multiplies hyperviscosity (damping).
2.  **`drag_scale`**: Multiplies Ekman drag (friction).
3.  **`eddy_diffusivity`**: Adds explicit subgrid mixing.
4.  **`smagorinsky_coeff`**: Dynamic scale-dependent viscosity.
5.  **`energy_correction`**: Backscatter energy to large scales.
6.  **`enstrophy_correction`**: Dissipates small-scale vorticity.

## ML Optimization Strategy

The tunable parameters are optimized using various ML techniques (Bayesian Opt, Genetic Algorithms, etc.) to minimize the error between Low-Res and High-Res statistics.

See `docs/experiments_and_data.md` for detailed experimental plans.

## References
- Pedlosky (1987) - Geophysical Fluid Dynamics
- Vallis (2017) - Atmospheric and Oceanic Fluid Dynamics
