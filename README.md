# ML_tuningQG

# QG Two-Layer Model: High-Res vs Low-Res Comparison

## Overview

This framework compares high-resolution (256x256) and low-resolution (64x64) quasi-geostrophic simulations to identify subgrid parameterization needs. The goal is to tune low-res parameters so it behaves like high-res, enabling ML-based optimization.

## Files

1. **qg_model.py** - Core QG model with subgrid parameterization support
2. **qg_plotting.py** - Visualization functions (from previous version)
3. **qg_comparison.py** - Statistical comparison functions
4. **main_comparison.py** - Main execution script

## Quick Start

```bash
python main_comparison.py
```

This will:
1. Run high-res simulation (256x256) as ground truth
2. Run low-res simulation (64x64) with default parameters
3. Compare statistics and generate plots
4. Save results to pickle files

## The Resolution Gap

**High-Resolution (256x256):**
- Grid spacing: ~5.9 km
- Resolves mesoscale eddies natively
- Accurate representation of turbulent cascade
- Computationally expensive

**Low-Resolution (64x64):**
- Grid spacing: ~23.4 km  
- Missing eddies < 50 km
- Needs subgrid parameterization
- 16x faster computation

## Tunable Subgrid Parameters

Located in `config_lowres['subgrid_params']` in `main_comparison.py`:

### 1. `viscosity_scale` (default: 1.0)
- **What it does**: Multiplies the hyperviscosity coefficient
- **Why important**: Low-res needs more dissipation to remove energy at unresolved scales
- **Tuning range**: 0.5 - 5.0
- **Effect**: Higher = more damping of small scales

### 2. `drag_scale` (default: 1.0)
- **What it does**: Multiplies the Ekman drag coefficient
- **Why important**: Controls energy dissipation rate in lower layer
- **Tuning range**: 0.5 - 3.0
- **Effect**: Higher = more friction, faster spin-down

### 3. `eddy_diffusivity` (default: 0.0 m²/s)
