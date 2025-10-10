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
- **What it does**: Adds explicit biharmonic eddy diffusion
- **Why important**: Represents subgrid eddy mixing processes
- **Tuning range**: 0.0 - 1e5
- **Effect**: Smooths gradients, represents unresolved turbulent mixing

### 4. `smagorinsky_coeff` (default: 0.0)
- **What it does**: Enables Smagorinsky-type eddy viscosity
- **Why important**: Scale-dependent viscosity based on local strain rate
- **Tuning range**: 0.0 - 0.3 (typical ~0.15)
- **Effect**: Dynamic viscosity stronger in high-shear regions

### 5. `energy_correction` (default: 0.0)
- **What it does**: Adds/removes energy tendency
- **Why important**: Can represent backscatter (inverse cascade)
- **Tuning range**: -0.01 to +0.01
- **Effect**: Negative = backscatter energy to large scales

### 6. `enstrophy_correction` (default: 0.0)
- **What it does**: Additional enstrophy dissipation
- **Why important**: Removes small-scale vorticity fluctuations
- **Tuning range**: 0.0 - 1e-6
- **Effect**: Smooths vorticity field

## Output Files

### Pickle Files
- `highres_results.pkl` - Complete high-res simulation data
- `lowres_results.pkl` - Complete low-res simulation data  
- `comparison_metrics.pkl` - Statistical comparison metrics

### Plots
- `comparison_statistics.png` - 12-panel comparison figure showing:
  - Energy and enstrophy evolution
  - Velocity statistics
  - Energy/enstrophy spectra
  - Reynolds stresses
  - Vorticity moments
  - Bias quantification

## Comparison Metrics

The code computes these key metrics (averaged over final 25% of simulation):

1. **Energy Bias (%)**: `(E_lowres - E_highres) / E_highres * 100`
2. **Energy RMSE (%)**: Root-mean-square difference
3. **Velocity Ratio**: `RMS_velocity_lowres / RMS_velocity_highres`
4. **Enstrophy Ratio**: `Enstrophy_lowres / Enstrophy_highres`
5. **Reynolds Stress Ratio**: `<u'u'>_lowres / <u'u'>_highres`

## Typical Biases (With Default Parameters)

When low-res uses the same parameters as high-res:

- **Energy**: Typically 5-15% higher (insufficient dissipation)
- **Enstrophy**: Typically 20-40% lower (missing small scales)
- **RMS Velocity**: Typically 10-20% higher
- **Reynolds Stresses**: Typically 30-50% lower (missing eddy activity)

## How to Tune Parameters

### Manual Tuning

1. Edit `config_lowres['subgrid_params']` in `main_comparison.py`
2. Run simulation
3. Check `comparison_statistics.png`
4. Iterate until biases are minimized

### Example: Reduce Energy Bias

If low-res has too much energy:
```python
'subgrid_params': {
    'viscosity_scale': 2.5,      # Increase dissipation
    'drag_scale': 1.5,           # Increase friction
    'eddy_diffusivity': 2e4,     # Add mixing
}
```

### Example: Improve Reynolds Stresses

If low-res has too little eddy activity:
```python
'subgrid_params': {
    'smagorinsky_coeff': 0.15,   # Dynamic viscosity
    'energy_correction': -0.005, # Backscatter
}
```

## ML Optimization Strategy

The tunable parameters can be optimized using ML:

1. **Target**: Metrics from `comparison_metrics.pkl`
2. **Parameters**: 6D vector from `subgrid_params`
3. **Loss Function**: Weighted sum of relative errors
4. **Methods**: 
   - Bayesian optimization
   - Genetic algorithms
   - Gradient-free optimization
   - Neural network surrogates

## Physical Interpretation

### Why Low-Res is Biased

1. **Missing Eddies**: Cannot resolve features < 50 km
2. **Spectral Gap**: Energy piles up at grid scale
3. **Wrong Dissipation**: Fixed viscosity not scale-aware
4. **No Backscatter**: Missing inverse cascade effects

### What Tuning Accomplishes

Tuned subgrid parameters effectively:
- Represent unresolved eddy effects
- Correct energy/enstrophy budgets
- Mimic proper spectral slopes
- Reproduce correct Reynolds stresses

## Code Structure

### qg_model.py
- `QGTwoLayerModel` class
- Handles both high-res and low-res
- Applies subgrid modifications when `subgrid_params` present
- Uses Adams-Bashforth 3 time stepping

### qg_comparison.py
- `compute_spatial_stats()` - Mean, STD, skewness, kurtosis, etc.
- `compute_spectral_stats()` - Energy and enstrophy spectra
- `compute_eddy_fluxes()` - Reynolds stresses
- `plot_comparison()` - Generate comprehensive comparison plot
- `print_comparison_summary()` - Print detailed metrics

### main_comparison.py
- Define high-res and low-res configurations
- Run both simulations
- Collect detailed statistics
- Generate comparison plots
- Save all results

## Tips for Best Results

1. **Run Time**: Use 20-30 days for statistics to converge
2. **Save Interval**: 6 hours captures evolution well
3. **Initial Conditions**: Same vortices for fair comparison
4. **Spin-Up**: Discard first 25% when computing metrics
5. **Multiple Runs**: Average over different ICs for robustness

## Extending the Code

### Add New Statistics
In `qg_comparison.py`, add to `compute_spatial_stats()`:
```python
'new_metric': your_calculation(q1, q2),
```

### Add New Subgrid Terms
In `qg_model.py`, modify `rhs()` method:
```python
if self.apply_subgrid:
    new_term = self.subgrid.get('new_param', 0.0)
    dq1dt += new_term * some_operator(q1)
```

### Change Resolution
Modify `config_lowres['nx']` and `config_lowres['ny']`:
- 32x32 for extreme coarsening
- 128x128 for moderate resolution

## References

This implementation follows standard QG turbulence practices:
- Pedlosky (1987) - Geophysical Fluid Dynamics
- Vallis (2017) - Atmospheric and Oceanic Fluid Dynamics
- McWilliams (2006) - Fundamentals of Geophysical Fluid Dynamics

## Support

Questions? Check:
1. Error messages for stability issues
2. `comparison_statistics.png` for visual diagnosis
3. Print statements for parameter values
4. Pickle files for detailed time series

Happy tuning! 🌊
