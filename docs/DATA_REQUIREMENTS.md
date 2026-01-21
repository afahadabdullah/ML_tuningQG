# Data Requirements for Publication Figures

This document specifies **exactly what data** each experiment must save for the publication figures.

---

## Quick Reference: Files to Generate

```
ML_tuning/
├── highres_results.pkl           ✅ (Already have - 763MB)
├── sensitivity_matrix.json       ✅ (Already have)
├── fidelity_data.pkl             ⚠️ (Need more samples - currently 5, need 20-50)
│
├── exp2_gp_results.pkl           ❌ Need from Exp2
├── exp3_nn_results.pkl           ❌ Need from Exp3  
├── exp5_gfgp_results.pkl         ❌ Need from Exp5
├── exp6_gfnn_results.pkl         ❌ Need from Exp6
│
└── lowres_default_results.pkl    ❌ Need (180d run with default params)
```

---

## Figure 1: High-Res Dynamics & Problem Setup

### Data Structure: `highres_results.pkl` ✅

```python
{
    'q1_history': List[np.ndarray],  # PV field snapshots [N_snapshots x nx x ny]
    'times': List[float],             # Simulation times in days
    'config': {
        'nx': int,      # Grid resolution (e.g., 256)
        'ny': int,
        'Lx': float,    # Domain size in meters
        'Ly': float,
    }
}
```

### Data Structure: `lowres_default_results.pkl` ❌

```python
# Run: 180 days with DEFAULT parameters at low resolution
{
    'q1_history': List[np.ndarray],  
    'times': List[float],            
    'config': dict,
    'params': dict,  # The default parameter values used
}
```

**How to generate:**
```python
from main_comparison import run_simulation, config_lowres
from exp6_GFNN import DEFAULT_PARAMS

config = config_lowres.copy()
config['subgrid_params'] = DEFAULT_PARAMS
results = run_simulation(config, sim_days=180, save_interval_hours=120)
results['params'] = DEFAULT_PARAMS

import pickle
with open('lowres_default_results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

---

## Figure 2: Parameter Sensitivity Ranking

### Data Structure: `sensitivity_matrix.json` ✅

```json
[
    {"parameter": "viscosity_scale", "sensitivity_score": 0.15},
    {"parameter": "drag_scale", "sensitivity_score": 0.08},
    {"parameter": "eddy_diffusivity", "sensitivity_score": 0.27},
    {"parameter": "smagorinsky_coeff", "sensitivity_score": 0.05},
    {"parameter": "energy_correction", "sensitivity_score": 0.12},
    {"parameter": "enstrophy_correction", "sensitivity_score": 0.03}
]
```

---

## Figure 3: Fidelity Correlation (Multi-Fidelity Justification)

### Data Structure: `fidelity_data.pkl` ⚠️

**Need 20-50 samples** (currently only 5). Each sample = same params run at BOTH fidelities.

```python
{
    'loss_30': List[float],    # Loss from 30-day runs (n_samples,)
    'loss_180': List[float],   # Loss from 180-day runs (n_samples,)
    'params': List[dict],      # Optional: The parameter configs tested
}
```

**How to generate:**
```python
# In your experiment script, after LHS sampling:
for params in lhs_samples:
    loss_30 = run_and_evaluate(params, sim_days=30)
    loss_180 = run_and_evaluate(params, sim_days=180)
    
    fidelity_data['loss_30'].append(loss_30)
    fidelity_data['loss_180'].append(loss_180)
    fidelity_data['params'].append(params)
```

---

## Figure 4: Computational Efficiency Comparison

### Data Structure: Each experiment should save `exp{N}_{method}_results.pkl`

```python
{
    # === REQUIRED FOR CONVERGENCE PLOT ===
    'y_samples': List[float],       # Loss at each iteration (length = n_iters)
    'x_samples': List[List[float]], # Parameter vectors (n_iters x n_params)
    
    # === REQUIRED FOR COST ACCOUNTING ===
    'sim_days_per_iter': List[int], # Days simulated at each iteration (30 or 180)
    'wall_times': List[float],      # Wall clock time per iteration (seconds)
    
    # === FINAL RESULTS ===
    'best_params': dict,            # Optimal parameters found
    'best_loss': float,             # Best loss achieved
    
    # === METADATA ===
    'method_name': str,             # e.g., "Hybrid GF-NN"
    'total_iterations': int,
    'total_sim_days': int,          # Cumulative sim days (cost metric)
    'total_wall_time': float,       # Total wall clock time
    
    # === CRITICAL FOR HYBRID METHODS (Exp5, Exp6) ===
    # This enables the multi-fidelity transition visualization
    'phases': {
        'screening': {
            'n_samples': int,           # Number of screening samples (REQUIRED)
            'sim_days': int,            # Fidelity level during screening (e.g., 30)
            'active_params': List[str]  # Parameters identified as active
        },
        'optimization': {
            'n_samples': int,           # Number of optimization samples
            'fidelity_schedule': List[int]  # sim_days per iteration
        }
    },
    
    # === OPTIONAL: For spectra comparison ===
    'best_run_results': {
        'q1_history': List[np.ndarray],  # Final validation run snapshots
        'times': List[float],
    }
}
```

---

## Figure 5: Energy Spectra Comparison

Uses data from Figure 1 + the `best_params` from each optimization method.

**Additional data needed in each experiment result:**
```python
{
    # ... other fields ...
    'best_run_results': {
        'q1_history': List[np.ndarray],  # Final validation run snapshots
        'times': List[float],
    }
}
```

**OR** just save `best_params` and the plotting script will re-run the simulation.

---

## Summary Table: What Each Experiment Must Save

| Experiment | Script | Output File | Key Fields |
|------------|--------|-------------|------------|
| **Exp 0 (Truth)** | `generate_highres.py` | `highres_results.pkl` | q1_history, times, config |
| **Exp 0 (Default)** | (manual run) | `lowres_default_results.pkl` | q1_history, times, params |
| **Exp 1 (GF)** | `exp1_GF.py` | `sensitivity_matrix.json` | parameter, sensitivity_score |
| **Exp 2 (GP)** | `exp2_GP.py` | `exp2_gp_results.pkl` | y_samples, best_params, sim_days_per_iter |
| **Exp 3 (NN)** | `exp3_NN.py` | `exp3_nn_results.pkl` | y_samples, best_params, sim_days_per_iter |
| **Exp 5 (GF-GP)** | `exp5_hybridGFGP.py` | `exp5_gfgp_results.pkl` | y_samples, best_params, phases |
| **Exp 6 (GF-NN)** | `exp6_GFNN.py` | `exp6_gfnn_results.pkl` | y_samples, best_params, phases |
| **Fidelity Valid** | (in any exp) | `fidelity_data.pkl` | loss_30, loss_180 (20-50 samples) |

---

## Saving Code Template

Add this to your experiment scripts:

```python
def save_experiment_results(
    y_samples, x_samples, best_params, best_loss,
    sim_days_per_iter, wall_times, method_name, 
    phases=None, best_run_results=None,
    filename=None
):
    """Standard save format for publication figures."""
    results = {
        'y_samples': y_samples,
        'x_samples': x_samples,
        'best_params': best_params,
        'best_loss': best_loss,
        'sim_days_per_iter': sim_days_per_iter,
        'wall_times': wall_times,
        'method_name': method_name,
        'total_iterations': len(y_samples),
        'total_sim_days': sum(sim_days_per_iter),
        'total_wall_time': sum(wall_times),
        'phases': phases,
        'best_run_results': best_run_results,
        'timestamp': datetime.now().isoformat()
    }
    
    if filename is None:
        filename = f"{method_name.lower().replace(' ', '_')}_results.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {filename}")
```
