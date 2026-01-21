"""
Main Script for High-Res vs Low-Res QG Comparison
Run both simulations and compare results
"""

import numpy as np
import pickle
from qg_model import QGTwoLayerModel
from qg_comparison import (compute_spectral_stats, compute_spatial_stats, 
                            compute_eddy_fluxes, plot_comparison, 
                            print_comparison_summary)

# ============================================================================
# CONFIGURATION: HIGH RESOLUTION (Ground Truth)
# ============================================================================

config_highres = {
    'name': 'HighRes_256x256',
    'nx': 256,
    'ny': 256,
    'Lx': 1.5e6,                  # 1500 km domain
    'Ly': 1.5e6,
    
    # Physical parameters
    'beta': 1.6e-11,              # Beta parameter (1/m/s)
    'f0': 1e-4,                   # Coriolis parameter (1/s)
    'g_prime': 0.02,              # Reduced gravity (m/s^2)
    'H1': 500.0,                  # Upper layer depth (m)
    'H2': 3500.0,                 # Lower layer depth (m)
    
    # Dissipation (tuned for 256x256)
    'r_drag': 1e-7,               # Ekman drag (1/s)
    'nu': 5e14,                   # Hyperviscosity (m^4/s)
    
    # Time stepping
    'dt': 1800.0,                 # 30 minutes
}

# ============================================================================
# CONFIGURATION: LOW RESOLUTION (With Subgrid Parameters to Tune)
# ============================================================================

config_lowres = {
    'name': 'LowRes_64x64',
    'nx': 64,                     # 4x coarser in each direction
    'ny': 64,
    'Lx': 1.5e6,                  # SAME physical domain
    'Ly': 1.5e6,
    
    # Physical parameters (IDENTICAL to high-res)
    'beta': 1.6e-11,
    'f0': 1e-4,
    'g_prime': 0.02,
    'H1': 500.0,
    'H2': 3500.0,
    
    # Dissipation (SAME as high-res - creates bias!)
    'r_drag': 1e-7,
    'nu': 5e14,
    
    # Time stepping
    'dt': 1800.0,
    
    # ========================================================================
    # TUNABLE SUBGRID PARAMETERS (For ML Optimization)
    # ========================================================================
    'subgrid_params': {
        # Each parameter modifies the model to better represent unresolved scales
        
        'viscosity_scale': 1.0,
        # What: Multiplies hyperviscosity coefficient
        # Why: Low-res needs more dissipation to remove energy at unresolved scales
        # Range: 0.5 - 5.0 (higher = more damping)
        
        'drag_scale': 1.0,
        # What: Multiplies Ekman drag coefficient
        # Why: Affects energy dissipation rate in lower layer
        # Range: 0.5 - 3.0 (higher = more friction)
        
        'eddy_diffusivity': 0.0,
        # What: Additional biharmonic diffusion (m^2/s)
        # Why: Represents subgrid eddy mixing
        # Range: 0.0 - 1e5 (higher = more mixing)
        
        'smagorinsky_coeff': 0.0,
        # What: Smagorinsky eddy viscosity coefficient
        # Why: Scale-dependent viscosity based on local strain rate
        # Range: 0.0 - 0.3 (typical ~0.15)
        
        'energy_correction': 0.0,
        # What: Backscatter energy from subgrid to resolved scales
        # Why: Represents upscale energy transfer (inverse cascade)
        # Range: -0.01 - 0.01 (negative = backscatter)
        
        'enstrophy_correction': 0.0,
        # What: Additional enstrophy dissipation rate
        # Why: Removes small-scale vorticity fluctuations
        # Range: 0.0 - 1e-6 (higher = more smoothing)
    }
}

# ============================================================================
# INITIAL CONDITIONS (Same for both resolutions)
# ============================================================================

def create_initial_conditions(nx, ny, Lx, Ly):
    """Create identical initial vortices scaled to grid"""
    x = np.arange(nx) * (Lx / nx)
    y = np.arange(ny) * (Ly / ny)
    X, Y = np.meshgrid(x, y)
    
    # Define vortices as fractions of domain
    vortex_params = [
        {'x': 0.5*Lx, 'y': 0.5*Ly, 'sigma': 0.15*Lx, 'amp1': 1e-6, 'amp2': 5e-7},
        {'x': 0.3*Lx, 'y': 0.7*Ly, 'sigma': 0.12*Lx, 'amp1': -8e-7, 'amp2': 0.0},
        {'x': 0.7*Lx, 'y': 0.3*Ly, 'sigma': 0.13*Lx, 'amp1': 0.0, 'amp2': 6e-7},
    ]
    
    q1 = np.zeros((ny, nx))
    q2 = np.zeros((ny, nx))
    
    for vortex in vortex_params:
        x0, y0 = vortex['x'], vortex['y']
        sigma = vortex['sigma']
        amp1, amp2 = vortex['amp1'], vortex['amp2']
        
        r2 = (X - x0)**2 + (Y - y0)**2
        gauss = np.exp(-r2 / (2 * sigma**2))
        
        q1 += amp1 * gauss
        q2 += amp2 * gauss
    
    return q1, q2

# ============================================================================
# RUN SIMULATION
# ============================================================================

def run_simulation(config, sim_days=20, save_interval_hours=6):
    """Run a single simulation and collect statistics"""
    
    print("\n" + "="*70)
    print(f"Running {config['name']} Simulation")
    print("="*70)
    print(f"Grid: {config['nx']} x {config['ny']}")
    print(f"Resolution: {config['Lx']/config['nx']/1e3:.1f} km per grid point")
    
    if 'subgrid_params' in config:
        print("\nSubgrid Parameters:")
        for key, val in config['subgrid_params'].items():
            print(f"  {key}: {val}")
    
    # Initialize model
    model = QGTwoLayerModel(config)
    q1, q2 = create_initial_conditions(config['nx'], config['ny'], 
                                       config['Lx'], config['Ly'])
    
    # Time stepping setup
    steps_per_day = int(86400 / config['dt'])
    total_steps = sim_days * steps_per_day
    save_every = int(save_interval_hours * 3600 / config['dt'])
    
    # Storage
    q1_history = [q1.copy()]
    q2_history = [q2.copy()]
    times = [0.0]
    
    energy_history = {'total': [], 'layer1': [], 'layer2': []}
    enstrophy_history = {'layer1': [], 'layer2': []}
    spatial_stats_history = []
    spectral_stats_history = []
    eddy_flux_history = []
    
    # Initial diagnostics
    KE1, KE2, KE = model.compute_energy(q1, q2)
    ens1, ens2 = model.compute_enstrophy(q1, q2)
    
    energy_history['total'].append(KE)
    energy_history['layer1'].append(KE1)
    energy_history['layer2'].append(KE2)
    enstrophy_history['layer1'].append(ens1)
    enstrophy_history['layer2'].append(ens2)
    
    spatial_stats = compute_spatial_stats(model, q1, q2)
    spatial_stats_history.append(spatial_stats)
    
    k, KE_spec, Z_spec = compute_spectral_stats(model, q1, q2)
    spectral_stats_history.append({'k': k, 'KE': KE_spec, 'Z': Z_spec})
    
    fluxes = compute_eddy_fluxes(model, q1, q2)
    eddy_flux_history.append(fluxes)
    
    print(f"\nInitial Energy: {KE:.3e}")
    print(f"Initial Enstrophy: {ens1+ens2:.3e}")
    print("\nIntegrating...")
    
    # Adams-Bashforth history
    dq1_hist = []
    dq2_hist = []
    
    for step in range(1, total_steps + 1):
        # Time step
        q1, q2 = model.step_ab3(q1, q2, dq1_hist, dq2_hist)
        
        # Check stability
        if not (np.isfinite(q1).all() and np.isfinite(q2).all()):
            print(f"\n*** Unstable at step {step} ***")
            break
        
        # Save diagnostics
        if step % save_every == 0:
            t = step * config['dt'] / 86400
            times.append(t)
            q1_history.append(q1.copy())
            q2_history.append(q2.copy())
            
            # Energy and enstrophy
            KE1, KE2, KE = model.compute_energy(q1, q2)
            ens1, ens2 = model.compute_enstrophy(q1, q2)
            
            energy_history['total'].append(KE)
            energy_history['layer1'].append(KE1)
            energy_history['layer2'].append(KE2)
            enstrophy_history['layer1'].append(ens1)
            enstrophy_history['layer2'].append(ens2)
            
            # Spatial statistics
            spatial_stats = compute_spatial_stats(model, q1, q2)
            spatial_stats_history.append(spatial_stats)
            
            # Spectral statistics
            k, KE_spec, Z_spec = compute_spectral_stats(model, q1, q2)
            spectral_stats_history.append({'k': k, 'KE': KE_spec, 'Z': Z_spec})
            
            # Eddy fluxes
            fluxes = compute_eddy_fluxes(model, q1, q2)
            eddy_flux_history.append(fluxes)
            
            # Progress update
            if step % (save_every * 4) == 0:
                print(f"  Day {t:5.1f} | Energy: {KE:.3e} | Enstrophy: {ens1+ens2:.3e}")
    
    print("\n" + "="*70)
    print(f"{config['name']} Simulation Complete!")
    print("="*70)
    
    # Package results
    results = {
        'config': config,
        'model': model,
        'times': np.array(times),
        'q1_history': q1_history,
        'q2_history': q2_history,
        'energy_history': energy_history,
        'enstrophy_history': enstrophy_history,
        'spatial_stats_history': spatial_stats_history,
        'spectral_stats_history': spectral_stats_history,
        'eddy_flux_history': eddy_flux_history,
    }
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run both simulations and compare"""
    
    print("\n" + "="*70)
    print("QG TWO-LAYER MODEL: HIGH-RES vs LOW-RES COMPARISON")
    print("="*70)
    print("\nThis script demonstrates the resolution bias in coarse models.")
    print("The goal: tune low-res subgrid parameters to match high-res statistics.")
    
    # Print tunable parameters
    print("\n" + "="*70)
    print("TUNABLE SUBGRID PARAMETERS (for ML optimization)")
    print("="*70)
    for param, value in config_lowres['subgrid_params'].items():
        print(f"\n{param}:")
        print(f"  Current value: {value}")
        # Add descriptions
        if param == 'viscosity_scale':
            print(f"  Purpose: Scale hyperviscosity to control small-scale damping")
            print(f"  Suggested range: 0.5 - 5.0")
        elif param == 'drag_scale':
            print(f"  Purpose: Scale Ekman drag to control energy dissipation")
            print(f"  Suggested range: 0.5 - 3.0")
        elif param == 'eddy_diffusivity':
            print(f"  Purpose: Add explicit eddy diffusion (m^2/s)")
            print(f"  Suggested range: 0.0 - 1e5")
        elif param == 'smagorinsky_coeff':
            print(f"  Purpose: Dynamic eddy viscosity based on strain rate")
            print(f"  Suggested range: 0.0 - 0.3 (typical ~0.15)")
        elif param == 'energy_correction':
            print(f"  Purpose: Backscatter energy to resolved scales")
            print(f"  Suggested range: -0.01 - 0.01")
        elif param == 'enstrophy_correction':
            print(f"  Purpose: Additional enstrophy dissipation")
            print(f"  Suggested range: 0.0 - 1e-6")
    
    print("\n" + "="*70)
    input("\nPress Enter to start simulations...")
    
    # Run high-resolution simulation
    highres_results = run_simulation(config_highres, sim_days=20, save_interval_hours=6)
    
    # Save high-res results
    with open('highres_results.pkl', 'wb') as f:
        pickle.dump(highres_results, f)
    print(f"\nHigh-res results saved to highres_results.pkl")
    
    # Run low-resolution simulation
    lowres_results = run_simulation(config_lowres, sim_days=20, save_interval_hours=6)
    
    # Save low-res results
    with open('lowres_results.pkl', 'wb') as f:
        pickle.dump(lowres_results, f)
    print(f"\nLow-res results saved to lowres_results.pkl")
    
    # Compare and plot
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS...")
    print("="*70)
    
    metrics = plot_comparison(highres_results, lowres_results, 'comparison_statistics.png')
    
    # Print detailed summary
    print_comparison_summary(highres_results, lowres_results, metrics)
    
    # Save comparison metrics
    with open('comparison_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("\nComparison metrics saved to comparison_metrics.pkl")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Examine comparison_statistics.png to see resolution biases")
    print("2. Modify config_lowres['subgrid_params'] in this script")
    print("3. Re-run to see if tuned parameters reduce bias")
    print("4. Use ML to automatically find optimal parameter values")
    print("="*70)
    
    return highres_results, lowres_results, metrics

if __name__ == "__main__":
    results = main()
