"""
Statistical Comparison Functions
Compare high-res and low-res simulations
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2

def compute_spectral_stats(model, q1, q2):
    """Compute spectral energy and enstrophy"""
    psi1, psi2 = model.q_to_psi(q1, q2)
    
    q1h = fft2(q1)
    q2h = fft2(q2)
    psi1h = fft2(psi1)
    psi2h = fft2(psi2)
    
    # Energy spectrum
    KE1 = 0.5 * model.H1 * (np.abs(model.KX * psi1h)**2 + np.abs(model.KY * psi1h)**2)
    KE2 = 0.5 * model.H2 * (np.abs(model.KX * psi2h)**2 + np.abs(model.KY * psi2h)**2)
    
    # Enstrophy spectrum
    Z1 = 0.5 * np.abs(q1h)**2
    Z2 = 0.5 * np.abs(q2h)**2
    
    # Radial averaging
    k = np.sqrt(model.KX**2 + model.KY**2)
    k_bins = np.arange(0, np.max(k), np.max(k)/20)
    
    KE_spectrum = np.zeros(len(k_bins)-1)
    Z_spectrum = np.zeros(len(k_bins)-1)
    
    for i in range(len(k_bins)-1):
        mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        if np.any(mask):
            KE_spectrum[i] = np.mean((KE1 + KE2)[mask])
            Z_spectrum[i] = np.mean((Z1 + Z2)[mask])
    
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    return k_centers, KE_spectrum, Z_spectrum

def compute_spatial_stats(model, q1, q2):
    """Compute spatial statistics"""
    psi1, psi2 = model.q_to_psi(q1, q2)
    u1, v1 = model.compute_velocity(psi1)
    u2, v2 = model.compute_velocity(psi2)
    
    stats = {
        # Vorticity statistics
        'q1_mean': np.mean(q1),
        'q1_std': np.std(q1),
        'q1_skew': np.mean((q1 - np.mean(q1))**3) / (np.std(q1)**3 + 1e-20),
        'q1_kurt': np.mean((q1 - np.mean(q1))**4) / (np.std(q1)**4 + 1e-20),
        'q2_mean': np.mean(q2),
        'q2_std': np.std(q2),
        'q2_skew': np.mean((q2 - np.mean(q2))**3) / (np.std(q2)**3 + 1e-20),
        'q2_kurt': np.mean((q2 - np.mean(q2))**4) / (np.std(q2)**4 + 1e-20),
        
        # Velocity statistics
        'u1_rms': np.sqrt(np.mean(u1**2)),
        'v1_rms': np.sqrt(np.mean(v1**2)),
        'u2_rms': np.sqrt(np.mean(u2**2)),
        'v2_rms': np.sqrt(np.mean(v2**2)),
        
        # Extrema
        'q1_max': np.max(np.abs(q1)),
        'q2_max': np.max(np.abs(q2)),
        
        # Spatial gradients
        'grad_q1': np.sqrt(np.mean(np.gradient(q1, axis=0)**2 + np.gradient(q1, axis=1)**2)),
        'grad_q2': np.sqrt(np.mean(np.gradient(q2, axis=0)**2 + np.gradient(q2, axis=1)**2)),
    }
    
    return stats

def compute_eddy_fluxes(model, q1, q2):
    """Compute Reynolds stresses"""
    psi1, psi2 = model.q_to_psi(q1, q2)
    u1, v1 = model.compute_velocity(psi1)
    u2, v2 = model.compute_velocity(psi2)
    
    # Eddy components
    u1_eddy = u1 - np.mean(u1)
    v1_eddy = v1 - np.mean(v1)
    u2_eddy = u2 - np.mean(u2)
    v2_eddy = v2 - np.mean(v2)
    
    fluxes = {
        'uu_layer1': np.mean(u1_eddy * u1_eddy),
        'vv_layer1': np.mean(v1_eddy * v1_eddy),
        'uv_layer1': np.mean(u1_eddy * v1_eddy),
        'uu_layer2': np.mean(u2_eddy * u2_eddy),
        'vv_layer2': np.mean(v2_eddy * v2_eddy),
        'uv_layer2': np.mean(u2_eddy * v2_eddy),
    }
    
    return fluxes

def plot_comparison(highres_results, lowres_results, output_file='comparison.png'):
    """Create comprehensive comparison plots"""
    
    fig = plt.figure(figsize=(20, 14))
    
    times_hr = highres_results['times']
    times_lr = lowres_results['times']
    
    # Row 1: Energy and Enstrophy
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(times_hr, highres_results['energy_history']['total'], 'b-', linewidth=2, label='High-Res (256x256)')
    ax1.plot(times_lr, lowres_results['energy_history']['total'], 'r--', linewidth=2, label='Low-Res (64x64)')
    ax1.set_xlabel('Time (days)', fontsize=11)
    ax1.set_ylabel('Total Energy', fontsize=11)
    ax1.set_title('Total Kinetic Energy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    ens_hr = np.array(highres_results['enstrophy_history']['layer1']) + \
             np.array(highres_results['enstrophy_history']['layer2'])
    ens_lr = np.array(lowres_results['enstrophy_history']['layer1']) + \
             np.array(lowres_results['enstrophy_history']['layer2'])
    ax2.plot(times_hr, ens_hr, 'b-', linewidth=2, label='High-Res')
    ax2.plot(times_lr, ens_lr, 'r--', linewidth=2, label='Low-Res')
    ax2.set_xlabel('Time (days)', fontsize=11)
    ax2.set_ylabel('Total Enstrophy', fontsize=11)
    ax2.set_title('Total Enstrophy', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 4, 3)
    q_std_hr = [s['q1_std'] for s in highres_results['spatial_stats_history']]
    q_std_lr = [s['q1_std'] for s in lowres_results['spatial_stats_history']]
    ax3.plot(times_hr, q_std_hr, 'b-', linewidth=2, label='High-Res')
    ax3.plot(times_lr, q_std_lr, 'r--', linewidth=2, label='Low-Res')
    ax3.set_xlabel('Time (days)', fontsize=11)
    ax3.set_ylabel('Vorticity STD', fontsize=11)
    ax3.set_title('Layer 1 Vorticity Variability', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 4, 4)
    urms_hr = [s['u1_rms'] for s in highres_results['spatial_stats_history']]
    urms_lr = [s['u1_rms'] for s in lowres_results['spatial_stats_history']]
    ax4.plot(times_hr, urms_hr, 'b-', linewidth=2, label='High-Res')
    ax4.plot(times_lr, urms_lr, 'r--', linewidth=2, label='Low-Res')
    ax4.set_xlabel('Time (days)', fontsize=11)
    ax4.set_ylabel('RMS Velocity (m/s)', fontsize=11)
    ax4.set_title('Layer 1 RMS Velocity', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Row 2: Spectra
    ax5 = plt.subplot(3, 4, 5)
    spec_hr = highres_results['spectral_stats_history'][-1]
    spec_lr = lowres_results['spectral_stats_history'][-1]
    ax5.loglog(spec_hr['k'], spec_hr['KE'], 'b-', linewidth=2, label='High-Res')
    ax5.loglog(spec_lr['k'], spec_lr['KE'], 'r--', linewidth=2, label='Low-Res')
    ax5.set_xlabel('Wavenumber k', fontsize=11)
    ax5.set_ylabel('Energy Spectrum', fontsize=11)
    ax5.set_title('Energy Spectrum (Final)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.loglog(spec_hr['k'], spec_hr['Z'], 'b-', linewidth=2, label='High-Res')
    ax6.loglog(spec_lr['k'], spec_lr['Z'], 'r--', linewidth=2, label='Low-Res')
    ax6.set_xlabel('Wavenumber k', fontsize=11)
    ax6.set_ylabel('Enstrophy Spectrum', fontsize=11)
    ax6.set_title('Enstrophy Spectrum (Final)', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 4, 7)
    uu_hr = [f['uu_layer1'] for f in highres_results['eddy_flux_history']]
    uu_lr = [f['uu_layer1'] for f in lowres_results['eddy_flux_history']]
    ax7.plot(times_hr, uu_hr, 'b-', linewidth=2, label='High-Res')
    ax7.plot(times_lr, uu_lr, 'r--', linewidth=2, label='Low-Res')
    ax7.set_xlabel('Time (days)', fontsize=11)
    ax7.set_ylabel("u'u' (m²/s²)", fontsize=11)
    ax7.set_title('Reynolds Stress <uu>', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 4, 8)
    uv_hr = [f['uv_layer1'] for f in highres_results['eddy_flux_history']]
    uv_lr = [f['uv_layer1'] for f in lowres_results['eddy_flux_history']]
    ax8.plot(times_hr, uv_hr, 'b-', linewidth=2, label='High-Res')
    ax8.plot(times_lr, uv_lr, 'r--', linewidth=2, label='Low-Res')
    ax8.set_xlabel('Time (days)', fontsize=11)
    ax8.set_ylabel("u'v' (m²/s²)", fontsize=11)
    ax8.set_title('Reynolds Stress <uv>', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Differences and Summary
    ax9 = plt.subplot(3, 4, 9)
    E_diff = 100 * (np.array(lowres_results['energy_history']['total']) - 
                    np.array(highres_results['energy_history']['total'])) / \
                   (np.array(highres_results['energy_history']['total']) + 1e-20)
    ax9.plot(times_lr, E_diff, 'k-', linewidth=2)
    ax9.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax9.set_xlabel('Time (days)', fontsize=11)
    ax9.set_ylabel('Energy Bias (%)', fontsize=11)
    ax9.set_title('Energy Difference (Low - High)', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    ax10 = plt.subplot(3, 4, 10)
    skew_hr = [s['q1_skew'] for s in highres_results['spatial_stats_history']]
    skew_lr = [s['q1_skew'] for s in lowres_results['spatial_stats_history']]
    ax10.plot(times_hr, skew_hr, 'b-', linewidth=2, label='High-Res')
    ax10.plot(times_lr, skew_lr, 'r--', linewidth=2, label='Low-Res')
    ax10.set_xlabel('Time (days)', fontsize=11)
    ax10.set_ylabel('Skewness', fontsize=11)
    ax10.set_title('Vorticity Skewness', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=10)
    ax10.grid(True, alpha=0.3)
    
    ax11 = plt.subplot(3, 4, 11)
    grad_hr = [s['grad_q1'] for s in highres_results['spatial_stats_history']]
    grad_lr = [s['grad_q1'] for s in lowres_results['spatial_stats_history']]
    ax11.plot(times_hr, grad_hr, 'b-', linewidth=2, label='High-Res')
    ax11.plot(times_lr, grad_lr, 'r--', linewidth=2, label='Low-Res')
    ax11.set_xlabel('Time (days)', fontsize=11)
    ax11.set_ylabel('Gradient Magnitude', fontsize=11)
    ax11.set_title('Small-Scale Activity', fontsize=12, fontweight='bold')
    ax11.legend(fontsize=10)
    ax11.grid(True, alpha=0.3)
    
    # Summary metrics
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Compute time-averaged metrics (last 25% of simulation)
    n_avg = len(E_diff) // 4
    E_bias = np.mean(E_diff[-n_avg:])
    E_rmse = np.sqrt(np.mean(E_diff[-n_avg:]**2))
    
    summary_text = f"""
RESOLUTION COMPARISON
{'='*35}

Grid Resolution:
  High-Res: {highres_results['config']['nx']}x{highres_results['config']['ny']}
  Low-Res:  {lowres_results['config']['nx']}x{lowres_results['config']['ny']}

Time-Averaged Bias (final 25%):
  Energy:      {E_bias:+.2f}%
  Energy RMSE: {E_rmse:.2f}%
  
Ratio (Low/High):
  RMS Velocity: {np.mean(urms_lr[-n_avg:])/np.mean(urms_hr[-n_avg:]):.3f}
  Enstrophy:    {np.mean(ens_lr[-n_avg:])/np.mean(ens_hr[-n_avg:]):.3f}
  <uu> Stress:  {np.mean(uu_lr[-n_avg:])/np.mean(uu_hr[-n_avg:]):.3f}
  <uv> Stress:  {np.mean(uv_lr[-n_avg:])/(np.mean(np.abs(uv_hr[-n_avg:]))+1e-20):+.3f}

These biases show the subgrid
scale parameterization gap that
ML can help close!
    """
    
    ax12.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', 
              facecolor='wheat', alpha=0.3))
    
    plt.suptitle('High-Resolution vs Low-Resolution QG Simulation Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {output_file}")
    plt.show()
    
    # Return metrics
    metrics = {
        'energy_bias_percent': E_bias,
        'energy_rmse_percent': E_rmse,
        'velocity_ratio': np.mean(urms_lr[-n_avg:])/np.mean(urms_hr[-n_avg:]),
        'enstrophy_ratio': np.mean(ens_lr[-n_avg:])/np.mean(ens_hr[-n_avg:]),
        'reynolds_uu_ratio': np.mean(uu_lr[-n_avg:])/np.mean(uu_hr[-n_avg:]),
    }
    
    return metrics

def compute_field_differences(highres_results, lowres_results):
    """Compute differences in vorticity and streamfunction fields"""
    
    # Get final states
    q1_hr = highres_results['q1_history'][-1]
    q2_hr = highres_results['q2_history'][-1]
    q1_lr = lowres_results['q1_history'][-1]
    q2_lr = lowres_results['q2_history'][-1]
    
    # Compute streamfunctions
    model_hr = highres_results['model']
    model_lr = lowres_results['model']
    
    psi1_hr, psi2_hr = model_hr.q_to_psi(q1_hr, q2_hr)
    psi2_hr, psi2_hr = model_hr.q_to_psi(q1_hr, q2_hr)
    psi1_lr, psi2_lr = model_lr.q_to_psi(q1_lr, q2_lr)
    
    # CORRECT APPROACH: Coarsen high-res to low-res grid
    # This represents what the low-res model SHOULD produce
    from scipy.ndimage import uniform_filter
    
    nx_hr = highres_results['config']['nx']
    ny_hr = highres_results['config']['ny']
    nx_lr = lowres_results['config']['nx']
    ny_lr = lowres_results['config']['ny']
    
    coarsen_factor = nx_hr // nx_lr
    
    # Apply box filter then subsample (proper coarsening)
    def coarsen_field(field_hr, factor):
        """Coarsen field by averaging then subsampling"""
        # Apply box filter
        filtered = uniform_filter(field_hr, size=factor, mode='wrap')
        # Subsample
        coarsened = filtered[::factor, ::factor]
        return coarsened
    
    q1_hr_coarse = coarsen_field(q1_hr, coarsen_factor)
    q2_hr_coarse = coarsen_field(q2_hr, coarsen_factor)
    psi1_hr_coarse = coarsen_field(psi1_hr, coarsen_factor)
    psi2_hr_coarse = coarsen_field(psi2_hr, coarsen_factor)
    
    # Now compare at low-res grid
    q1_diff = q1_lr - q1_hr_coarse
    q2_diff = q2_lr - q2_hr_coarse
    psi1_diff = psi1_lr - psi1_hr_coarse
    psi2_diff = psi2_lr - psi2_hr_coarse
    
    # Compute RMS differences
    q1_rmse = np.sqrt(np.mean(q1_diff**2))
    q2_rmse = np.sqrt(np.mean(q2_diff**2))
    psi1_rmse = np.sqrt(np.mean(psi1_diff**2))
    psi2_rmse = np.sqrt(np.mean(psi2_diff**2))
    
    # Compute normalized differences (relative to coarsened high-res)
    q1_nrmse = q1_rmse / (np.std(q1_hr_coarse) + 1e-20)
    q2_nrmse = q2_rmse / (np.std(q2_hr_coarse) + 1e-20)
    psi1_nrmse = psi1_rmse / (np.std(psi1_hr_coarse) + 1e-20)
    psi2_nrmse = psi2_rmse / (np.std(psi2_hr_coarse) + 1e-20)
    
    # For plotting, also keep uncoarsened high-res
    field_stats = {
        # Original high-res fields
        'q1_hr_full': q1_hr, 'q2_hr_full': q2_hr,
        'psi1_hr_full': psi1_hr, 'psi2_hr_full': psi2_hr,
        
        # Coarsened high-res (what low-res should match)
        'q1_hr': q1_hr_coarse, 'q2_hr': q2_hr_coarse,
        'psi1_hr': psi1_hr_coarse, 'psi2_hr': psi2_hr_coarse,
        
        # Low-res fields
        'q1_lr': q1_lr, 'q2_lr': q2_lr,
        'psi1_lr': psi1_lr, 'psi2_lr': psi2_lr,
        
        # Differences (Low-res minus coarsened high-res)
        'q1_diff': q1_diff, 'q2_diff': q2_diff,
        'psi1_diff': psi1_diff, 'psi2_diff': psi2_diff,
        
        # Error metrics
        'q1_rmse': q1_rmse, 'q2_rmse': q2_rmse,
        'psi1_rmse': psi1_rmse, 'psi2_rmse': psi2_rmse,
        'q1_nrmse': q1_nrmse, 'q2_nrmse': q2_nrmse,
        'psi1_nrmse': psi1_nrmse, 'psi2_nrmse': psi2_nrmse,
    }
    
    return field_stats

def plot_field_comparison(highres_results, lowres_results, field_stats, output_file='field_comparison.png'):
    """Plot vorticity and streamfunction field comparisons"""
    
    fig = plt.figure(figsize=(22, 18))
    
    # Use low-res grid for ALL comparison plots (columns 2-5)
    X_lr = lowres_results['model'].X / 1e3
    Y_lr = lowres_results['model'].Y / 1e3
    
    # Use high-res grid only for first column
    X_hr = highres_results['model'].X / 1e3
    Y_hr = highres_results['model'].Y / 1e3
    
    # Row 1: Layer 1 Vorticity
    ax1 = plt.subplot(4, 5, 1)
    levels = np.linspace(np.percentile(field_stats['q1_hr_full'], 1), 
                         np.percentile(field_stats['q1_hr_full'], 99), 30)
    cf = ax1.contourf(X_hr, Y_hr, field_stats['q1_hr_full'], levels=levels, cmap='RdBu_r',extend='both')
    ax1.set_title('High-Res (256x256)\nLayer 1 Vorticity', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(4, 5, 2)
    levels_coarse = np.linspace(np.percentile(field_stats['q1_hr'], 1), 
                                np.percentile(field_stats['q1_hr'], 99), 30)
    cf = ax2.contourf(X_lr, Y_lr, field_stats['q1_hr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax2.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(4, 5, 3)
    cf = ax3.contourf(X_lr, Y_lr, field_stats['q1_lr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax3.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(4, 5, 4)
    diff_max = np.max(np.abs(field_stats['q1_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax4.contourf(X_lr, Y_lr, field_stats['q1_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax4.set_title(f'Difference\nNRMSE={field_stats["q1_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(4, 5, 5)
    ax5.hist(field_stats['q1_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax5.hist(field_stats['q1_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax5.set_xlabel('Vorticity', fontsize=9)
    ax5.set_ylabel('PDF', fontsize=9)
    ax5.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Row 2: Layer 2 Vorticity
    ax6 = plt.subplot(4, 5, 6)
    levels = np.linspace(np.percentile(field_stats['q2_hr_full'], 1), 
                         np.percentile(field_stats['q2_hr_full'], 99), 30)
    cf = ax6.contourf(X_hr, Y_hr, field_stats['q2_hr_full'], levels=levels, cmap='RdBu_r',extend='both')
    ax6.set_title('High-Res (256x256)\nLayer 2 Vorticity', fontweight='bold', fontsize=10)
    ax6.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax6, fraction=0.046)
    
    ax7 = plt.subplot(4, 5, 7)
    levels_coarse = np.linspace(np.percentile(field_stats['q2_hr'], 1), 
                                np.percentile(field_stats['q2_hr'], 99), 30)
    cf = ax7.contourf(X_lr, Y_lr, field_stats['q2_hr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax7.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax7, fraction=0.046)
    
    ax8 = plt.subplot(4, 5, 8)
    cf = ax8.contourf(X_lr, Y_lr, field_stats['q2_lr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax8.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax8, fraction=0.046)
    
    ax9 = plt.subplot(4, 5, 9)
    diff_max = np.max(np.abs(field_stats['q2_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax9.contourf(X_lr, Y_lr, field_stats['q2_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax9.set_title(f'Difference\nNRMSE={field_stats["q2_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax9, fraction=0.046)
    
    ax10 = plt.subplot(4, 5, 10)
    ax10.hist(field_stats['q2_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax10.hist(field_stats['q2_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax10.set_xlabel('Vorticity', fontsize=9)
    ax10.set_ylabel('PDF', fontsize=9)
    ax10.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # Row 3: Layer 1 Streamfunction
    ax11 = plt.subplot(4, 5, 11)
    levels = np.linspace(np.percentile(field_stats['psi1_hr_full'], 1), 
                         np.percentile(field_stats['psi1_hr_full'], 99), 30)
    cf = ax11.contourf(X_hr, Y_hr, field_stats['psi1_hr_full'], levels=levels, cmap='viridis',extend='both')
    ax11.set_title('High-Res (256x256)\nLayer 1 Streamfunction', fontweight='bold', fontsize=10)
    ax11.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax11, fraction=0.046)
    
    ax12 = plt.subplot(4, 5, 12)
    levels_coarse = np.linspace(np.percentile(field_stats['psi1_hr'], 1), 
                                np.percentile(field_stats['psi1_hr'], 99), 30)
    cf = ax12.contourf(X_lr, Y_lr, field_stats['psi1_hr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax12.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax12, fraction=0.046)
    
    ax13 = plt.subplot(4, 5, 13)
    cf = ax13.contourf(X_lr, Y_lr, field_stats['psi1_lr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax13.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax13, fraction=0.046)
    
    ax14 = plt.subplot(4, 5, 14)
    diff_max = np.max(np.abs(field_stats['psi1_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax14.contourf(X_lr, Y_lr, field_stats['psi1_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax14.set_title(f'Difference\nNRMSE={field_stats["psi1_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax14, fraction=0.046)
    
    ax15 = plt.subplot(4, 5, 15)
    ax15.hist(field_stats['psi1_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax15.hist(field_stats['psi1_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax15.set_xlabel('Streamfunction', fontsize=9)
    ax15.set_ylabel('PDF', fontsize=9)
    ax15.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax15.legend(fontsize=8)
    ax15.grid(True, alpha=0.3)
    
    # Row 4: Layer 2 Streamfunction
    ax16 = plt.subplot(4, 5, 16)
    levels = np.linspace(np.percentile(field_stats['psi2_hr_full'], 1), 
                         np.percentile(field_stats['psi2_hr_full'], 99), 30)
    cf = ax16.contourf(X_hr, Y_hr, field_stats['psi2_hr_full'], levels=levels, cmap='viridis',extend='both')
    ax16.set_title('High-Res (256x256)\nLayer 2 Streamfunction', fontweight='bold', fontsize=10)
    ax16.set_xlabel('X (km)')
    ax16.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax16, fraction=0.046)
    
    ax17 = plt.subplot(4, 5, 17)
    levels_coarse = np.linspace(np.percentile(field_stats['psi2_hr'], 1), 
                                np.percentile(field_stats['psi2_hr'], 99), 30)
    cf = ax17.contourf(X_lr, Y_lr, field_stats['psi2_hr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax17.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    ax17.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax17, fraction=0.046)
    
    ax18 = plt.subplot(4, 5, 18)
    cf = ax18.contourf(X_lr, Y_lr, field_stats['psi2_lr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax18.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    ax18.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax18, fraction=0.046)
    
    ax19 = plt.subplot(4, 5, 19)
    diff_max = np.max(np.abs(field_stats['psi2_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax19.contourf(X_lr, Y_lr, field_stats['psi2_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax19.set_title(f'Difference\nNRMSE={field_stats["psi2_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    ax19.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax19, fraction=0.046)
    
    ax20 = plt.subplot(4, 5, 20)
    ax20.hist(field_stats['psi2_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax20.hist(field_stats['psi2_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax20.set_xlabel('Streamfunction', fontsize=9)
    ax20.set_ylabel('PDF', fontsize=9)
    ax20.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax20.legend(fontsize=8)
    ax20.grid(True, alpha=0.3)
    
    plt.suptitle('Field Comparison: Low-Res vs Coarsened High-Res (Final Time)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nField comparison plot saved to {output_file}")
    plt.show()
    
    # Row 1: Layer 1 Vorticity
    ax1 = plt.subplot(4, 5, 1)
    levels = np.linspace(np.percentile(field_stats['q1_hr_full'], 1), 
                         np.percentile(field_stats['q1_hr_full'], 99), 30)
    cf = ax1.contourf(X_hr, Y_hr, field_stats['q1_hr_full'], levels=levels, cmap='RdBu_r',extend='both')
    ax1.set_title('High-Res (256x256)\nLayer 1 Vorticity', fontweight='bold', fontsize=10)
    ax1.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(4, 5, 2)
    levels_coarse = np.linspace(np.percentile(field_stats['q1_hr'], 1), 
                                np.percentile(field_stats['q1_hr'], 99), 30)
    cf = ax2.contourf(X_lr, Y_lr, field_stats['q1_hr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax2.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax2, fraction=0.046)
    
    ax3 = plt.subplot(4, 5, 3)
    cf = ax3.contourf(X_lr, Y_lr, field_stats['q1_lr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax3.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax3, fraction=0.046)
    
    ax4 = plt.subplot(4, 5, 4)
    diff_max = np.max(np.abs(field_stats['q1_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax4.contourf(X_lr, Y_lr, field_stats['q1_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax4.set_title(f'Difference\nNRMSE={field_stats["q1_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax4, fraction=0.046)
    
    ax5 = plt.subplot(4, 5, 5)
    ax5.hist(field_stats['q1_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax5.hist(field_stats['q1_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax5.set_xlabel('Vorticity', fontsize=9)
    ax5.set_ylabel('PDF', fontsize=9)
    ax5.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Row 2: Layer 2 Vorticity
    ax6 = plt.subplot(4, 5, 6)
    levels = np.linspace(np.percentile(field_stats['q2_hr_full'], 1), 
                         np.percentile(field_stats['q2_hr_full'], 99), 30)
    cf = ax6.contourf(X_hr, Y_hr, field_stats['q2_hr_full'], levels=levels, cmap='RdBu_r',extend='both')
    ax6.set_title('High-Res (256x256)\nLayer 2 Vorticity', fontweight='bold', fontsize=10)
    ax6.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax6, fraction=0.046)
    
    ax7 = plt.subplot(4, 5, 7)
    levels_coarse = np.linspace(np.percentile(field_stats['q2_hr'], 1), 
                                np.percentile(field_stats['q2_hr'], 99), 30)
    cf = ax7.contourf(X_lr, Y_lr, field_stats['q2_hr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax7.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax7, fraction=0.046)
    
    ax8 = plt.subplot(4, 5, 8)
    cf = ax8.contourf(X_lr, Y_lr, field_stats['q2_lr'], levels=levels_coarse, cmap='RdBu_r',extend='both')
    ax8.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax8, fraction=0.046)
    
    ax9 = plt.subplot(4, 5, 9)
    diff_max = np.max(np.abs(field_stats['q2_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax9.contourf(X_lr, Y_lr, field_stats['q2_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax9.set_title(f'Difference\nNRMSE={field_stats["q2_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax9, fraction=0.046)
    
    ax10 = plt.subplot(4, 5, 10)
    ax10.hist(field_stats['q2_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax10.hist(field_stats['q2_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax10.set_xlabel('Vorticity', fontsize=9)
    ax10.set_ylabel('PDF', fontsize=9)
    ax10.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # Row 3: Layer 1 Streamfunction
    ax11 = plt.subplot(4, 5, 11)
    levels = np.linspace(np.percentile(field_stats['psi1_hr_full'], 1), 
                         np.percentile(field_stats['psi1_hr_full'], 99), 30)
    cf = ax11.contourf(X_hr, Y_hr, field_stats['psi1_hr_full'], levels=levels, cmap='viridis',extend='both')
    ax11.set_title('High-Res (256x256)\nLayer 1 Streamfunction', fontweight='bold', fontsize=10)
    ax11.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax11, fraction=0.046)
    
    ax12 = plt.subplot(4, 5, 12)
    levels_coarse = np.linspace(np.percentile(field_stats['psi1_hr'], 1), 
                                np.percentile(field_stats['psi1_hr'], 99), 30)
    cf = ax12.contourf(X_lr, Y_lr, field_stats['psi1_hr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax12.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax12, fraction=0.046)
    
    ax13 = plt.subplot(4, 5, 13)
    cf = ax13.contourf(X_lr, Y_lr, field_stats['psi1_lr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax13.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax13, fraction=0.046)
    
    ax14 = plt.subplot(4, 5, 14)
    diff_max = np.max(np.abs(field_stats['psi1_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax14.contourf(X_lr, Y_lr, field_stats['psi1_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax14.set_title(f'Difference\nNRMSE={field_stats["psi1_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    plt.colorbar(cf, ax=ax14, fraction=0.046)
    
    ax15 = plt.subplot(4, 5, 15)
    ax15.hist(field_stats['psi1_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax15.hist(field_stats['psi1_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax15.set_xlabel('Streamfunction', fontsize=9)
    ax15.set_ylabel('PDF', fontsize=9)
    ax15.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax15.legend(fontsize=8)
    ax15.grid(True, alpha=0.3)
    
    # Row 4: Layer 2 Streamfunction
    ax16 = plt.subplot(4, 5, 16)
    levels = np.linspace(np.percentile(field_stats['psi2_hr_full'], 1), 
                         np.percentile(field_stats['psi2_hr_full'], 99), 30)
    cf = ax16.contourf(X_hr, Y_hr, field_stats['psi2_hr_full'], levels=levels, cmap='viridis',extend='both')
    ax16.set_title('High-Res (256x256)\nLayer 2 Streamfunction', fontweight='bold', fontsize=10)
    ax16.set_xlabel('X (km)')
    ax16.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax16, fraction=0.046)
    
    ax17 = plt.subplot(4, 5, 17)
    levels_coarse = np.linspace(np.percentile(field_stats['psi2_hr'], 1), 
                                np.percentile(field_stats['psi2_hr'], 99), 30)
    cf = ax17.contourf(X_lr, Y_lr, field_stats['psi2_hr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax17.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    ax17.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax17, fraction=0.046)
    
    ax18 = plt.subplot(4, 5, 18)
    cf = ax18.contourf(X_lr, Y_lr, field_stats['psi2_lr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax18.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    ax18.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax18, fraction=0.046)
    
    ax19 = plt.subplot(4, 5, 19)
    diff_max = np.max(np.abs(field_stats['psi2_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax19.contourf(X_lr, Y_lr, field_stats['psi2_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax19.set_title(f'Difference\nNRMSE={field_stats["psi2_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    ax19.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax19, fraction=0.046)
    
    ax20 = plt.subplot(4, 5, 20)
    ax20.hist(field_stats['psi2_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax20.hist(field_stats['psi2_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax20.set_xlabel('Streamfunction', fontsize=9)
    ax20.set_ylabel('PDF', fontsize=9)
    ax20.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax20.legend(fontsize=8)
    ax20.grid(True, alpha=0.3)
    
    plt.suptitle('Field Comparison: Low-Res vs Coarsened High-Res (Final Time)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nField comparison plot saved to {output_file}")
    plt.show()
    # Row 4: Layer 2 Streamfunction
    ax16 = plt.subplot(4, 5, 16)
    levels = np.linspace(np.percentile(field_stats['psi2_hr_full'], 1), 
                         np.percentile(field_stats['psi2_hr_full'], 99), 30)
    cf = ax16.contourf(X_hr, Y_hr, field_stats['psi2_hr_full'], levels=levels, cmap='viridis',extend='both')
    ax16.set_title('High-Res (256x256)\nLayer 2 Streamfunction', fontweight='bold', fontsize=10)
    ax16.set_xlabel('X (km)')
    ax16.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax16, fraction=0.046)
    
    ax17 = plt.subplot(4, 5, 17)
    levels_coarse = np.linspace(np.percentile(field_stats['psi2_hr'], 1), 
                                np.percentile(field_stats['psi2_hr'], 99), 30)
    cf = ax17.contourf(X_lr, Y_lr, field_stats['psi2_hr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax17.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
    ax17.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax17, fraction=0.046)
    
    ax18 = plt.subplot(4, 5, 18)
    cf = ax18.contourf(X_lr, Y_lr, field_stats['psi2_lr'], levels=levels_coarse, cmap='viridis',extend='both')
    ax18.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
    ax18.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax18, fraction=0.046)
    
    ax19 = plt.subplot(4, 5, 19)
    diff_max = np.max(np.abs(field_stats['psi2_diff']))
    diff_levels = np.linspace(-diff_max, diff_max, 30)
    cf = ax19.contourf(X_lr, Y_lr, field_stats['psi2_diff'], levels=diff_levels, cmap='seismic',extend='both')
    ax19.set_title(f'Difference\nNRMSE={field_stats["psi2_nrmse"]:.3f}', fontweight='bold', fontsize=10)
    ax19.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax19, fraction=0.046)
    
    ax20 = plt.subplot(4, 5, 20)
    ax20.hist(field_stats['psi2_hr'].flatten(), bins=40, alpha=0.6, label='HR Coarse', density=True, color='blue')
    ax20.hist(field_stats['psi2_lr'].flatten(), bins=40, alpha=0.6, label='Low-Res', density=True, color='red')
    ax20.set_xlabel('Streamfunction', fontsize=9)
    ax20.set_ylabel('PDF', fontsize=9)
    ax20.set_title('PDF Comparison', fontweight='bold', fontsize=10)
    ax20.legend(fontsize=8)
    ax20.grid(True, alpha=0.3)
    
    plt.suptitle('Field Comparison: Low-Res vs Coarsened High-Res (Final Time)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nField comparison plot saved to {output_file}")
    plt.show()

def print_comparison_summary(highres_results, lowres_results, metrics):
    """Print detailed comparison summary"""
    
    print("\n" + "="*70)
    print("DETAILED COMPARISON SUMMARY")
    print("="*70)
    
    print("\nGRID INFORMATION:")
    print(f"  High-Res: {highres_results['config']['nx']}x{highres_results['config']['ny']} "
          f"(dx = {highres_results['config']['Lx']/highres_results['config']['nx']/1e3:.1f} km)")
    print(f"  Low-Res:  {lowres_results['config']['nx']}x{lowres_results['config']['ny']} "
          f"(dx = {lowres_results['config']['Lx']/lowres_results['config']['nx']/1e3:.1f} km)")
    print(f"  Resolution Ratio: {highres_results['config']['nx']//lowres_results['config']['nx']}x coarser")
    
    print("\nKEY STATISTICS (Time-Averaged, Final 25%):")
    print(f"  Energy Bias:        {metrics['energy_bias_percent']:+.2f}%")
    print(f"  Energy RMSE:        {metrics['energy_rmse_percent']:.2f}%")
    print(f"  Velocity Ratio:     {metrics['velocity_ratio']:.3f}")
    print(f"  Enstrophy Ratio:    {metrics['enstrophy_ratio']:.3f}")
    print(f"  Reynolds uu Ratio:  {metrics['reynolds_uu_ratio']:.3f}")
    
    print("\nFIELD DIFFERENCES (Final Time, Low-Res vs Coarsened High-Res):")
    print(f"  Vorticity Layer 1 NRMSE:      {metrics.get('q1_nrmse', 0):.3f}")
    print(f"  Vorticity Layer 2 NRMSE:      {metrics.get('q2_nrmse', 0):.3f}")
    print(f"  Streamfunction Layer 1 NRMSE: {metrics.get('psi1_nrmse', 0):.3f}")
    print(f"  Streamfunction Layer 2 NRMSE: {metrics.get('psi2_nrmse', 0):.3f}")
    print("\n  Note: NRMSE = Normalized RMS Error (difference / std_deviation)")
    print("        Values > 0.3 indicate significant bias needing parameterization")
    
    print("\n" + "="*70)