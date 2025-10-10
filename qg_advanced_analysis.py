"""
Advanced Analysis Functions for QG Comparison
Temporal averaging and barotropic analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import pickle

def compute_time_averaged_fields(results, n_days=10):
    """
    Compute time-averaged fields over last n_days
    
    Parameters:
    -----------
    results : dict
        Simulation results dictionary
    n_days : float
        Number of days to average over (from the end)
    
    Returns:
    --------
    dict : Time-averaged fields
    """
    times = results['times']
    
    # Find indices for last n_days
    time_threshold = times[-1] - n_days
    indices = np.where(times >= time_threshold)[0]
    
    print(f"Averaging over last {n_days} days ({len(indices)} snapshots)")
    print(f"Time range: {times[indices[0]]:.1f} to {times[indices[-1]]:.1f} days")
    
    # Average q1 and q2
    q1_avg = np.mean([results['q1_history'][i] for i in indices], axis=0)
    q2_avg = np.mean([results['q2_history'][i] for i in indices], axis=0)
    
    # Compute averaged streamfunctions
    model = results['model']
    psi1_avg, psi2_avg = model.q_to_psi(q1_avg, q2_avg)
    
    # Compute barotropic and baroclinic components
    # Barotropic = depth-weighted average
    H_total = model.H1 + model.H2
    q_bt = (model.H1 * q1_avg + model.H2 * q2_avg) / H_total
    psi_bt = (model.H1 * psi1_avg + model.H2 * psi2_avg) / H_total
    
    # Baroclinic = difference
    q_bc = q1_avg - q2_avg
    psi_bc = psi1_avg - psi2_avg
    
    # Compute velocities
    u1_avg, v1_avg = model.compute_velocity(psi1_avg)
    u2_avg, v2_avg = model.compute_velocity(psi2_avg)
    u_bt, v_bt = model.compute_velocity(psi_bt)
    u_bc, v_bc = model.compute_velocity(psi_bc)
    
    averaged_fields = {
        'q1': q1_avg,
        'q2': q2_avg,
        'psi1': psi1_avg,
        'psi2': psi2_avg,
        'q_barotropic': q_bt,
        'psi_barotropic': psi_bt,
        'q_baroclinic': q_bc,
        'psi_baroclinic': psi_bc,
        'u1': u1_avg,
        'v1': v1_avg,
        'u2': u2_avg,
        'v2': v2_avg,
        'u_barotropic': u_bt,
        'v_barotropic': v_bt,
        'time_range': (times[indices[0]], times[indices[-1]]),
        'n_snapshots': len(indices),
    }
    
    return averaged_fields

def compare_time_averaged_fields(highres_results, lowres_results, n_days=10):
    """
    Compare time-averaged fields between high-res and low-res
    
    Parameters:
    -----------
    highres_results : dict
        High-resolution results
    lowres_results : dict
        Low-resolution results
    n_days : float
        Number of days to average
    
    Returns:
    --------
    dict : Comparison statistics
    """
    print("\n" + "="*70)
    print(f"COMPUTING TIME-AVERAGED FIELDS (Last {n_days} days)")
    print("="*70)
    
    # Get time-averaged fields
    hr_avg = compute_time_averaged_fields(highres_results, n_days)
    lr_avg = compute_time_averaged_fields(lowres_results, n_days)
    
    # Coarsen high-res to low-res grid
    nx_hr = highres_results['config']['nx']
    nx_lr = lowres_results['config']['nx']
    coarsen_factor = nx_hr // nx_lr
    
    def coarsen_field(field_hr):
        """Coarsen field by averaging then subsampling"""
        filtered = uniform_filter(field_hr, size=coarsen_factor, mode='wrap')
        return filtered[::coarsen_factor, ::coarsen_factor]
    
    # Coarsen all high-res fields
    hr_avg_coarse = {}
    for key in ['q1', 'q2', 'psi1', 'psi2', 'q_barotropic', 'psi_barotropic', 
                'q_baroclinic', 'psi_baroclinic', 'u1', 'v1', 'u2', 'v2',
                'u_barotropic', 'v_barotropic']:
        hr_avg_coarse[key] = coarsen_field(hr_avg[key])
    
    # Compute differences
    differences = {}
    nrmse = {}
    
    for key in ['q1', 'q2', 'psi1', 'psi2', 'q_barotropic', 'psi_barotropic', 
                'q_baroclinic', 'psi_baroclinic']:
        diff = lr_avg[key] - hr_avg_coarse[key]
        rmse = np.sqrt(np.mean(diff**2))
        norm = np.std(hr_avg_coarse[key]) + 1e-20
        
        differences[key] = diff
        nrmse[key] = rmse / norm
    
    # Print summary
    print("\n" + "="*70)
    print("TIME-AVERAGED FIELD COMPARISON")
    print("="*70)
    print(f"\nVorticity NRMSE:")
    print(f"  Layer 1:    {nrmse['q1']:.4f}")
    print(f"  Layer 2:    {nrmse['q2']:.4f}")
    print(f"  Barotropic: {nrmse['q_barotropic']:.4f}")
    print(f"  Baroclinic: {nrmse['q_baroclinic']:.4f}")
    
    print(f"\nStreamfunction NRMSE:")
    print(f"  Layer 1:    {nrmse['psi1']:.4f}")
    print(f"  Layer 2:    {nrmse['psi2']:.4f}")
    print(f"  Barotropic: {nrmse['psi_barotropic']:.4f}")
    print(f"  Baroclinic: {nrmse['psi_baroclinic']:.4f}")
    print("="*70)
    
    comparison = {
        'hr_avg': hr_avg,
        'lr_avg': lr_avg,
        'hr_avg_coarse': hr_avg_coarse,
        'differences': differences,
        'nrmse': nrmse,
        'n_days': n_days,
    }
    
    return comparison

def plot_time_averaged_comparison(comparison, highres_results, lowres_results, 
                                   output_file='time_averaged_comparison.png'):
    """
    Plot time-averaged field comparison with barotropic PV
    
    Parameters:
    -----------
    comparison : dict
        Output from compare_time_averaged_fields()
    highres_results : dict
        High-res results for grid info
    lowres_results : dict
        Low-res results for grid info
    output_file : str
        Output filename
    """
    
    hr_avg = comparison['hr_avg']
    lr_avg = comparison['lr_avg']
    hr_avg_coarse = comparison['hr_avg_coarse']
    differences = comparison['differences']
    nrmse = comparison['nrmse']
    
    X_lr = lowres_results['model'].X / 1e3
    Y_lr = lowres_results['model'].Y / 1e3
    X_hr = highres_results['model'].X / 1e3
    Y_hr = highres_results['model'].Y / 1e3
    
    fig = plt.figure(figsize=(22, 16))
    
    fields_to_plot = [
        ('q1', 'Layer 1 Vorticity', 'RdBu_r'),
        ('q2', 'Layer 2 Vorticity', 'RdBu_r'),
        ('q_barotropic', 'Barotropic PV', 'RdBu_r'),
        ('psi_barotropic', 'Barotropic Streamfunction', 'viridis'),
    ]
    
    for row, (field_key, field_name, cmap) in enumerate(fields_to_plot):
        
        # Column 1: High-res full resolution
        ax1 = plt.subplot(4, 5, row*5 + 1)
        field_hr = hr_avg[field_key]
        levels = np.linspace(np.percentile(field_hr, 1), 
                            np.percentile(field_hr, 99), 30)
        cf = ax1.contourf(X_hr, Y_hr, field_hr, levels=levels, cmap=cmap,extend='both')
        if row == 0:
            ax1.set_title('High-Res (256x256)', fontweight='bold', fontsize=10)
        ax1.set_ylabel(field_name + '\nY (km)', fontsize=9)
        plt.colorbar(cf, ax=ax1, fraction=0.046)
        
        # Column 2: High-res coarsened (target)
        ax2 = plt.subplot(4, 5, row*5 + 2)
        field_hr_coarse = hr_avg_coarse[field_key]
        levels_coarse = np.linspace(np.percentile(field_hr_coarse, 1),
                                    np.percentile(field_hr_coarse, 99), 30)
        cf = ax2.contourf(X_lr, Y_lr, field_hr_coarse, levels=levels_coarse, cmap=cmap,extend='both')
        if row == 0:
            ax2.set_title('High-Res Coarsened\n(Target)', fontweight='bold', fontsize=10)
        plt.colorbar(cf, ax=ax2, fraction=0.046)
        
        # Column 3: Low-res actual
        ax3 = plt.subplot(4, 5, row*5 + 3)
        field_lr = lr_avg[field_key]
        cf = ax3.contourf(X_lr, Y_lr, field_lr, levels=levels_coarse, cmap=cmap,extend='both')
        if row == 0:
            ax3.set_title('Low-Res (64x64)\n(Actual)', fontweight='bold', fontsize=10)
        plt.colorbar(cf, ax=ax3, fraction=0.046)
        
        # Column 4: Difference
        ax4 = plt.subplot(4, 5, row*5 + 4)
        diff = differences[field_key]
        diff_max = np.max(np.abs(diff))
        diff_levels = np.linspace(-diff_max, diff_max, 30)
        cf = ax4.contourf(X_lr, Y_lr, diff, levels=diff_levels, cmap='seismic',extend='both')
        if row == 0:
            ax4.set_title('Difference\n(LR - HR Coarse)', fontweight='bold', fontsize=10)
        ax4.text(0.5, 0.95, f'NRMSE={nrmse[field_key]:.3f}', 
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9, fontweight='bold')
        plt.colorbar(cf, ax=ax4, fraction=0.046)
        
        # Column 5: PDF comparison
        ax5 = plt.subplot(4, 5, row*5 + 5)
        ax5.hist(field_hr_coarse.flatten(), bins=40, alpha=0.6, 
                label='HR Coarse', density=True, color='blue')
        ax5.hist(field_lr.flatten(), bins=40, alpha=0.6, 
                label='Low-Res', density=True, color='red')
        if row == 0:
            ax5.set_title('PDF Comparison', fontweight='bold', fontsize=10)
        ax5.set_xlabel('Value', fontsize=8)
        ax5.set_ylabel('PDF', fontsize=8)
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(labelsize=7)
    
    n_days = comparison['n_days']
    plt.suptitle(f'Time-Averaged Fields Comparison (Last {n_days} days)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nTime-averaged comparison plot saved to {output_file}")
    plt.show()

def plot_barotropic_baroclinic_decomposition(comparison, highres_results, lowres_results,
                                             output_file='bt_bc_decomposition.png'):
    """
    Plot barotropic and baroclinic mode decomposition
    """
    
    hr_avg = comparison['hr_avg']
    lr_avg = comparison['lr_avg']
    hr_avg_coarse = comparison['hr_avg_coarse']
    
    X_lr = lowres_results['model'].X / 1e3
    Y_lr = lowres_results['model'].Y / 1e3
    
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Barotropic PV
    ax1 = plt.subplot(3, 3, 1)
    field = hr_avg_coarse['q_barotropic']
    levels = np.linspace(np.percentile(field, 1), np.percentile(field, 99), 30)
    cf = ax1.contourf(X_lr, Y_lr, field, levels=levels, cmap='RdBu_r',extend='both')
    ax1.set_title('High-Res Coarsened\nBarotropic PV', fontweight='bold')
    ax1.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax1)
    
    ax2 = plt.subplot(3, 3, 2)
    field = lr_avg['q_barotropic']
    cf = ax2.contourf(X_lr, Y_lr, field, levels=levels, cmap='RdBu_r',extend='both')
    ax2.set_title('Low-Res\nBarotropic PV', fontweight='bold')
    plt.colorbar(cf, ax=ax2)
    
    ax3 = plt.subplot(3, 3, 3)
    diff = lr_avg['q_barotropic'] - hr_avg_coarse['q_barotropic']
    diff_max = np.max(np.abs(diff))
    levels_diff = np.linspace(-diff_max, diff_max, 30)
    cf = ax3.contourf(X_lr, Y_lr, diff, levels=levels_diff, cmap='seismic',extend='both')
    ax3.set_title('Difference\nBarotropic PV', fontweight='bold')
    plt.colorbar(cf, ax=ax3)
    
    # Row 2: Baroclinic PV
    ax4 = plt.subplot(3, 3, 4)
    field = hr_avg_coarse['q_baroclinic']
    levels = np.linspace(np.percentile(field, 1), np.percentile(field, 99), 30)
    cf = ax4.contourf(X_lr, Y_lr, field, levels=levels, cmap='RdBu_r',extend='both')
    ax4.set_title('High-Res Coarsened\nBaroclinic PV', fontweight='bold')
    ax4.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax4)
    
    ax5 = plt.subplot(3, 3, 5)
    field = lr_avg['q_baroclinic']
    cf = ax5.contourf(X_lr, Y_lr, field, levels=levels, cmap='RdBu_r',extend='both')
    ax5.set_title('Low-Res\nBaroclinic PV', fontweight='bold')
    plt.colorbar(cf, ax=ax5)
    
    ax6 = plt.subplot(3, 3, 6)
    diff = lr_avg['q_baroclinic'] - hr_avg_coarse['q_baroclinic']
    diff_max = np.max(np.abs(diff))
    levels_diff = np.linspace(-diff_max, diff_max, 30)
    cf = ax6.contourf(X_lr, Y_lr, diff, levels=levels_diff, cmap='seismic',extend='both')
    ax6.set_title('Difference\nBaroclinic PV', fontweight='bold')
    plt.colorbar(cf, ax=ax6)
    
    # Row 3: Barotropic streamfunction and velocity
    ax7 = plt.subplot(3, 3, 7)
    field = hr_avg_coarse['psi_barotropic']
    levels = np.linspace(np.percentile(field, 1), np.percentile(field, 99), 30)
    cf = ax7.contourf(X_lr, Y_lr, field, levels=levels, cmap='viridis',extend='both')
    ax7.set_title('High-Res Coarsened\nBarotropic ψ', fontweight='bold')
    ax7.set_xlabel('X (km)')
    ax7.set_ylabel('Y (km)')
    plt.colorbar(cf, ax=ax7)
    
    ax8 = plt.subplot(3, 3, 8)
    field = lr_avg['psi_barotropic']
    cf = ax8.contourf(X_lr, Y_lr, field, levels=levels, cmap='viridis',extend='both')
    ax8.set_title('Low-Res\nBarotropic ψ', fontweight='bold')
    ax8.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax8)
    
    ax9 = plt.subplot(3, 3, 9)
    # Velocity magnitude comparison
    speed_hr = np.sqrt(hr_avg_coarse['u_barotropic']**2 + hr_avg_coarse['v_barotropic']**2)
    speed_lr = np.sqrt(lr_avg['u_barotropic']**2 + lr_avg['v_barotropic']**2)
    speed_diff = speed_lr - speed_hr
    diff_max = np.max(np.abs(speed_diff))
    levels_diff = np.linspace(-diff_max, diff_max, 30)
    cf = ax9.contourf(X_lr, Y_lr, speed_diff, levels=levels_diff, cmap='seismic',extend='both')
    ax9.set_title('Difference\nBarotropic Speed', fontweight='bold')
    ax9.set_xlabel('X (km)')
    plt.colorbar(cf, ax=ax9)
    
    plt.suptitle('Barotropic and Baroclinic Mode Decomposition', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nBarotropic/Baroclinic decomposition plot saved to {output_file}")
    plt.show()

def main():
    """
    Main function to run advanced analysis
    Load existing results and perform temporal averaging analysis
    """
    
    print("\n" + "="*70)
    print("ADVANCED QG ANALYSIS: TIME AVERAGING & BAROTROPIC PV")
    print("="*70)
    
    # Load results
    print("\nLoading simulation results...")
    try:
        with open('highres_results.pkl', 'rb') as f:
            highres_results = pickle.load(f)
        print("  ✓ High-res results loaded")
    except:
        print("  ✗ Could not load highres_results.pkl")
        return
    
    try:
        with open('lowres_results.pkl', 'rb') as f:
            lowres_results = pickle.load(f)
        print("  ✓ Low-res results loaded")
    except:
        print("  ✗ Could not load lowres_results.pkl")
        return
    
    # Compute time-averaged comparison
    comparison = compare_time_averaged_fields(highres_results, lowres_results, n_days=10)
    
    # Save comparison results
    with open('time_averaged_comparison.pkl', 'wb') as f:
        pickle.dump(comparison, f)
    print("\nTime-averaged comparison saved to time_averaged_comparison.pkl")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_time_averaged_comparison(comparison, highres_results, lowres_results,
                                  'time_averaged_comparison.png')
    
    plot_barotropic_baroclinic_decomposition(comparison, highres_results, lowres_results,
                                            'bt_bc_decomposition.png')
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - time_averaged_comparison.pkl")
    print("  - time_averaged_comparison.png")
    print("  - bt_bc_decomposition.png")
    print("="*70)
    
    return comparison

if __name__ == "__main__":
    comparison = main()