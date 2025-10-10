"""
Plotting and Visualization Functions for QG Model
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

def setup_figure():
    """Create figure layout for visualization"""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]

def plot_vorticity_field(ax, X, Y, q, title, time):
    """Plot vorticity field with contours"""
    ax.clear()
    
    # Handle NaN/Inf values
    if not np.isfinite(q).all():
        ax.text(0.5, 0.5, 'Data contains NaN/Inf', 
                ha='center', va='center', transform=ax.transAxes)
        return None
    
    # Get safe levels
    q_min, q_max = np.percentile(q[np.isfinite(q)], [1, 99])
    if q_min == q_max or not np.isfinite([q_min, q_max]).all():
        q_min, q_max = np.min(q[np.isfinite(q)]), np.max(q[np.isfinite(q)])
    if q_min == q_max:
        q_min, q_max = -1, 1
    
    levels = np.linspace(q_min, q_max, 30)
    cf = ax.contourf(X, Y, q, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(X, Y, q, levels=10, colors='k', linewidths=0.5, alpha=0.3)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(f'{title}\nTime = {time:.2f} days')
    ax.set_aspect('equal')
    return cf

def plot_streamfunction(ax, X, Y, psi, title):
    """Plot streamfunction with streamlines"""
    ax.clear()
    
    # Handle NaN/Inf values
    if not np.isfinite(psi).all():
        ax.text(0.5, 0.5, 'Data contains NaN/Inf', 
                ha='center', va='center', transform=ax.transAxes)
        return None
    
    psi_min, psi_max = np.percentile(psi[np.isfinite(psi)], [1, 99])
    if psi_min == psi_max or not np.isfinite([psi_min, psi_max]).all():
        psi_min, psi_max = np.min(psi[np.isfinite(psi)]), np.max(psi[np.isfinite(psi)])
    if psi_min == psi_max:
        psi_min, psi_max = -1, 1
        
    levels = np.linspace(psi_min, psi_max, 30)
    cf = ax.contourf(X, Y, psi, levels=levels, cmap='viridis', extend='both')
    ax.contour(X, Y, psi, levels=15, colors='k', linewidths=0.5, alpha=0.4)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(title)
    ax.set_aspect('equal')
    return cf

def plot_velocity_field(ax, X, Y, u, v, q, title, stride=8):
    """Plot velocity vectors over vorticity"""
    ax.clear()
    levels = np.linspace(np.percentile(q, 1), np.percentile(q, 99), 30)
    cf = ax.contourf(X, Y, q, levels=levels, cmap='RdBu_r', extend='both', alpha=0.7)
    
    speed = np.sqrt(u**2 + v**2)
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              u[::stride, ::stride], v[::stride, ::stride],
              speed[::stride, ::stride], cmap='plasma', scale=50, width=0.003)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title(title)
    ax.set_aspect('equal')
    return cf

def plot_statistics(ax, time_array, data_array, ylabel, title, color='b'):
    """Plot time series statistics"""
    ax.clear()
    ax.plot(time_array, data_array, color=color, linewidth=2)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

def create_animation(model, q1_history, q2_history, times, output_file='qg_animation.mp4'):
    """Create animation of the simulation"""
    fig, axes = setup_figure()
    
    # Storage for colorbar objects
    cbar_objs = [None] * 6
    
    def update(frame):
        q1 = q1_history[frame]
        q2 = q2_history[frame]
        time = times[frame]
        
        psi1, psi2 = model.q_to_psi(q1, q2)
        u1, v1 = model.compute_velocity(psi1)
        u2, v2 = model.compute_velocity(psi2)
        
        # Remove old colorbars
        for cbar in cbar_objs:
            if cbar is not None:
                cbar.remove()
        
        # Layer 1 vorticity
        cf1 = plot_vorticity_field(axes[0], model.X/1e3, model.Y/1e3, q1, 
                                    'Upper Layer Vorticity', time)
        cbar_objs[0] = plt.colorbar(cf1, ax=axes[0], fraction=0.046)
        
        # Layer 2 vorticity
        cf2 = plot_vorticity_field(axes[1], model.X/1e3, model.Y/1e3, q2, 
                                    'Lower Layer Vorticity', time)
        cbar_objs[1] = plt.colorbar(cf2, ax=axes[1], fraction=0.046)
        
        # Layer 1 streamfunction
        cf3 = plot_streamfunction(axes[2], model.X/1e3, model.Y/1e3, psi1, 
                                   'Upper Layer Streamfunction')
        cbar_objs[2] = plt.colorbar(cf3, ax=axes[2], fraction=0.046)
        
        # Layer 2 streamfunction
        cf4 = plot_streamfunction(axes[3], model.X/1e3, model.Y/1e3, psi2, 
                                   'Lower Layer Streamfunction')
        cbar_objs[3] = plt.colorbar(cf4, ax=axes[3], fraction=0.046)
        
        # Layer 1 velocity
        cf5 = plot_velocity_field(axes[4], model.X/1e3, model.Y/1e3, u1, v1, q1,
                                   'Upper Layer Velocity')
        cbar_objs[4] = plt.colorbar(cf5, ax=axes[4], fraction=0.046)
        
        # Layer 2 velocity
        cf6 = plot_velocity_field(axes[5], model.X/1e3, model.Y/1e3, u2, v2, q2,
                                   'Lower Layer Velocity')
        cbar_objs[5] = plt.colorbar(cf6, ax=axes[5], fraction=0.046)
        
        return axes
    
    anim = animation.FuncAnimation(fig, update, frames=len(times), 
                                   interval=50, blit=False)
    
    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    anim.save(output_file, writer=writer)
    
    plt.close()
    print(f"Animation saved to {output_file}")
    return anim

def plot_diagnostics(times, energy_history, enstrophy_history, output_file='diagnostics.png'):
    """Plot diagnostic statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total energy
    plot_statistics(axes[0, 0], times, energy_history['total'], 
                   'Total Energy', 'Total Kinetic Energy Evolution', 'darkblue')
    
    # Layer energies
    axes[0, 1].plot(times, energy_history['layer1'], 'r-', linewidth=2, label='Upper Layer')
    axes[0, 1].plot(times, energy_history['layer2'], 'b-', linewidth=2, label='Lower Layer')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].set_title('Layer Kinetic Energies')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total enstrophy
    plot_statistics(axes[1, 0], times, 
                   np.array(enstrophy_history['layer1']) + np.array(enstrophy_history['layer2']), 
                   'Total Enstrophy', 'Total Enstrophy Evolution', 'darkgreen')
    
    # Layer enstrophies
    axes[1, 1].plot(times, enstrophy_history['layer1'], 'r-', linewidth=2, label='Upper Layer')
    axes[1, 1].plot(times, enstrophy_history['layer2'], 'b-', linewidth=2, label='Lower Layer')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Enstrophy')
    axes[1, 1].set_title('Layer Enstrophies')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Diagnostics saved to {output_file}")
    plt.show()

def plot_snapshot(model, q1, q2, time, output_file='snapshot.png'):
    """Plot a single snapshot of the system"""
    fig, axes = setup_figure()
    
    psi1, psi2 = model.q_to_psi(q1, q2)
    u1, v1 = model.compute_velocity(psi1)
    u2, v2 = model.compute_velocity(psi2)
    
    # Layer 1 vorticity
    cf1 = plot_vorticity_field(axes[0], model.X/1e3, model.Y/1e3, q1, 
                                'Upper Layer Vorticity', time)
    plt.colorbar(cf1, ax=axes[0], fraction=0.046)
    
    # Layer 2 vorticity
    cf2 = plot_vorticity_field(axes[1], model.X/1e3, model.Y/1e3, q2, 
                                'Lower Layer Vorticity', time)
    plt.colorbar(cf2, ax=axes[1], fraction=0.046)
    
    # Layer 1 streamfunction
    cf3 = plot_streamfunction(axes[2], model.X/1e3, model.Y/1e3, psi1, 
                               'Upper Layer Streamfunction')
    plt.colorbar(cf3, ax=axes[2], fraction=0.046)
    
    # Layer 2 streamfunction
    cf4 = plot_streamfunction(axes[3], model.X/1e3, model.Y/1e3, psi2, 
                               'Lower Layer Streamfunction')
    plt.colorbar(cf4, ax=axes[3], fraction=0.046)
    
    # Layer 1 velocity
    cf5 = plot_velocity_field(axes[4], model.X/1e3, model.Y/1e3, u1, v1, q1,
                               'Upper Layer Velocity')
    plt.colorbar(cf5, ax=axes[4], fraction=0.046)
    
    # Layer 2 velocity
    cf6 = plot_velocity_field(axes[5], model.X/1e3, model.Y/1e3, u2, v2, q2,
                               'Lower Layer Velocity')
    plt.colorbar(cf6, ax=axes[5], fraction=0.046)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Snapshot saved to {output_file}")
    plt.show()