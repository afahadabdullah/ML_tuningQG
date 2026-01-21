"""
QG Beta-Plane Two-Layer Model Functions
Supports both high-resolution and low-resolution with subgrid parameterization
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

class QGTwoLayerModel:
    """Two-layer quasi-geostrophic model on a beta-plane"""
    
    def __init__(self, config):
        """Initialize model with configuration parameters"""
        self.name = config.get('name', 'QG_Model')
        self.nx = config['nx']
        self.ny = config['ny']
        self.Lx = config['Lx']
        self.Ly = config['Ly']
        self.dt = config['dt']
        self.beta = config['beta']
        self.f0 = config['f0']
        self.g_prime = config['g_prime']
        self.H1 = config['H1']
        self.H2 = config['H2']
        self.r_ek = config['r_drag']
        self.nu4 = config['nu']
        
        # Subgrid parameters (for low-res models)
        self.subgrid = config.get('subgrid_params', {})
        self.apply_subgrid = len(self.subgrid) > 0
        
        # Grid
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.x = np.arange(self.nx) * self.dx
        self.y = np.arange(self.ny) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Wavenumbers
        kx = 2 * np.pi * fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.ny, self.dy)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0
        
        # Rossby deformation wavenumber
        self.kd = np.sqrt(self.f0**2 / (self.g_prime * self.H1 * self.H2 / (self.H1 + self.H2)))
        self.kd2 = self.kd**2
        
        # Apply subgrid modifications to dissipation
        nu_effective = self.nu4
        r_effective = self.r_ek
        
        if self.apply_subgrid:
            nu_effective *= self.subgrid.get('viscosity_scale', 1.0)
            r_effective *= self.subgrid.get('drag_scale', 1.0)
        
        # Filters
        self.filter = np.exp(-nu_effective * self.dt * self.K2**4)
        
        # 2/3 dealiasing
        kmax = 2.0 * np.pi / (3.0 * self.dx)
        self.dealias = (np.abs(self.KX) < kmax) & (np.abs(self.KY) < kmax)
        
        # Additional subgrid operators
        if self.apply_subgrid:
            # Smagorinsky eddy viscosity wavenumber damping
            self.smag_coeff = self.subgrid.get('smagorinsky_coeff', 0.0)
            
            # Biharmonic eddy diffusion
            self.eddy_diff = self.subgrid.get('eddy_diffusivity', 0.0)
            if self.eddy_diff > 0:
                self.eddy_filter = np.exp(-self.eddy_diff * self.dt * self.K2**2)
            else:
                self.eddy_filter = 1.0
        
        # Store effective dissipation
        self.r_effective = r_effective
        self.nu_effective = nu_effective
        
    def initialize_vorticity(self, vortex_params):
        """Initialize vorticity fields with Gaussian vortices"""
        q1 = np.zeros((self.ny, self.nx))
        q2 = np.zeros((self.ny, self.nx))
        
        for vortex in vortex_params:
            x0, y0 = vortex['x'], vortex['y']
            sigma = vortex['sigma']
            amp1 = vortex['amp1']
            amp2 = vortex['amp2']
            
            r2 = (self.X - x0)**2 + (self.Y - y0)**2
            gauss = np.exp(-r2 / (2 * sigma**2))
            
            q1 += amp1 * gauss
            q2 += amp2 * gauss
        
        return q1, q2
    
    def q_to_psi(self, q1, q2):
        """Invert PV to streamfunction"""
        q1h = fft2(q1)
        q2h = fft2(q2)
        
        # Simple inversion
        psi1h = -q1h / self.K2
        psi2h = -q2h / self.K2
        
        psi1h[0, 0] = 0
        psi2h[0, 0] = 0
        
        psi1 = np.real(ifft2(psi1h))
        psi2 = np.real(ifft2(psi2h))
        
        return psi1, psi2
    
    def jacobian(self, a, b):
        """Arakawa Jacobian J(a,b)"""
        ah = fft2(a) * self.dealias
        bh = fft2(b) * self.dealias
        
        ax = np.real(ifft2(1j * self.KX * ah))
        ay = np.real(ifft2(1j * self.KY * ah))
        bx = np.real(ifft2(1j * self.KX * bh))
        by = np.real(ifft2(1j * self.KY * bh))
        
        jac = ax * by - ay * bx
        jach = fft2(jac) * self.dealias
        
        return np.real(ifft2(jach))
    
    def compute_smagorinsky_viscosity(self, psi):
        """Compute Smagorinsky eddy viscosity if enabled"""
        if self.smag_coeff <= 0:
            return 0.0
        
        psih = fft2(psi)
        
        # Compute velocity gradients
        dudx = np.real(ifft2(-1j * self.KX * 1j * self.KY * psih))
        dudy = np.real(ifft2(-1j * self.KY * 1j * self.KY * psih))
        dvdx = np.real(ifft2(1j * self.KX * 1j * self.KX * psih))
        dvdy = np.real(ifft2(1j * self.KX * 1j * self.KY * psih))
        
        # Strain rate magnitude
        S = np.sqrt(2 * (dudx**2 + dvdy**2) + (dudy + dvdx)**2)
        
        # Smagorinsky viscosity: nu_eddy = (C_s * dx)^2 * |S|
        nu_smag = (self.smag_coeff * self.dx)**2 * S
        
        return np.mean(nu_smag)
    
    def rhs(self, q1, q2):
        """Right hand side of QG equations"""
        psi1, psi2 = self.q_to_psi(q1, q2)
        
        # Advection
        adv1 = -self.jacobian(psi1, q1)
        adv2 = -self.jacobian(psi2, q2)
        
        # Beta effect
        psi1h = fft2(psi1)
        psi2h = fft2(psi2)
        beta1 = -self.beta * np.real(ifft2(1j * self.KX * psi1h))
        beta2 = -self.beta * np.real(ifft2(1j * self.KX * psi2h))
        
        # Ekman friction on layer 2
        ekman = self.r_effective * np.real(ifft2(-self.K2 * psi2h))
        
        dq1dt = adv1 + beta1
        dq2dt = adv2 + beta2 + ekman
        
        # Add subgrid energy/enstrophy corrections if specified
        if self.apply_subgrid:
            energy_corr = self.subgrid.get('energy_correction', 0.0)
            enstrophy_corr = self.subgrid.get('enstrophy_correction', 0.0)
            
            if energy_corr != 0.0:
                # Energy correction acts like backscatter
                dq1dt += energy_corr * np.random.randn(*q1.shape) * 1e-8
                dq2dt += energy_corr * np.random.randn(*q2.shape) * 1e-8
            
            if enstrophy_corr != 0.0:
                # Enstrophy correction acts like additional dissipation
                dq1dt -= enstrophy_corr * q1
                dq2dt -= enstrophy_corr * q2
        
        return dq1dt, dq2dt
    
    def step_ab3(self, q1, q2, dq1_hist, dq2_hist):
        """Adams-Bashforth 3rd order step"""
        dq1, dq2 = self.rhs(q1, q2)
        
        dq1_hist.append(dq1.copy())
        dq2_hist.append(dq2.copy())
        
        if len(dq1_hist) > 3:
            dq1_hist.pop(0)
            dq2_hist.pop(0)
        
        # Time stepping
        if len(dq1_hist) == 1:
            q1_new = q1 + self.dt * dq1_hist[0]
            q2_new = q2 + self.dt * dq2_hist[0]
        elif len(dq1_hist) == 2:
            q1_new = q1 + self.dt * (1.5 * dq1_hist[1] - 0.5 * dq1_hist[0])
            q2_new = q2 + self.dt * (1.5 * dq2_hist[1] - 0.5 * dq2_hist[0])
        else:
            q1_new = q1 + self.dt * (23/12 * dq1_hist[2] - 16/12 * dq1_hist[1] + 5/12 * dq1_hist[0])
            q2_new = q2 + self.dt * (23/12 * dq2_hist[2] - 16/12 * dq2_hist[1] + 5/12 * dq2_hist[0])
        
        # Apply filters
        q1h = fft2(q1_new) * self.filter * self.dealias
        q2h = fft2(q2_new) * self.filter * self.dealias
        
        if self.apply_subgrid and self.eddy_diff > 0:
            q1h *= self.eddy_filter
            q2h *= self.eddy_filter
        
        q1_new = np.real(ifft2(q1h))
        q2_new = np.real(ifft2(q2h))
        
        return q1_new, q2_new
    
    def compute_energy(self, q1, q2):
        """Compute kinetic energy"""
        psi1, psi2 = self.q_to_psi(q1, q2)
        psi1h = fft2(psi1)
        psi2h = fft2(psi2)
        
        u1 = np.real(ifft2(-1j * self.KY * psi1h))
        v1 = np.real(ifft2(1j * self.KX * psi1h))
        u2 = np.real(ifft2(-1j * self.KY * psi2h))
        v2 = np.real(ifft2(1j * self.KX * psi2h))
        
        KE1 = 0.5 * self.H1 * np.mean(u1**2 + v1**2)
        KE2 = 0.5 * self.H2 * np.mean(u2**2 + v2**2)
        
        return KE1, KE2, KE1 + KE2
    
    def compute_enstrophy(self, q1, q2):
        """Compute enstrophy"""
        return 0.5 * np.mean(q1**2), 0.5 * np.mean(q2**2)
    
    def compute_velocity(self, psi):
        """Get velocity from streamfunction"""
        psih = fft2(psi)
        u = np.real(ifft2(-1j * self.KY * psih))
        v = np.real(ifft2(1j * self.KX * psih))
        return u, v
