# Experimental Plan & Data Requirements

This document outlines the specific experiments required to validate the "Multi-Fidelity Hybrid" paper and lists the data that must be saved to generate the necessary publication-quality plots.

---

## Part 1: List of Experiments

### **Experiment 0: Ground Truth & Control**
*   **Goal:** Establish the "Target" climate and the "Untuned" baseline.
*   **Setup:**
    *   **Truth Run:** High-Resolution (e.g., $512^2$ or $256^2$) simulation run for long duration (e.g., 30 years). This is the "Observation".
    *   **Control Run:** Low-Resolution (e.g., $64^2$ or $96^2$) simulation with *default/standard* parameters.
*   **Output:**
    *   "True" Climate Statistics (Mean $\psi$, $U_{zonal}$, EKE levels).
    *   "Control" Error (RMSE against Truth).

### **Experiment 1: Linear Sensitivity Screening (Green's Functions)**
*   **Goal:** Identify active parameter subspace and demonstrate the "Linearity limit".
*   **Method:**
    *   Perturb each of the $N$ parameters individually by $\pm \delta$.
    *   Run short simulations ($T_{short} = 30$ days).
    *   Calculate Response Matrix $J = \Delta \text{Stats} / \Delta \text{Param}$.
*   **Outcome:**
    *   Ranked list of sensitivities.
    *   Selection of Top-K "Active" parameters for Stage 2.

### **Experiment 2: The Benchmark Baselines (Competitors)**
*   **Goal:** Establish the cost/accuracy curve for standard methods.
*   **Run 2a: Pure Random Search (LHS):**
    *   Sampling: Latin Hypercube of full parameter space.
    *   Fidelity: Fixed High Fidelity (180 days) or Adaptive.
*   **Run 2b: Standard Gaussian Process (GP):**
    *   Optimizer: GP-UCB or EI.
    *   Fidelity: Single Fidelity (High Res, 180 days).
    *   Purpose: Represents state-of-the-art "Black Box" tuning.

### **Experiment 3: Deterministic/Robust NN Optimization**
*   **Goal:** Demonstrate Deep Learning optimization performance without hybrid filtering.
*   **Method:**
    *   Standard Bayesian Optimization loop.
    *   Surrogate: Neural Network (with MC Dropout/Ensemble).
    *   Fidelity: Single or Multi-fidelity (Adaptive).
*   **Script:** `exp3_NN.py`

### **Experiment 4: The NN Emulator (Optimization Strategy)**
*   **Goal:** Optimize based on learned physics mapping, not just error.
*   **Method:**
    *   **Step 1 (Train):** Learn mapping $f: \theta \to \mathbf{S}$ (Parameters $\to$ Statistics) using LHS data (from Exp 2a).
    *   **Step 2 (Optimize):** Define loss $L(\theta) = ||f(\theta) - \mathbf{S}_{true}||^2$.
    *   **Step 3 (Solver):** Use gradient descent on the differentiable NN emulator to find $\theta^*$.
    *   **Step 4 (Verify):** Run simulation with $\theta^*$ to validate.
*   **Script:** `exp4_Emulator.py`

### **Experiment 5: The Hybrid Multi-Fidelity Method (GF + BNN)**
*   **Goal:** Demonstrate superior efficiency by combining Linear Screening and Non-Linear Tuning.
*   **Phase 5a (Scout):** Run short simulations on *Screened* subspace (from Exp 1).
*   **Phase 5b (Expert):** Use pre-trained BNN to propose candidates at high fidelity.
*   **Script:** `exp5_Hybrid.py`

### **Experiment 6: Validating Hybrid Optimization (GF-GP vs GF-NN)**
*   **Goal:** Compare Gaussian Process and Neural Network surrogates within the Hybrid GF framework.
*   **Method:**
    *   **Phase 1 (Screening):** Green's Function (GF) screening (180d) to identify Top-K active parameters and optimize baseline.
    *   **Phase 2 (Optimization):** Multi-fidelity Bayesian Optimization (GP or NN) on active parameters, fixing others to GF-optimized values.
*   **Scripts:** `exp6_hybridGFGP.py` (GP-based), `exp6_GFNN.py` (NN-based).
*   **Key Innovation:** Uses GF-optimized values as warm-start baseline and fixed parameter values, effectively reducing dimensionality while improving the starting point.

### **Experiment 7: The Inverse Diagnostic (Analysis)**
*   **Goal:** Quantify parameter identifiability (Can we recover parameters from statistics?).
*   **Method:**
    *   Use the database of runs from Exp 2a/3a (Input: Stats, Output: Params).
    *   Train an "Inverse NN".
    *   Test on a held-out set: Predict parameters given observations.
*   **Outcome:** Scatter plots of True Param vs Predicted Param. High scatter = Low Identifiability (Equifinality).

---

## Part 2: Data & Metrics to Save

To generate the plots discussed in the critique (Convergence, Alignment, Fidelity Correlation), you MUST save the following structural data in a standardized JSON/CSV/NetCDF format.

### **A. The Master Optimization Log (JSON/CSV)**
*File: `optimization_log_{method_name}.json`*
For every single simulation run during optimization, save:
1.  **`experiment_id`**: Unique ID (e.g., `run_001`).
2.  **`method`**: "Hybrid", "GP", "Random".
3.  **`fidelity_level`**: "30_days", "180_days", etc.
4.  **`wall_clock_time`**: Actual compute time (seconds).
5.  **`simulated_days`**: Cost metric.
6.  **`parameters`**: Dictionary of parameter values $\{p_1, p_2, ..., p_n\}$.
7.  **`loss_total`**: The scalar objective function value.
8.  **`loss_components`**: Breakdown of loss (e.g., `{'psi_rmse': 0.1, 'eke_error': 0.05}`). *Crucial for debugging why the optimizer chose a specific point.*

### **B. Statistical Fields (NetCDF/Pickle)**
*File: `stats_{experiment_id}.nc`*
Do **NOT** save full 4D snapshots. Only save the time-averaged statistics used for the loss function. This allows for post-hoc analysis (e.g., "What did the flow actually look like for the best run?").
1.  **`psi_mean`**: Time-averaged streamfunction $\bar{\psi}(x,y)$.
2.  **`u_zonal_mean`**: Zonal mean velocity $\bar{U}(y)$.
3.  **`eke_spectrum`**: Energy spectrum $E(k)$ (1D wavenumber).
4.  **`q_flux`**: (Optional) Potential vorticity flux for physics diagnostics.

### **C. Sensitivity Data (For Exp 1)**
*File: `sensitivity_matrix.json`*
1.  **`parameter_names`**: List of perturbed parameters.
2.  **`perturbation_size`**: $\delta$.
3.  **`response_gradients`**: Vector of derivatives $\partial L / \partial p_i$.

---

## Part 3: Required Plots & Data Mapping

1.  **The Convergence Race (Cost vs Error)**
    *   *X-axis:* Cumulative Simulated Days (Cost).
    *   *Y-axis:* Validated Loss (RMSE of High-Fidelity run).
    *   *Data Needed:* `optimization_log` from Exp 2, 3.
    *   *Note:* Need to account for the "pre-training cost" of the Hybrid method (add the cost of the 30-day scouts to the start time).

2.  **Fidelity Correlation (Scatter)**
    *   *X-axis:* Loss (30-day run).
    *   *Y-axis:* Loss (180-day run).
    *   *Data Needed:* A subset of runs involving the *same* parameters run at *both* fidelities. (You may need to explicitly run a "Validation Set" where you take 50 random parameters and run them at BOTH 30 and 180 days to prove this correlation exists).

3.  **Identifiability Matrix (The Inverse Plot)**
    *   *Grid of Scatter Plots:* True Parameter vs Predicted Parameter.
    *   *Data Needed:* Results from Exp 4.

4.  **Physics Consistency Check (Spectra)**
    *   *Plot:* Energy Spectrum $E(k)$ vs Wavenumber $k$.
    *   *Lines:* Truth (Black), Baseline (Gray), Hybrid Optimized (Blue), GP Optimized (Red).
    *   *Data Needed:* `eke_spectrum` from the *Best* run of each method in Exp 1, 2, 3.
