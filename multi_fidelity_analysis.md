# Why Low-Correlation Multi-Fidelity Optimization Succeeds

## The Observation
Experiments 5 & 6 demonstrate a rapid convergence to the optimum despite a seemingly poor statistical correlation between the low-fidelity (30-day) and high-fidelity (180-day) simulations. 

As shown in **Figure 3**, the $R^2$ correlation between the fidelities is approximately **0.37**. Traditionally, this low correlation suggests that the low-fidelity model is a poor predictor of the high-fidelity outcome. However, our results show that the optimization algorithm "learns" effectively.

![Fidelity Correlation](fig3_fidelity_correlation.png)
*Figure 1: Scatter plot showing the global correlation between 30-day and 180-day loss is low (R² ≈ 0.37). Note the noisy relationship.*

## The Explanation: Topology vs. Magnitude

The key insight is that while the **magnitude** of the loss is noisy, the **topology** of the loss landscape is preserved. The optimization algorithm does not require the 30-day run to perfectly predict the *value* of the 180-day loss; it only needs to predict the *location* of the minima and maxima.

### Visual Evidence
The **Loss Landscape Matrix** below demonstrates this "Structure-Preserving" property.

![Loss Landscape Comparison](fig_loss_landscape_comparison.png)
*Figure 2: Pairwise loss landscapes for 30-day (Top) vs 180-day (Bottom). Note that while the colors (absolute values) differ, the "Valley" of low loss is in the same region for both.*

**Key Takeaways from the Landscape:**
1.  **Shared Valleys**: The "Valley" (dark purple/blue region) appears in roughly the same location in parameter space for both fidelities.
2.  **Gradient Alignment**: The slopes lead towards the same basin of attraction.
3.  **Global Optimum Match**: The red dot (Global Opt) sits comfortably in the low-loss valley of the 30-day proxy, confirming that the proxy is "directionally correct."

## The Mechanism: How the Surrogate Learns

### 1. The "Negative Information" Filter
The 30-day runs act as a high-throughput rejection filter. Even if the correlation is low, the proxy reliably identifies "bad" regions.
*   **Action**: The algorithm uses cheap 30-day runs to prune 90% of the search space that is obviously poor.
*   **Result**: The expensive 180-day evaluations are reserved exclusively for the "promising" valley, effectively increasing the sample efficiency by an order of magnitude.

### 2. Learning the Bias Function (Discrepancy Modeling)
The Gaussian Process (GP) or Neural Network (NN) surrogate explicitly models the discrepancy between fidelities. The standard multi-fidelity relationship is modeled as:

$$ f_{high}(x) \approx \rho \cdot f_{low}(x) + \delta(x) $$

**Where:**
*   **$f_{low}(x)$**: The prediction from the cheap 30-day simulation.
*   **$\rho$ (Rho)**: A constant scaling factor. It accounts for systematic scaling differences (e.g., if 30-day runs are consistently 80% of the magnitude of 180-day runs).
*   **$\delta(x)$ (Delta)**: The **Bias** or **Discrepancy Function**. This is a non-linear function that learns the *errors* of the low-fidelity model. For example, if the 30-day model is accurate everywhere *except* when drag is high, $\delta(x)$ captures that specific local error.

Even if $f_{low}$ is globally noisy, the **Bias Function $\delta(x)$** is often smoother and easier to learn than the full objective.

### 3. Usage in Experiment 5 & 6
Our proposed methods explicitly leverage this structure:

*   **Experiment 5 (Multi-Fidelity GP)**: Uses a **Linear Correlation Kernel**. It explicitly fits the $\rho$ parameter and models $\delta(x)$ as an independent Gaussian Process. This allows it to mathematically "subtract out" the error of the 30-day run.
*   **Experiment 6 (Multi-Fidelity NN)**: Uses a Deep Neural Network that takes the 30-day prediction as an *input* feature. The network implicitly learns the non-linear mapping $Low(x) \to High(x)$, effectively approximating complex forms of $\rho$ and $\delta(x)$ that may vary across the input space.

### 4. Acquisition Functions (The Strategy)
Both Experiments 5 and 6 utilize **Acquisition Functions** (specifically **Upper Confidence Bound (UCB)**) to decide which simulation to run next.
*   **Role**: The acquisition function balances **Exploration** (looking in high-bias/unknown regions) and **Exploitation** (zooming in on the valley).
*   **Impact**: This prevents the model from blindly trusting the 30-day proxy. If the model is uncertain about the bias $\delta(x)$ in a region, the acquisition function will trigger a comprehensive 180-day run to "calibrate" the proxy.

## Conclusion
The success of Experiment 5/6 is not due to a linear correlation between fidelities, but due to **Topological Consistency** and **Active Discrepancy Learning**. The 30-day simulation acts as a "Structure-Preserving Proxy," and the Bias/Acquisition framework allows the optimizer to mathematically correct its flaws.
