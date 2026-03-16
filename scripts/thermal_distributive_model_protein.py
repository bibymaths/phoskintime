import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. System Parameters ---
S = 10.0  # Basal synthesis rate
D = 1.0  # Basal degradation rate
k_phos = 5.0  # Forward phosphorylation rate (Kinase activity)
k_dephos = 2.0  # Backward dephosphorylation rate (Phosphatase activity)

# Thermal Parameters
Tm = 40.0  # Melting temperature of our specific protein
c_fold = 0.8  # Steepness of the unfolding curve
k_penalty = 4.0  # Degradation multiplier for unfolded proteins


# --- 2. The Standard (Old) ODE Model ---
def standard_model(t, y):
    P, Pp = y  # Unphosphorylated (P) and Phosphorylated (Pp)

    # Standard mass-action kinetics
    dP_dt = S - D * P - k_phos * P + k_dephos * Pp
    dPp_dt = k_phos * P - D * Pp - k_dephos * Pp

    return [dP_dt, dPp_dt]


# --- 3. The Improved (Thermal) ODE Model ---
def thermal_model(t, y, T):
    P, Pp = y

    # Calculate folded fraction based on Temperature (T) and Melting Point (Tm)
    f_folded = 1.0 / (1.0 + np.exp(c_fold * (T - Tm)))
    f_unfolded = 1.0 - f_folded

    # Apply thermal penalties
    D_therm = D + (D * k_penalty * f_unfolded)
    P_active = P * f_folded

    # Thermodynamic-kinetic equations
    dP_dt = S - D_therm * P - k_phos * P_active + k_dephos * Pp
    dPp_dt = k_phos * P_active - D_therm * Pp - k_dephos * Pp

    return [dP_dt, dPp_dt]


# --- 4. Run the Simulations ---
# Define your new time and resolution parameters
FORWARD_TIME = 5.0   # Increase this to simulate further into the future (e.g., 50, 100, 500)
NUM_POINTS = 10000     # Increase this to get a much smoother, high-resolution curve

t_span = (0, FORWARD_TIME)
t_eval = np.linspace(0, FORWARD_TIME, NUM_POINTS)

y0 = [0.0, 0.0]  # Start with 0 protein

# Run Standard Model (Temperature independent, so we just run it once)
sol_standard = solve_ivp(standard_model, t_span, y0, t_eval=t_eval)

# Run Thermal Model at 37°C
sol_thermal_37 = solve_ivp(lambda t, y: thermal_model(t, y, T=37.0), t_span, y0, t_eval=t_eval)

# Run Thermal Model at 42°C (Heat Shock)
sol_thermal_42 = solve_ivp(lambda t, y: thermal_model(t, y, T=42.0), t_span, y0, t_eval=t_eval)

# --- 5. Plot the Results ---
plt.figure(figsize=(12, 6))

# Plot Phosphorylated Protein (Active Signal)
plt.plot(sol_standard.t, sol_standard.y[1], 'k--', linewidth=2, label='Standard Model (Ignores Temp)')
plt.plot(sol_thermal_37.t, sol_thermal_37.y[1], 'b-', linewidth=2.5, label='Thermal Model @ 37°C')
plt.plot(sol_thermal_42.t, sol_thermal_42.y[1], 'r-', linewidth=2.5, label='Thermal Model @ 42°C (Heat Shock)')

plt.title('Phosphorylated Protein Accumulation: Standard vs. Thermal Model', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Concentration of Phosphorylated Protein ($P_p$)', fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()