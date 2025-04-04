import numpy as np
from scipy.optimize import minimize
from config.logconf import setup_logger
from models.testmod import TEST_MODEL, HILL_N, K_HALF, PROCESSIVITY, NEGATIVE_FEEDBACK_CONSTANT

logger = setup_logger()

def initial_condition(num_psites: int) -> list:
    """
    Compute steady-state conditions for the given model.

    Common parameters:
      A, B, C, D = 1, 1, 1, 1
      S_rates = ones(num_psites)
      D_rates = ones(num_psites)

    The parameter vector for the ODE models is:
      [A, B, C, D, S1, ..., S_n, D1, ..., D_n]

    Extra (hardcoded) parameters per model:
      - semi_processive: p_proc = 0.8
      - oscillating:      A_osc = 1.0, omega = 0.5 (steady state computed at t=0)
      - noise:            noise_std = 0.01 (set to 0 for steady state)
      - cooperative:      hill_n = 2, K_half = 0.5
    """
    A, B, C, D = 1, 1, 1, 1
    S_rates = np.ones(num_psites)
    D_rates = np.ones(num_psites)

    if TEST_MODEL == 'semi_processive':
        # Differential Equations (Semi-processive):
        #   dR/dt = A - B·R
        #   dP/dt = C·R - D·P - p_proc·S1·P + P1
        #   For n = 1: dP1/dt = p_proc·S1·P - (1 + D1)·P1
        #   For n > 1:
        #     dP1/dt = p_proc·S1·P - (1 + p_proc·S2 + D1)·P1 + P2
        #     dPᵢ/dt = p_proc·Sᵢ·Pᵢ₋₁ - (1 + p_proc·Sᵢ₊₁ + Dᵢ)·Pᵢ + Pᵢ₊₁  (i=2..n-1)
        #     dPₙ/dt = p_proc·Sₙ·Pₙ₋₁ - (1 + Dₙ)·Pₙ
        p_proc = PROCESSIVITY

        def steady_state_equations(y):
            R, P, *P_sites = y
            eq_R = A - B * R
            eq_P = C * R - D * P - p_proc * S_rates[0] * P + (P_sites[0] if num_psites >= 1 else 0)
            if num_psites == 1:
                eq_P1 = p_proc * S_rates[0] * P - (1 + D_rates[0]) * P_sites[0]
                return [eq_R, eq_P, eq_P1]
            else:
                eqs = []
                eq_P1 = p_proc * S_rates[0] * P - (1 + p_proc * S_rates[1] + D_rates[0]) * P_sites[0] + P_sites[1]
                eqs.append(eq_P1)
                for i in range(1, num_psites - 1):
                    eq_pi = p_proc * S_rates[i] * P_sites[i - 1] - (1 + p_proc * S_rates[i + 1] + D_rates[i]) * P_sites[
                        i] + P_sites[i + 1]
                    eqs.append(eq_pi)
                eq_last = p_proc * S_rates[-1] * P_sites[-2] - (1 + D_rates[-1]) * P_sites[-1]
                eqs.append(eq_last)
                return [eq_R, eq_P] + eqs

    elif TEST_MODEL == 'neg_feedback':
        k_fb = NEGATIVE_FEEDBACK_CONSTANT
        def steady_state_equations(y):
            R, P, *P_sites = y
            # Compute the feedback factor: f = 1/(1 + k_fb * (P1 + P2 + ... + Pn))
            f = 1.0 / (1.0 + k_fb * sum(P_sites))
            eq_R = A - B * R
            eq_P = C * R - D * P - f * S_rates[0] * P + (P_sites[0] if num_psites >= 1 else 0)
            if num_psites == 1:
                eq_P1 = f * S_rates[0] * P - (1 + D_rates[0]) * P_sites[0]
                return [eq_R, eq_P, eq_P1]
            else:
                eqs = []
                eq_P1 = f * S_rates[0] * P - (1 + f * S_rates[1] + D_rates[0]) * P_sites[0] + P_sites[1]
                eqs.append(eq_P1)
                for i in range(1, num_psites - 1):
                    eq_pi = f * S_rates[i] * P_sites[i - 1] - (1 + f * S_rates[i + 1] + D_rates[i]) * P_sites[i] + P_sites[
                        i + 1]
                    eqs.append(eq_pi)
                eq_last = f * S_rates[-1] * P_sites[-2] - (1 + D_rates[-1]) * P_sites[-1]
                eqs.append(eq_last)
                return [eq_R, eq_P] + eqs

    elif TEST_MODEL == 'site_affinity':
        # Differential Equations (Site-specific Binding Affinity):
        #   dR/dt = A - B·R
        #   dP/dt = C·R - D·P - S1·P + P1
        #   For n = 1: dP1/dt = S1·P - (1 + D1)·P1
        #   For n > 1:
        #     dP1/dt = S1·P - (1 + S2 + D1)·P1 + P2
        #     dPᵢ/dt = Sᵢ·Pᵢ₋₁ - (1 + Sᵢ₊₁ + Dᵢ)·Pᵢ + Pᵢ₊₁, i = 2..n-1
        #     dPₙ/dt = Sₙ·Pₙ₋₁ - (1 + Dₙ)·Pₙ
        def steady_state_equations(y):
            R, P, *P_sites = y
            eq_R = A - B * R
            eq_P = C * R - D * P - S_rates[0] * P + (P_sites[0] if num_psites >= 1 else 0)
            if num_psites == 1:
                eq_P1 = S_rates[0] * P - (1 + D_rates[0]) * P_sites[0]
                return [eq_R, eq_P, eq_P1]
            else:
                eqs = []
                eq_P1 = S_rates[0] * P - (1 + S_rates[1] + D_rates[0]) * P_sites[0] + P_sites[1]
                eqs.append(eq_P1)
                for i in range(1, num_psites - 1):
                    eq_pi = S_rates[i] * P_sites[i - 1] - (1 + S_rates[i + 1] + D_rates[i]) * P_sites[i] + P_sites[
                        i + 1]
                    eqs.append(eq_pi)
                eq_last = S_rates[-1] * P_sites[-2] - (1 + D_rates[-1]) * P_sites[-1]
                eqs.append(eq_last)
                return [eq_R, eq_P] + eqs

    elif TEST_MODEL == 'crosstalk':
        # Differential Equations (Crosstalk):
        #   Let φ = (P1 + ... + Pn)/n.
        #   dR/dt = A - B·R
        #   dP/dt = C·R - D·P - S1·P·(1 + φ) + P1
        #   For n = 1: dP1/dt = S1·P·(1 + φ) - (1 + D1)·P1
        #   For n > 1:
        #     dP1/dt = S1·P·(1 + φ) - (1 + S2·(1 + φ) + D1)·P1 + P2
        #     dPᵢ/dt = Sᵢ·Pᵢ₋₁·(1 + φ) - (1 + Sᵢ₊₁·(1 + φ) + Dᵢ)·Pᵢ + Pᵢ₊₁
        def steady_state_equations(y):
            R, P, *P_sites = y
            mod_factor = np.sum(P_sites) / num_psites if num_psites > 0 else 0
            eq_R = A - B * R
            eq_P = C * R - D * P - S_rates[0] * P * (1 + mod_factor) + (P_sites[0] if num_psites >= 1 else 0)
            if num_psites == 1:
                eq_P1 = S_rates[0] * P * (1 + mod_factor) - (1 + D_rates[0]) * P_sites[0]
                return [eq_R, eq_P, eq_P1]
            else:
                eqs = []
                eq_P1 = S_rates[0] * P * (1 + mod_factor) - (1 + S_rates[1] * (1 + mod_factor) + D_rates[0]) * P_sites[
                    0] + P_sites[1]
                eqs.append(eq_P1)
                for i in range(1, num_psites - 1):
                    eq_pi = S_rates[i] * P_sites[i - 1] * (1 + mod_factor) - (
                                1 + S_rates[i + 1] * (1 + mod_factor) + D_rates[i]) * P_sites[i] + P_sites[i + 1]
                    eqs.append(eq_pi)
                eq_last = S_rates[-1] * P_sites[-2] * (1 + mod_factor) - (1 + D_rates[-1]) * P_sites[-1]
                eqs.append(eq_last)
                return [eq_R, eq_P] + eqs

    elif TEST_MODEL == 'cooperative':
        # Differential Equations (Cooperative, Hill-type):
        #   Let activation φ = (ΣP_i)^hill_n / (K_half^hill_n + (ΣP_i)^hill_n + ε), with hill_n = 2, K_half = 0.5.
        #   dR/dt = A - B·R
        #   dP/dt = C·R - D·P - S1·P·φ + P1
        #   For n = 1: dP1/dt = S1·P·φ - (1 + D1)·P1
        #   For n > 1:
        #     dP1/dt = S1·P·φ - (1 + S2·φ + D1)·P1 + P2
        #     dPᵢ/dt = Sᵢ·Pᵢ₋₁·φ - (1 + Sᵢ₊₁·φ + Dᵢ)·Pᵢ + Pᵢ₊₁, for i = 2,...,n-1
        #     dPₙ/dt = Sₙ·Pₙ₋₁·φ - (1 + Dₙ)·Pₙ

        # Works well for hill_n = 0.5 and K_half = 2
        hill_n = HILL_N
        K_half = K_HALF

        def steady_state_equations(y):
            R, P, *P_sites = y
            sum_P = np.sum(P_sites)
            activation = (sum_P ** hill_n) / (K_half ** hill_n + sum_P ** hill_n + 1e-8)
            eq_R = A - B * R
            eq_P = C * R - D * P - S_rates[0] * P * activation + (P_sites[0] if num_psites >= 1 else 0)
            if num_psites == 1:
                eq_P1 = S_rates[0] * P * activation - (1 + D_rates[0]) * P_sites[0]
                return [eq_R, eq_P, eq_P1]
            else:
                eqs = []
                eq_P1 = S_rates[0] * P * activation - (1 + S_rates[1] * activation + D_rates[0]) * P_sites[0] + P_sites[
                    1]
                eqs.append(eq_P1)
                for i in range(1, num_psites - 1):
                    eq_pi = S_rates[i] * P_sites[i - 1] * activation - (1 + S_rates[i + 1] * activation + D_rates[i]) * \
                            P_sites[i] + P_sites[i + 1]
                    eqs.append(eq_pi)
                eq_last = S_rates[-1] * P_sites[-2] * activation - (1 + D_rates[-1]) * P_sites[-1]
                eqs.append(eq_last)
                return [eq_R, eq_P] + eqs
    else:
        raise ValueError("Unknown model type: " + TEST_MODEL)

    y0_guess = np.ones(num_psites + 2)
    bounds_local = [(1e-8, None)] * (num_psites + 2)

    result = minimize(lambda y: 0, y0_guess, method='SLSQP', bounds=bounds_local,
                      constraints={'type': 'eq', 'fun': steady_state_equations},
                      options={'maxiter': 10000})

    logger.info("Steady-State conditions calculated for TEST model: " + TEST_MODEL)
    if result.success:
        return result.x.tolist()
    else:
        logger.info(f"result: {result}")
        raise ValueError("Failed to find steady-state conditions for model: " + TEST_MODEL)
