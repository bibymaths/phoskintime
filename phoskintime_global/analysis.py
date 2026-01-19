import os
import numpy as np
import matplotlib.pyplot as plt
from phoskintime_global.simulate import simulate_odeint


def simulate_until_steady(sys, t_max=1440.0, n_points=1000):
    """
    Simulates the system from t=0 to t_max (default 24h) to observe convergence.
    Uses log-spacing to capture fast initial phosphorylation and slow transcription.
    """
    # Log-space time grid (0, 0.001 ... t_max)
    t_log = np.logspace(np.log10(1e-3), np.log10(t_max), n_points - 1)
    t_eval = np.concatenate(([0.0], t_log))

    print(f"[SteadyState] Simulating for {t_max} minutes...")

    # Run simulation (Tight tolerances for accuracy)
    Y = simulate_odeint(sys, t_eval, rtol=1e-6, atol=1e-8, mxstep=50000)

    # Check rate of change at the end
    dt = t_eval[-1] - t_eval[-2]
    dist = np.linalg.norm(Y[-1] - Y[-2])
    rate = dist / dt

    print(f"[SteadyState] Final rate of change: {rate:.2e}")
    return t_eval, Y


def plot_steady_state_all(t, Y, sys, idx, output_dir):
    """
    Plots RNA, Protein, and Phospho trajectories for EVERY protein.
    Saves one PNG per protein.
    """
    save_dir = os.path.join(output_dir, "steady_state_plots")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[Plot] Saving plots for {len(idx.proteins)} proteins to: {save_dir}/")

    for i, p_name in enumerate(idx.proteins):
        st = idx.offset_y[i]

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax_rna, ax_prot, ax_phos = axes

        # 1. RNA
        ax_rna.plot(t, Y[:, st], color='#1f77b4', linewidth=2)
        ax_rna.set_title(f"{p_name} RNA")
        ax_rna.set_ylabel("Conc (a.u.)")

        # 2. Protein & Phospho
        p_unphos = Y[:, st + 1]
        ns = idx.n_sites[i]

        if ns > 0:
            # Sum phospho states
            p_phos_states = Y[:, st + 2: st + 2 + ns]
            total_phos = np.sum(p_phos_states, axis=1)
            total_prot = p_unphos + total_phos

            # Plot Phospho
            ax_phos.plot(t, total_phos, color='#d62728', linewidth=2, label="Total Phospho")
            # Plot individual sites (top 3 to avoid clutter)
            for j in range(min(ns, 3)):
                site_name = idx.sites[i][j]
                ax_phos.plot(t, p_phos_states[:, j], linestyle="--", alpha=0.6, label=site_name)
            ax_phos.legend(fontsize=8)
        else:
            total_prot = p_unphos
            ax_phos.text(0.5, 0.5, "No Phospho Sites", ha='center', transform=ax_phos.transAxes)

        # 3. Total Protein
        ax_prot.plot(t, total_prot, color='#2ca02c', linewidth=2)
        ax_prot.set_title(f"{p_name} Total Protein")

        # Styling
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (min)")
            if t[-1] > 100:  # Log scale for long times
                ax.set_xscale("symlog", linthresh=10.0)

        plt.suptitle(f"Dynamics: {p_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{p_name}_dynamics.png"), dpi=80)
        plt.close(fig)  # Free memory

    print("[Plot] Simulate until steady-state [Done].")