"""
Native Optuna Optimization Engine (MOTPE).

This module provides an alternative optimization backend using Optuna's
Multi-Objective Tree-structured Parzen Estimator (MOTPE). It is designed to replace
or coexist with the Pymoo genetic algorithm.

**Key Features:**
1.  **Bayesian Optimization:** Uses MOTPE to efficiently explore the parameter space
    by modeling the probability density of good vs. bad solutions.
2.  **Persistent Storage:** Saves all trial data to a local SQLite database (`optimization.db`),
    allowing pause/resume and post-hoc analysis.
3.  **Live Dashboard:** Optionally launches the `optuna-dashboard` for real-time
    visualization of the Pareto front and parameter importances.
4.  **Vectorized Loss:** Implements a high-performance objective function that avoids
    Python loops during evaluation.


"""

import numpy as np
import optuna
import logging
import time
import os
import threading
import webbrowser
from dataclasses import dataclass
from typing import List

# Check for dashboard support (optional dependency)
try:
    from optuna_dashboard import run_server

    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

from global_model.params import unpack_params
from global_model.utils import _base_idx
from global_model.jacspeedup import solve_custom

logger = logging.getLogger(__name__)


@dataclass
class OptunaResult:
    """
    Standardized result container to ensure compatibility with existing export scripts.

    This mimics the structure of Pymoo's Result object so that downstream plotting
    and export functions don't need to change.
    """
    X: np.ndarray  # Best Parameters (n_solutions, n_vars)
    F: np.ndarray  # Best Objectives (n_solutions, n_objs)
    history: List
    algorithm: str = "Optuna-MOTPE"
    exec_time: float = 0.0


class NativeOptunaObjective:
    """
    The optimization kernel designed specifically for Optuna.

    This class is callable `objective(trial)` and handles the entire evaluation pipeline:
    Parameter Suggestion -> System Update -> ODE Simulation -> Loss Calculation.

    It pre-calculates array indices (`_build_fast_indices`) to enable fully vectorized
    loss computation, which is critical for performance when running thousands of trials.
    """

    def __init__(self, sys, slices, loss_data, time_grid, lambdas, xl, xu):
        self.sys = sys
        self.slices = slices
        self.loss_data = loss_data
        self.time_grid = time_grid
        self.lambdas = lambdas
        self.xl = xl
        self.xu = xu

        # Unpack weights for fast access
        self.w_prot = loss_data.get("w_prot", 1.0)
        self.w_rna = loss_data.get("w_rna", 1.0)
        self.w_pho = loss_data.get("w_pho", 1.0)

        # Build fast indices to avoid dictionary lookups inside the loop
        self._build_fast_indices()

    def _build_fast_indices(self):
        """Helper to vectorize the sparse data lookup using numpy arrays."""
        # Protein Indices
        if "prot_rows" in self.loss_data:
            self.p_rows = self.loss_data["prot_rows"]
            self.p_cols = self.loss_data["prot_cols"]
        else:
            pr, pc = [], []
            for (r, c) in self.loss_data["prot_idx"]:
                pr.append(r);
                pc.append(c)
            self.p_rows = np.array(pr, dtype=np.int32)
            self.p_cols = np.array(pc, dtype=np.int32)

        # RNA Indices
        if "rna_rows" in self.loss_data:
            self.r_rows = self.loss_data["rna_rows"]
            self.r_cols = self.loss_data["rna_cols"]
        else:
            rr, rc = [], []
            for (r, c) in self.loss_data["rna_idx"]:
                rr.append(r);
                rc.append(c)
            self.r_rows = np.array(rr, dtype=np.int32)
            self.r_cols = np.array(rc, dtype=np.int32)

        # Phospho Indices
        if "pho_rows" in self.loss_data:
            self.ph_rows = self.loss_data["pho_rows"]
            self.ph_cols = self.loss_data["pho_cols"]
        else:
            phr, phc = [], []
            for (r, c) in self.loss_data["pho_idx"]:
                phr.append(r);
                phc.append(c)
            self.ph_rows = np.array(phr, dtype=np.int32)
            self.ph_cols = np.array(phc, dtype=np.int32)

    def __call__(self, trial):
        """
        The main evaluation step called by Optuna.
        """
        # 1. Define Search Space
        # Suggest float values for all variables based on bounds xl/xu
        n_vars = len(self.xl)
        theta = np.zeros(n_vars)

        for i in range(n_vars):
            theta[i] = trial.suggest_float(f"p_{i}", self.xl[i], self.xu[i])

        # 2. Update System Physics
        # Map flat vector 'theta' back to physical parameters (A_i, c_k, etc.)
        params = unpack_params(theta, self.slices)
        self.sys.update(**params)

        # 3. Simulate
        try:
            # Generate Initial Conditions (usually steady state or data-driven)
            y0 = self.sys.y0()

            # Solve using the fast JIT backend
            # Note: We use relaxed tolerances for optimization speed (1e-4)
            # Final simulation for plotting will be higher precision.
            Y = solve_custom(self.sys, y0, self.time_grid, rtol=1e-4, atol=1e-6)

            # --- COMPUTE MSE LOSS (Vectorized) ---
            # 
            # Instead of looping, we index Y using pre-calculated row/col arrays.

            # Protein Loss
            flat_prot_pred = Y[self.p_rows, self.p_cols]
            # Use weighted difference if weights are in loss_data, else global
            diff_p = (flat_prot_pred - self.loss_data["prot_target"])
            mse_p = np.mean(diff_p ** 2)

            # RNA Loss
            flat_rna_pred = Y[self.r_rows, self.r_cols]
            diff_r = (flat_rna_pred - self.loss_data["rna_target"])
            mse_r = np.mean(diff_r ** 2)

            # Phospho Loss
            flat_pho_pred = Y[self.ph_rows, self.ph_cols]
            diff_ph = (flat_pho_pred - self.loss_data["pho_target"])
            mse_ph = np.mean(diff_ph ** 2)

            # Return tuple for multi-objective optimization
            return mse_p, mse_r, mse_ph

        except Exception as e:
            # Prune trials that crash the solver (stiffness or numerical instability)
            # This allows the optimizer to learn to avoid unstable regions.
            # logger.warning(f"Trial {trial.number} solver failed: {e}")
            raise optuna.TrialPruned()


def _augment_loss_data(idx, loss_data, time_grid, df_prot, df_rna, df_pho):
    """
    Pre-processes experimental dataframes into numerical indices for fast lookup.

    This ensures 'prot_idx' (Time Index, State Index) tuples exist in loss_data.
    It resolves string names (e.g., "AKT1") to integer state indices using the `Index` object.

    Args:
        idx: The Index object containing mappings.
        loss_data: The dictionary to populate.
        time_grid: The simulation time grid.
        df_*: Pandas DataFrames containing observed data.
    """
    # 1. PROTEIN
    if "prot_idx" not in loss_data:
        logger.info("[Optuna] Building Protein indices...")
        p_list = []
        targets = []

        if df_prot is not None and not df_prot.empty:
            for _, row in df_prot.iterrows():
                t_val = row['time']
                prot_name = row['protein']
                t_idx = _base_idx(time_grid, t_val)

                # Use p2i from Index class
                if prot_name in idx.p2i:
                    p_idx = idx.p2i[prot_name]
                    # Map to state index: offset_y[i] is RNA, +1 is Protein
                    state_idx = idx.offset_y[p_idx] + 1
                    p_list.append((t_idx, state_idx))
                    targets.append(row['fc'])

        loss_data["prot_idx"] = p_list
        loss_data["prot_target"] = np.array(targets, dtype=float)

    # 2. RNA
    if "rna_idx" not in loss_data:
        logger.info("[Optuna] Building RNA indices...")
        r_list = []
        targets = []
        if df_rna is not None and not df_rna.empty:
            for _, row in df_rna.iterrows():
                t_val = row['time']
                prot_name = row['protein']
                t_idx = _base_idx(time_grid, t_val)

                if prot_name in idx.p2i:
                    p_idx = idx.p2i[prot_name]
                    # Map to state index: offset_y[i] is RNA
                    state_idx = idx.offset_y[p_idx]
                    r_list.append((t_idx, state_idx))
                    targets.append(row['fc'])
        loss_data["rna_idx"] = r_list
        loss_data["rna_target"] = np.array(targets, dtype=float)

    # 3. PHOSPHO
    if "pho_idx" not in loss_data:
        logger.info("[Optuna] Building Phospho indices...")
        ph_list = []
        targets = []
        if df_pho is not None and not df_pho.empty:
            for _, row in df_pho.iterrows():
                t_val = row['time']
                site = row['psite']
                prot_name = row['protein']
                t_idx = _base_idx(time_grid, t_val)

                if prot_name in idx.p2i:
                    p_idx = idx.p2i[prot_name]
                    # Look up site in the list of sites for this protein
                    # idx.sites is a list of lists: [['S_1', 'Y_4'], ['T_9'], ...]
                    protein_sites = idx.sites[p_idx]

                    if site in protein_sites:
                        local_s = protein_sites.index(site)
                        y_start = idx.offset_y[p_idx]
                        # Site state = RNA(1) + Protein(1) + local_site_index
                        state_idx = y_start + 2 + local_s
                        ph_list.append((t_idx, state_idx))
                        targets.append(row['fc'])

        loss_data["pho_idx"] = ph_list
        loss_data["pho_target"] = np.array(targets, dtype=float)

    return loss_data


def run_optuna_solver(args, sys, loss_data, slices, xl, xu, defaults, lambdas, time_grid,
                      df_prot, df_rna, df_pho, n_trials=5000):
    """
    Main driver function for the Optuna Optimization pipeline.

    It replaces the Pymoo genetic algorithm with Optuna's MOTPE.

    Workflow:
    1.  Augment loss data with fast indices.
    2.  Setup persistent SQLite storage (`optimization.db`).
    3.  Launch the Optuna Dashboard (background thread) for real-time monitoring.
    4.  Run the optimization loop.
    5.  Extract the Pareto front from the database.

    Args:
        args: Configuration namespace (seed, output_dir, etc.).
        sys: The system object.
        loss_data: Pre-computed data for loss function.
        slices: Parameter slices.
        xl, xu: Lower/Upper parameter bounds.
        n_trials: Number of evaluations to run.

    Returns:
        OptunaResult: Standardized result object containing Best Parameters (X) and Objectives (F).
    """
    start_time = time.time()

    # --- 0. Ensure Indices Exist ---
    loss_data = _augment_loss_data(sys.idx, loss_data, time_grid, df_prot, df_rna, df_pho)

    # --- 1. Setup Persistent Storage & Dashboard ---
    db_dir = os.path.join(args.output_dir, "optuna_solver")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "optimization.db")
    storage_url = f"sqlite:///{db_path}"
    study_name = "GlobalModel_Optimization"

    logger.info("=" * 60)
    logger.info("🚀 STARTING NATIVE OPTUNA OPTIMIZATION (MOTPE)")
    logger.info(f"    Storage: {storage_url}")
    logger.info(f"    Trials:  {n_trials}")
    logger.info("=" * 60)

    # --- 2. Create Study ---
    # Use TPESampler with multivariate=True for capturing parameter dependencies.
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        n_startup_trials=min(args.pop, n_trials // 10)
    )

    storage = optuna.storages.RDBStorage(url=storage_url)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["minimize", "minimize", "minimize"],  # Multi-objective
        sampler=sampler,
        load_if_exists=True
    )

    # --- 3. Dashboard Launch ---
    # 
    if HAS_DASHBOARD:
        def start_dashboard():
            try:
                run_server(storage, host="127.0.0.1", port=8081)
            except Exception:
                pass

        t = threading.Thread(target=start_dashboard, daemon=True)
        t.start()
        logger.info("[Optuna] Dashboard live at http://127.0.0.1:8081")

    # --- 4. Optimization Loop ---
    objective = NativeOptunaObjective(sys, slices, loss_data, time_grid, lambdas, xl, xu)

    try:
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        logger.info("[Optuna] Interrupted by user. Finalizing results...")

    # --- 5. Extract Pareto Front ---
    # 
    logger.info("[Optuna] Extracting Pareto Front from Database...")

    pareto_trials = study.best_trials
    n_pareto = len(pareto_trials)
    n_vars = len(xl)
    n_obj = 3

    X_out = np.zeros((n_pareto, n_vars))
    F_out = np.zeros((n_pareto, n_obj))

    for i, trial in enumerate(pareto_trials):
        # Extract params in correct order based on "p_0", "p_1"... keys
        for k_str, val in trial.params.items():
            idx_p = int(k_str.split("_")[1])
            X_out[i, idx_p] = val
        F_out[i, :] = trial.values

    elapsed = time.time() - start_time

    res = OptunaResult(X=X_out, F=F_out, history=[], exec_time=elapsed)

    logger.info("=" * 60)
    logger.info(f"🏆 OPTIMIZATION COMPLETE")
    logger.info(f"    Pareto Solutions: {n_pareto}")
    logger.info(f"    Total Time:       {elapsed / 60:.2f} min")
    logger.info("=" * 60)

    return res
