import optuna
import pandas as pd
import numpy as np
import logging
import os
import threading
import time
import webbrowser
import matplotlib.pyplot as plt
import datetime

# Check if optuna-dashboard is installed
try:
    from optuna_dashboard import run_server

    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

# Check if optuna has matplotlib visualization
try:
    from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, \
        plot_parallel_coordinate

    HAS_OPTUNA_VIZ = True
except ImportError:
    HAS_OPTUNA_VIZ = False

from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.callback import Callback
from global_model.optproblem import GlobalODE_MOO

logger = logging.getLogger(__name__)


# --- Pymoo Callback for Pruning ---
class OptunaPruningCallback(Callback):
    def __init__(self, trial, gen_step=5):
        super().__init__()
        self.trial = trial
        self.gen_step = gen_step

    def notify(self, algorithm):
        if algorithm.n_gen % self.gen_step == 0:
            pop = algorithm.pop
            F = pop.get("F")
            # Metric: Sum of MSEs of the best solution
            current_score = np.min(np.sum(F, axis=1))

            self.trial.report(current_score, algorithm.n_gen)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


class BioObjective:
    """
    Wraps the Pymoo optimization pipeline for Optuna.
    """

    def __init__(self, sys, loss_data, defaults, time_grid, runner, args, slices, xl, xu):
        self.sys = sys
        self.loss_data = loss_data
        self.defaults = defaults
        self.time_grid = time_grid
        self.runner = runner
        self.args = args
        self.slices = slices
        self.xl = xl
        self.xu = xu

    def __call__(self, trial):
        try:
            # 1. Suggest Hyperparameters
            l_prot = trial.suggest_float("lambda_protein", 1.0, 20.0)
            l_phos = trial.suggest_float("lambda_phospho", 0.5, 5.0)
            l_rna = trial.suggest_float("lambda_rna", 0.1, 2.0)
            l_prior = trial.suggest_float("lambda_prior", 0.01, 1.0, log=True)

            lambdas = {
                "protein": l_prot,
                "phospho": l_phos,
                "rna": l_rna,
                "prior": l_prior
            }

            logger.info(f"[Scan] Trial {trial.number}: Testing {lambdas}")

            if self.runner is None:
                raise ValueError("Elementwise runner is None! Check runner.py initialization.")

            # 2. Build Problem
            problem = GlobalODE_MOO(
                sys=self.sys,
                slices=self.slices,
                loss_data=self.loss_data,
                defaults=self.defaults,
                lambdas=lambdas,
                time_grid=self.time_grid,
                xl=self.xl,
                xu=self.xu,
                elementwise_runner=self.runner
            )

            # 3. Fast Optimization
            SCAN_GEN = max(40, self.args.n_gen // 5)
            ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

            algorithm = UNSGA3(
                pop_size=self.args.pop,
                ref_dirs=ref_dirs,
                sampling=LHS(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=1 / problem.n_var, eta=10),
            )

            pruning_callback = OptunaPruningCallback(trial, gen_step=5)

            res = pymoo_minimize(
                problem,
                algorithm,
                termination=DefaultMultiObjectiveTermination(n_max_gen=SCAN_GEN),
                callback=pruning_callback,
                verbose=False
            )

            # 4. Evaluate Metric
            if res.F is None or len(res.F) == 0:
                return float("inf")

            scores = (l_prot * res.F[:, 0]) + (l_rna * res.F[:, 1]) + (l_phos * res.F[:, 2])
            best_score = np.min(scores)
            best_idx = np.argmin(scores)

            trial.set_user_attr("mse_prot", float(res.F[best_idx, 0]))
            trial.set_user_attr("mse_rna", float(res.F[best_idx, 1]))
            trial.set_user_attr("mse_phos", float(res.F[best_idx, 2]))

            logger.info(f"[Scan] Trial {trial.number} Score: {best_score:.4f}")
            return best_score

        except optuna.TrialPruned:
            logger.info(f"[Scan] Trial {trial.number} pruned.")
            raise
        except Exception as e:
            logger.error(f"[Scan] Trial {trial.number} FAILED: {e}", exc_info=True)
            return float("inf")


def run_hyperparameter_scan(args, sys, loss_data, defaults, time_grid, runner, slices, xl, xu):
    """
    Orchestrates the Optuna study, launches dashboard, and saves results.
    """
    logger.info("=" * 60)
    logger.info("STARTING HYPERPARAMETER SCAN (OPTUNA + DASHBOARD)")
    logger.info("=" * 60)

    objective = BioObjective(sys, loss_data, defaults, time_grid, runner, args, slices, xl, xu)

    # Output setup
    scan_dir = os.path.join(args.output_dir, "hyperparam_scan")
    os.makedirs(scan_dir, exist_ok=True)

    # --- 1. Setup Persistent Storage (Required for Dashboard) ---
    db_path = os.path.join(scan_dir, "scan.db")
    storage_url = f"sqlite:///{db_path}"

    # Use unique name to prevent loading broken/empty previous studies
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"PhosKinTime_Scan_{timestamp}"

    logger.info(f"[Scan] Using database: {storage_url}")
    logger.info(f"[Scan] Study Name: {study_name}")

    storage = optuna.storages.RDBStorage(url=storage_url)

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=args.n_gen // 5, reduction_factor=3),
        load_if_exists=True
    )

    # --- 2. Launch Dashboard (Background Thread) ---
    if HAS_DASHBOARD:
        def start_dashboard():
            logger.info("[Dashboard] Starting server on http://127.0.0.1:8080 ...")
            try:
                # Removed 'quiet=True' as it caused errors in your version
                run_server(storage, host="127.0.0.1", port=8080)
            except Exception as e:
                logger.warning(f"[Dashboard] Failed to start: {e}")

        t = threading.Thread(target=start_dashboard, daemon=True)
        t.start()

        time.sleep(3)
        try:
            webbrowser.open("http://127.0.0.1:8080")
        except:
            pass
    else:
        logger.warning("[Dashboard] optuna-dashboard library not found.")

    # --- 3. Run Optimization ---
    try:
        # Run at least 20 trials
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        logger.info("[Scan] Interrupted by user. Saving current progress...")
    except Exception as e:
        logger.error(f"[Scan] Optimization crashed: {e}", exc_info=True)

    # --- 4. Robust Results Export ---
    # Check if we actually have results before crashing on 'best_params'
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(complete_trials) == 0:
        logger.error("[Scan] NO SUCCESSFUL TRIALS. Cannot save best params or plots.")
        logger.error("Check the logs above for 'Trial FAILED' messages to debug the root cause.")
        # Return defaults to prevent runner.py from crashing
        return {
            "lambda_protein": args.lambda_protein,
            "lambda_phospho": args.lambda_phospho,
            "lambda_rna": args.lambda_rna,
            "lambda_prior": args.lambda_prior
        }

    # Save Excel
    df = study.trials_dataframe()
    clean_cols = [c.replace("params_", "") for c in df.columns]
    df.columns = clean_cols
    excel_path = os.path.join(scan_dir, "scan_results.xlsx")
    df.to_excel(excel_path, index=False)
    logger.info(f"[Scan] Saved full results table to: {excel_path}")

    # Generate Plots
    if HAS_OPTUNA_VIZ:
        try:
            fig_imp = plot_param_importances(study)
            fig_imp.figure.savefig(os.path.join(scan_dir, "param_importance.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_imp.figure)

            fig_hist = plot_optimization_history(study)
            fig_hist.figure.savefig(os.path.join(scan_dir, "optimization_history.png"), dpi=300)
            plt.close(fig_hist.figure)

            fig_par = plot_parallel_coordinate(study)
            fig_par.figure.savefig(os.path.join(scan_dir, "parallel_coordinates.png"), dpi=300)
            plt.close(fig_par.figure)

            logger.info("[Scan] Saved visualization plots.")
        except Exception as e:
            logger.warning(f"[Scan] Visualization failed: {e}")

    logger.info("=" * 60)
    logger.info("SCAN COMPLETE")
    logger.info(f"Best Params: {study.best_params}")
    logger.info(f"Best Score:  {study.best_value:.4f}")
    logger.info("=" * 60)

    return study.best_params