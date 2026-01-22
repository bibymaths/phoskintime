import argparse
import atexit
import json
import logging
import os
from pathlib import Path

from global_model.optuna_solver import run_optuna_solver
from global_model.scan import run_hyperparameter_scan
from global_model.sensitivity import run_sensitivity_analysis

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
import numpy as np
import multiprocessing as mp
import pandas as pd

from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize

from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.cache import prepare_fast_loss_data
from global_model.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, RESULTS_DIR, MAX_ITERATIONS, \
    POPULATION_SIZE, SEED, REGULARIZATION_LAMBDA, REGULARIZATION_RNA, REGULARIZATION_PHOSPHO, TIME_POINTS_PHOSPHO, \
    REGULARIZATION_PROTEIN, NORMALIZE_FC_STEADY, USE_INITIAL_CONDITION_FROM_DATA, KINASE_NET_FILE, TF_NET_FILE, \
    MS_DATA_FILE, RNA_DATA_FILE, PHOSPHO_DATA_FILE, KINOPT_RESULTS_FILE, TFOPT_RESULTS_FILE, REFINE, NUM_REFINE, \
    WEIGHTING_METHOD_PROTEIN, WEIGHTING_METHOD_RNA, APP_NAME, VERSION, PARENT_PACKAGE, CITATION, DOI, GITHUB_URL, \
    DOCS_URL, SENSITIVITY_METRIC, SENSITIVITY_ANALYSIS, N_TRIALS, AVAILABLE_MODELS, OPTIMIZER, HYPERPARAM_SCAN
from global_model.io import load_data
from global_model.network import Index, KinaseInput, System
from global_model.optproblem import GlobalODE_MOO, get_weight_options, build_weight_functions
from global_model.params import init_raw_params, unpack_params
from global_model.refine import run_iterative_refinement
from global_model.simulate import simulate_and_measure
from global_model.utils import normalize_fc_to_t0, _base_idx, get_parameter_labels, calculate_bio_bounds, \
    load_config_toml
from global_model.export import export_pareto_front_to_excel, plot_gof_from_pareto_excel, plot_goodness_of_fit, \
    export_results, save_pareto_3d, save_parallel_coordinates, create_convergence_video, save_gene_timeseries_plots, \
    scan_prior_reg, export_S_rates, plot_s_rates_report, process_convergence_history, export_kinase_activities, \
    export_param_correlations, export_residuals, export_parameter_distributions
from global_model.analysis import simulate_until_steady, plot_steady_state_all
from frechet import frechet_distance
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


@atexit.register
def _close_log_handlers():
    lg = logging.getLogger()
    for h in list(lg.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass


def main():
    global problem
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinase-net", default=KINASE_NET_FILE)
    parser.add_argument("--tf-net", default=TF_NET_FILE)
    parser.add_argument("--ms", default=MS_DATA_FILE)
    parser.add_argument("--rna", default=RNA_DATA_FILE)
    parser.add_argument("--phospho", default=PHOSPHO_DATA_FILE)

    # kinopt and tfopt results
    parser.add_argument("--kinopt", default=KINOPT_RESULTS_FILE)
    parser.add_argument("--tfopt", default=TFOPT_RESULTS_FILE)

    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--cores", type=int, default=os.cpu_count())

    # Pymoo
    parser.add_argument("--n-gen", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)

    # Loss weights
    parser.add_argument("--lambda-prior", type=float, default=REGULARIZATION_LAMBDA)
    parser.add_argument("--lambda-protein", type=float, default=REGULARIZATION_PROTEIN)
    parser.add_argument("--lambda-rna", type=float, default=REGULARIZATION_RNA)
    parser.add_argument("--lambda-phospho", type=float, default=REGULARIZATION_PHOSPHO)

    # Data inference
    parser.add_argument("--normalize-fc-steady", action="store_true", default=NORMALIZE_FC_STEADY)
    parser.add_argument("--use-initial-condition-from-data", action="store_true",
                        default=USE_INITIAL_CONDITION_FROM_DATA)
    parser.add_argument("--refine", action="store_true",
                        help="Run a second optimization pass with tighter bounds around the Pareto front.",
                        default=REFINE)
    parser.add_argument("--scan", action="store_true",
                        help="Run a hyperparameter scan using Optuna to find the best regularization parameters.",
                        default=HYPERPARAM_SCAN)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run a sensitivity analysis after optimization.",
                        default=SENSITIVITY_ANALYSIS)
    parser.add_argument("--solver", type=str, choices=["pymoo", "optuna"], default=OPTIMIZER,
                        help="Choice of optimization solver.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # logger.info Arguments
    logger.info("============================================================")
    logger.info("PhosKinTime Global Model")
    logger.info("------------------------------------------------------------")
    logger.info(f"Application        : {APP_NAME}")
    logger.info(f"Version            : {VERSION}")
    logger.info(f"Available Models   : {AVAILABLE_MODELS}")
    logger.info("------------------------------------------------------------")
    logger.info(f"Parent Package     : {PARENT_PACKAGE}")
    logger.info(f"Citation           : {CITATION}")
    logger.info(f"DOI                : {DOI}")
    logger.info(f"Source Code        : {GITHUB_URL}")
    logger.info(f"Documentation      : {DOCS_URL}")
    logger.info("============================================================")
    logger.info(f"[Args] Output directory: {args.output_dir}")
    logger.info(f"[Args] Number of cores: {args.cores}")
    logger.info(f"[Args] Number of generations: {args.n_gen}")
    logger.info(f"[Args] Population size: {args.pop}")
    logger.info(f"[Args] Seed: {args.seed}")
    logger.info(f"[Args] Lambda prior: {args.lambda_prior}")
    logger.info(f"[Args] Lambda protein: {args.lambda_protein}")
    logger.info(f"[Args] Lambda RNA: {args.lambda_rna}")
    logger.info(f"[Args] Lambda phospho: {args.lambda_phospho}")

    # 1) Load
    df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(args)

    if args.normalize_fc_steady:
        df_prot = normalize_fc_to_t0(df_prot)
        df_pho = normalize_fc_to_t0(df_pho)
        logger.info("[Data] Normalized protein and phospho FC to t=0.")

    base = df_rna[df_rna["time"] == 4.0].set_index("protein")["fc"]
    df_rna["fc"] = df_rna.apply(lambda r: r["fc"] / base.get(r["protein"], np.nan), axis=1)
    df_rna = df_rna.dropna(subset=["fc"])

    # 2) Model index to include all TF and kinases
    idx = Index(df_kin, tf_interactions=df_tf, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)
    kin_in = KinaseInput(idx.kinases, df_prot)
    logger.info(f"[KinaseInput] Initialized with {len(idx.kinases)} kinases.")

    # Restrict obs to model proteins
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    # Build weight functions
    w_prot_pho, w_rna = build_weight_functions(
        TIME_POINTS_PROTEIN,
        TIME_POINTS_RNA,
        scheme_prot_pho=WEIGHTING_METHOD_PROTEIN,
        scheme_rna=WEIGHTING_METHOD_RNA,
        early_window_prot_pho=120.0,
        early_window_rna=30.0,
    )

    df_prot["w"] = w_prot_pho(df_prot["time"].to_numpy(dtype=float))
    df_pho["w"] = w_prot_pho(df_pho["time"].to_numpy(dtype=float))
    df_rna["w"] = w_rna(df_rna["time"].to_numpy(dtype=float))

    # 3) Build W + TF
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf, idx, tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map)

    # --- Robust Labeled Export Block ---
    logger.info("[Output] Exporting network matrices with labels...")

    # Generate labels for EVERY ROW in the matrix (one for every site in the model)
    all_site_labels = []
    for i, p in enumerate(idx.proteins):
        for s in idx.sites[i]:
            all_site_labels.append(f"{p}_{s}")

    # Convert to COO format to get row/col indices directly
    w_coo = W_global.tocoo()

    # Create a mapping array (faster than list comprehension for large networks)
    site_labels_arr = np.array(all_site_labels)
    kinase_labels_arr = np.array(idx.kinases)

    df_w_export = pd.DataFrame({
        'Site': site_labels_arr[w_coo.row],
        'Kinase': kinase_labels_arr[w_coo.col],
        'Weight': w_coo.data
    })

    df_w_export.to_csv(os.path.join(args.output_dir, "network_W_global.csv"), index=False)

    # 2. Save tf_mat (TF -> Target mRNA)
    tf_coo = tf_mat.tocoo()

    # (Rows = Targets, Cols = Source TFs)
    protein_labels_arr = np.array(idx.proteins)

    df_tf_export = pd.DataFrame({
        'Target': protein_labels_arr[tf_coo.row],  # Row indices -> Target Gene Names
        'Source_TF': protein_labels_arr[tf_coo.col],  # Col indices -> Source TF Names
        'Weight': tf_coo.data
    })

    df_tf_export.to_csv(os.path.join(args.output_dir, "network_tf_mat.csv"), index=False)

    # 3. Save KinaseInput (Observed Kinase Trajectories)
    df_kin_input = pd.DataFrame(
        kin_in.Kmat,
        index=idx.kinases,
        columns=[f"t_{t}" for t in kin_in.grid]
    )
    df_kin_input.to_csv(os.path.join(args.output_dir, "network_kinase_inputs.csv"))

    # Calculate TF degree normalization (using absolute sum to handle repressors)
    # This effectively normalizes the 'regulatory input' so genes with many TFs
    # don't have explode synthesis rates compared to genes with few TFs.
    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    # Initialize Kinase Activity Multipliers (c_k) from Beta Priors
    # We use max(0.01) to prevent zero/negative activity in the log-space parameterization
    c_k_init = np.array([max(0.01, float(kin_beta_map.get(k, 1.0))) for k in idx.kinases])

    # 4) Defaults/system
    defaults = {
        "c_k": c_k_init,
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1
    }

    # Model system of Data IO + ODE + solver + optimization
    # Create System object
    # This object encapsulates the model system and provides methods for data IO, ODE integration, solver setup, and optimization
    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    if args.use_initial_condition_from_data:
        sys.attach_initial_condition_data(
            df_prot=df_prot,
            df_rna=df_rna,
            df_pho=df_pho
        )
        logger.info("[Model] Initial conditions set from data.")

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO]))

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, solver_times)
    loss_data["prot_base_idx"] = _base_idx(solver_times, 0.0)
    loss_data["rna_base_idx"] = _base_idx(solver_times, 4.0)
    loss_data["pho_base_idx"] = _base_idx(solver_times, 0.0)

    # 6) Decision vector bounds

    # Calculate optimal bounds based on network topology and data constraints
    custom_bounds = calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in)

    # Initialize raw params using these custom bounds for optimization
    theta0, slices, xl, xu = init_raw_params(defaults, custom_bounds=custom_bounds)

    runner = None
    pool = None
    # 7) Pymoo parallel runner
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        runner = StarmapParallelization(pool.starmap)
        logger.info(f"[Fit] Parallel evaluation enabled with {args.cores} workers.")
    else:
        logger.info("[Fit] Parallel evaluation disabled (or unavailable).")

    # --- HYPERPARAMETER SCAN BLOCK ---
    if args.scan:

        # This function will run the loop, save Excel/PNGs, and return the best dict
        best_lambdas = run_hyperparameter_scan(
            args, sys, loss_data, defaults, solver_times, runner, slices, xl, xu
        )

        logger.info(f"[Runner] Adopting optimized hyperparameters: {best_lambdas}")
        lambdas = {
            "protein": best_lambdas["lambda_protein"],
            "phospho": best_lambdas["lambda_phospho"],
            "rna": best_lambdas["lambda_rna"],
            "prior": best_lambdas["lambda_prior"]
        }
    else:
        # Standard manual arguments
        lambdas = {
            "protein": args.lambda_protein,
            "rna": args.lambda_rna,
            "phospho": args.lambda_phospho,
            "prior": args.lambda_prior
        }

    logger.info(f"[Scan] Using lambdas: {lambdas}")

    if args.solver == "optuna":

        total_trials = N_TRIALS

        logger.info(f"[Optuna] Running Optuna solver with {total_trials} total trials "
                    f"({args.pop} pop x {args.n_gen} generations).")

        res = run_optuna_solver(
            args=args,
            sys=sys,
            loss_data=loss_data,
            slices=slices,
            xl=xl,
            xu=xu,
            defaults=defaults,
            lambdas=lambdas,
            time_grid=solver_times,
            n_trials=total_trials,
            df_prot=df_prot,
            df_rna=df_rna,
            df_pho=df_pho
        )

    else:

        # 8) Problem
        problem = GlobalODE_MOO(
            sys=sys,
            slices=slices,
            loss_data=loss_data,
            defaults=defaults,
            lambdas=lambdas,
            time_grid=solver_times,
            xl=xl,
            xu=xu,
            elementwise_runner=runner
        )

        # 9) UNSGA3 needs reference directions
        ref_dirs = get_reference_directions(
            "das-dennis",
            problem.n_obj,
            n_partitions=30,
            seed=args.seed
        )

        # logger.info number of reference directions
        logger.info(f"[Fit] Number of reference directions: {len(ref_dirs)}")

        algorithm = UNSGA3(
            pop_size=args.pop,
            ref_dirs=ref_dirs,
            eliminate_duplicates=True,
            sampling=LHS(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(prob=1 / problem.n_var, eta=10),
        )

        termination = DefaultMultiObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=0.0025,
            period=30,
            n_max_gen=args.n_gen,
            n_max_evals=100000
        )

        logger.info(
            f"[Data] Number of points: {loss_data['n_p']} protein, {loss_data['n_r']} RNA, {loss_data['n_ph']} phospho | Total {loss_data['n_p'] + loss_data['n_r'] + loss_data['n_ph']} data points")
        logger.info(f"[Fit] UNSGA3: pop={args.pop}, n_gen={args.n_gen}, n_var={problem.n_var}, n_obj={problem.n_obj}")

        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=args.seed,
            save_history=True,
            verbose=True
        )

    if pool is not None:
        pool.close()
        pool.join()


    # Save full result object
    with open(os.path.join(args.output_dir, f"{args.solver}_optimization_result.pkl"), "wb") as f:
        pickle.dump(res, f)
    logger.info("[Output] Saved full optimization state (pickle).")

    if args.solver != "optuna":
        # Export convergence history
        df_hist = process_convergence_history(res, args.output_dir)
        df_hist.to_csv(Path(args.output_dir) / "convergence_history.csv", index=False)

        if args.refine:
            logger.info("[Refinement] Recursive refinement started.")

            # Pass the result of the first run (res) as the starting point
            res = run_iterative_refinement(
                problem,
                res,
                args,
                idx=sys.idx,
                max_passes=NUM_REFINE,
                padding=0.25
            )

            logger.info("[Refinement] Recursive refinement complete.")

    # 10) Save Pareto set
    X = res.X
    F = res.F
    np.save(os.path.join(args.output_dir, "pareto_X.npy"), X)
    np.save(os.path.join(args.output_dir, "pareto_F.npy"), F)

    # Also write a CSV summary
    df_pareto = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "phospho_mse"])
    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_F.csv"), index=False)
    logger.info(f"[Output] Saved Pareto front: {len(df_pareto)} solutions")

    excel_path = os.path.join(args.output_dir, "pareto_front.xlsx")

    export_pareto_front_to_excel(
        res=res,
        sys=sys,
        idx=idx,
        slices=slices,
        output_path=excel_path,
        weights=(args.lambda_protein, args.lambda_rna, args.lambda_phospho),
        top_k_trajectories=None,
    )

    logger.info(f"[Output] Saved Pareto front Excel: {excel_path}")

    plot_gof_from_pareto_excel(
        excel_path=excel_path,
        output_dir=os.path.join(args.output_dir, "goodness_of_fits_all_solutions"),
        plot_goodness_of_fit_func=plot_goodness_of_fit,
        df_prot_obs_all=df_prot,
        df_rna_obs_all=df_rna,
        df_phos_obs_all=df_pho,
        top_k=None,
        score_col="scalar_score",
    )

    logger.info(f"[Output] Saved Goodness of Fit plots for all Pareto solutions.")

    # 11) Pick one solution
    # Modified solution selection using Fréchet distance
    F = res.F
    solution_breakdowns = []

    # Compute Fréchet distances for each solution
    frechet_scores = []
    for i in range(len(X)):
        theta = X[i].astype(float)
        params_temp = unpack_params(theta, slices)
        sys.update(**params_temp)

        # Simulate with current parameters
        dfp_temp, dfr_temp, dfph_temp = simulate_and_measure(
            sys, idx, TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO
        )

        detailed_scores = {"prot": {}, "rna": {}, "phospho": {}}

        # Calculate Fréchet distance for each modality
        frechet_prot = 0.0
        frechet_rna = 0.0
        frechet_phospho = 0.0

        # Protein Fréchet distance
        if dfp_temp is not None and len(df_prot) > 0:
            for protein in df_prot['protein'].unique():
                obs = df_prot[df_prot['protein'] == protein][['time', 'fc']].values
                pred = dfp_temp[dfp_temp['protein'] == protein][['time', 'pred_fc']].values
                obs = obs[np.argsort(obs[:, 0])]
                pred = pred[np.argsort(pred[:, 0])]
                if len(obs) > 1 and len(pred) > 1:
                    d = frechet_distance(np.ascontiguousarray(obs), np.ascontiguousarray(pred))
                    frechet_prot += d
                    detailed_scores["prot"][protein] = d

        # RNA Fréchet distance
        if dfr_temp is not None and len(df_rna) > 0:
            for protein in df_rna['protein'].unique():
                obs = df_rna[df_rna['protein'] == protein][['time', 'fc']].values
                pred = dfr_temp[dfr_temp['protein'] == protein][['time', 'pred_fc']].values
                obs = obs[np.argsort(obs[:, 0])]
                pred = pred[np.argsort(pred[:, 0])]
                if len(obs) > 1 and len(pred) > 1:
                    d = frechet_distance(np.ascontiguousarray(obs), np.ascontiguousarray(pred))
                    frechet_rna += d
                    detailed_scores["rna"][protein] = d

        # Phospho Fréchet distance
        if dfph_temp is not None and len(df_pho) > 0:
            for site in df_pho['psite'].unique():
                obs = df_pho[df_pho['psite'] == site][['time', 'fc']].values
                pred = dfph_temp[dfph_temp['psite'] == site][['time', 'pred_fc']].values
                obs = obs[np.argsort(obs[:, 0])]
                pred = pred[np.argsort(pred[:, 0])]
                if len(obs) > 1 and len(pred) > 1:
                    d = frechet_distance(np.ascontiguousarray(obs), np.ascontiguousarray(pred))
                    frechet_phospho += d
                    detailed_scores["phospho"][site] = d

        # Weighted combination of Fréchet distances
        frechet_score = (args.lambda_protein * frechet_prot +
                         args.lambda_rna * frechet_rna +
                         args.lambda_phospho * frechet_phospho)
        frechet_scores.append(frechet_score)
        solution_breakdowns.append(detailed_scores)

    # Select solution with minimum Fréchet distance
    I = np.argmin(frechet_scores)

    best_breakdown = solution_breakdowns[I]

    logger.info("=" * 60)
    logger.info(f"FRECHET DISTANCE BREAKDOWN FOR BEST SOLUTION (Index {I})")
    logger.info("=" * 60)
    logger.info(f"{'Modality':<10} | {'Protein/Site':<20} | {'Fréchet Dist':<12}")
    logger.info("-" * 60)

    for modality, scores in best_breakdown.items():
        # Sort by worst fit first to spot "ballooning" or "flat" issues
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for name, val in sorted_items:
            logger.info(f"{modality:<10} | {name:<20} | {val:>12.6f}")

    logger.info("-" * 60)
    logger.info(f"TOTAL WEIGHTED FRECHET SCORE: {frechet_scores[I]:.6f}")
    logger.info("=" * 60)

    theta_best = X[I].astype(float)
    F_best = F[I]
    params = unpack_params(theta_best, slices)
    sys.update(**params)

    if args.sensitivity:
        run_sensitivity_analysis(
            sys=sys,
            idx=idx,
            fitted_params=params,
            output_dir=args.output_dir,
            metric=SENSITIVITY_METRIC
        )

    # 1. Export Dynamic Kinase Activities (Mechanism check)
    export_kinase_activities(sys, idx, args.output_dir, t_max=120)
    logger.info("[Output] Saved dynamic kinase activities.")

    # 2. Export Parameter Correlations (Identifiability check)
    export_param_correlations(res, slices, idx, args.output_dir, best_idx=I)
    logger.info("[Output] Saved parameter correlation analysis.")

    # 3. Residuals (Check for systematic bias in best solution)
    export_residuals(sys, idx, df_prot, df_rna, df_pho, args.output_dir)
    logger.info("[Output] Saved residual analysis.")

    # 4. Parameter Uncertainty (Check robustness across Pareto front)
    export_parameter_distributions(res, slices, idx, args.output_dir)
    logger.info("[Output] Saved parameter uncertainty analysis.")

    logger.info("[Model] System updated with optimized parameters.")
    logger.info(f"[Selection] Best solution index: {I}, Fréchet score: {frechet_scores[I]:.6f}")

    # Save the phosphorylation rates
    export_S_rates(sys, idx, args.output_dir, filename="S_rates_picked.csv", long=True)
    logger.info("[Output] Saved phosphorylation rates for picked solution.")

    plot_s_rates_report(
        f"{args.output_dir}/S_rates_picked.csv",
        f"{args.output_dir}/S_rates_report.pdf",
        top_k_sites_per_protein=24,
        max_sites_per_page=12,
        ncols=3,
        normalize_per_site=False,
        heatmap_per_protein=True,
        heatmap_cap_sites=80,
    )

    logger.info(
        f"[Output] Saved phosphorylation rates report for picked solution {args.output_dir}/S_rates_report.pdf.")

    # 12) Export picked solution
    dfp, dfr, dfph = simulate_and_measure(sys, idx, TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO)

    # Save raw preds
    if dfp is not None: dfp.to_csv(os.path.join(args.output_dir, "pred_prot_picked.csv"), index=False)
    if dfr is not None: dfr.to_csv(os.path.join(args.output_dir, "pred_rna_picked.csv"), index=False)
    if dfph is not None: dfph.to_csv(os.path.join(args.output_dir, "pred_phospho_picked.csv"), index=False)

    p_out = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in params.items()}
    with open(os.path.join(args.output_dir, "fitted_params_picked.json"), "w") as f:
        json.dump(p_out, f, indent=2)

    # Write picked objective values
    picked = {"prot_mse": float(F[I, 0]), "rna_mse": float(F[I, 1]), "phospho_mse": float(F[I, 2]),
              "scalar_score": float(
                  args.lambda_protein * F[I, 0] + args.lambda_rna * F[I, 1] + args.lambda_phospho * F[I, 2])}
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    logger.info(
        f"[Loss] Solution: prot_mse={picked['prot_mse']:.6f}, rna_mse={picked['rna_mse']:.6f}, phospho_mse={picked['phospho_mse']:.6f}, scalar_score={picked['scalar_score']:.6f}")

    plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, df_pho, dfph, output_dir=args.output_dir)
    logger.info("[Done] Goodness of Fit plot saved.")

    ts_dir = os.path.join(args.output_dir, "timeseries_plots")
    for g in idx.proteins:
        save_gene_timeseries_plots(
            gene=g,
            df_prot_obs=df_prot,
            df_prot_pred=dfp,
            df_rna_obs=df_rna,
            df_rna_pred=dfr,
            df_phos_obs=df_pho,
            df_phos_pred=dfph,
            output_dir=ts_dir,
            prot_times=TIME_POINTS_PROTEIN,
            rna_times=TIME_POINTS_RNA,
            filename_prefix="fit"
        )

    logger.info("[Done] Time series plots saved.")

    logger.info("[Simulate] Running post-optimization dynamics check...")

    # 1. Simulate for 7 days (10080 min) to see long-term behavior
    logger.info("[Simulate] Simulating system for 7 days to assess steady-state behavior.")
    t_check, Y_check = simulate_until_steady(sys, t_max=24 * 7 * 60)

    # Log for each protein whether it reached steady state
    for i, protein in enumerate(idx.proteins):
        reached_steady_state = Y_check[i, -1] > 0.99 * Y_check[i, 0]
        logger.info(f"Protein {protein}: Steady state reached? {reached_steady_state}")

    # 2. Plot every single protein
    plot_steady_state_all(
        t_check,
        Y_check,
        sys,
        idx,
        output_dir=args.output_dir
    )

    logger.info("[Simulate] Check complete. Inspect 'steady_state_plots' folder.")

    if dfp is not None and dfr is not None and dfph is not None:
        export_results(sys, idx, df_prot, df_rna, df_pho, dfp, dfr, dfph, args.output_dir)

    logger.info("[Done] Exported results saved.")

    # 1. 3D Pareto Front
    save_pareto_3d(res, selected_solution=F_best, output_dir=args.output_dir)
    logger.info("[Done] 3D Pareto plot saved.")

    # 2. Parallel Coordinate Plot
    save_parallel_coordinates(res, selected_solution=F_best, output_dir=args.output_dir)
    logger.info("[Done] Parallel Coordinate plot saved.")

    # 3. Convergence Video
    create_convergence_video(res, output_dir=args.output_dir)
    logger.info("[Done] Convergence video saved.")

    # 4. Prior Regularization Scan
    scan_prior_reg(out_dir=args.output_dir)
    logger.info("[Done] Prior regularization scan saved.")

    # Display all parameters from the configuration class
    global_config = load_config_toml("config.toml")
    logger.info("=" * 80)
    logger.info("GLOBAL MODEL CONFIGURATION")
    logger.info("=" * 80)
    for key in dir(global_config):
        if not key.startswith('_'):  # Skip private/magic attributes
            value = getattr(global_config, key)
            logger.info(f"  {key:<40} = {value}")
    logger.info("=" * 80)

    # Display all parameters 
    param_names = get_parameter_labels(sys.idx)
    logger.info("=" * 80)
    logger.info("OPTIMIZED NETWORK PARAMETERS FOR PICKED SOLUTION")
    logger.info("=" * 80)
    for param_type, labels in param_names.items():
        logger.info(f"\n{param_type}:")
        logger.info("-" * 80)
        for label, value in labels.items():
            logger.info(f"  {label:<40} = {value:>12.6g}")

    logger.info("=" * 80)
    # Finalize logging
    logger.info(f"[Complete] All results saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    main()
