#! usr/bin/python

"""Run the command-line networkmodel workflow that loads data, builds topology, optimizes parameters, and writes outputs; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.analysis, networkmodel.buildmat, networkmodel.cache, networkmodel.config, networkmodel.dashboard_bundle, networkmodel.export, networkmodel.inference, networkmodel.io, networkmodel.jax_backend, networkmodel.mode_outputs, networkmodel.network, networkmodel.optproblem, networkmodel.params, networkmodel.scan, networkmodel.sensitivity, networkmodel.simulate, networkmodel.steadystate, networkmodel.utils."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

POSTERIOR_NUM_CHAINS_ENV = os.environ.get("POSTERIOR_NUM_CHAINS", "8")

os.environ.setdefault(
    "XLA_FLAGS",
    (
        f"--xla_force_host_platform_device_count={POSTERIOR_NUM_CHAINS_ENV} "
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1"
    ),
)

import argparse
import atexit
import json
import logging
import pickle
import multiprocessing as mp

import numpy as np
import pandas as pd

from networkmodel.dashboard_bundle import save_dashboard_bundle
from networkmodel.scan import run_hyperparameter_scan
from networkmodel.sensitivity import run_sensitivity_analysis
from networkmodel.steadystate import _dump_y0
from networkmodel.buildmat import build_W_parallel, build_tf_matrix
from networkmodel.cache import prepare_fast_loss_data
from networkmodel.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, RESULTS_DIR, MAX_ITERATIONS, \
    SEED, REGULARIZATION_LAMBDA, REGULARIZATION_RNA, REGULARIZATION_PHOSPHO, TIME_POINTS_PHOSPHO, \
    REGULARIZATION_PROTEIN, NORMALIZE_FC_STEADY, USE_INITIAL_CONDITION_FROM_DATA, KINASE_NET_FILE, TF_NET_FILE, \
    MS_DATA_FILE, RNA_DATA_FILE, PHOSPHO_DATA_FILE, KINOPT_RESULTS_FILE, TFOPT_RESULTS_FILE, \
    WEIGHTING_METHOD_PROTEIN, WEIGHTING_METHOD_RNA, APP_NAME, VERSION, PARENT_PACKAGE, CITATION, DOI, GITHUB_URL, \
    DOCS_URL, SENSITIVITY_METRIC, SENSITIVITY_ANALYSIS, AVAILABLE_MODELS, HYPERPARAM_SCAN, MODEL, CORES, \
    N_STARTS, PROFILE_LIKELIHOOD, PROFILE_INDICES, PROFILE_GRID_SIZE, POSTERIOR_SAMPLING, \
    POSTERIOR_NUM_WARMUP, POSTERIOR_NUM_SAMPLES
from networkmodel.io import load_data
from networkmodel.network import Index, KinaseInput, System
from networkmodel.optproblem import GlobalODEScalarObjective, build_weight_functions
from networkmodel.params import init_raw_params, unpack_params
from networkmodel.simulate import simulate_and_measure
from networkmodel.utils import normalize_fc_to_t0, _base_idx, calculate_bio_bounds, \
    get_optimized_sets
from networkmodel.export import export_pareto_front_to_excel, plot_goodness_of_fit, \
    export_results, save_gene_timeseries_plots, \
    scan_prior_reg, export_S_rates, plot_s_rates_report, export_kinase_activities, \
    export_param_correlations, export_residuals, export_parameter_distributions
from networkmodel.analysis import simulate_until_steady, plot_steady_state_all
from networkmodel.backend import warn_deprecated_backend_options, detect_data_mode, JaxoptResult
from networkmodel.bayesianinference import (
    InferenceContext, run_multistart,
    run_profile_likelihood, run_numpyro_posterior,
    configure_jax_parallelism,
)
from networkmodel.mode_outputs import write_scalar_result_tables
from common.frechet import frechet_distance
from config_loader import load_config_toml
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)
global problem

@atexit.register
def _close_log_handlers():
    """Handle internal close log handlers"""
    lg = logging.getLogger()
    for h in list(lg.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass


def main():
    """Run the networkmodel entry point
    
    Returns:
        Computed result from this routine.
    
    Raises:
        ValueError: When inputs are inconsistent or unsupported.
    """
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
    parser.add_argument("--cores", type=int, default=CORES)

    # JAXopt
    parser.add_argument("--n-gen", type=int, default=MAX_ITERATIONS)
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
    parser.add_argument("--scan", action="store_true",
                        help="Run a hyperparameter scan using Optuna to find the best regularization parameters.",
                        default=HYPERPARAM_SCAN)
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run a sensitivity analysis after optimization.",
                        default=SENSITIVITY_ANALYSIS)
    parser.add_argument("--solver", type=str, choices=["jaxopt", "pymoo", "optuna"], default="jaxopt",
                        help="Choice of optimization solver. Legacy pymoo/optuna values map to jaxopt.")

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

    logger.info("[Solver] Using Diffrax Kvaerno Solver")

    if MODEL == 0:
        logger.info("[Model] Using Distributive Model")
    elif MODEL == 1:
        logger.info("[Model] Using Sequential Model")
    elif MODEL == 2:
        logger.info("[Model] Using Combinatorial Model")
    elif MODEL == 4:
        logger.info("[Model] Using Saturating Model")
    else:
        raise ValueError(f"Unknown MODEL value in config file: {MODEL}")

    logger.info(f"[Args] Output directory: {args.output_dir}")
    logger.info(f"[Args] Number of cores: {args.cores}")
    logger.info(f"[Args] Number of generations: {args.n_gen}")
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

    # base = df_rna[df_rna["time"] == 4.0].set_index("protein")["fc"]
    # df_rna["fc"] = df_rna.apply(lambda r: r["fc"] / base.get(r["protein"], np.nan), axis=1)
    # df_rna = df_rna.dropna(subset=["fc"])

    df_prot_raw = df_prot.copy()

    # -------------------------------------------------------------------------
    # STRICT MECHANISTIC PHOSPHO FILTER (drop any (protein, psite) not in df_kin)
    # -------------------------------------------------------------------------
    for _df in (df_kin, df_pho):
        _df["protein"] = _df["protein"].astype(str).str.strip()
    df_kin["psite"] = df_kin["psite"].astype(str).str.strip()
    df_pho["psite"] = df_pho["psite"].astype(str).str.strip()

    kin_site_pairs = set(zip(df_kin["protein"].values, df_kin["psite"].values))

    n_before = len(df_pho)
    pairs = list(zip(df_pho["protein"].values, df_pho["psite"].values))
    keep = np.fromiter(((p, s) in kin_site_pairs for (p, s) in pairs), dtype=bool, count=len(pairs))
    df_pho = df_pho.loc[keep].copy()

    logger.info(f"[Phospho] Mechanistic site filter: {n_before} → {len(df_pho)} (dropped {n_before - len(df_pho)})")

    # -------------------------------------------------------------------------
    # Keep ONLY proteins that are observed in at least one modality (prot/rna/phospho)
    # -------------------------------------------------------------------------
    # observed_proteins = (
    #         set(df_prot["protein"].unique())
    #         | set(df_rna["protein"].unique())
    #         | set(df_pho["protein"].unique())
    # )
    #
    # logger.info(f"[Data] Observed proteins (union prot/rna/phospho): {len(observed_proteins)}")

    # -------------------------------------------------------------------------
    # Sophisticated TF handling:
    # - Keep TF edges even if TF is not present in kinase network
    # - Proxy orphan TFs so build_tf_matrix can still be used:
    #     Priority 1: proxy to a target that is a kinase (df_kin["kinase"])  [feedback-like]
    #     Priority 2: proxy to a target that is a signaling protein with sites (df_kin["protein"])
    # - Rewrite TF edges to use proxy TF names (so TF columns exist in idx universe)
    # - Drop only edges that still cannot be represented after proxying
    # -------------------------------------------------------------------------

    if df_tf is None or df_tf.empty:
        df_tf_model = df_tf  # empty is fine
        idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)
        logger.info("[TF] TF net empty; building Index without TF edges.")
    else:
        required = {"tf", "target"}
        missing = required - set(df_tf.columns)
        if missing:
            raise ValueError(f"TF net missing columns: {missing}. Found columns: {list(df_tf.columns)}")

        # proteins_with_sites = set(df_kin["protein"].unique())  # signaling layer “state-capable” proteins
        # kinase_set = set(df_kin["kinase"].unique())  # kinases (drivers / feedback proxies)

        proteins_with_sites = set(df_kin["protein"].unique())  # state-capable signaling proteins
        kinase_set = set(df_kin["kinase"].unique())

        # Build a target universe that includes anything the model might plausibly score or represent
        target_universe = (
                set(df_kin["protein"].unique())
                | set(df_kin["kinase"].unique())
                | set(df_prot["protein"].unique())
                | set(df_rna["protein"].unique())
                | set(df_pho["protein"].unique())
        )

        # Start with TF edges whose TARGET is in target_universe (not just proteins_with_sites)
        df_tf_model = df_tf[df_tf["target"].isin(target_universe)].copy()

        # Identify orphan TFs relative to signaling proteins-with-sites
        orphan_tfs = sorted(set(df_tf_model["tf"].unique()) - proteins_with_sites)

        # Build proxy map without modifying Index:
        # orphan TF -> proxy protein name (must be representable in idx universe)
        TF_PROXY_MAP = {}

        # Helper: score candidate proxies (optional; keeps deterministic but informed choice)
        def _proxy_score(orphan: str, candidate: str) -> float:
            score = 0.0
            if tf_beta_map and orphan in tf_beta_map:
                score += float(tf_beta_map[orphan])
            if kin_beta_map and candidate in kin_beta_map:
                score += float(kin_beta_map[candidate])
            return score

        for orphan in orphan_tfs:
            targets = df_tf_model.loc[df_tf_model["tf"] == orphan, "target"].astype(str)

            # Priority 1: targets that are kinases
            cand1 = [t for t in targets if t in kinase_set]

            # Priority 2: any signaling proteins-with-sites targets
            cand2 = [t for t in targets if t in proteins_with_sites]

            candidates = cand1 if cand1 else cand2
            if candidates:
                # Choose best candidate by score; deterministic tie-break by name
                best = sorted(candidates, key=lambda c: (-_proxy_score(orphan, c), c))[0]
                TF_PROXY_MAP[orphan] = best

        # Rewrite TF names to proxies where available
        if TF_PROXY_MAP:
            df_tf_model["tf_original"] = df_tf_model["tf"]
            df_tf_model["tf"] = df_tf_model["tf"].replace(TF_PROXY_MAP)

        # After proxying, enforce representability:
        # TF source must now be a signaling protein-with-sites (so idx/protein universe can include it)
        n_before_src = len(df_tf_model)
        df_tf_model = df_tf_model[df_tf_model["tf"].isin(proteins_with_sites)].copy()
        n_after_src = len(df_tf_model)

        # Optional: keep only columns used by build_tf_matrix
        keep_cols = [c for c in df_tf_model.columns if c in ("tf", "target", "alpha")]
        df_tf_model = df_tf_model[keep_cols].drop_duplicates()

        # Build Index with proxied TF net (Index proxy_map may still be used internally; we did not modify it)
        idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)

        # Logging
        logger.info(f"[TF] TF edges in (raw): {len(df_tf)}")
        logger.info(
            f"[TF] TF edges after target-in-sites filter: {len(df_tf[df_tf['target'].isin(proteins_with_sites)])}")
        logger.info(f"[TF] Orphan TFs detected (vs proteins_with_sites): {len(orphan_tfs)}")
        logger.info(f"[TF] Orphan TFs proxied (external map): {len(TF_PROXY_MAP)}")
        if TF_PROXY_MAP:
            sample = list(TF_PROXY_MAP.items())[:25]
            logger.info(f"[TF] Proxy examples (orphan->proxy): {sample}" + (" ..." if len(TF_PROXY_MAP) > 25 else ""))
        logger.info(f"[TF] TF edges after proxy + TF-source representability: {n_before_src} → {n_after_src}")
        logger.info(f"[TF] TF edges final (deduped): {len(df_tf_model)}")

    # -------------------------------------------------------------------------
    # Restrict observations to proteins present in the model index
    # -------------------------------------------------------------------------
    n_prot_before = len(df_prot)
    n_rna_before = len(df_rna)
    n_pho_before = len(df_pho)

    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    logger.info(
        f"[Data] Observation filtering by model index | "
        f"Protein: {n_prot_before} → {len(df_prot)}, "
        f"RNA: {n_rna_before} → {len(df_rna)}, "
        f"Phospho: {n_pho_before} → {len(df_pho)}"
    )

    # -------------------------------------------------------------------------
    # Build kinase input using observed protein trajectories
    # -------------------------------------------------------------------------
    df_prot_kin = df_prot_raw[df_prot_raw["protein"].isin(idx.kinases)].copy()
    kin_in = KinaseInput(idx.kinases, df_prot_kin)

    # Coverage diagnostics
    observed_kinases = set(df_prot_kin["protein"].unique())
    missing_kinases = [k for k in idx.kinases if k not in observed_kinases]

    logger.info(
        f"[KinaseInput] Initialized with {len(idx.kinases)} kinases | "
        f"Observed trajectories: {len(idx.kinases) - len(missing_kinases)} / {len(idx.kinases)}"
    )

    if missing_kinases:
        logger.warning(
            f"[KinaseInput] {len(missing_kinases)} kinases have NO observed protein FC "
            f"(defaulting to 1.0 driver): {missing_kinases[:25]}"
            + (" ..." if len(missing_kinases) > 25 else "")
        )

    # -------------------------------------------------------------------------
    # Final index summary
    # -------------------------------------------------------------------------
    logger.info(
        f"[Index] Model proteins: {len(idx.proteins)} | "
        f"Kinases: {len(idx.kinases)} | "
        f"Total sites: {idx.total_sites}"
    )

    # -----------------------------------------------------------------------------
    # FULL-UNIVERSE MODE (LEGACY / “MODEL EVERYTHING” BEHAVIOR)
    # -----------------------------------------------------------------------------
    # Rationale
    # ---------
    # This pathway intentionally constructs the *largest possible* model universe by
    # ingesting BOTH:
    #   (i) the kinase–substrate signaling network (df_kin), and
    #   (ii) the TF–target transcriptional network (df_tf).
    #
    # Consequence: idx.proteins becomes the UNION of:
    #   - signaling targets ("protein" in df_kin)
    #   - kinases ("kinase" in df_kin) if they also appear as proteins/TFs in unions
    #   - TF sources ("tf" in df_tf)
    #   - TF targets ("target" in df_tf)
    #
    # This is the “model everything” philosophy: every entity that appears anywhere
    # in the regulatory or signaling topology is treated as part of the modeled state
    # space (mRNA + protein + phospho states, where applicable).
    #
    # Practical implications
    # ----------------------
    # 1) Orphan TFs:
    #    If a TF exists only in df_tf but does not have signaling sites in df_kin,
    #    Index may still include it in idx.proteins. The Index class can optionally
    #    *redirect* such orphan TFs to kinase proxies (proxy_map) to avoid creating
    #    unsupported signaling states; however the protein name still exists as a
    #    label in idx.proteins.
    #
    # 2) Kinase driving:
    #    KinaseInput is constructed using df_prot trajectories. If a kinase has no
    #    observed protein FC data, it defaults to a flat driver (1.0), and can still
    #    be scaled by the optimized multiplier c_k. This preserves solvability while
    #    implicitly assuming missing kinase dynamics.
    #
    # 3) Observation restriction:
    #    After idx is built, each observation table is restricted to idx.proteins.
    #    This ensures the optimizer scores only entities that the model explicitly
    #    represents, and prevents “unknown proteins” from leaking into the loss.
    #
    # When to use
    # -----------
    # Use this mode when you explicitly want the ODE system to represent the full
    # mechanistic + regulatory scope implied by the input networks, even if this
    # increases dimensionality and introduces entities with incomplete observability.
    # -----------------------------------------------------------------------------

    # # 2) Model index to include all TFs and kinases (full topology union)
    # idx = Index(df_kin, tf_interactions=df_tf, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)
    #
    # # Kinase activity input trajectories (data-driven when present; defaults otherwise)
    # kin_in = KinaseInput(idx.kinases, df_prot)
    # logger.info(f"[KinaseInput] Initialized with {len(idx.kinases)} kinases.")
    #
    # # Restrict observations to the modeled protein universe (TFs + kinases + targets)
    # df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    # df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    # df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()
    #
    # logger.info(f"[Index] Proteins in model (full-universe): {len(idx.proteins)}")

    # New: keep the config for logging, but don't treat it as a callable
    weight_cfg = build_weight_functions(
        method_protein=WEIGHTING_METHOD_PROTEIN,
        method_rna=WEIGHTING_METHOD_RNA,
        time_grid=TIME_POINTS_PROTEIN,
    )

    logger.info("[Weights] Protein weighting scheme: %s", weight_cfg["protein"])
    logger.info("[Weights] RNA weighting scheme: %s", weight_cfg["rna"])

    # The scalar JAX objective uses global lambdas, so use uniform per-row weights here
    df_prot["w"] = 1.0
    df_rna["w"] = 1.0

    # 3) Build W + TF 
    # -------------------------------------------------------------------------
    # Network Matrix Construction
    # -------------------------------------------------------------------------
    # Build the kinase-substrate interaction matrix (W_global) and the 
    # transcription factor regulatory matrix (tf_mat) in parallel for efficiency.
    #
    # W_global: Sparse matrix (sites × kinases) encoding kinase-substrate relationships
    # tf_mat:   Sparse matrix (genes × TFs) encoding TF-target regulatory relationships
    #
    # Both matrices form the mechanistic backbone of the ODE system and are used
    # throughout optimization to compute phosphorylation rates and transcriptional
    # regulation, respectively.
    # -------------------------------------------------------------------------
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf_model, idx, tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map)

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

    logger.info("[Output] Exporting network matrices with labels.")

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
        "B_i": np.full(idx.N, 0.05),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.1),
        "E_i": np.ones(idx.N),
        "tf_scale": 1.0
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
        sys.set_initial_conditions()
        _dump_y0(sys, args.output_dir)
        logger.info("[Model] Initial conditions set from data.")

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO]))

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, solver_times)
    loss_data["prot_base_idx"] = _base_idx(solver_times, 0.0)
    loss_data["rna_base_idx"] = _base_idx(solver_times, 4.0)
    loss_data["pho_base_idx"] = _base_idx(solver_times, 0.0)
    logger.info(
        "[GlobalObjective] Loss data mapped with %s state layout; phospho rows=%d; lambda_phospho=%g",
        loss_data.get("state_layout", "standard"),
        int(loss_data.get("n_ph", 0)),
        float(args.lambda_phospho),
    )
    if float(args.lambda_phospho) == 0.0 and int(loss_data.get("n_ph", 0)) > 0:
        logger.warning("[GlobalObjective] Phospho data are present but --lambda-phospho is zero.")

    # 6) Decision vector bounds

    # Calculate optimal bounds based on network topology and data constraints
    custom_bounds = calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in)

    # Initialize raw params using these custom bounds for optimization.
    # The theta layout intentionally excludes network alpha/beta construction weights.
    theta0, slices, xl, xu = init_raw_params(defaults, custom_bounds=custom_bounds)

    parameter_names = np.empty(theta0.shape[0], dtype=object)

    for name, sl in slices.items():
        if name == "c_k":
            labels = [f"c_k[{k}]" for k in idx.kinases]
        elif name == "Dp_i":
            labels = [
                f"Dp_i[{p}_{s}]"
                for p, sites in zip(idx.proteins, idx.sites)
                for s in sites
            ]
        elif name == "tf_scale":
            labels = ["tf_scale"]
        else:
            labels = [f"{name}[{p}]" for p in idx.proteins]

        parameter_names[sl] = labels

    parameter_names = parameter_names.tolist()

    logger.info("[Optimizer] theta0.shape=%s xl.shape=%s xu.shape=%s", theta0.shape, xl.shape, xu.shape)
    if theta0.shape != xl.shape or theta0.shape != xu.shape:
        raise ValueError(f"theta0/xl/xu shape mismatch: {theta0.shape}, {xl.shape}, {xu.shape}")
    if "alpha" in slices or "beta" in slices:
        raise ValueError("alpha/beta are fixed network weights and must not be optimized")
    for _name, _sl in slices.items():
        logger.info("[Optimizer] theta group %-8s slice=(%d,%d) size=%d", _name, _sl.start, _sl.stop,
                    _sl.stop - _sl.start)

    opt_proteins, opt_sites, opt_kinases = get_optimized_sets(idx, slices, xl, xu)

    logger.info(f"[Optimized] Proteins with free vars: {len(opt_proteins)} / {idx.N}")
    logger.info(f"[Optimized] Sites with free Dp_i:  {len(opt_sites)} / {idx.total_sites}")
    logger.info(f"[Optimized] Kinases with free c_k: {len(opt_kinases)} / {len(idx.kinases)}")

    with open(os.path.join(args.output_dir, "optimized_entities.json"), "w") as f:
        json.dump(
            {
                "proteins": sorted(opt_proteins),
                "sites": sorted(opt_sites),
                "kinases": sorted(opt_kinases),
            },
            f,
            indent=2,
        )

    # --- HYPERPARAMETER SCAN ---
    if args.scan:

        runner = None
        pool = None
        logger.warning(
            "[Deprecated Config] Hyperparameter scan is retained for compatibility but runs without legacy evolutionary parallelization in JAXopt mode.")

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

        if pool is not None:
            pool.close()
            pool.join()

    else:
        # Standard manual arguments
        lambdas = {
            "protein": args.lambda_protein,
            "rna": args.lambda_rna,
            "phospho": args.lambda_phospho,
            "prior": args.lambda_prior
        }

    logger.info(f"[Scan] Using lambdas: {lambdas}")

    warn_deprecated_backend_options(vars(args), logger_obj=logger)
    args.solver = "jaxopt"
    mode = detect_data_mode(loss_data=loss_data, logger_obj=logger)
    problem = GlobalODEScalarObjective(
        sys=sys,
        slices=slices,
        loss_data=loss_data,
        defaults=defaults,
        lambdas=lambdas,
        time_grid=solver_times,
        xl=xl,
        xu=xu,
        data_mode=mode,
    )
    logger.info(
        f"[Data] Number of points: {loss_data['n_p']} protein, {loss_data['n_r']} RNA, "
        f"{loss_data['n_ph']} phospho | Total {loss_data['n_p'] + loss_data['n_r'] + loss_data['n_ph']} data points"
    )
    configure_jax_parallelism(max_workers=args.cores, logger_obj=logger)
    ctx = InferenceContext(
        objective_fun=problem.objective,
        theta0=theta0,
        lower=xl,
        upper=xu,
        mode=mode,
        output_dir=args.output_dir,
        parameter_names=parameter_names,
        maxiter=args.n_gen,
        tol=1e-6,
    )
    if N_STARTS > 1:
        try:
            ms = run_multistart(ctx, n_starts=N_STARTS, seed=args.seed, max_workers=args.cores)
            best_row = ms["best"].drop(
                columns=["start_id", "seed", "success", "selected_best"],
                errors="ignore",
            ).iloc[0]
            best_x = np.asarray(best_row.values, dtype=np.float64)
            best_f = float(ms["summary"].loc[ms["summary"]["selected_best"], "final_objective"].iloc[0])
            opt_state = None
            # Repopulate problem.final_loss_breakdown from the winning parameters.
            try:
                objective_raw = getattr(problem, "objective_raw", problem._objective_raw)
                _, breakdown = objective_raw(best_x)
                problem.final_loss_breakdown = {k: float(v) for k, v in breakdown.items()}
            except Exception as e:
                logger.warning("Could not repopulate loss breakdown after multistart: %s", e)
        except RuntimeError:
            logger.warning("Multistart failed; falling back to single-start solve.")
            best_x, opt_state, best_f = problem.solve(theta0, maxiter=args.n_gen)
    else:
        best_x, opt_state, best_f = problem.solve(theta0, maxiter=args.n_gen)

    res = JaxoptResult(
        X=np.asarray([best_x], dtype=float),
        F=np.asarray([[best_f]], dtype=float),
        objective_value=float(best_f),
        params=np.asarray(best_x, dtype=float),
        state=opt_state,
        data_mode=mode,
        loss_breakdown=dict(problem.final_loss_breakdown),
    )
    # Replace the original ctx with a post-optimization version.
    # Only theta0 changes — everything else is identical.
    ctx = InferenceContext(
        objective_fun=problem.objective,
        theta0=best_x,
        lower=xl,
        upper=xu,
        mode=mode,
        output_dir=args.output_dir,
        parameter_names=parameter_names,
        maxiter=args.n_gen,
        tol=1e-6,
    )
    if PROFILE_LIKELIHOOD and PROFILE_INDICES.strip():
        logger.info(f"Profile likelihood requested for indices: {PROFILE_INDICES}")
        profile_indices = [int(x) for x in PROFILE_INDICES.split(",") if x.strip()]
        run_profile_likelihood(ctx, parameter_indices=profile_indices,
                               grid_size=PROFILE_GRID_SIZE)
        logger.info("Profile likelihood complete.")
    if POSTERIOR_SAMPLING:
        try:
            logger.info(
                f"Posterior sampling requested with warmup={POSTERIOR_NUM_WARMUP}, samples={POSTERIOR_NUM_SAMPLES}")
            run_numpyro_posterior(
                ctx,
                num_warmup=POSTERIOR_NUM_WARMUP,
                num_samples=POSTERIOR_NUM_SAMPLES,
                seed=args.seed,
                num_chains=min(8, int(args.cores)),
                chain_method="parallel",
            )
            logger.info("Posterior sampling complete.")
        except RuntimeError as e:
            logger.warning("Posterior sampling skipped: %s", e)
    # Save full result object
    with open(os.path.join(args.output_dir, f"{args.solver}_optimization_result.pkl"), "wb") as f:
        pickle.dump(res, f)
    logger.info("[Output] Saved full optimization state (pickle).")

    logger.info("[Refinement] Scalar JAXopt mode uses deterministic local optimization; legacy refinement is not run.")

    # 10) Save scalar optimum with backward-compatible filenames
    X = res.X
    F = res.F

    np.save(os.path.join(args.output_dir, "pareto_X.npy"), X)
    np.save(os.path.join(args.output_dir, "pareto_F.npy"), F)

    # Also write CSV/JSON summaries with mode metadata.
    output_paths = write_scalar_result_tables(args.output_dir, mode, F.reshape(-1))
    df_pareto = pd.read_csv(output_paths["legacy_objective"])
    logger.info(f"[Output] Saved scalar objective table: {len(df_pareto)} solution(s)")

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

    logger.info(f"[Output] Saved scalar objective Excel compatibility file: {excel_path}")

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

    # 4. Parameter Uncertainty (Check robustness around scalar optimum)
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

    # Write picked objective values. Global JAXopt now returns a scalar F with
    # per-modality components captured from the scalar objective breakdown.
    scalar_score = float(np.asarray(F[I]).reshape(-1)[0])
    picked = {"scalar_score": scalar_score}
    picked.update({k: float(v) for k, v in getattr(res, "loss_breakdown", {}).items()})
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    logger.info("[Loss] Picked scalar objective breakdown: %s", picked)

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

    # Display all parameters structured by type
    logger.info("=" * 80)
    logger.info("OPTIMIZED NETWORK PARAMETERS FOR PICKED SOLUTION")
    logger.info("=" * 80)

    # 1. Kinase Activities
    if 'c_k' in params:
        logger.info("Kinase Activities (c_k) [Fold-Change vs Prior]:")
        logger.info("-" * 80)
        for i, k_name in enumerate(sys.idx.kinases):
            val = params['c_k'][i]
            logger.info(f"  {k_name:<40} = {val:>12.6g}")

    # 2. Protein-Specific Rates
    rate_labels = {
        'A_i': 'Transcription Rates (A)',
        'B_i': 'RNA Degradation Rates (B)',
        'C_i': 'Translation Rates (C)',
        'D_i': 'Protein Degradation Rates (D)',
        'E_i': 'Initial Condition Multipliers (E)'
    }

    for p_key, p_desc in rate_labels.items():
        if p_key in params:
            logger.info(f"{p_desc}:")
            logger.info("-" * 80)
            for i, p_name in enumerate(sys.idx.proteins):
                val = params[p_key][i]
                logger.info(f"  {p_name:<40} = {val:>12.6g}")

    # 3. Phosphatase Rates
    if 'Dp_i' in params:
        logger.info("De-phosphorylation Rates (Dp):")
        logger.info("-" * 80)
        flat_idx = 0
        for i, p_name in enumerate(sys.idx.proteins):
            sites = sys.idx.sites[i]
            if not sites:
                continue
            for s_name in sites:
                val = params['Dp_i'][flat_idx]
                label = f"{p_name}_{s_name}"
                logger.info(f"  {label:<40} = {val:>12.6g}")
                flat_idx += 1

    # 4. Global Scalars
    if 'tf_scale' in params:
        logger.info("Global Parameters:")
        logger.info("-" * 80)
        val = float(params['tf_scale'])
        logger.info(f"  {'TF Saturation Scale':<40} = {val:>12.6g}")

    bundle_path = save_dashboard_bundle(
        output_dir=args.output_dir,
        args=args,
        res=res,
        slices=slices,
        xl=xl,
        xu=xu,
        defaults=defaults,
        lambdas=lambdas,
        solver_times=solver_times,
        df_prot=df_prot,
        df_rna=df_rna,
        df_pho=df_pho,
        frechet_scores=frechet_scores,
        picked_index=int(I),
    )
    logger.info(f"[Dashboard] Saved dashboard bundle: {bundle_path}")

    # Finalize logging
    logger.info(f"[Complete] All results saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    main()
