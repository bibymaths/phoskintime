"""Rebuild networkmodel posterior objective from saved run artifacts.

This module exists so posterior chains can be launched as independent Python
processes without pickling a live JAX/Diffrax objective.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from networkmodel.config import RESULTS_DIR

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import numpy as np
import pandas as pd

from networkmodel.BuildMatrix import build_W_parallel, build_tf_matrix
from networkmodel.OptimizationProblem import GlobalODEScalarObjective, build_weight_functions
from networkmodel.BayesianInference import InferenceContext
from networkmodel.backend import detect_data_mode
from networkmodel.cache import prepare_fast_loss_data
from networkmodel.config import (
    TIME_POINTS_PROTEIN,
    TIME_POINTS_RNA,
    TIME_POINTS_PHOSPHO,
    WEIGHTING_METHOD_PROTEIN,
    WEIGHTING_METHOD_RNA,
)
from networkmodel.io import load_data
from networkmodel.network import Index, KinaseInput, System
from networkmodel.params import init_raw_params
from networkmodel.utils import normalize_fc_to_t0, _base_idx, calculate_bio_bounds

from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def _build_parameter_names(theta0, slices, idx) -> list[str]:
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

    return parameter_names.tolist()


def _prepare_tf_model(df_tf, df_kin, df_prot, df_rna, df_pho, kin_beta_map, tf_beta_map):
    """Reproduce runner.py TF filtering/proxy logic."""
    if df_tf is None or df_tf.empty:
        return df_tf, Index(
            df_kin,
            tf_interactions=df_tf,
            kin_beta_map=kin_beta_map,
            tf_beta_map=tf_beta_map,
        )

    required = {"tf", "target"}
    missing = required - set(df_tf.columns)
    if missing:
        raise ValueError(f"TF net missing columns: {missing}. Found columns: {list(df_tf.columns)}")

    proteins_with_sites = set(df_kin["protein"].unique())
    kinase_set = set(df_kin["kinase"].unique())

    target_universe = (
            set(df_kin["protein"].unique())
            | set(df_kin["kinase"].unique())
            | set(df_prot["protein"].unique())
            | set(df_rna["protein"].unique())
            | set(df_pho["protein"].unique())
    )

    df_tf_model = df_tf[df_tf["target"].isin(target_universe)].copy()
    orphan_tfs = sorted(set(df_tf_model["tf"].unique()) - proteins_with_sites)

    proxy_map = {}

    def _proxy_score(orphan: str, candidate: str) -> float:
        score = 0.0
        if tf_beta_map and orphan in tf_beta_map:
            score += float(tf_beta_map[orphan])
        if kin_beta_map and candidate in kin_beta_map:
            score += float(kin_beta_map[candidate])
        return score

    for orphan in orphan_tfs:
        targets = df_tf_model.loc[df_tf_model["tf"] == orphan, "target"].astype(str)

        cand1 = [t for t in targets if t in kinase_set]
        cand2 = [t for t in targets if t in proteins_with_sites]

        candidates = cand1 if cand1 else cand2
        if candidates:
            best = sorted(candidates, key=lambda c: (-_proxy_score(orphan, c), c))[0]
            proxy_map[orphan] = best

    if proxy_map:
        df_tf_model["tf_original"] = df_tf_model["tf"]
        df_tf_model["tf"] = df_tf_model["tf"].replace(proxy_map)

    df_tf_model = df_tf_model[df_tf_model["tf"].isin(proteins_with_sites)].copy()

    keep_cols = [c for c in df_tf_model.columns if c in ("tf", "target", "alpha")]
    df_tf_model = df_tf_model[keep_cols].drop_duplicates()

    idx = Index(
        df_kin,
        tf_interactions=df_tf_model,
        kin_beta_map=kin_beta_map,
        tf_beta_map=tf_beta_map,
    )

    logger.info("[PosteriorWorker] TF edges final: %d", len(df_tf_model))
    logger.info("[PosteriorWorker] Orphan TFs proxied: %d", len(proxy_map))

    return df_tf_model, idx


def _load_data_for_posterior_context(args, cfg: dict):
    """Load worker data, replacing df_kin with persisted preprocessed network when present."""
    df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(args)
    preprocessed_path = cfg.get("preprocessed_kinase_net")
    if preprocessed_path:
        path = Path(preprocessed_path)
        if not path.is_file():
            raise FileNotFoundError(f"preprocessed kinase network not found: {path}")
        df_kin = pd.read_csv(path)
    return df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map


def build_networkmodel_posterior_context(run_config_path: str | Path, chain_output_dir: str | Path) -> InferenceContext:
    """Rebuild an InferenceContext from saved posterior run config.

    This function intentionally reconstructs the objective in a fresh process.
    """
    run_config_path = Path(run_config_path)
    with open(run_config_path) as f:
        cfg = json.load(f)

    base_output_dir = Path(cfg["output_dir"])
    payload_dir = Path(cfg["payload_dir"])

    theta0 = np.load(payload_dir / "posterior_theta0.npy")
    lower = np.load(payload_dir / "posterior_lower.npy")
    upper = np.load(payload_dir / "posterior_upper.npy")

    with open(payload_dir / "posterior_parameter_names.json") as f:
        saved_parameter_names = json.load(f)

    args = SimpleNamespace(
        kinase_net=cfg["kinase_net"],
        tf_net=cfg["tf_net"],
        ms=cfg["ms"],
        rna=cfg["rna"],
        phospho=cfg["phospho"],
        kinopt=cfg["kinopt"],
        tfopt=cfg["tfopt"],
        output_dir=str(base_output_dir),
        cores=int(cfg.get("rebuild_cores", 1)),
        n_gen=int(cfg.get("n_gen", 50)),
        seed=int(cfg.get("seed", 0)),
        lambda_prior=float(cfg["lambdas"]["prior"]),
        lambda_protein=float(cfg["lambdas"]["protein"]),
        lambda_rna=float(cfg["lambdas"]["rna"]),
        lambda_phospho=float(cfg["lambdas"]["phospho"]),
        normalize_fc_steady=bool(cfg.get("normalize_fc_steady", False)),
        use_initial_condition_from_data=bool(cfg.get("use_initial_condition_from_data", False)),
        scan=False,
        sensitivity=False,
        solver="jaxopt",
    )

    df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = _load_data_for_posterior_context(args, cfg)

    if args.normalize_fc_steady:
        df_prot = normalize_fc_to_t0(df_prot)
        df_pho = normalize_fc_to_t0(df_pho)

    df_prot_raw = df_prot.copy()

    for _df in (df_kin, df_pho):
        _df["protein"] = _df["protein"].astype(str).str.strip()

    df_kin["psite"] = df_kin["psite"].astype(str).str.strip()
    df_pho["psite"] = df_pho["psite"].astype(str).str.strip()

    kin_site_pairs = set(zip(df_kin["protein"].values, df_kin["psite"].values))
    pairs = list(zip(df_pho["protein"].values, df_pho["psite"].values))
    keep = np.fromiter(((p, s) in kin_site_pairs for (p, s) in pairs), dtype=bool, count=len(pairs))
    df_pho = df_pho.loc[keep].copy()

    df_tf_model, idx = _prepare_tf_model(
        df_tf=df_tf,
        df_kin=df_kin,
        df_prot=df_prot,
        df_rna=df_rna,
        df_pho=df_pho,
        kin_beta_map=kin_beta_map,
        tf_beta_map=tf_beta_map,
    )

    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    df_prot_kin = df_prot_raw[df_prot_raw["protein"].isin(idx.kinases)].copy()
    kin_in = KinaseInput(idx.kinases, df_prot_kin)

    weight_cfg = build_weight_functions(
        method_protein=WEIGHTING_METHOD_PROTEIN,
        method_rna=WEIGHTING_METHOD_RNA,
        time_grid=TIME_POINTS_PROTEIN,
    )
    logger.info("[PosteriorWorker] Protein weighting scheme: %s", weight_cfg["protein"])
    logger.info("[PosteriorWorker] RNA weighting scheme: %s", weight_cfg["rna"])

    df_prot["w"] = 1.0
    df_rna["w"] = 1.0

    W_global = build_W_parallel(df_kin, idx, n_cores=int(args.cores))
    tf_mat = build_tf_matrix(
        df_tf_model,
        idx,
        tf_beta_map=tf_beta_map,
        kin_beta_map=kin_beta_map,
    )

    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    c_k_init = np.array([max(0.01, float(kin_beta_map.get(k, 1.0))) for k in idx.kinases])

    defaults = {
        "c_k": c_k_init,
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.05),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.1),
        "E_i": np.ones(idx.N),
        "tf_scale": 1.0,
    }

    sys_obj = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    if args.use_initial_condition_from_data:
        sys_obj.attach_initial_condition_data(
            df_prot=df_prot,
            df_rna=df_rna,
            df_pho=df_pho,
        )
        sys_obj.set_initial_conditions()

    solver_times = np.unique(
        np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO])
    )

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, solver_times)
    loss_data["prot_base_idx"] = _base_idx(solver_times, 0.0)
    loss_data["rna_base_idx"] = _base_idx(solver_times, 4.0)
    loss_data["pho_base_idx"] = _base_idx(solver_times, 0.0)

    custom_bounds = calculate_bio_bounds(idx, df_prot, df_rna, tf_mat, kin_in)
    theta0_build, slices, xl_build, xu_build = init_raw_params(defaults, custom_bounds=custom_bounds)

    if theta0_build.shape != theta0.shape:
        raise ValueError(
            f"Rebuilt theta layout does not match saved theta0: "
            f"rebuilt={theta0_build.shape}, saved={theta0.shape}"
        )

    rebuilt_parameter_names = _build_parameter_names(theta0_build, slices, idx)
    if len(saved_parameter_names) != len(rebuilt_parameter_names):
        raise ValueError(
            "Saved parameter_names length does not match rebuilt parameter layout: "
            f"{len(saved_parameter_names)} vs {len(rebuilt_parameter_names)}"
        )

    lambdas = {
        "protein": float(cfg["lambdas"]["protein"]),
        "rna": float(cfg["lambdas"]["rna"]),
        "phospho": float(cfg["lambdas"]["phospho"]),
        "prior": float(cfg["lambdas"]["prior"]),
    }

    mode = detect_data_mode(loss_data=loss_data, logger_obj=logger)

    problem = GlobalODEScalarObjective(
        sys=sys_obj,
        slices=slices,
        loss_data=loss_data,
        defaults=defaults,
        lambdas=lambdas,
        time_grid=solver_times,
        xl=lower,
        xu=upper,
        data_mode=mode,
    )

    ctx = InferenceContext(
        objective_fun=problem.objective,
        theta0=theta0,
        lower=lower,
        upper=upper,
        mode=mode,
        output_dir=chain_output_dir,
        parameter_names=saved_parameter_names,
        maxiter=int(cfg.get("n_gen", 50)),
        tol=float(cfg.get("tol", 1e-6)),
    )

    return ctx


def write_posterior_payload(
        *,
        ctx: InferenceContext,
        runner_args,
        lambdas: dict,
        output_dir: str | Path,
) -> Path:
    """Write posterior payload needed by standalone posterior worker."""
    output_dir = Path(output_dir)
    payload_dir = output_dir / "posterior_payload"
    payload_dir.mkdir(parents=True, exist_ok=True)

    np.save(payload_dir / "posterior_theta0.npy", np.asarray(ctx.theta0, dtype=np.float64))
    np.save(payload_dir / "posterior_lower.npy", np.asarray(ctx.lower, dtype=np.float64))
    np.save(payload_dir / "posterior_upper.npy", np.asarray(ctx.upper, dtype=np.float64))

    with open(payload_dir / "posterior_parameter_names.json", "w") as f:
        json.dump(list(ctx.parameter_names or []), f, indent=2)

    run_config = {
        "output_dir": str(output_dir),
        "payload_dir": str(payload_dir),
        "kinase_net": str(runner_args.kinase_net),
        "tf_net": str(runner_args.tf_net),
        "ms": str(runner_args.ms),
        "rna": str(runner_args.rna),
        "phospho": str(runner_args.phospho),
        "kinopt": str(runner_args.kinopt),
        "tfopt": str(runner_args.tfopt),
        "cores": int(getattr(runner_args, "cores", 1)),
        "rebuild_cores": 1,
        "n_gen": int(getattr(runner_args, "n_gen", 50)),
        "seed": int(getattr(runner_args, "seed", 0)),
        "normalize_fc_steady": bool(getattr(runner_args, "normalize_fc_steady", False)),
        "use_initial_condition_from_data": bool(getattr(runner_args, "use_initial_condition_from_data", False)),
        "lambdas": {
            "protein": float(lambdas["protein"]),
            "rna": float(lambdas["rna"]),
            "phospho": float(lambdas["phospho"]),
            "prior": float(lambdas["prior"]),
        },
        "tol": float(getattr(ctx, "tol", 1e-6)),
    }

    preprocessed_kinase_net = getattr(runner_args, "preprocessed_kinase_net", "")
    if preprocessed_kinase_net:
        run_config["preprocessed_kinase_net"] = str(preprocessed_kinase_net)

    run_config_path = payload_dir / "posterior_run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    return run_config_path
