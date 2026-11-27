from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib  # py>=3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # py<3.11


def _project_root() -> Path:
    """
    Find repo root robustly by walking upwards until config.toml is found.
    This avoids 'root = folder containing config_loader.py' which is wrong
    when config/ is a subdir.

    Returns:
        The root directory of the project.
    """
    start = Path(__file__).resolve().parent
    for p in [start, *start.parents]:
        if (p / "config.toml").is_file():
            return p
    return start


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries, with override taking precedence.

    Args:
        base (dict[str, Any]): The base dictionary to merge into.
        override (dict[str, Any]): The dictionary containing overrides.

    Returns:
        dict[str, Any]: The merged dictionary.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=8)
def load(mode: str, section: str) -> dict[str, Any]:
    """
    Load configuration for a specific mode and section.

    Args:
        mode (str): The mode to load configuration for.
        section (str): The section to load configuration for.

    Returns:
        dict[str, Any]: The loaded configuration.
    """
    root = _project_root()
    with (root / "config.toml").open("rb") as f:
        raw = tomllib.load(f)

    base = raw.get(section, {}) or {}
    modes = (base.get("modes", {}) or {})
    merged = _deep_merge(base, modes.get(mode, {}))

    merged["_paths"] = raw.get("paths", {}) or {}
    merged["_root"] = str(root)

    return merged


# Phokintime Global Config
# sync with networkmodel/config.py
# TODO: Both config_loader.py and networkmodel/config.py load overlapping configuration
# surfaces from config.toml. They should be unified in a future refactor so that
# networkmodel/config.py imports from config_loader.py rather than duplicating logic.
# See: https://github.com/bibymaths/phoskintime/issues (technical debt)

@dataclass(frozen=True)
class PhosKinConfig:
    kinase_net: str | Path
    tf_net: str | Path
    ms_data: str | Path
    rna_data: str | Path
    phospho_data: str | Path | None
    kinopt_results: str | Path
    tfopt_results: str | Path

    normalize_fc_steady: bool
    use_initial_condition_from_data: bool

    time_points_prot: np.ndarray
    time_points_rna: np.ndarray
    time_points_phospho: np.ndarray

    bounds_config: dict[str, tuple[float, float]]
    model: str

    ode_abs_tol: float
    ode_rel_tol: float
    ode_max_steps: int

    loss_mode: int
    maximum_iterations: int
    seed: int
    cores: int

    regularization_rna: float
    regularization_lambda: float
    regularization_phospho: float
    regularization_protein: float

    results_dir: str | Path

    app_name: str = "Phoskintime-Global"
    version: str = "0.1.0"
    parent_package: str = "phoskintime"
    citation: str = ""
    doi: str = ""
    github_url: str = ""
    docs_url: str = ""

    hyperparam_scan: bool = False

    # Inference / post-optimization analyses
    n_starts: int = 5
    optimizer: str = "lbfgsb"
    # Optional L-BFGS-B stopping criteria (None = networkmodel.optimizers.LBFGSB_DEFAULTS).
    lbfgsb_maxiter: int | None = None
    lbfgsb_maxfun: int | None = None
    lbfgsb_ftol: float | None = None
    lbfgsb_pgtol: float | None = None
    lbfgsb_maxcor: int | None = None
    lbfgsb_maxls: int | None = None
    profile_likelihood: bool = False
    profile_indices: str = ""
    profile_grid_size: int = 10
    posterior_sampling: bool = False
    posterior_num_warmup: int = 20
    posterior_num_samples: int = 30

    # Data scaling & weighting
    scaling_method: str = "none"
    fc_baseline_near_zero: float = 1e-12
    drive_extrapolation: str = "hold"
    drive_baseline_tau: float = 60.0
    weighting_method_protein: str = "uniform"
    weighting_method_rna: str = "uniform"
    weighting_method_phospho: str = "uniform"

    # Sensitivity analysis
    sensitivity_analysis: bool = False
    sensitivity_perturbation: float = 0.2
    sensitivity_trajectories: int = 1000
    sensitivity_levels: int = 400
    sensitvity_top_curves: int = 50
    sensitivity_metric: str = "total_signal"

    # Models metadata
    available_models: tuple[str, ...] = ()

    # Optional pure-JAX hyperedge/network preprocessing
    enable_hyperedge_preprocessing: bool = False
    hyperedge_preprocessing_output_subdir: str = "networkpruning"
    hyperedge_discovery_threshold: float = 0.0
    hyperedge_pruning_threshold: float = 0.0
    hyperedge_motif_detection: bool = True
    hyperedge_sparse_tensor_export: bool = True
    hyperedge_identifiability_preprocessing: bool = True
    hyperedge_max_triplets: int | None = None
    hyperedge_batch_size: int = 65536
    hyperedge_plot_generation: bool = True
    hyperedge_csv_export: bool = True
    # Optional PINN / NeuralODE extension
    enable_pinn: bool = False
    pinn_mode: str = "off"
    pinn_hidden_size: int = 32
    pinn_depth: int = 2
    pinn_activation: str = "tanh"
    pinn_output_scale: float = 1e-2
    pinn_weight_bound: float = 0.25
    pinn_l2_regularization: float = 1e-6
    pinn_t_scale: float = 1.0
    pinn_y_scale: float = 1.0
    pinn_seed: int = 42


def _parse_hyperedge_preprocessing_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate optional pure-JAX hyperedge preprocessing settings."""
    prep = cfg.get("hyperedge_preprocessing", {}) or {}

    def _bool(name: str, default: bool) -> bool:
        raw = prep.get(name, default)
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(raw)

    def _float_nonnegative(name: str, default: float) -> float:
        value = float(prep.get(name, default))
        if value < 0.0:
            raise ValueError(f"networkmodel.hyperedge_preprocessing.{name} must be non-negative, got {value}")
        return value

    def _nonnegative_int_cap(name: str, default: int | None) -> int | None:
        raw = prep.get(name, default)
        if raw is None:
            return None
        value = int(raw)
        if value < 0:
            raise ValueError(f"networkmodel.hyperedge_preprocessing.{name} must be non-negative, got {value}")
        return None if value == 0 else value

    def _positive_int(name: str, default: int) -> int:
        value = int(prep.get(name, default))
        if value <= 0:
            raise ValueError(f"networkmodel.hyperedge_preprocessing.{name} must be positive, got {value}")
        return value

    output_subdir = str(prep.get("output_subdir", "networkpruning")).strip()
    if not output_subdir:
        raise ValueError("networkmodel.hyperedge_preprocessing.output_subdir must be a non-empty string")

    return {
        "enable_hyperedge_preprocessing": _bool("enable_hyperedge_preprocessing", False),
        "hyperedge_preprocessing_output_subdir": output_subdir,
        "hyperedge_discovery_threshold": _float_nonnegative("hyperedge_discovery_threshold", 0.0),
        "hyperedge_pruning_threshold": _float_nonnegative("hyperedge_pruning_threshold", 0.0),
        "hyperedge_motif_detection": _bool("motif_detection", True),
        "hyperedge_sparse_tensor_export": _bool("sparse_tensor_export", True),
        "hyperedge_identifiability_preprocessing": _bool("identifiability_preprocessing", True),
        "hyperedge_max_triplets": _nonnegative_int_cap("max_triplets", 0),
        "hyperedge_batch_size": _positive_int("batch_size", 65536),
        "hyperedge_plot_generation": _bool("plot_generation", True),
        "hyperedge_csv_export": _bool("csv_export", True),
    }


# Keys load_config_toml reads directly from the [networkmodel] table (X8).
_NETWORKMODEL_TOP_LEVEL_KEYS = frozenset({
    "app_name", "version", "parent_package", "citation", "doi", "github_url", "docs_url", "cores", "seed",
    "kinase_net", "tf_net", "ms", "rna", "phospho", "kinopt", "tfopt", "output_dir", "output_directory",
    "normalize_fc_steady", "use_initial_condition_from_data", "hyperparam_scan", "n_gen", "loss", "n_starts",
    "optimizer", "lbfgsb_maxiter", "lbfgsb_maxfun", "lbfgsb_ftol", "lbfgsb_pgtol", "lbfgsb_maxcor", "lbfgsb_maxls",
    "profile_likelihood", "profile_indices", "profile_grid_size", "posterior_sampling", "posterior_num_warmup",
    "posterior_num_samples", "scaling_method", "weighting_method_protein", "weighting_method_rna",
    "weighting_method_phospho", "sensitivity_analysis", "sensitivity_levels", "sensitivity_metric",
    "sensitivity_perturbation", "sensitivity_top_curves", "sensitivity_trajectories", "enable_pinn",
    "lambda_prior", "lambda_protein", "lambda_rna", "lambda_phospho",
})


def _warn_misplaced_networkmodel_keys(cfg: dict) -> None:
    """Warn when a known [networkmodel] key sits in a sub-table, where the loader never reads it (X8)."""
    import warnings

    for sub_name, sub in cfg.items():
        if not isinstance(sub, dict):
            continue
        # "rna"/"phospho"/"protein" are also legitimate time-grid names in [networkmodel.timepoints].
        legit = {"rna", "phospho", "protein"} if sub_name == "timepoints" else set()
        misplaced = sorted(k for k in sub if k in _NETWORKMODEL_TOP_LEVEL_KEYS and k not in legit)
        if misplaced:
            warnings.warn(
                f"config: {misplaced} sit under [networkmodel.{sub_name}] but are read from [networkmodel]; "
                "they are ignored there (see KNOWN_ISSUES.md X8).",
                UserWarning, stacklevel=3,
            )


def load_config_toml(path: str | Path) -> PhosKinConfig:
    path = Path(path)

    with path.open("rb") as f:
        full_cfg = tomllib.load(f)

    cfg = (full_cfg or {}).get("networkmodel", {}) or {}
    _warn_misplaced_networkmodel_keys(cfg)

    # -------------------------
    # 0) Metadata
    # -------------------------
    app_name = cfg.get("app_name", "Phoskintime-Global")
    version = cfg.get("version", "0.1.0")
    parent_package = cfg.get("parent_package", "phoskintime")
    citation = cfg.get("citation", "")
    doi = cfg.get("doi", "")
    github_url = cfg.get("github_url", "")
    docs_url = cfg.get("docs_url", "")

    # -------------------------
    # 1) Inputs
    # -------------------------
    kinase_net = cfg.get("kinase_net", "data/input2.csv")
    tf_net = cfg.get("tf_net", "data/input4.csv")
    ms_data = cfg.get("ms", "data/input1.csv")
    rna_data = cfg.get("rna", "data/input3.csv")

    phospho_data = cfg.get("phospho", None)
    if not phospho_data:
        phospho_data = cfg.get("ms", "data/input1.csv")

    kinopt_res = cfg.get("kinopt", "data/kinopt_results.xlsx")
    tfopt_res = cfg.get("tfopt", "data/tfopt_results.xlsx")

    # -------------------------
    # 2) Output & Run Settings
    # -------------------------
    res_dir = cfg.get("output_dir", cfg.get("output_directory", "results_global"))
    cores = int(cfg.get("cores", 0))
    seed = int(cfg.get("seed", 42))

    # -------------------------
    # 3) Data inference flags
    # -------------------------
    normalize_fc_steady = bool(cfg.get("normalize_fc_steady", False))
    use_initial_condition_from_data = bool(cfg.get("use_initial_condition_from_data", True))

    # -------------------------
    # 4) Timepoints
    # -------------------------
    tp_cfg = cfg.get("timepoints", {}) or {}
    tp_prot = tp_cfg.get("protein", []) or []
    tp_rna = tp_cfg.get("rna", []) or []
    tp_phospho = tp_cfg.get("phospho_protein", None)

    # If phospho timepoints are missing, fall back to protein timepoints (sane default)
    if not tp_phospho:
        tp_phospho = tp_prot

    time_points_prot = np.asarray(tp_prot, dtype=float)
    time_points_rna = np.asarray(tp_rna, dtype=float)
    time_points_phospho = np.asarray(tp_phospho, dtype=float)

    # -------------------------
    # 5) Model(s)
    # -------------------------
    models_cfg = cfg.get("models", {}) or {}
    model = models_cfg.get("default_model", cfg.get("model", "combinatorial"))
    available_models = tuple(models_cfg.get("available_models", []) or [])

    # -------------------------
    # 6) JAXopt optimizer params
    # -------------------------
    hyp_scan = bool(cfg.get("hyperparam_scan", False))

    max_iter = int(cfg.get("n_gen", 200))

    # Loss
    loss_mode = int(cfg.get("loss", 0))
    if loss_mode != 0:
        raise ValueError(
            f"[networkmodel] loss = {loss_mode} is not supported: the only loss is 0 (weighted mean squared "
            "error). Robust losses were never used by the fit; see KNOWN_ISSUES.md N3."
        )

    # Inference / post-optimization analyses
    n_starts = int(cfg.get("n_starts", 5))
    optimizer = str(cfg.get("optimizer", "lbfgsb")).strip().lower()
    if optimizer not in ("projected_gradient", "lbfgsb"):
        raise ValueError(f"[networkmodel] optimizer must be 'projected_gradient' or 'lbfgsb', got {optimizer!r}.")
    lbfgsb = {}
    for _k, _cast in (("maxiter", int), ("maxfun", int), ("ftol", float), ("pgtol", float), ("maxcor", int),
                      ("maxls", int)):
        _v = cfg.get(f"lbfgsb_{_k}")
        lbfgsb[_k] = None if _v is None else _cast(_v)
        if lbfgsb[_k] is not None and not lbfgsb[_k] > 0:
            raise ValueError(f"[networkmodel] lbfgsb_{_k} must be > 0, got {_v!r}.")
    profile_likelihood = bool(cfg.get("profile_likelihood", False))
    profile_indices = str(cfg.get("profile_indices", ""))
    profile_grid_size = int(cfg.get("profile_grid_size", 10))
    posterior_sampling = bool(cfg.get("posterior_sampling", False))
    posterior_num_warmup = int(cfg.get("posterior_num_warmup", 20))
    posterior_num_samples = int(cfg.get("posterior_num_samples", 30))

    # -------------------------
    # 7) Regularization (loss weights)
    # -------------------------
    # Support both the new flat keys and any legacy nested dicts if they exist.
    reg_cfg = cfg.get("regularization", {}) or {}
    reg_lambda = float(cfg.get("lambda_prior", reg_cfg.get("lambda", 0.01)))
    reg_protein = float(cfg.get("lambda_protein", reg_cfg.get("protein", 1.0)))
    reg_rna = float(cfg.get("lambda_rna", reg_cfg.get("rna", 1.0)))
    reg_phospho = float(cfg.get("lambda_phospho", reg_cfg.get("phospho", 1.0)))

    # -------------------------
    # 8) Solver
    # -------------------------
    sol_cfg = cfg.get("solver", {}) or {}
    ode_abs_tol = float(sol_cfg.get("absolute_tolerance", 1e-8))
    ode_rel_tol = float(sol_cfg.get("relative_tolerance", 1e-8))
    ode_max_steps = int(sol_cfg.get("max_timesteps", 200000))

    # -------------------------
    # 9) Bounds
    # -------------------------
    if cfg.get("bounds"):
        raise ValueError(
            "[networkmodel.bounds] is not used: the optimizer bounds are computed from the data by "
            "networkmodel.utils.calculate_bio_bounds (logged at run start, recorded in metadata.json). "
            "Remove the [networkmodel.bounds] table; see KNOWN_ISSUES.md N2."
        )
    bounds_config: dict[str, tuple[float, float]] = {}

    # -------------------------
    # 10) Scaling & weighting
    # -------------------------
    scaling_method = str(cfg.get("scaling_method", "none"))
    fc_baseline_near_zero = float(cfg.get("fc_baseline_near_zero", 1e-12))
    drive_extrapolation = str(cfg.get("drive_extrapolation", "hold"))
    drive_baseline_tau = float(cfg.get("drive_baseline_tau", 60.0))
    if drive_extrapolation not in ("hold", "baseline"):
        raise ValueError(f"[networkmodel] drive_extrapolation must be 'hold' or 'baseline', got {drive_extrapolation!r}.")
    if drive_extrapolation == "baseline" and not drive_baseline_tau > 0.0:
        raise ValueError(f"[networkmodel] drive_baseline_tau must be > 0, got {drive_baseline_tau}.")
    if not fc_baseline_near_zero >= 0.0:
        raise ValueError(f"[networkmodel] fc_baseline_near_zero must be >= 0, got {fc_baseline_near_zero}.")
    weighting_method_protein = str(cfg.get("weighting_method_protein", "uniform"))
    weighting_method_rna = str(cfg.get("weighting_method_rna", "uniform"))
    weighting_method_phospho = str(cfg.get("weighting_method_phospho", "uniform"))
    for _key, _val in (("weighting_method_protein", weighting_method_protein),
                       ("weighting_method_rna", weighting_method_rna),
                       ("weighting_method_phospho", weighting_method_phospho)):
        if _val != "uniform":
            raise ValueError(
                f"[networkmodel] {_key} = {_val!r} is not supported: only 'uniform' weighting is implemented "
                "(see KNOWN_ISSUES.md N4)."
            )

    # -------------------------
    # 11) Sensitivity analysis
    # -------------------------
    sensitivity_analysis = bool(cfg.get("sensitivity_analysis", False))
    sensitivity_perturbation = float(cfg.get("sensitivity_perturbation", 0.2))
    sensitivity_trajectories = int(cfg.get("sensitivity_trajectories", 1000))
    sensitivity_levels = int(cfg.get("sensitivity_levels", 400))
    sensitivity_top_curves = int(cfg.get("sensitivity_top_curves", 50))  # key matches TOML
    sensitivity_metric = str(cfg.get("sensitivity_metric", "total_signal"))

    # -------------------------
    # 12) Optional hyperedge/network preprocessing
    # -------------------------
    hyperedge_cfg = _parse_hyperedge_preprocessing_config(cfg)

    # -------------------------
    # 13) Optional PINN / NeuralODE
    # -------------------------
    enable_pinn = bool(cfg.get("enable_pinn", False))
    pinn_mode = str(cfg.get("pinn_mode", "off"))
    pinn_hidden_size = int(cfg.get("pinn_hidden_size", 32))
    pinn_depth = int(cfg.get("pinn_depth", 2))
    pinn_activation = str(cfg.get("pinn_activation", "tanh"))
    pinn_output_scale = float(cfg.get("pinn_output_scale", 1e-2))
    pinn_weight_bound = float(cfg.get("pinn_weight_bound", 0.25))
    pinn_l2_regularization = float(cfg.get("pinn_l2_regularization", 1e-6))
    pinn_t_scale = float(cfg.get("pinn_t_scale", 1.0))
    pinn_y_scale = float(cfg.get("pinn_y_scale", 1.0))
    pinn_seed = int(cfg.get("pinn_seed", seed))

    return PhosKinConfig(
        kinase_net=kinase_net,
        tf_net=tf_net,

        ms_data=ms_data,
        rna_data=rna_data,
        phospho_data=phospho_data,

        kinopt_results=kinopt_res,
        tfopt_results=tfopt_res,

        normalize_fc_steady=normalize_fc_steady,
        use_initial_condition_from_data=use_initial_condition_from_data,

        time_points_prot=time_points_prot,
        time_points_rna=time_points_rna,
        time_points_phospho=time_points_phospho,

        bounds_config=bounds_config,

        model=model,

        ode_abs_tol=ode_abs_tol,
        ode_rel_tol=ode_rel_tol,
        ode_max_steps=ode_max_steps,

        loss_mode=loss_mode,
        maximum_iterations=max_iter,

        seed=seed,
        cores=cores,

        regularization_rna=reg_rna,
        regularization_lambda=reg_lambda,
        regularization_phospho=reg_phospho,
        regularization_protein=reg_protein,

        results_dir=res_dir,

        app_name=app_name,
        version=version,
        parent_package=parent_package,
        citation=citation,
        doi=doi,
        github_url=github_url,
        docs_url=docs_url,

        hyperparam_scan=hyp_scan,

        n_starts=n_starts,
        optimizer=optimizer,
        lbfgsb_maxiter=lbfgsb["maxiter"],
        lbfgsb_maxfun=lbfgsb["maxfun"],
        lbfgsb_ftol=lbfgsb["ftol"],
        lbfgsb_pgtol=lbfgsb["pgtol"],
        lbfgsb_maxcor=lbfgsb["maxcor"],
        lbfgsb_maxls=lbfgsb["maxls"],
        profile_likelihood=profile_likelihood,
        profile_indices=profile_indices,
        profile_grid_size=profile_grid_size,
        posterior_sampling=posterior_sampling,
        posterior_num_warmup=posterior_num_warmup,
        posterior_num_samples=posterior_num_samples,

        scaling_method=scaling_method,
        fc_baseline_near_zero=fc_baseline_near_zero,
        drive_extrapolation=drive_extrapolation,
        drive_baseline_tau=drive_baseline_tau,

        weighting_method_protein=weighting_method_protein,
        weighting_method_rna=weighting_method_rna,
        weighting_method_phospho=weighting_method_phospho,

        sensitivity_analysis=sensitivity_analysis,
        sensitivity_perturbation=sensitivity_perturbation,
        sensitivity_trajectories=sensitivity_trajectories,
        sensitivity_levels=sensitivity_levels,
        sensitvity_top_curves=sensitivity_top_curves,
        sensitivity_metric=sensitivity_metric,

        available_models=available_models,

        enable_pinn=enable_pinn,
        pinn_mode=pinn_mode,
        pinn_hidden_size=pinn_hidden_size,
        pinn_depth=pinn_depth,
        pinn_activation=pinn_activation,
        pinn_output_scale=pinn_output_scale,
        pinn_weight_bound=pinn_weight_bound,
        pinn_l2_regularization=pinn_l2_regularization,
        pinn_t_scale=pinn_t_scale,
        pinn_y_scale=pinn_y_scale,
        pinn_seed=pinn_seed,

        **hyperedge_cfg,
    )
