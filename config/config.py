import os
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib
from numba import njit

from config.constants import (
    ALPHA_WEIGHT,
    BETA_WEIGHT,
    GAMMA_WEIGHT,
    DELTA_WEIGHT,
    INPUT_EXCEL_PROTEIN,
    INPUT_EXCEL_PSITE, DEV_TEST, MU_WEIGHT, INPUT_EXCEL_RNA, TIME_POINTS, BOOTSTRAPS, UB_mRNA_prod, UB_mRNA_deg,
    UB_Protein_prod, UB_Protein_deg, UB_Phospho_prod
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from config.logconf import setup_logger

logging = setup_logger()


def parse_bound_pair(val):
    """
    Parse a string representing a pair of bounds (lower, upper) into a tuple of floats.
    The upper bound can be 'inf' or 'infinity' to represent infinity.
    Raises ValueError if the input is not in the correct format.
    Args:
        val (str): The string to parse, e.g., "0,3" or "0,infinity".
    Returns:
        tuple: A tuple containing the lower and upper bounds as floats.
    """
    try:
        parts = val.split(',')
        if len(parts) != 2:
            raise ValueError("Bounds must be provided as 'lower,upper'")
        lower = float(parts[0])
        upper_str = parts[1].strip().lower()
        if upper_str in ["inf", "infinity"]:
            upper = float("inf")
        else:
            upper = float(parts[1])
        return lower, upper
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid bound pair '{val}': {e}")


def parse_fix_value(val):
    """
    Parse a fixed value or a list of fixed values from a string.
    If the input is a single value, it returns that value as a float.
    If the input is a comma-separated list, it returns a list of floats.
    Raises ValueError if the input is not in the correct format.
    Args:
        val (str): The string to parse, e.g., "1.0" or "1.0,2.0".
    Returns:
        float or list: The parsed fixed value(s) as a float or a list of floats.
    """
    if val is None:
        return None
    if ',' in val:
        try:
            return [float(x) for x in val.split(',')]
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid fixed value list '{val}': {e}")
    else:
        try:
            return float(val)
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid fixed value '{val}': {e}")


def ensure_output_directory(directory):
    """
    Args:
        directory (str): The path to the directory to create.
    Returns:
        None
    """
    os.makedirs(directory, exist_ok=True)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def default_config_path() -> Path:
    return PROJECT_ROOT / "config.toml"


def parse_config_path(argv: list[str] | None = None) -> Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--conf", default=None)
    known, _ = pre_parser.parse_known_args(argv)
    return Path(known.conf).expanduser() if known.conf else None


def _model_type(model: str) -> str:
    return {
        "protwise": "Protein-wise",
        "distmod": "Distributive",
        "succmod": "Successive",
        "randmod": "Random",
    }.get(str(model), "Unknown")


def load_selected_config(conf_path: str | Path | None = None) -> dict[str, Any]:
    selected = Path(conf_path).expanduser().resolve() if conf_path else default_config_path().resolve()
    with selected.open("rb") as handle:
        raw = tomllib.load(handle)
    ode = raw.get("ode", {}) or {}
    modes = ode.get("modes", {}) or {}
    merged = _deep_merge(ode, modes.get("local", {}) or {})
    merged["_paths"] = raw.get("paths", {}) or {}
    merged["_root"] = str(PROJECT_ROOT)
    merged["_config_path"] = str(selected)
    merged["_config_source"] = "custom" if conf_path else "default"
    return merged


def _path_from_config(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _defaults_from_loaded_config(loaded_config: dict[str, Any]) -> dict[str, Any]:
    root = Path(loaded_config.get("_root", PROJECT_ROOT))
    paths = loaded_config.get("_paths", {}) or {}
    bounds = loaded_config.get("bounds", {}) or {}
    bootstrap = loaded_config.get("bootstrap", {}) or {}
    time = loaded_config.get("time", {}) or {}
    inputs = loaded_config.get("inputs", {}) or {}
    output = loaded_config.get("output", {}) or {}
    fit = loaded_config.get("fit", {}) or {}
    weights = fit.get("composite_weights", {}) or {}
    sensitivity = loaded_config.get("sensitivity", {}) or {}
    morris = sensitivity.get("morris", {}) or {}

    model = str(loaded_config.get("model", "randmod"))
    model_type = _model_type(model)
    # Treat empty strings in [paths] as missing values.
    results_dir = _path_from_config(root, paths.get("results_dir") or "results")
    out_dir_name = str(output.get("out_dir_name") or "").strip()
    out_xlsx_name = str(output.get("out_xlsx_name") or "").strip()
    out_dir = results_dir / (out_dir_name or f"{model_type}_results")

    return {
        "conf": None if loaded_config.get("_config_source") == "default" else loaded_config.get("_config_path"),
        "resolved_config_path": loaded_config.get("_config_path"),
        "config_source": loaded_config.get("_config_source", "default"),
        "A_bound": (0.0, float(bounds.get("mRNA_prod", 20))),
        "B_bound": (0.0, float(bounds.get("mRNA_deg", 20))),
        "C_bound": (0.0, float(bounds.get("protein_prod", 20))),
        "D_bound": (0.0, float(bounds.get("protein_deg", 20))),
        "Ssite_bound": (0.0, float(bounds.get("phospho_prod", 20))),
        "Dsite_bound": (0.0, float(bounds.get("phospho_deg", bounds.get("protein_deg", 20)))),
        "bootstraps": int(bootstrap.get("n", 0)),
        "input_excel_protein": _path_from_config(root, inputs["protein_excel"]) if inputs.get("protein_excel") else "",
        "input_excel_psite": _path_from_config(root, inputs["psite_excel"]) if inputs.get("psite_excel") else "",
        "input_excel_rna": _path_from_config(root, inputs["rna_excel"]) if inputs.get("rna_excel") else "",
        "outdir": out_dir,
        "out_results_dir": out_dir / (out_xlsx_name or f"{model_type}_results.xlsx"),
        "time_points": np.asarray(
            time.get("protein", [0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]),
            dtype=float),
        "time_points_rna": np.asarray(time.get("rna", [4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0]),
                                      dtype=float),
        "model": model,
        "model_type": model_type,
        "dev_test": bool(loaded_config.get("dev_test", False)),
        "use_regularization": bool(fit.get("use_regularization", True)),
        "alpha_ci": float(loaded_config.get("alpha_ci", 0.95)),
        "weights": {
            "alpha": float(weights.get("rmse", 1.0)),
            "beta": float(weights.get("mae", 1.0)),
            "gamma": float(weights.get("var", 1.0)),
            "delta": float(weights.get("mse", 1.0)),
            "mu": float(weights.get("l2", 1.0)),
        },
        "sensitivity": {
            "enabled": bool(sensitivity.get("enabled", True)),
            "metric": str(loaded_config.get("y_metric", "total_signal")),
            "num_trajectories": int(morris.get("num_trajectories", 1000)),
            "num_levels": int(morris.get("num_levels", 400)),
            "perturbation": float(sensitivity.get("perturbation", 0.5)),
        },
    }


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PhosKinTime - ODE Parameter Estimation of Cell Signalling Events in Temporal Space"
    )
    parser.add_argument("--conf", default=defaults["conf"], help="Path to ProtWise/ODE TOML config file.")
    parser.add_argument("--A-bound", type=parse_bound_pair, default=defaults["A_bound"])
    parser.add_argument("--B-bound", type=parse_bound_pair, default=defaults["B_bound"])
    parser.add_argument("--C-bound", type=parse_bound_pair, default=defaults["C_bound"])
    parser.add_argument("--D-bound", type=parse_bound_pair, default=defaults["D_bound"])
    parser.add_argument("--Ssite-bound", type=parse_bound_pair, default=defaults["Ssite_bound"])
    parser.add_argument("--Dsite-bound", type=parse_bound_pair, default=defaults["Dsite_bound"])
    parser.add_argument("--bootstraps", type=int, default=defaults["bootstraps"])
    parser.add_argument("--input-excel-protein", type=str, default=str(defaults["input_excel_protein"]),
                        help="Path to the original protein data file")
    parser.add_argument("--input-excel-psite", type=str, default=str(defaults["input_excel_psite"]),
                        help="Path to the estimated optimized phosphorylation-residue file")
    parser.add_argument("--input-excel-rna", type=str, default=str(defaults["input_excel_rna"]),
                        help="Path to the estimated optimized mRNA-TF file")
    parser.add_argument("--outdir", "--output-dir", dest="outdir", type=str, default=str(defaults["outdir"]),
                        help="Directory where all run outputs and provenance files are written.")
    return parser


def parse_args(argv: list[str] | None = None):
    selected_config = parse_config_path(argv)
    if selected_config is not None:
        os.environ["PHOSKINTIME_ODE_CONFIG"] = str(selected_config.expanduser().resolve())
    loaded_config = load_selected_config(selected_config)
    defaults = _defaults_from_loaded_config(loaded_config)
    parser = build_parser(defaults)
    args = parser.parse_args(argv)
    args.resolved_config_path = str(Path(defaults["resolved_config_path"]).resolve())
    args.config_source = defaults["config_source"]
    args.time_points = defaults["time_points"]
    args.time_points_rna = defaults["time_points_rna"]
    args.out_results_dir = str(Path(args.outdir) / Path(defaults["out_results_dir"]).name)
    args.model = defaults["model"]
    args.model_type = defaults["model_type"]
    args.dev_test = defaults["dev_test"]
    args.use_regularization = defaults["use_regularization"]
    args.alpha_ci = defaults["alpha_ci"]
    args.weights = defaults["weights"]
    args.sensitivity = defaults["sensitivity"]
    return args


def log_config(logger, bounds, args):
    """
    Log the configuration settings for the PhosKinTime script.
    This function logs the parameter bounds
    bootstrapping iterations.
    It uses the provided logger to output the information.

    Args:
        logger (logging.Logger): The logger to use for logging.
        bounds (dict): The parameter bounds.
        args (argparse.Namespace): The command-line arguments.
    Returns:
        None
    """
    logger.info("Parameter Bounds:")
    for key, val in bounds.items():
        logger.info(f"      {key}      : {val}")
    logger.info(f"      Bootstrapping Iterations: {args.bootstraps}")
    logger.info("           --------------------------------")
    np.set_printoptions(suppress=True)


def extract_config(args, loaded_config: dict[str, Any] | None = None):
    """
    Extract effective ProtWise runtime settings after config and CLI precedence
    have been resolved. CLI values override values loaded from --conf/default config.
    """
    bounds = {
        "A": args.A_bound,
        "B": args.B_bound,
        "C": args.C_bound,
        "D": args.D_bound,
        "S(i)": args.Ssite_bound,
        "D(i)": args.Dsite_bound,
    }
    config = {
        "bounds": bounds,
        "bootstraps": args.bootstraps,
        "input_excel_protein": args.input_excel_protein,
        "input_excel_psite": args.input_excel_psite,
        "input_excel_rna": args.input_excel_rna,
        "max_workers": 1 if getattr(args, "dev_test", DEV_TEST) else os.cpu_count(),
        "outdir": args.outdir,
        "out_results_dir": args.out_results_dir,
        "time_points": args.time_points,
        "time_points_rna": args.time_points_rna,
        "supplied_config_path": args.conf,
        "resolved_config_path": args.resolved_config_path,
        "config_source": args.config_source,
        "model": args.model,
        "model_type": args.model_type,
        "dev_test": args.dev_test,
        "use_regularization": args.use_regularization,
        "alpha_ci": args.alpha_ci,
        "weights": args.weights,
        "sensitivity": args.sensitivity,
    }
    return config


@njit(cache=True)
def score_fit(params, target, prediction,
              alpha=ALPHA_WEIGHT,
              beta=BETA_WEIGHT,
              gamma=GAMMA_WEIGHT,
              delta=DELTA_WEIGHT,
              mu=MU_WEIGHT):
    """
    Calculate the score for the fit of a model to target data.
    The score is a weighted combination of various metrics including
    mean squared error (MSE), root mean squared error (RMSE),
    mean absolute error (MAE), variance, and regularization penalty.
    The weights for each metric can be adjusted using the parameters
    alpha, beta, gamma, and delta.
    The regularization penalty is controlled by the reg_penalty parameter.
    The function returns the calculated score.
    Args:
        params (np.ndarray): The model parameters.
        target (np.ndarray): The target data.
        prediction (np.ndarray): The predicted data.
        alpha (float): Weight for RMSE.
        beta (float): Weight for MAE.
        gamma (float): Weight for variance.
        delta (float): Weight for MSE.
        mu (float): Regularization penalty weight.
    Returns:
        float: The calculated score.
    """

    # Compute scaled absolute residuals (error per data point).
    residual = np.abs(target - prediction) / target.size

    # Compute mean squared error (MSE) from residuals.
    mse = np.sum(residual ** 2)

    # Compute root mean squared error (RMSE) from residuals.
    rmse = np.sqrt(np.mean(residual ** 2))

    # Compute mean absolute error (MAE) from residuals.
    mae = np.mean(residual)

    # Compute variance of residuals.
    variance = np.var(residual)

    # L2 norm of parameters.
    l2_norm = np.linalg.norm(params, ord=2) / len(params)

    # Calculate weighted total score combining errors
    score = delta * mse + alpha * rmse + beta * mae + gamma * variance + mu * l2_norm

    return score


def future_times(n_new: int, ratio: Optional[float] = None, tp: np.ndarray = TIME_POINTS) -> np.ndarray:
    """
    Extend ttime points by n_new points, each spaced by multiplying the previous interval by ratio.
    If ratio is None, it is inferred from the last two points.

    Args:
        n_new (int): Number of new time points to generate.
        ratio (float, optional): Ratio to multiply the previous interval. Defaults to None.
        tp (np.ndarray, optional): Existing time points. Defaults to TIME_POINTS.
    Returns:
        np.ndarray: Extended time points.
    """
    times = tp.tolist()
    if ratio is None:
        # avoid divide-by-zero if the last point is zero
        ratio = times[-1] / times[-2]
    for _ in range(n_new):
        times.append(times[-1] * ratio)
    return np.array(times)
