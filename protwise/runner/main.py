import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import atexit
import logging
import sys
from pathlib import Path

ODE_CONFIG_ENV = "PHOSKINTIME_ODE_CONFIG"


def _parse_config_path(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--conf", default=None)
    known, _ = parser.parse_known_args(argv)
    return Path(known.conf).expanduser().resolve() if known.conf else None


def _select_config_for_runtime(argv: list[str] | None = None) -> None:
    conf_path = _parse_config_path(argv)
    if conf_path is not None:
        os.environ[ODE_CONFIG_ENV] = str(conf_path)
        for module_name in list(sys.modules):
            if module_name in {"config.constants", "config.config"} or module_name.startswith((
                "protwise.paramest",
                "protwise.plotting",
                "protwise.models",
            )):
                sys.modules.pop(module_name, None)


@atexit.register
def _close_log_handlers():
    lg = logging.getLogger("phoskintime")
    for h in list(lg.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass


def _metadata_extra(config: dict) -> dict:
    return {
        "supplied_config_path": config.get("supplied_config_path"),
        "resolved_config_path": config.get("resolved_config_path"),
        "config_source": config.get("config_source"),
        "effective_inputs": {
            "protein": config.get("input_excel_protein"),
            "phosphosite": config.get("input_excel_psite"),
            "rna": config.get("input_excel_rna"),
        },
        "effective_bounds": config.get("bounds"),
        "effective_bootstraps": config.get("bootstraps"),
        "effective_time_grid": config.get("time_points"),
        "output_directory": config.get("outdir"),
    }



def initialize_run_contract(config: dict, args, logger):
    from common.results import (
        attach_file_console_logger,
        ensure_result_dir,
        write_command,
        write_metadata,
        write_resolved_config,
    )

    out_dir = ensure_result_dir(config["outdir"])["root"]
    out_results_dir = out_dir / Path(config["out_results_dir"]).name
    config["outdir"] = str(out_dir)
    config["out_results_dir"] = str(out_results_dir)
    attach_file_console_logger(logger, out_dir)
    write_command(out_dir)
    write_resolved_config(out_dir, config)
    write_metadata(
        out_dir,
        workflow="protwise.runner",
        args=args,
        inputs=[config.get("input_excel_protein"), config.get("input_excel_psite"), config.get("input_excel_rna")],
        extra=_metadata_extra(config),
    )
    return out_dir, out_results_dir

def main(argv: list[str] | None = None):
    """
    Main function to run the phosphorylation modelling process.
    It reads the selected configuration, loads the data, and processes each gene.
    """
    _select_config_for_runtime(argv)

    import pandas as pd
    from config.helpers import location
    from config.config import parse_args, extract_config, log_config
    from config.constants import (
        NUM_TRAJECTORIES,
        PARAMETER_SPACE,
        PERTURBATIONS_VALUE,
        Y_METRIC_DESCRIPTIONS,
        validate_ode_inputs,
    )
    from config.logconf import setup_logger
    from protwise.paramest.core import process_gene_wrapper
    from protwise.plotting import Plotter
    from common.utils import latexit
    from common.results import populate_standard_subdirs
    from common.utils.display import ensure_output_directory, save_result, organize_output_files, create_report, merge_obs_est

    logger = setup_logger()
    args = parse_args(argv)
    config = extract_config(args)
    if not config:
        logger.error("Invalid configuration. Exiting.")
        return

    validate_ode_inputs(config)
    out_dir, out_results_dir = initialize_run_contract(config, args, logger)

    # Set up the logger
    weights = config.get("weights", {})
    sensitivity = config.get("sensitivity", {})
    logger.info("           --------------------------------")
    logger.info(f"{config['model_type']} Phosphorylation Modelling Configuration")
    logger.info("           --------------------------------")
    log_config(logger, config["bounds"], args)
    logger.info("      i = Number of phosphorylation sites (Residue_Position) in the model")
    logger.info(f"      L2 Regularization: {config['use_regularization']}")
    logger.info(f"      Confidence Interval: {config['alpha_ci'] * 100}")
    logger.info("           --------------------------------")
    logger.info("       Composite Scoring Function:")
    logger.info("       score = α * RMSE + β * MAE + γ * Var(residuals) + δ * MSE + μ * L2 norm")
    logger.info("           --------------------------------")
    logger.info("       Definitions:")
    logger.info("       - RMSE: Root Mean Squared Error")
    logger.info("       - MAE: Mean Absolute Error")
    logger.info("       - Var(residuals): Variance of residuals")
    logger.info("       - MSE: Mean Squared Error")
    logger.info("       - L2 norm: L2 norm of parameter estimates")
    logger.info("           --------------------------------")
    logger.info("       Weights:")
    logger.info(f"      - α (RMSE): {weights.get('alpha')}")
    logger.info(f"      - β (MAE): {weights.get('beta')}")
    logger.info(f"      - γ (Var): {weights.get('gamma')}")
    logger.info(f"      - δ (MSE): {weights.get('delta')}")
    logger.info(f"      - μ (L2 norm): {weights.get('mu')}")
    logger.info("           --------------------------------")
    logger.info("       Lower score indicates a better fit.")
    logger.info("           --------------------------------")
    logger.info(f"      Sensitivity Analysis: {sensitivity.get('enabled')}")
    if sensitivity.get("enabled"):
        metric = str(sensitivity.get("metric", "total_signal"))
        logger.info(f"      - Metric: {' '.join(part.upper() for part in metric.split('_'))}")
        logger.info(f"      - {Y_METRIC_DESCRIPTIONS.get(metric, 'No description available.')}")
        logger.info(f"      - Number of Trajectories: {sensitivity.get('num_trajectories', NUM_TRAJECTORIES)}")
        logger.info(f"      - Parameter Space: {sensitivity.get('num_levels', PARAMETER_SPACE)}")
        logger.info(f"      - Perturbations: {sensitivity.get('perturbation', PERTURBATIONS_VALUE)}")
    logger.info("           --------------------------------")

    ensure_output_directory(out_dir)

    protein_data = pd.read_csv(config["input_excel_protein"])
    kinase_data = pd.read_excel(config["input_excel_psite"], sheet_name="Estimated")
    mrna_data = pd.read_excel(config["input_excel_rna"], sheet_name="Estimated")

    if mrna_data.empty and kinase_data.empty and protein_data.empty:
        logger.error("No data found in the input files.")
        return

    required_columns = ["Gene", "Psite"] + [f"x{i}" for i in range(1, 15)]
    missing_columns = [col for col in required_columns if col not in kinase_data.columns and col not in protein_data.columns]
    if missing_columns:
        logger.error(f"Missing columns in the phosphorylation data: {', '.join(missing_columns)}")
        return

    required_mrna_columns = ["mRNA"] + [f"x{i}" for i in range(1, 10)]
    missing_mrna_columns = [col for col in required_mrna_columns if col not in mrna_data.columns]
    if missing_mrna_columns:
        logger.error(f"Missing columns in the mRNA data: {', '.join(missing_mrna_columns)}")
        return

    proteins = set(kinase_data["Gene"].dropna().unique())
    mrnas = set(mrna_data["mRNA"].dropna().unique())
    common_proteins = sorted(proteins.intersection(mrnas))
    non_common = sorted(proteins.symmetric_difference(mrnas))

    if not common_proteins:
        logger.warning("No common proteins found between phosphorylation and mRNA data.")
    else:
        logger.info(f"Genes found in phosphorylation data: {len(proteins)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in proteins))
        logger.info(f"Genes found in mRNA data: {len(mrnas)}")
        logger.info("  " + " ".join(f"[{rna}]" for rna in mrnas))
        logger.info(f"Genes found common between phosphorylation and mRNA data: {len(common_proteins)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in common_proteins))
        logger.info(f"Genes NOT found in both datasets: {len(non_common)}")
        logger.info("  " + " ".join(f"[{gene}]" for gene in non_common))
        logger.info("           --------------------------------")

    if config.get("dev_test"):
        test_gene = "ABL2"
        if test_gene in kinase_data["Gene"].values:
            genes = kinase_data[kinase_data["Gene"] == test_gene]["Gene"].unique().tolist()
        else:
            raise ValueError(f"{test_gene} not found in the input data.")
    else:
        genes = common_proteins

    if not genes:
        logger.error("No genes found in the input data.")
        return

    results = []
    for gene in genes:
        logger.info(f"[{gene}]      Processing...")
        result = process_gene_wrapper(
            gene,
            protein_data,
            kinase_data,
            mrna_data,
            config["time_points"],
            config["bounds"],
            config["bootstraps"],
            out_dir=out_dir,
        )
        results.append(result)

    if not results:
        logger.error("No results found after processing.")
        return

    save_result(results, excel_filename=out_results_dir)
    merged_df = merge_obs_est(out_results_dir)

    Plotter("", out_dir).plot_gof(merged_df)
    Plotter("", out_dir).plot_kld(merged_df)
    Plotter("", out_dir).plot_top_param_pairs(out_results_dir)
    Plotter("", out_dir).plot_regularization(out_results_dir)
    Plotter("", out_dir).plot_model_error(out_results_dir)
    logger.info("Plotting completed.")

    latexit.main(out_dir)
    logger.info("LateX generated.")

    organize_output_files([out_dir])
    create_report(out_dir)

    logger.info("           --------------------------------")
    logger.info(f"          Report & Results {location(str(out_dir))}")
    for fpath in [out_dir / "report.html"]:
        logger.info(f"          {fpath.as_uri()}")
    logger.info("           --------------------------------")

    populate_standard_subdirs(out_dir)


if __name__ == "__main__":
    main()
