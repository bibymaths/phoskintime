import argparse
import json
import os
import numpy as np
import multiprocessing as mp

import pandas as pd
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize

from phoskintime_global.buildmat import build_W_parallel, build_tf_matrix
from phoskintime_global.cache import prepare_fast_loss_data
from phoskintime_global.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, RESULTS_DIR, MAX_ITERATIONS, \
    POPULATION_SIZE, SEED, REGULARIZATION_LAMBDA, REGULARIZATION_RNA, REGULARIZATION_PHOSPHO, TIME_POINTS_PHOSPHO, \
    REGULARIZATION_PROTEIN
from phoskintime_global.io import load_data
from phoskintime_global.network import Index, KinaseInput, System
from phoskintime_global.optproblem import GlobalODE_MOO, get_weight_options
from phoskintime_global.params import init_raw_params, unpack_params
from phoskintime_global.simulate import simulate_and_measure
from phoskintime_global.utils import normalize_fc_to_t0, _base_idx, slen
from phoskintime_global.export import export_pareto_front_to_excel, plot_gof_from_pareto_excel, plot_goodness_of_fit, \
    export_results, save_pareto_3d, save_parallel_coordinates, create_convergence_video, save_gene_timeseries_plots, \
    scan_prior_reg, export_S_rates, export_S_rates_with_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinase-net", required=True)
    parser.add_argument("--tf-net", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--phospho", required=False)

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

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Print Arguments
    print(f"[Args] Output directory: {args.output_dir}")
    print(f"[Args] Number of cores: {args.cores}")
    print(f"[Args] Number of generations: {args.n_gen}")
    print(f"[Args] Population size: {args.pop}")
    print(f"[Args] Seed: {args.seed}")
    print(f"[Args] Lambda prior: {args.lambda_prior}")
    print(f"[Args] Lambda protein: {args.lambda_protein}")
    print(f"[Args] Lambda RNA: {args.lambda_rna}")
    print(f"[Args] Lambda phospho: {args.lambda_phospho}")

    # 1) Load
    df_kin, df_tf, df_prot, df_pho, df_rna = load_data(args)
    df_prot = normalize_fc_to_t0(df_prot)
    df_pho = normalize_fc_to_t0(df_pho)
    base = df_rna[df_rna["time"] == 4.0].set_index("protein")["fc"]
    df_rna["fc"] = df_rna.apply(lambda r: r["fc"] / base.get(r["protein"], np.nan), axis=1)
    df_rna = df_rna.dropna(subset=["fc"])

    # 2) Model index
    idx = Index(df_kin)

    # Restrict obs to model proteins
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    # Weights - Piecewise Early Boost
    # times
    tp_prot_pho = np.asarray(TIME_POINTS_PROTEIN, dtype=float)  # protein/phospho grid
    tp_rna = np.asarray(TIME_POINTS_RNA, dtype=float)  # rna grid

    # early windows: "first 5" and "first 3"
    ew_prot_pho = float(tp_prot_pho[4])  # 5th time point
    ew_rna = float(tp_rna[2])  # 3rd time point

    # build schemes
    schemes_prot_pho = get_weight_options(tp_prot_pho, early_window=ew_prot_pho)
    schemes_rna = get_weight_options(tp_rna, early_window=ew_rna)

    # pick weights
    w_prot_pho = schemes_prot_pho["piecewise_early_boost_mean1"]
    w_rna = schemes_rna["piecewise_early_boost_mean1"]

    # apply
    df_prot["w"] = w_prot_pho(df_prot["time"].values)
    df_pho["w"] = w_prot_pho(df_pho["time"].values)
    df_rna["w"] = w_rna(df_rna["time"].values)

    # 3) Build W + TF
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf, idx)
    kin_in = KinaseInput(idx.kinases, df_prot)

    # print("[Debug] KinaseInput coverage:",
    #       (kin_in.Kmat != 1.0).any(axis=1).sum(), "/", kin_in.Kmat.shape[0], "kinases have non-1 input")

    tf_deg = np.asarray(tf_mat.sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg == 0.0] = 1.0

    # 4) Defaults/system
    defaults = {
        "c_k": np.ones(len(idx.kinases)),
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1
    }
    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA, TIME_POINTS_PHOSPHO]))

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, df_pho, solver_times)
    loss_data["prot_base_idx"] = _base_idx(solver_times, 0.0)
    loss_data["rna_base_idx"] = _base_idx(solver_times, 4.0)
    loss_data["pho_base_idx"] = _base_idx(solver_times, 0.0)

    # 6) Decision vector bounds (raw space)
    theta0, slices, xl, xu = init_raw_params(defaults)
    lambdas = {
        "protein": args.lambda_protein,
        "rna": args.lambda_rna,
        "phospho": args.lambda_phospho,
        "prior": args.lambda_prior
    }

    # 7) Pymoo parallel runner
    runner = None
    pool = None
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        runner = StarmapParallelization(pool.starmap)
        print(f"[Fit] Parallel evaluation enabled with {args.cores} workers.")
    else:
        print("[Fit] Parallel evaluation disabled (or unavailable).")

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

    # Sanity Check - Decision vector
    sizes = {
        "c_k": slen(slices["c_k"]),
        "A_i": slen(slices["A_i"]),
        "B_i": slen(slices["B_i"]),
        "C_i": slen(slices["C_i"]),
        "D_i": slen(slices["D_i"]),
        "Dp_i": slen(slices["Dp_i"]),
        "E_i": slen(slices["E_i"]),
        "tf_scale": slen(slices["tf_scale"]),
    }

    print(f"n_var = {problem.n_var}")
    for k, v in sizes.items():
        print(f"{k}[{v}]")

    total = sum(sizes.values())
    print(f"sum_slices = {total}")

    assert total == problem.n_var, f"Mismatch: sum_slices={total} != n_var={problem.n_var}"

    # 9) UNSGA3 needs reference directions
    ref_dirs = get_reference_directions(
        "das-dennis",
        problem.n_obj,
        n_partitions=20
    )

    # Print number of reference directions
    print(f"[Fit] Number of reference directions: {len(ref_dirs)}")

    algorithm = UNSGA3(
        pop_size=args.pop,
        ref_dirs=ref_dirs,
        eliminate_duplicates=True,
        sampling=LHS()
    )

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=args.n_gen,
        n_max_evals=100000
    )

    print(f"[Data] Number of points: {loss_data['n_p']} protein, {loss_data['n_r']} RNA, {loss_data['n_ph']} phospho | Total {loss_data['n_p'] + loss_data['n_r'] + loss_data['n_ph']} data points")
    print(f"[Fit] UNSGA3: pop={args.pop}, n_gen={args.n_gen}, n_var={problem.n_var}, n_obj={problem.n_obj}")

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

    # 10) Save Pareto set
    X = res.X
    F = res.F
    np.save(os.path.join(args.output_dir, "pareto_X.npy"), X)
    np.save(os.path.join(args.output_dir, "pareto_F.npy"), F)

    # Also write a CSV summary
    df_pareto = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "phospho_mse"])
    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_F.csv"), index=False)
    print(f"[Output] Saved Pareto front: {len(df_pareto)} solutions")

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

    print(f"[Output] Saved Pareto front Excel: {excel_path}")

    plot_gof_from_pareto_excel(
        excel_path=excel_path,
        output_dir=os.path.join(args.output_dir, "gof_all"),
        plot_goodness_of_fit_func=plot_goodness_of_fit,
        df_prot_obs_all=df_prot,
        df_rna_obs_all=df_rna,
        df_phos_obs_all=df_pho,
        top_k=None,
        score_col="scalar_score",
    )

    print(f"[Output] Saved Goodness of Fit plots for all Pareto solutions.")

    # 11) Pick one solution
    F = res.F
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    w = np.array([args.lambda_protein, args.lambda_rna, args.lambda_phospho], dtype=float)
    I = np.argmin((Fn * w).sum(axis=1))
    theta_best = X[I].astype(float)
    F_best = F[I]
    params = unpack_params(theta_best, slices)
    sys.update(**params)

    # Save the phosphorylation rates
    export_S_rates(sys, idx, args.output_dir, filename="S_rates_picked.csv", long=True)
    print("[Output] Saved phosphorylation rates for picked solution.")

    export_S_rates_with_times(
        sys,
        idx,
        args.output_dir,
        filename="S_rates_picked_times.csv",
        times=TIME_POINTS_PHOSPHO,
    )

    print("[Output] Saved phosphorylation rates for picked solution at specific times.")

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
              "scalar_score": float(args.lambda_protein * F[I, 0] + args.lambda_rna * F[I, 1] + args.lambda_phospho * F[I, 2])}
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    print("[Done] Picked solution:")
    print(json.dumps(picked, indent=2))

    plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, df_pho, dfph, output_dir=args.output_dir)
    print("[Done] Goodness of Fit plot saved.")

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

    print("[Done] Time series plots saved.")

    if dfp is not None and dfr is not None and dfph is not None:
        export_results(sys, idx, df_prot, df_rna, df_pho, dfp, dfr, dfph, args.output_dir)

    print("[Done] Exported results saved.")

    # --- VISUALIZATION BLOCK ---

    # 1. 3D Pareto Front
    #
    save_pareto_3d(res, selected_solution=F_best, output_dir=args.output_dir)
    print("[Done] 3D Pareto plot saved.")

    # 2. Parallel Coordinate Plot
    #
    save_parallel_coordinates(res, selected_solution=F_best, output_dir=args.output_dir)
    print("[Done] Parallel Coordinate plot saved.")

    # 3. Convergence Video
    create_convergence_video(res, output_dir=args.output_dir)
    print("[Done] Convergence video saved.")

    # 4. Prior Regularization Scan
    scan_prior_reg(out_dir=args.output_dir)
    print("[Done] Prior regularization scan saved.")

if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except Exception:
        pass
    main()
