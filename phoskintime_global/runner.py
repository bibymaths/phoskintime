import argparse
import json
import os
import numpy as np
import multiprocessing as mp

import pandas as pd
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import StarmapParallelization
from pymoo.indicators.hv import HV
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize as pymoo_minimize

from phoskintime_global.buildmat import build_W_parallel, build_tf_matrix
from phoskintime_global.cache import prepare_fast_loss_data
from phoskintime_global.config import TIME_POINTS_PROTEIN, TIME_POINTS_RNA, RESULTS_DIR, MAX_ITERATIONS, \
    POPULATION_SIZE, SEED, REGULARIZATION_LAMBDA, REGULARIZATION_RNA
from phoskintime_global.io import load_data
from phoskintime_global.network import Index, KinaseInput, System
from phoskintime_global.optproblem import GlobalODE_MOO
from phoskintime_global.params import init_raw_params, unpack_params
from phoskintime_global.simulate import simulate_and_measure
from phoskintime_global.utils import normalize_fc_to_t0
from phoskintime_global.export import export_pareto_front_to_excel, plot_gof_from_pareto_excel, plot_goodness_of_fit, \
    export_results, save_pareto_3d, save_parallel_coordinates, create_convergence_video, save_gene_timeseries_plots, \
    scan_prior_reg, plot_hypervolume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kinase-net", required=True)
    parser.add_argument("--tf-net", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--output-dir", default=RESULTS_DIR)
    parser.add_argument("--cores", type=int, default=os.cpu_count())

    # Pymoo
    parser.add_argument("--n-gen", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--pop", type=int, default=POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)

    # Loss weights
    parser.add_argument("--lambda-prior", type=float, default=REGULARIZATION_LAMBDA)
    parser.add_argument("--lambda-rna", type=float, default=REGULARIZATION_RNA)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load
    df_kin, df_tf, df_prot, df_rna = load_data(args)
    df_prot = normalize_fc_to_t0(df_prot)
    base = df_rna[df_rna["time"] == 4.0].set_index("protein")["fc"]
    df_rna["fc"] = df_rna.apply(lambda r: r["fc"] / base.get(r["protein"], np.nan), axis=1)
    df_rna = df_rna.dropna(subset=["fc"])

    # 2) Model index
    idx = Index(df_kin)

    # Restrict obs to model proteins
    df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()

    # Weights (early emphasis)
    all_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA]))
    wmap = {t: 1.0 + (all_times.max() - t) / all_times.max() for t in all_times}
    df_prot["w"] = df_prot["time"].map(wmap).fillna(1.0)
    df_rna["w"] = df_rna["time"].map(wmap).fillna(1.0)

    # 3) Build W + TF
    W_global = build_W_parallel(df_kin, idx, n_cores=args.cores)
    tf_mat = build_tf_matrix(df_tf, idx)
    kin_in = KinaseInput(idx.kinases, df_prot)

    tf_deg = np.asarray(tf_mat.sum(axis=1)).ravel().astype(np.float64)  # row sums = in-degree per target
    tf_deg[tf_deg == 0.0] = 1.0

    # 4) Defaults/system
    defaults = {
        "c_k": np.ones(len(idx.kinases)),
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1
    }
    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # 5) Precompute loss data on solver time grid
    solver_times = np.unique(np.concatenate([TIME_POINTS_PROTEIN, TIME_POINTS_RNA]))
    rna_base_time = 4.0
    rna_base_idx = int(np.where(solver_times == rna_base_time)[0][0])

    loss_data = prepare_fast_loss_data(idx, df_prot, df_rna, solver_times)
    loss_data["rna_base_idx"] = np.int32(rna_base_idx)

    # 6) Decision vector bounds (raw space)
    theta0, slices, xl, xu = init_raw_params(defaults)
    lambdas = {"rna": args.lambda_rna, "prior": args.lambda_prior}

    # 7) Pymoo parallel runner (optional)
    runner = None
    pool = None
    if args.cores > 1:
        pool = mp.Pool(args.cores)
        runner = StarmapParallelization(pool.starmap)
        print(f"[Pymoo] Parallel evaluation enabled with {args.cores} workers.")
    else:
        print("[Pymoo] Parallel evaluation disabled (or unavailable).")

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

    # 9) UNSGA3 needs reference directions for n_obj=3
    ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=20)
    algorithm = UNSGA3(pop_size=args.pop, ref_dirs=ref_dirs, eliminate_duplicates=True, sampling=LHS())
    evaluator = Evaluator(evaluate_values_of=["F"])

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=args.n_gen,
        n_max_evals=100000
    )

    print(f"[Fit] UNSGA3: pop={args.pop}, n_gen={args.n_gen}, n_var={problem.n_var}, n_obj={problem.n_obj}")

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        evaluator=evaluator,
        seed=args.seed,
        save_history=True,
        verbose=True
    )

    if pool is not None:
        pool.close()
        pool.join()

    F_all = np.vstack([algo.pop.get("F") for algo in res.history if algo.pop is not None])
    ref_point = np.max(F_all, axis=0) * 1.05

    hv_indicator = HV(ref_point=ref_point)

    hv_history = []
    gen_history = []

    for algo in res.history:
        if algo.pop is None:
            continue
        F = algo.pop.get("F")
        hv = hv_indicator.do(F)
        hv_history.append(hv)
        gen_history.append(algo.n_gen)

    hv_history = np.array(hv_history)
    gen_history = np.array(gen_history)

    plot_hypervolume(
        gen_history=gen_history,
        hv_history=hv_history,
        out_path=os.path.join(args.output_dir, "hypervolume.png")
    )

    print("[Output] Saved hypervolume plot.")

    # 10) Save Pareto set
    X = res.X
    F = res.F
    np.save(os.path.join(args.output_dir, "pareto_X.npy"), X)
    np.save(os.path.join(args.output_dir, "pareto_F.npy"), F)

    # Also write a CSV summary
    df_pareto = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "reg_loss"])
    df_pareto.to_csv(os.path.join(args.output_dir, "pareto_F.csv"), index=False)
    print(f"[Output] Saved Pareto front: {len(df_pareto)} solutions")

    excel_path = os.path.join(args.output_dir, "pareto_front.xlsx")

    export_pareto_front_to_excel(
        res=res,
        sys=sys,
        idx=idx,
        slices=slices,
        output_path=excel_path,
        weights=(1.0, args.lambda_rna, args.lambda_prior),
        top_k_trajectories=None,
    )

    print(f"[Output] Saved Pareto front Excel: {excel_path}")

    plot_gof_from_pareto_excel(
        excel_path=excel_path,
        output_dir=os.path.join(args.output_dir, "gof_all"),
        plot_goodness_of_fit_func=plot_goodness_of_fit,
        df_prot_obs_all=df_prot,
        df_rna_obs_all=df_rna,
        top_k=None,
        score_col="scalar_score",
    )

    print(f"[Output] Saved Goodness of Fit plots for all Pareto solutions.")

    # 11) Pick one solution
    F = res.F
    Fn = (F - F.min(axis=0)) / (np.ptp(F, axis=0) + 1e-12)
    w = np.array([1.0, args.lambda_rna, args.lambda_prior])
    I = np.argmin((Fn * w).sum(axis=1))
    theta_best = X[I].astype(float)
    F_best = F[I]
    params = unpack_params(theta_best, slices)
    sys.update(**params)

    # 12) Export picked solution
    dfp, dfr = simulate_and_measure(sys, idx, TIME_POINTS_PROTEIN, TIME_POINTS_RNA)
    # Save raw preds
    if dfp is not None: dfp.to_csv(os.path.join(args.output_dir, "pred_prot_picked.csv"), index=False)
    if dfr is not None: dfr.to_csv(os.path.join(args.output_dir, "pred_rna_picked.csv"), index=False)

    p_out = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in params.items()}
    with open(os.path.join(args.output_dir, "fitted_params_picked.json"), "w") as f:
        json.dump(p_out, f, indent=2)

    # Write picked objective values
    picked = {"prot_mse": float(F[I, 0]), "rna_mse": float(F[I, 1]), "reg_loss": float(F[I, 2]),
              "scalar_score": float(F[I, 0] + args.lambda_rna * F[I, 1] + F[I, 2])}
    with open(os.path.join(args.output_dir, "picked_objectives.json"), "w") as f:
        json.dump(picked, f, indent=2)

    print("[Done] Picked solution:")
    print(json.dumps(picked, indent=2))

    plot_goodness_of_fit(df_prot, dfp, df_rna, dfr, output_dir=args.output_dir)
    print("[Done] Goodness of Fit plot saved.")

    ts_dir = os.path.join(args.output_dir, "timeseries_plots")
    for g in idx.proteins:
        save_gene_timeseries_plots(
            gene=g,
            df_prot_obs=df_prot,
            df_prot_pred=dfp,
            df_rna_obs=df_rna,
            df_rna_pred=dfr,
            output_dir=ts_dir,
            prot_times=TIME_POINTS_PROTEIN,
            rna_times=TIME_POINTS_RNA,
            filename_prefix="fit"
        )

    print("[Done] Time series plots saved.")

    # Use the NEW export function (pass dfp, dfr directly)
    if dfp is not None and dfr is not None:
        export_results(sys, idx, df_prot, df_rna, dfp, dfr, args.output_dir)

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
