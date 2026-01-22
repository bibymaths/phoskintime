from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from global_model.dashboard_bundle import load_dashboard_bundle


def _load_outputs(output_dir: Path):
    # Prefer bundle (rich objects). Also load standard artifacts if present.
    bundle = load_dashboard_bundle(output_dir)

    # Pareto
    pareto_F_csv = output_dir / "pareto_F.csv"
    if pareto_F_csv.exists():
        df_pareto = pd.read_csv(pareto_F_csv)
    else:
        F = bundle["res"].F
        df_pareto = pd.DataFrame(F, columns=["prot_mse", "rna_mse", "phospho_mse"])

    # Convergence
    conv_csv = output_dir / "convergence_history.csv"
    df_conv = pd.read_csv(conv_csv) if conv_csv.exists() else None

    # Predictions (picked)
    pred_prot = output_dir / "pred_prot_picked.csv"
    pred_rna = output_dir / "pred_rna_picked.csv"
    pred_pho = output_dir / "pred_phospho_picked.csv"

    df_pred_prot = pd.read_csv(pred_prot) if pred_prot.exists() else None
    df_pred_rna = pd.read_csv(pred_rna) if pred_rna.exists() else None
    df_pred_pho = pd.read_csv(pred_pho) if pred_pho.exists() else None

    return bundle, df_pareto, df_conv, df_pred_prot, df_pred_rna, df_pred_pho


def _fig_pareto_3d(df_pareto: pd.DataFrame, picked_index: int | None):
    df = df_pareto.copy()
    df["idx"] = np.arange(len(df))

    fig = px.scatter_3d(
        df,
        x="prot_mse",
        y="rna_mse",
        z="phospho_mse",
        hover_data=["idx"],
    )

    if picked_index is not None and 0 <= picked_index < len(df):
        picked = df.iloc[[picked_index]]
        fig.add_trace(
            go.Scatter3d(
                x=picked["prot_mse"],
                y=picked["rna_mse"],
                z=picked["phospho_mse"],
                mode="markers",
                marker=dict(size=8),
                name="picked",
                text=[f"picked={picked_index}"],
            )
        )
    fig.update_layout(height=650, margin=dict(l=0, r=0, b=0, t=30))
    return fig


def _fig_convergence(df_conv: pd.DataFrame):
    # Expect columns from your process_convergence_history export; adapt if needed.
    # Try common patterns.
    cols = df_conv.columns.tolist()

    # Heuristic: show first numeric columns by default.
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df_conv[c])]
    if not numeric_cols:
        return None

    y = st.selectbox("Convergence metric", numeric_cols, index=0)
    fig = px.line(df_conv, x=df_conv.index, y=y)
    fig.update_layout(height=350, margin=dict(l=0, r=0, b=0, t=30))
    return fig


def _plot_timeseries_obs_pred(
        df_obs: pd.DataFrame,
        df_pred: pd.DataFrame | None,
        entity_col: str,
        x_col: str,
        y_obs: str,
        y_pred: str,
        title: str,
        entity: str,
):
    obs = df_obs[df_obs[entity_col] == entity].copy()
    obs = obs.sort_values(x_col)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs[x_col], y=obs[y_obs], mode="markers+lines", name="obs"))

    if df_pred is not None and entity_col in df_pred.columns:
        pred = df_pred[df_pred[entity_col] == entity].copy()
        if len(pred) > 0:
            pred = pred.sort_values(x_col)
            fig.add_trace(go.Scatter(x=pred[x_col], y=pred[y_pred], mode="lines", name="pred"))

    fig.update_layout(title=title, height=350, margin=dict(l=0, r=0, b=0, t=40))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    st.set_page_config(page_title="PhosKinTime Global Dashboard", layout="wide")

    bundle, df_pareto, df_conv, df_pred_prot, df_pred_rna, df_pred_pho = _load_outputs(output_dir)

    picked_index = bundle.get("picked_index", None)
    frechet_scores = bundle.get("frechet_scores", None)

    # Header
    meta_left, meta_right = st.columns([2, 1])
    with meta_left:
        st.title("PhosKinTime Global Dashboard")
        st.caption(str(output_dir.resolve()))

    with meta_right:
        args_dict = bundle.get("args", {})
        st.markdown("**Run summary**")
        st.write(
            {
                "solver": args_dict.get("solver", ""),
                "pop": args_dict.get("pop", ""),
                "n_gen": args_dict.get("n_gen", ""),
                "cores": args_dict.get("cores", ""),
                "picked_index": picked_index,
            }
        )

    # Tabs
    tab_overview, tab_timeseries, tab_network, tab_params = st.tabs(
        ["Overview", "Time series", "Network", "Parameters"]
    )

    with tab_overview:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Pareto front")
            st.plotly_chart(_fig_pareto_3d(df_pareto, picked_index), use_container_width=True)

        with c2:
            st.subheader("Selection")
            if frechet_scores is not None:
                fs = np.asarray(frechet_scores, dtype=float)
                st.write(
                    {
                        "n_solutions": int(len(fs)),
                        "picked_frechet": float(fs[picked_index]) if picked_index is not None else None,
                        "min_frechet": float(np.min(fs)),
                        "median_frechet": float(np.median(fs)),
                    }
                )
                df_fs = pd.DataFrame({"idx": np.arange(len(fs)), "frechet": fs})
                st.plotly_chart(px.line(df_fs, x="idx", y="frechet"), use_container_width=True)
            else:
                st.info("No frechet_scores found in bundle.")

            st.subheader("Convergence")
            if df_conv is not None and len(df_conv) > 0:
                fig = _fig_convergence(df_conv)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No convergence_history.csv found.")

    with tab_timeseries:
        df_prot_obs = bundle["df_prot_obs"]
        df_rna_obs = bundle["df_rna_obs"]
        df_pho_obs = bundle["df_pho_obs"]

        # Protein
        st.subheader("Protein trajectories (obs vs pred)")
        proteins = sorted(df_prot_obs["protein"].unique().tolist())
        prot = st.selectbox("Protein", proteins, index=0 if proteins else None)
        if prot:
            st.plotly_chart(
                _plot_timeseries_obs_pred(
                    df_obs=df_prot_obs,
                    df_pred=df_pred_prot,
                    entity_col="protein",
                    x_col="time",
                    y_obs="fc",
                    y_pred="pred_fc",
                    title=f"Protein: {prot}",
                    entity=prot,
                ),
                use_container_width=True,
            )

        # RNA
        st.subheader("RNA trajectories (obs vs pred)")
        genes = sorted(df_rna_obs["protein"].unique().tolist())
        g = st.selectbox("Gene", genes, index=0 if genes else None)
        if g:
            st.plotly_chart(
                _plot_timeseries_obs_pred(
                    df_obs=df_rna_obs,
                    df_pred=df_pred_rna,
                    entity_col="protein",
                    x_col="time",
                    y_obs="fc",
                    y_pred="pred_fc",
                    title=f"RNA: {g}",
                    entity=g,
                ),
                use_container_width=True,
            )

        # Phospho
        st.subheader("Phospho trajectories (obs vs pred)")
        sites = sorted(df_pho_obs["psite"].unique().tolist()) if "psite" in df_pho_obs.columns else []
        site = st.selectbox("P-site", sites, index=0 if sites else None)
        if site:
            st.plotly_chart(
                _plot_timeseries_obs_pred(
                    df_obs=df_pho_obs,
                    df_pred=df_pred_pho,
                    entity_col="psite",
                    x_col="time",
                    y_obs="fc",
                    y_pred="pred_fc",
                    title=f"Phospho: {site}",
                    entity=site,
                ),
                use_container_width=True,
            )

    with tab_network:
        st.subheader("Network exports")

        w_csv = output_dir / "network_W_global.csv"
        tf_csv = output_dir / "network_tf_mat.csv"

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Kinase → Site (W)**")
            if w_csv.exists():
                dfw = pd.read_csv(w_csv)
                st.write(dfw.head(20))
                fig = px.histogram(dfw, x="Weight")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("network_W_global.csv not found.")

        with c2:
            st.markdown("**TF → Target (TF matrix)**")
            if tf_csv.exists():
                dft = pd.read_csv(tf_csv)
                st.write(dft.head(20))
                fig = px.histogram(dft, x="Weight")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("network_tf_mat.csv not found.")

    with tab_params:
        st.subheader("Fitted parameter summary")
        fitted = output_dir / "fitted_params_picked.json"
        picked = output_dir / "picked_objectives.json"

        c1, c2 = st.columns(2)
        with c1:
            if picked.exists():
                st.markdown("**Picked objectives**")
                st.json(picked.read_text())
            else:
                st.info("picked_objectives.json not found.")

        with c2:
            if fitted.exists():
                st.markdown("**Fitted parameters (picked)**")
                st.json(fitted.read_text())
            else:
                st.info("fitted_params_picked.json not found.")


if __name__ == "__main__":
    main()
