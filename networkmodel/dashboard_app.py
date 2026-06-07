#!/usr/bin/env python
"""Render saved networkmodel outputs in a Streamlit dashboard; it does not describe planned backends or execute unrelated optimization workflows on import, and it depends on networkmodel.dashboard_bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from networkmodel.dashboard_bundle import load_dashboard_bundle

import base64
import json


def _list_files(output_dir: Path, patterns: list[str]):
    """Handle internal list files"""
    files = []
    for pat in patterns:
        files.extend(output_dir.glob(pat))
    return sorted(files)


def _show_image(path: Path, caption: str | None = None):
    """Handle internal show image"""
    st.image(str(path), caption=caption, use_container_width=True)


def _show_video(path: Path):
    """Handle internal show video"""
    st.video(str(path))


def _show_pdf(path: Path, height: int = 700):
    """Handle internal show pdf"""
    pdf_bytes = path.read_bytes()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}px"></iframe>'
    st.markdown(html, unsafe_allow_html=True)


def _read_json(path: Path):
    """Handle internal read json"""
    return json.loads(path.read_text())


def _load_outputs(output_dir: Path):
    """Handle internal load outputs"""
    # Prefer bundle (rich objects). Also load standard artifacts if present.
    bundle = load_dashboard_bundle(output_dir)

    # Scalar objective table. Legacy pareto_F.csv is still accepted as a filename
    # but now stores a scalar_objective column in the JAXopt/Diffrax path.
    objective_csv = output_dir / "scalar_objective.csv"
    legacy_objective_csv = output_dir / "pareto_F.csv"
    if objective_csv.exists():
        df_objective = pd.read_csv(objective_csv)
    elif legacy_objective_csv.exists():
        df_objective = pd.read_csv(legacy_objective_csv)
    else:
        F = bundle.get("objective_values", bundle.get("pareto_F"))
        arr = np.asarray(F, dtype=float).reshape(-1, 1)
        df_objective = pd.DataFrame(arr, columns=["scalar_objective"])
    if "scalar_objective" not in df_objective.columns:
        numeric = [c for c in df_objective.columns if pd.api.types.is_numeric_dtype(df_objective[c])]
        df_objective["scalar_objective"] = df_objective[numeric].sum(axis=1) if numeric else np.nan

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

    return bundle, df_objective, df_conv, df_pred_prot, df_pred_rna, df_pred_pho


def _load_inference_outputs(output_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Handle internal load inference outputs"""
    files = {
        "best_fit": output_dir / "optimization" / "best_fit.csv",
        "multistart_summary": output_dir / "optimization" / "multistart_summary.csv",
        "multistart_parameters": output_dir / "optimization" / "multistart_parameters.csv",
        "profile_likelihood": output_dir / "profiles" / "profile_likelihood_summary.csv",
        "posterior_summary": output_dir / "posterior" / "posterior_summary.csv",
        "posterior_samples": output_dir / "posterior" / "posterior_samples.csv",
        "posterior_predictive": output_dir / "posterior" / "posterior_predictive.csv",
    }
    return {name: (pd.read_csv(path) if path.exists() else None) for name, path in files.items()}


def _fig_scalar_objective(df_objective: pd.DataFrame, picked_index: int | None):
    """Handle internal fig scalar objective"""
    df = df_objective.copy()
    df["idx"] = np.arange(len(df))
    fig = px.scatter(df, x="idx", y="scalar_objective", hover_data=["idx"])
    if picked_index is not None and 0 <= picked_index < len(df):
        picked = df.iloc[[picked_index]]
        fig.add_trace(
            go.Scatter(
                x=picked["idx"],
                y=picked["scalar_objective"],
                mode="markers",
                marker=dict(size=12),
                name="picked",
                text=[f"picked={picked_index}"],
            )
        )
    fig.update_layout(height=450, margin=dict(l=0, r=0, b=0, t=30))
    return fig


def _fig_convergence(df_conv: pd.DataFrame):
    """Handle internal fig convergence"""
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
    """Handle internal plot timeseries obs pred"""
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
    """Run the networkmodel entry point"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    st.set_page_config(page_title="PhosKinTime Global Dashboard", layout="wide")

    bundle, df_objective, df_conv, df_pred_prot, df_pred_rna, df_pred_pho = _load_outputs(output_dir)
    inference_outputs = _load_inference_outputs(output_dir)

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
                "solver": args_dict.get("solver", "jaxopt"),
                "data_mode": bundle.get("data_mode", ""),
                "active_layers": bundle.get("active_layers", ""),
                "picked_index": picked_index,
            }
        )

    # Tabs
    tab_overview, tab_timeseries, tab_network, tab_params, tab_inference, tab_browser = st.tabs(
        ["Overview", "Time series", "Network", "Parameters", "Inference", "Browse results"]
    )

    with tab_browser:
        st.subheader("Quick gallery")

        gallery = [
            "scalar_objective.png",
            "mode_metadata.json",
            "goodness_of_fit.png",
            "residuals_analysis.png",
            "kinase_activities_plot.png",
            "sensitivity_mu_star.png",
            "convergence_plot.png",
        ]
        cols = st.columns(3)
        for i, fn in enumerate(gallery):
            p = output_dir / fn
            if p.exists():
                with cols[i % 3]:
                    _show_image(p, caption=fn)

        st.divider()
        st.subheader("Open a result file")

        kind = st.selectbox("Type", ["Images", "CSVs", "JSON", "PDF", "Video", "Subfolders"], index=0)

        if kind == "Images":
            imgs = _list_files(output_dir, ["*.png", "*.jpg", "*.jpeg"])
            choice = st.selectbox("File", [p.name for p in imgs], index=0 if imgs else None)
            if choice:
                _show_image(output_dir / choice, caption=choice)

        elif kind == "CSVs":
            csvs = _list_files(output_dir, ["*.csv"])
            choice = st.selectbox("File", [p.name for p in csvs], index=0 if csvs else None)
            if choice:
                df = pd.read_csv(output_dir / choice)
                st.write(df.head(50))
                st.download_button("Download CSV", data=(output_dir / choice).read_bytes(), file_name=choice)

        elif kind == "JSON":
            jss = _list_files(output_dir, ["*.json"])
            choice = st.selectbox("File", [p.name for p in jss], index=0 if jss else None)
            if choice:
                st.json(_read_json(output_dir / choice))

        elif kind == "PDF":
            pdfs = _list_files(output_dir, ["*.pdf"])
            choice = st.selectbox("File", [p.name for p in pdfs], index=0 if pdfs else None)
            if choice:
                _show_pdf(output_dir / choice)

        elif kind == "Video":
            vids = _list_files(output_dir, ["*.mp4"])
            choice = st.selectbox("File", [p.name for p in vids], index=0 if vids else None)
            if choice:
                _show_video(output_dir / choice)

        else:  # Subfolders
            sub = st.selectbox("Folder", ["sensitivity_perturbations", "steady_state_plots", "timeseries_plots",
                                          "steady_state_summary"])
            folder = output_dir / sub
            if folder.exists():
                imgs = sorted(folder.glob("*.png"))
                choice = st.selectbox("File", [p.name for p in imgs], index=0 if imgs else None)
                if choice:
                    _show_image(folder / choice, caption=f"{sub}/{choice}")
            else:
                st.info(f"{sub} not found.")

    with tab_inference:
        st.subheader("Inference diagnostics")
        for name, df in inference_outputs.items():
            if df is not None:
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.write(df.head(50))
        plot_patterns = [
            "plots/multistart/*.png",
            "plots/profile_likelihood/*.png",
            "plots/posterior/*.png",
        ]
        for path in _list_files(output_dir, plot_patterns):
            _show_image(path, caption=str(path.relative_to(output_dir)))

    with tab_overview:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("Scalar objective")
            st.plotly_chart(_fig_scalar_objective(df_objective, picked_index), use_container_width=True)

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

        # adjust if your column is not literally "protein"
        prot_col = "protein"  # or "GeneID" / "gene" depending on your table
        site_col = "psite"

        if prot_col not in df_pho_obs.columns or site_col not in df_pho_obs.columns:
            st.info(f"Phospho table must contain '{prot_col}' and '{site_col}' columns.")
        else:
            # 1) pick protein (drives the site menu)
            pho_proteins = sorted(df_pho_obs[prot_col].dropna().unique().tolist())
            pho_prot = st.selectbox("Phospho protein", pho_proteins, index=0 if pho_proteins else None)

            if pho_prot:
                df_pho_obs_p = df_pho_obs[df_pho_obs[prot_col] == pho_prot]
                sites = sorted(df_pho_obs_p[site_col].dropna().unique().tolist())

                # 2) pick site among sites for that protein
                site = st.selectbox("P-site", sites, index=0 if sites else None)

                # 3) filter predictions the same way (if present)
                df_pred_pho_p = None
                if df_pred_pho is not None and prot_col in df_pred_pho.columns and site_col in df_pred_pho.columns:
                    df_pred_pho_p = df_pred_pho[df_pred_pho[prot_col] == pho_prot]

                if site:
                    # filter obs/pred to the chosen site
                    fig = _plot_timeseries_obs_pred(
                        df_obs=df_pho_obs_p,  # already filtered by protein
                        df_pred=df_pred_pho_p,  # already filtered by protein (or None)
                        entity_col=site_col,
                        x_col="time",
                        y_obs="fc",
                        y_pred="pred_fc",
                        title=f"Phospho: {pho_prot} — {site}",
                        entity=site,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No phosphosites found for this protein.")

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
