import plotly.express as px
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path

from global_model import config
from global_model.network import Index, KinaseInput, System
from global_model.simulate import simulate_and_measure
from global_model.analysis import simulate_until_steady
from global_model.params import init_raw_params, unpack_params
from global_model.io import load_data
from global_model.buildmat import build_W_parallel, build_tf_matrix
from global_model.utils import normalize_fc_to_t0

st.set_page_config(page_title="PhoskinTime Global Knockout", layout="wide")


def _standardize_tf_columns(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Map TF network columns to expected ('tf','target') names where possible."""
    if df_tf is None or df_tf.empty:
        return df_tf
    cols = df_tf.columns.tolist()
    if "tf" in cols and "target" in cols:
        return df_tf

    # common alternates used in exports
    src_col = "Source" if "Source" in cols else ("source" if "source" in cols else None)
    tgt_col = "Target" if "Target" in cols else ("target" if "target" in cols else None)

    if src_col is None or tgt_col is None:
        # last-resort: assume first two columns are (src,tgt)
        src_col = cols[0]
        tgt_col = cols[1] if len(cols) > 1 else cols[0]

    df = df_tf.copy()
    df = df.rename(columns={src_col: "tf", tgt_col: "target"})
    return df


def _build_df_tf_model(
    df_tf_raw: pd.DataFrame,
    df_kin: pd.DataFrame,
    df_prot: pd.DataFrame,
    df_rna: pd.DataFrame,
    df_pho: pd.DataFrame,
    kin_beta_map: dict | None,
    tf_beta_map: dict | None,
) -> tuple[pd.DataFrame | None, dict]:
    """
    Port of runner.py 'Sophisticated TF handling' (proxy orphan TFs, filter to representable universe).
    Returns (df_tf_model, TF_PROXY_MAP).
    """
    if df_tf_raw is None or df_tf_raw.empty:
        return df_tf_raw, {}

    df_tf = _standardize_tf_columns(df_tf_raw)

    required = {"tf", "target"}
    missing = required - set(df_tf.columns)
    if missing:
        raise ValueError(f"TF net missing columns: {missing}. Found columns: {list(df_tf.columns)}")

    proteins_with_sites = set(df_kin["protein"].astype(str).str.strip().unique())
    kinase_set = set(df_kin["kinase"].astype(str).str.strip().unique())

    target_universe = (
        set(df_kin["protein"].astype(str).str.strip().unique())
        | set(df_kin["kinase"].astype(str).str.strip().unique())
        | set(df_prot["protein"].astype(str).str.strip().unique())
        | set(df_rna["protein"].astype(str).str.strip().unique())
        | set(df_pho["protein"].astype(str).str.strip().unique())
    )

    df_tf_model = df_tf[df_tf["target"].astype(str).str.strip().isin(target_universe)].copy()
    df_tf_model["tf"] = df_tf_model["tf"].astype(str).str.strip()
    df_tf_model["target"] = df_tf_model["target"].astype(str).str.strip()

    orphan_tfs = sorted(set(df_tf_model["tf"].unique()) - proteins_with_sites)

    TF_PROXY_MAP: dict[str, str] = {}

    def _proxy_score(orphan: str, candidate: str) -> float:
        score = 0.0
        if tf_beta_map and orphan in tf_beta_map:
            score += float(tf_beta_map[orphan])
        if kin_beta_map and candidate in kin_beta_map:
            score += float(kin_beta_map[candidate])
        return score

    for orphan in orphan_tfs:
        targets = df_tf_model.loc[df_tf_model["tf"] == orphan, "target"].astype(str)
        cand1 = [t for t in targets if t in kinase_set]               # Priority 1
        cand2 = [t for t in targets if t in proteins_with_sites]      # Priority 2
        candidates = cand1 if cand1 else cand2
        if candidates:
            best = sorted(candidates, key=lambda c: (-_proxy_score(orphan, c), c))[0]
            TF_PROXY_MAP[orphan] = best

    if TF_PROXY_MAP:
        df_tf_model["tf_original"] = df_tf_model["tf"]
        df_tf_model["tf"] = df_tf_model["tf"].replace(TF_PROXY_MAP)

    # enforce representability: TF source must be a signaling protein-with-sites
    df_tf_model = df_tf_model[df_tf_model["tf"].isin(proteins_with_sites)].copy()

    keep_cols = [c for c in df_tf_model.columns if c in ("tf", "target", "alpha", "tf_original")]
    df_tf_model = df_tf_model[keep_cols].drop_duplicates()

    return df_tf_model, TF_PROXY_MAP


def _mechanistic_phospho_filter(df_kin: pd.DataFrame, df_pho: pd.DataFrame) -> pd.DataFrame:
    """Exact runner.py: keep only (protein, psite) in df_kin."""
    if df_pho is None or df_pho.empty:
        return df_pho

    df_kin2 = df_kin.copy()
    df_pho2 = df_pho.copy()

    for _df in (df_kin2, df_pho2):
        _df["protein"] = _df["protein"].astype(str).str.strip()
    df_kin2["psite"] = df_kin2["psite"].astype(str).str.strip()
    df_pho2["psite"] = df_pho2["psite"].astype(str).str.strip()

    kin_site_pairs = set(zip(df_kin2["protein"].values, df_kin2["psite"].values))
    pairs = list(zip(df_pho2["protein"].values, df_pho2["psite"].values))
    keep = np.fromiter(((p, s) in kin_site_pairs for (p, s) in pairs), dtype=bool, count=len(pairs))
    return df_pho2.loc[keep].copy()


@st.cache_resource
def load_system():
    # IMPORTANT: ensure model selection matches run *before* building System
    config.MODEL = 0

    results_dir = Path("./results_model_global_distributive")

    class Args:
        kinase_net, tf_net = config.KINASE_NET_FILE, config.TF_NET_FILE
        ms, rna, phospho = config.MS_DATA_FILE, config.RNA_DATA_FILE, config.PHOSPHO_DATA_FILE
        kinopt, tfopt = None, None
        normalize_fc_steady = False

    df_kin, df_tf_raw, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(Args())

    # optional (runner has a flag; keep identical default behavior = False)
    if getattr(Args, "normalize_fc_steady", False):
        df_prot = normalize_fc_to_t0(df_prot)
        df_pho = normalize_fc_to_t0(df_pho)

    # runner: strict mechanistic phospho filter
    df_pho = _mechanistic_phospho_filter(df_kin, df_pho)

    # runner: TF proxying + representability filtering
    df_tf_model, TF_PROXY_MAP = _build_df_tf_model(
        df_tf_raw=df_tf_raw,
        df_kin=df_kin,
        df_prot=df_prot,
        df_rna=df_rna,
        df_pho=df_pho,
        kin_beta_map=kin_beta_map,
        tf_beta_map=tf_beta_map,
    )

    # runner: build Index with beta maps
    idx = Index(df_kin, tf_interactions=df_tf_model, kin_beta_map=kin_beta_map, tf_beta_map=tf_beta_map)

    # runner: restrict observations to idx.proteins
    if df_prot is not None and not df_prot.empty:
        df_prot = df_prot[df_prot["protein"].isin(idx.proteins)].copy()
    if df_rna is not None and not df_rna.empty:
        df_rna = df_rna[df_rna["protein"].isin(idx.proteins)].copy()
    if df_pho is not None and not df_pho.empty:
        df_pho = df_pho[df_pho["protein"].isin(idx.proteins)].copy()

    # runner: build kinase input from observed kinase trajectories only
    df_prot_kin = df_prot[df_prot["protein"].isin(idx.kinases)].copy() if df_prot is not None else df_prot
    kin_in = KinaseInput(idx.kinases, df_prot_kin)

    # runner: build matrices
    W_global = build_W_parallel(df_kin, idx, n_cores=1)
    tf_mat = build_tf_matrix(df_tf_model, idx, tf_beta_map=tf_beta_map, kin_beta_map=kin_beta_map)

    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    # runner: initialize c_k from beta priors
    c_k_init = np.array([max(0.01, float(kin_beta_map.get(k, 1.0))) for k in idx.kinases])

    defaults = {
        "c_k": c_k_init,
        "A_i": np.ones(idx.N),
        "B_i": np.full(idx.N, 0.2),
        "C_i": np.full(idx.N, 0.5),
        "D_i": np.full(idx.N, 0.05),
        "Dp_i": np.full(idx.total_sites, 0.05),
        "E_i": np.ones(idx.N),
        "tf_scale": 0.1,
    }

    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)

    # slices/schema must match the run that produced pareto_X.npy
    _, slices, _, _ = init_raw_params(defaults)

    X = np.load(results_dir / "pareto_X.npy")
    theta = X[0].astype(float)

    expected_dim = max(s.stop for s in slices.values())
    if theta.size != expected_dim:
        raise ValueError(
            f"Incompatible pareto_X dimension: got {theta.size}, expected {expected_dim}. "
            "Dashboard reconstruction does not match optimization run. "
            "Fix compare_mechanisms reconstruction or re-run optimization."
        )

    best_params = unpack_params(theta, slices)

    s_rates_path = results_dir / "S_rates_picked.csv"
    s_rates = pd.read_csv(s_rates_path) if s_rates_path.exists() else pd.DataFrame()

    # return df_tf_model for the graph layer (it may include tf_original column)
    return sys, idx, best_params, df_tf_model, s_rates


def run_sim(sys, idx, mod_params):
    sys.update(**mod_params)
    return simulate_and_measure(sys, idx, config.TIME_POINTS_PROTEIN, config.TIME_POINTS_RNA, config.TIME_POINTS_PHOSPHO)


# --- UI Setup ---
st.title("🧪 Global Signaling & Transcriptional Knockout Explorer")
sys, idx, best_params, df_tf_model, s_rates = load_system()

st.sidebar.header("🕹️ Control Panel")
ko_type = st.sidebar.selectbox("1. Choose Perturbation Type", ["None", "Protein (Synthesis)", "Kinase (Activity)"])

ko_params = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in best_params.items()}

Kmat_backup = None  # default

def restore_Kinase():
    return None

if ko_type == "Protein (Synthesis)":
    target = st.sidebar.selectbox("Select Target Protein", idx.proteins)
    p_idx = idx.p2i[target]
    scale = st.sidebar.slider("Protein Synthesis Scale (0 = KO, 1 = WT)", 0.0, 1.0, 0.0, 0.05)
    ko_params["A_i"][p_idx] *= scale  # scale synthesis rate
    sys.update(**ko_params)

elif ko_type == "Kinase (Activity)":
    target = st.sidebar.selectbox("Select Kinase to Inhibit", idx.kinases)
    k_idx = idx.k2i[target]
    scale = st.sidebar.slider("Kinase Activity Scale (0 = KO, 1 = WT)", 0.0, 1.0, 0.0, 0.05)

    # Backup Kmat and c_k for later restoration
    Kmat_backup = sys.kin.Kmat.copy()
    c_k_backup = sys.c_k.copy()

    # Apply multiplicative inhibition to both dynamic and static components
    sys.kin.Kmat[k_idx, :] *= scale
    ko_params["c_k"][k_idx] *= scale

    # Restore both after KO simulation
    def restore_Kinase():
        sys.kin.Kmat = Kmat_backup
        sys.c_k = c_k_backup

    sys.update(**ko_params)
else:
    target = None

# --- Simulation Logic ---
wt_dfp, wt_dfr, wt_pho = run_sim(sys, idx, best_params)
ko_dfp, ko_dfr, ko_pho = run_sim(sys, idx, ko_params)

# --- Versatile Visualization: The Impact Scatter ---
st.header("🎯 System-Wide Impact Analysis")

final_t = wt_dfr["time"].max()
comparison = pd.DataFrame(
    {
        "protein": wt_dfr[wt_dfr["time"] == final_t]["protein"].values,
        "WT_final": wt_dfr[wt_dfr["time"] == final_t]["pred_fc"].values,
        "KO_final": ko_dfr[ko_dfr["time"] == final_t]["pred_fc"].values,
    }
)

comparison["log2_fc"] = np.log2((comparison["KO_final"] + 1e-6) / (comparison["WT_final"] + 1e-6))

sensitivity = []
for p in comparison["protein"]:
    wt_vals = wt_dfr[wt_dfr["protein"] == p]["pred_fc"].values
    ko_vals = ko_dfr[ko_dfr["protein"] == p]["pred_fc"].values
    sensitivity.append(np.sum(np.abs(wt_vals - ko_vals)))
comparison["sensitivity_score"] = sensitivity

fig_scatter = px.scatter(
    comparison,
    x="log2_fc",
    y="sensitivity_score",
    text="protein",
    color="log2_fc",
    color_continuous_scale="RdBu_r",
    labels={"log2_fc": "Log2 Impact (KO/WT)", "sensitivity_score": "Cumulative Perturbation (Area)"},
    title="Sensitivity vs. Magnitude: Which genes are most 'fragile'?",
)
fig_scatter.update_traces(textposition="top center")
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Delta Trajectory ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")
st.markdown("This shows the **absolute loss of signal** over time ($WT - KO$).")

delta_df = wt_dfr.copy()
delta_df["delta"] = wt_dfr["pred_fc"] - ko_dfr["pred_fc"]

max_deltas = delta_df.groupby("protein")["delta"].transform(lambda x: x.abs().max())
delta_df["is_significant"] = max_deltas > 0.1
delta_df["Impact Group"] = delta_df["is_significant"].map({True: "Significant Effect", False: "Minimal/No Effect"})

fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    labels={"time": "Time (min)", "delta": "Δ Signal (WT - KO)", "protein": "Protein"},
    title="Propagation of Signal Loss Over Time",
)

fig_delta.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white",
)

for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df["protein"] == p_name]["delta"].abs().max()
    if max_d < 0.1:
        trace.line.width = 1
        trace.opacity = 0.2
        trace.showlegend = False
    else:
        trace.line.width = 3
        trace.opacity = 0.9

st.plotly_chart(fig_delta, use_container_width=True)

# --- Delta Logic: Cross-Modality Toggle ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")

modality_choice = st.radio("Select Modality for Delta Analysis:", ["mRNA", "Protein"], horizontal=True)
active_wt = wt_dfr if modality_choice == "mRNA" else wt_dfp
active_ko = ko_dfr if modality_choice == "mRNA" else ko_dfp

delta_df = active_wt.copy()
delta_df["delta"] = active_wt["pred_fc"] - active_ko["pred_fc"]

max_deltas = delta_df.groupby("protein")["delta"].transform(lambda x: x.abs().max())
delta_df["is_significant"] = max_deltas > (0.05 if modality_choice == "Protein" else 0.1)
delta_df["Impact Group"] = delta_df["is_significant"].map({True: "Significant", False: "Minimal"})

fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Safe,
    labels={"time": "Time (min)", "delta": f"Δ {modality_choice} Signal (WT - KO)", "protein": "Protein"},
    title=f"Propagation of {modality_choice} Signal Loss",
)

for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df["protein"] == p_name]["delta"].abs().max()
    if max_d < (0.05 if modality_choice == "Protein" else 0.1):
        trace.line.width = 1
        trace.opacity = 0.2
        trace.showlegend = False
    else:
        trace.line.width = 3
        trace.opacity = 0.9

fig_delta.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_delta, use_container_width=True)

# --- Summary Statistics Table ---
st.header("📋 Knockout Impact Summary")
top_impact = comparison.copy()
top_impact["absolute_impact"] = top_impact["log2_fc"].abs()
top_impact = top_impact.sort_values("absolute_impact", ascending=False).head(15)

st.dataframe(
    top_impact[["protein", "log2_fc", "sensitivity_score"]],
    column_config={
        "log2_fc": st.column_config.NumberColumn("Log2 Impact", format="%.2f"),
        "sensitivity_score": st.column_config.NumberColumn("Total Area Delta", format="%.2f"),
    },
    use_container_width=True,
)

# --- Steady-State Kinase Drive Analysis ---
st.header("⚙️ Steady-State Kinase Drive")

# Recompute steady-state kinetics
_, Y = simulate_until_steady(sys, t_max=960)
y_last = Y[-1].astype(float, copy=False)
t_last = 960.0

Kt = sys.kin.eval(t_last) * sys.c_k
W = sys.W_global.tocoo()
edge_contrib = W.data * Kt[W.col]

# Sum phospho drive per kinase
nK = len(idx.kinases)
kin_sum = np.zeros(nK, dtype=float)
np.add.at(kin_sum, W.col, edge_contrib)

df_kin = pd.DataFrame({"kinase": idx.kinases, "Kt": Kt, "phospho_drive": kin_sum})
df_kin = df_kin.sort_values("phospho_drive", ascending=False)

# Plot top 25
top_df = df_kin.head(10)
fig_kinase = px.bar(
    top_df,
    x="kinase",
    y="phospho_drive",
    color="Kt",
    title="Top 10 Kinases by Phosphorylation Drive at Steady State",
    labels={"phospho_drive": "Σ W_ik * Kt_k", "Kt": "Kinase Activity"},
    color_continuous_scale="Viridis",
)
fig_kinase.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_kinase, use_container_width=True)

# --- Dominant Kinase Per Site Summary ---
n_sites_total = sys.W_global.shape[0]
best_val = np.full(n_sites_total, -np.inf, dtype=float)
best_k = np.full(n_sites_total, -1, dtype=np.int32)
second_val = np.full(n_sites_total, -np.inf, dtype=float)

for r, c, v in zip(W.row, W.col, edge_contrib):
    if v > best_val[r]:
        second_val[r] = best_val[r]
        best_val[r] = v
        best_k[r] = c
    elif v > second_val[r]:
        second_val[r] = v

# Build kinase labels
dominant_kins = np.asarray(idx.kinases, dtype=object)[best_k[best_k >= 0]]
df_dom = pd.DataFrame({"kinase": dominant_kins})
dom_counts = df_dom["kinase"].value_counts().reset_index()
dom_counts.columns = ["kinase", "n_sites_dominated"]

# Plot top10
fig_dom = px.bar(
    dom_counts.head(10),
    x="kinase",
    y="n_sites_dominated",
    title="Top 10 Kinases by Dominated Phospho-Sites (Steady State)",
    labels={"n_sites_dominated": "# Dominated Sites"},
    color="n_sites_dominated",
    color_continuous_scale="Blues",
)
fig_dom.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_dom, use_container_width=True)

# --- Download Section ---
st.subheader("📥 Export Plot Data")
pivot_delta = delta_df.pivot(index="time", columns="protein", values="delta")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        label=f"Download Δ-{modality_choice} CSV",
        data=pivot_delta.to_csv(),
        file_name=f"delta_{modality_choice.lower()}_knockout.csv",
        mime="text/csv",
    )
with c2:
    st.info(f"The CSV contains the absolute loss (WT-KO) for all {len(idx.proteins)} entities.")

st.divider()
st.header("🔬 Deep Dive: Single Protein Inspector")
selected_p = st.selectbox("Select a protein to inspect in detail:", idx.proteins)

wt_p_data = wt_dfp[wt_dfp["protein"] == selected_p]
ko_p_data = ko_dfp[ko_dfp["protein"] == selected_p]
wt_r_data = wt_dfr[wt_dfr["protein"] == selected_p]
ko_r_data = ko_dfr[ko_dfr["protein"] == selected_p]
wt_pho_data = wt_pho[wt_pho["protein"] == selected_p]
ko_pho_data = ko_pho[ko_pho["protein"] == selected_p]

col1, col2 = st.columns(2)

with col1:
    fig_insp_r = go.Figure()
    fig_insp_r.add_trace(
        go.Scatter(x=wt_r_data["time"], y=wt_r_data["pred_fc"], name="Wild Type", line=dict(dash="dash", color="black", width=2))
    )
    fig_insp_r.add_trace(
        go.Scatter(x=ko_r_data["time"], y=ko_r_data["pred_fc"], name="Knockout", line=dict(color="red", width=3))
    )
    fig_insp_r.update_layout(title=f"{selected_p} mRNA Response", xaxis_title="Time (min)", yaxis_title="Fold Change", template="plotly_white")
    st.plotly_chart(fig_insp_r, use_container_width=True)

with col2:
    fig_insp_p = go.Figure()
    fig_insp_p.add_trace(
        go.Scatter(x=wt_p_data["time"], y=wt_p_data["pred_fc"], name="Wild Type", line=dict(dash="dash", color="black", width=2))
    )
    fig_insp_p.add_trace(
        go.Scatter(x=ko_p_data["time"], y=ko_p_data["pred_fc"], name="Knockout", line=dict(color="blue", width=3))
    )
    fig_insp_p.update_layout(title=f"{selected_p} Protein Abundance", xaxis_title="Time (min)", yaxis_title="Fold Change", template="plotly_white")
    st.plotly_chart(fig_insp_p, use_container_width=True)


st.divider()
col1, col2 = st.columns(2)


with col1:
    if wt_pho_data.empty:
        st.info("No phospho-site data available for this protein.")
    else:
        fig_sites = go.Figure()
        for site in wt_pho_data["psite"].unique():
            color = px.colors.qualitative.Plotly[hash(site) % len(px.colors.qualitative.Plotly)]
            site_wt = wt_pho_data[wt_pho_data["psite"] == site]
            site_ko = ko_pho_data[ko_pho_data["psite"] == site]
            fig_sites.add_trace(
                go.Scatter(x=site_wt["time"], y=site_wt["pred_fc"], name=f"WT Site: {site}", line=dict(dash="dash", color=color))
            )
            fig_sites.add_trace(
                go.Scatter(x=site_ko["time"], y=site_ko["pred_fc"], name=f"KO Site: {site}", line=dict(color=color))
            )
        fig_sites.update_layout(title=f"{selected_p} Phospho-site Dynamics", xaxis_title="Time (min)", yaxis_title="Fold Change", template="plotly_white")
    st.plotly_chart(fig_sites, use_container_width=True)

with col2:
    if s_rates is None or s_rates.empty:
        st.info("S_rates_picked.csv not found or empty.")
    else:
        p_s = s_rates[s_rates["protein"] == selected_p]
        fig_s = go.Figure()
        for site in p_s["psite"].unique():
            color = px.colors.qualitative.Plotly[hash(site) % len(px.colors.qualitative.Plotly)]
            site_data = p_s[p_s["psite"] == site]
            fig_s.add_trace(go.Scatter(x=site_data["time"], y=site_data["S"], name=f"Site: {site}", mode="lines", line=dict(color=color)))
        fig_s.update_layout(title=f"{selected_p} Phosphorylation Rate (S)",xaxis_title="Time (min)", yaxis_title="Kinase Signaling Drive", template="plotly_white")
        st.plotly_chart(fig_s, use_container_width=True)

st.divider()

# --- Forward Simulation Panel with Finer Resolution ---
st.subheader(f"🔁 Forward Simulation of {selected_p}")

# Slider for t_max (in minutes): range from 1 hour to 14 days
t_max = st.slider(
    "Simulation Time (minutes)",
    min_value=2,
    max_value=14 * 24 * 60,  # 14 days
    value=960,       # default: 16 hrs
    step=60
)

n_points = 10000

# Simulate WT to steady state
t_fine, Y_wt = simulate_until_steady(sys, t_max=t_max, n_points=n_points)

# Extract per-protein output
def extract_fc_from_Y(Y, idx, t, protein, normalize=True):
    p_idx = idx.p2i[protein]
    st_y = idx.offset_y[p_idx]
    rna_vals = Y[:, st_y]
    prot_vals = Y[:, st_y + 1]
    ns = idx.n_sites[p_idx]
    site_names = idx.sites[p_idx]

    if ns > 0:
        psite_vals = Y[:, st_y + 2: st_y + 2 + ns]
        phos_sum = np.sum(psite_vals, axis=1)
        total_prot = prot_vals + phos_sum
    else:
        psite_vals = np.zeros((len(t), 0))
        total_prot = prot_vals

    if normalize:
        rna_vals = rna_vals / rna_vals[0]
        total_prot = total_prot / total_prot[0]
        if ns > 0:
            psite_vals = psite_vals / psite_vals[0, :]  # normalize each site

    # Base result
    df = pd.DataFrame({
        "time": t,
        "rna": rna_vals,
        "protein": total_prot,
    })

    # Add phospho sites in long format
    if ns > 0:
        df_ps = pd.DataFrame(psite_vals, columns=site_names)
        df_ps["time"] = t
        df_long = df_ps.melt(id_vars="time", var_name="psite", value_name="psite_value")
        df = df.merge(df_long, on="time", how="left")

    return df

# WT values
df_wt = extract_fc_from_Y(Y_wt, idx, t_fine, selected_p)

# KO simulation
sys.update(**ko_params)
t_ko, Y_ko = simulate_until_steady(sys, t_max=t_max, n_points=n_points)
df_ko = extract_fc_from_Y(Y_ko, idx, t_ko, selected_p)
sys.update(**best_params)  # restore baseline

# --- Plot mRNA
col1, col2 = st.columns(2)

with col1:
    fig_fine_r = go.Figure()
    fig_fine_r.add_trace(go.Scatter(x=df_wt["time"], y=df_wt["rna"], name="WT", line=dict(color="black", dash="dash")))
    fig_fine_r.add_trace(go.Scatter(x=df_ko["time"], y=df_ko["rna"], name="KO", line=dict(color="red")))
    fig_fine_r.update_layout(title="mRNA Simulation", xaxis_title="Time", yaxis_title="Fold Change", template="plotly_white")
    # fig_fine_r.update_xaxes(type="log")
    st.plotly_chart(fig_fine_r, use_container_width=True)

# --- Plot Protein
with col2:
    fig_fine_p = go.Figure()
    fig_fine_p.add_trace(go.Scatter(x=df_wt["time"], y=df_wt["protein"], name="WT", line=dict(color="black", dash="dash")))
    fig_fine_p.add_trace(go.Scatter(x=df_ko["time"], y=df_ko["protein"], name="KO", line=dict(color="blue")))
    fig_fine_p.update_layout(title="Protein Simulation", xaxis_title="Time", yaxis_title="Fold Change", template="plotly_white")
    # fig_fine_p.update_xaxes(type="log")
    st.plotly_chart(fig_fine_p, use_container_width=True)

col1, col2 = st.columns(2)

# --- Signaling Drive Panel (Phosphorylation S)
with col1:
    if selected_p in idx.p2i:
        p_idx = idx.p2i[selected_p]
        ns = idx.n_sites[p_idx]

        # Simulate WT again (ensures S is from WT context)
        t_S, Y_S = t_ko, Y_ko  # Use already simulated KO values
        kin_vals = Y_S[:, [idx.k2i[k] for k in idx.kinases]].T  # (n_kinases, time)
        kin_scaled = kin_vals * sys.c_k[:, None]
        S_t = sys.W_global @ kin_scaled  # (n_sites, time)

        site_names, site_rows = [], []

        # Recompute correct global site indices for selected_p
        site_counter = 0
        for i, p in enumerate(idx.proteins):
            for j, site in enumerate(idx.sites[i]):
                if p == selected_p:
                    site_names.append(site)
                    site_rows.append(site_counter)
                site_counter += 1

        fig_s_time = go.Figure()

        for site_name, site_idx in zip(site_names, site_rows):
            color = px.colors.qualitative.Plotly[hash(site_name) % len(px.colors.qualitative.Plotly)]
            fig_s_time.add_trace(go.Scatter(
                x=t_S,
                y=S_t[site_idx, :],
                name=site_name,
                mode="lines",
                line=dict(dash="solid", color=color),
                opacity=0.9,
            ))

        fig_s_time.update_layout(
            title=f"{selected_p} – Phosphorylation (S)",
            xaxis_title="Time (min)",
            yaxis_title="S (Signaling Rate)",
            template="plotly_white"
        )
        # fig_s_time.update_xaxes(type="log")
        st.plotly_chart(fig_s_time, use_container_width=True)
    else:
        st.warning(f"{selected_p} not found in protein index.")

# Plot Phospho-sites states from simulation
with col2:
    fig_sites_fine = go.Figure()

    # Safety check for valid data
    if "psite" in df_wt.columns and not df_wt["psite"].isna().all():
        for site in df_wt["psite"].dropna().unique():
            site_wt = df_wt[df_wt["psite"] == site]
            site_ko = df_ko[df_ko["psite"] == site]
            color = px.colors.qualitative.Plotly[hash(site) % len(px.colors.qualitative.Plotly)]

            if not site_wt.empty:
                fig_sites_fine.add_trace(go.Scatter(
                    x=site_wt["time"],
                    y=site_wt["psite_value"],
                    name=f"WT {site}",
                    line=dict(dash="dash", color=color)
                ))
            if not site_ko.empty:
                fig_sites_fine.add_trace(go.Scatter(
                    x=site_ko["time"],
                    y=site_ko["psite_value"],
                    name=f"KO {site}",
                    line=dict(color=color)
                ))

        fig_sites_fine.update_layout(
            title=f"{selected_p} Phospho-site State Dynamics",
            xaxis_title="Time (min)",
            yaxis_title="Phospho-site Level (a.u.)",
            template="plotly_white"
        )
        st.plotly_chart(fig_sites_fine, use_container_width=True)
    else:
        st.info("No phospho site data available for this protein.")

# --- Graph Options ---
st.sidebar.divider()
st.sidebar.header("🕸️ Graph Options")
depth = st.sidebar.slider("Cascade Depth", min_value=1, max_value=3, value=1, help="1: Direct targets only. 2+: Includes targets of targets.")

# --- Functional Hierarchy Map ---
if ko_type != "None" and df_tf_model is not None and not df_tf_model.empty and target is not None:
    st.divider()
    st.header(f"🕸️ Functional Influence Map: {target}")

    # df_tf_model is standardized to ('tf','target')
    src_col, tgt_col = "tf", "target"

    edges_list = []
    current_layer_nodes = {target}
    processed_nodes = set()

    final_kin_act = sys.kin.Kmat[:, -1] * sys.c_k
    S_final = sys.W_global.dot(final_kin_act)

    for d in range(depth):
        next_layer_nodes = set()
        if not current_layer_nodes:
            break

        if ko_type == "Kinase (Activity)" or d > 0:
            restore_Kinase()
            for k_name in current_layer_nodes:
                if k_name in idx.kinases:
                    k_idx = idx.k2i[k_name]
                    W_csc = sys.W_global.tocsc()
                    col = W_csc.getcol(k_idx)
                    rows, _ = col.nonzero()

                    for site_idx in rows:
                        prot_idx = np.searchsorted(idx.offset_s, site_idx, side="right") - 1
                        target_prot = idx.proteins[prot_idx]

                        beta_val = W_csc[site_idx, k_idx]
                        s_flux = S_final[site_idx]

                        edges_list.append(
                            {
                                "src": k_name,
                                "tgt": target_prot,
                                "type": "signaling",
                                "weight": float(beta_val),
                                "label": f"β (Kin-Site): {float(beta_val):.3f}, S-Flux: {float(s_flux):.3f}",
                            }
                        )
                        next_layer_nodes.add(target_prot)

        tf_hits = df_tf_model[df_tf_model[src_col].isin(current_layer_nodes)]
        for _, row in tf_hits.iterrows():
            s, t = row[src_col], row[tgt_col]
            t_idx = idx.p2i[t]
            alpha_val = sys.E_i[t_idx]

            edges_list.append(
                {
                    "src": s,
                    "tgt": t,
                    "type": "transcription",
                    "weight": float(alpha_val),
                    "label": f"α (TF Efficacy): {float(alpha_val):.3f}",
                }
            )
            next_layer_nodes.add(t)

        processed_nodes.update(current_layer_nodes)
        current_layer_nodes = next_layer_nodes - processed_nodes

    if not edges_list:
        st.warning("No functional connections found.")
    else:
        plot_df = pd.DataFrame(edges_list).drop_duplicates()
        G = nx.from_pandas_edgelist(plot_df, "src", "tgt", create_using=nx.DiGraph())
        pos = nx.spring_layout(G, k=1.0, seed=42)

        edge_traces = []
        midpoint_node_x, midpoint_node_y, midpoint_hover_text = [], [], []

        for _, row in plot_df.iterrows():
            x0, y0 = pos[row["src"]]
            x1, y1 = pos[row["tgt"]]

            width = np.clip(row["weight"] * 5, 1.0, 7)
            color = "#3498db" if row["type"] == "signaling" else "#e67e22"

            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color=color),
                    hoverinfo="none",
                    mode="lines",
                    opacity=0.6,
                )
            )

            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            midpoint_node_x.append(mid_x)
            midpoint_node_y.append(mid_y)
            midpoint_hover_text.append(row["label"])

        edge_hover_trace = go.Scatter(
            x=midpoint_node_x,
            y=midpoint_node_y,
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=midpoint_hover_text,
            name="Edge Data",
        )

        node_x, node_y, node_color, node_text = [], [], [], []
        # Inside your node plotting loop
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Get impact value if available
            impact = comparison.loc[comparison["protein"] == node, "log2_fc"]
            impact = float(impact.iloc[0]) if not impact.empty else 0.0
            node_color.append(impact)

            # Handle missing nodes in idx.p2i
            if node in idx.p2i:
                n_idx = idx.p2i[node]
                c_val = float(sys.C_i[n_idx])
            else:
                c_val = float('nan')  # or 0.0 or "N/A" depending on display preference

            node_text.append(
                f"<b>{node}</b><br>Log2 Impact: {impact:.2f}<br>Internal α (C_i): {c_val:.3f}"
                if not np.isnan(c_val)
                else f"<b>{node}</b><br>Log2 Impact: {impact:.2f}<br>Internal α (C_i): N/A"
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[f"<b>{n}</b>" for n in G.nodes()],
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale="RdBu_r",
                size=35,
                colorbar=dict(title="Log2 FC", thickness=15),
                color=node_color,
                cmin=-2,
                cmax=2,
                line=dict(width=2, color="white"),
            ),
            hovertext=node_text,
            hoverinfo="text",
        )

        fig = go.Figure(
            data=edge_traces + [edge_hover_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(0,0,0,0)",
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.info("🔵 Blue Edges: Signaling (S-flux & β) | 🟠 Orange Edges: Transcription (α)")
