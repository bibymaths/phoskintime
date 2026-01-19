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
from global_model.params import init_raw_params, unpack_params
from global_model.io import load_data
from global_model.buildmat import build_W_parallel, build_tf_matrix

st.set_page_config(page_title="PhoskinTime Global Knockout", layout="wide")


@st.cache_resource
def load_system():
    # Path to your results
    results_dir = Path("./results_global_sequential")

    class Args:
        kinase_net, tf_net = config.KINASE_NET_FILE, config.TF_NET_FILE
        ms, rna, phospho = config.MS_DATA_FILE, config.RNA_DATA_FILE, config.PHOSPHO_DATA_FILE
        kinopt, tfopt = None, None
        normalize_fc_steady = False

    df_kin, df_tf, df_prot, df_pho, df_rna, kin_beta_map, tf_beta_map = load_data(Args())
    idx = Index(df_kin, tf_interactions=df_tf)
    W_global = build_W_parallel(df_kin, idx, n_cores=1)
    tf_mat = build_tf_matrix(df_tf, idx, tf_beta_map=tf_beta_map)
    kin_in = KinaseInput(idx.kinases, df_prot)
    tf_deg = np.asarray(np.abs(tf_mat).sum(axis=1)).ravel().astype(np.float64)
    tf_deg[tf_deg < 1e-12] = 1.0

    defaults = {"c_k": np.ones(len(idx.kinases)), "A_i": np.ones(idx.N), "B_i": np.full(idx.N, 0.2),
                "C_i": np.full(idx.N, 0.5), "D_i": np.full(idx.N, 0.05),
                "Dp_i": np.full(idx.total_sites, 0.05), "E_i": np.ones(idx.N), "tf_scale": 0.1}
    sys = System(idx, W_global, tf_mat, kin_in, defaults, tf_deg)
    _, slices, _, _ = init_raw_params(defaults)

    X = np.load(results_dir / "pareto_X.npy")
    best_params = unpack_params(X[0], slices)
    s_rates = pd.read_csv(results_dir / "S_rates_picked.csv")
    return sys, idx, best_params, df_tf, s_rates

# --- Simulation Logic ---
def run_sim(sys, idx, mod_params):
    sys.update(**mod_params)
    return simulate_and_measure(sys, idx, config.TIME_POINTS_PROTEIN, config.TIME_POINTS_RNA,
                                config.TIME_POINTS_PHOSPHO)


# --- UI Setup ---
st.title("🧪 Global Signaling & Transcriptional Knockout Explorer")
sys, idx, best_params, df_tf, s_rates = load_system()
config.MODEL = 0  # Reconstruct using Distributive kinetics

st.sidebar.header("🕹️ Control Panel")
ko_type = st.sidebar.selectbox("1. Choose Knockout Type", ["None", "Protein (Synthesis)", "Kinase (Activity)"])

ko_params = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in best_params.items()}

if ko_type == "Protein (Synthesis)":
    target = st.sidebar.selectbox("Select Target Protein", idx.proteins)
    p_idx = list(idx.proteins).index(target)
    ko_params["A_i"][p_idx] = 0.0
elif ko_type == "Kinase (Activity)":
    target = st.sidebar.selectbox("Select Kinase to Inhibit", idx.kinases)
    k_idx = list(idx.kinases).index(target)
    ko_params["c_k"][k_idx] = 0.0

# --- Simulation Logic ---
wt_dfp, wt_dfr, _ = run_sim(sys, idx, best_params)
ko_dfp, ko_dfr, _ = run_sim(sys, idx, ko_params)

# --- Versatile Visualization: The Impact Scatter ---
st.header("🎯 System-Wide Impact Analysis")

# Calculate metrics for the scatter plot
final_t = wt_dfr['time'].max()
comparison = pd.DataFrame({
    'protein': wt_dfr[wt_dfr['time'] == final_t]['protein'].values,
    'WT_final': wt_dfr[wt_dfr['time'] == final_t]['pred_fc'].values,
    'KO_final': ko_dfr[ko_dfr['time'] == final_t]['pred_fc'].values
})

comparison['log2_fc'] = np.log2((comparison['KO_final'] + 1e-6) / (comparison['WT_final'] + 1e-6))
# Calculate "Sensitivity": Total area between curves (Integral of the difference)
sensitivity = []
for p in comparison['protein']:
    wt_vals = wt_dfr[wt_dfr['protein'] == p]['pred_fc'].values
    ko_vals = ko_dfr[ko_dfr['protein'] == p]['pred_fc'].values
    sensitivity.append(np.sum(np.abs(wt_vals - ko_vals)))
comparison['sensitivity_score'] = sensitivity

# Interactive Scatter Plot
fig_scatter = px.scatter(
    comparison,
    x="log2_fc",
    y="sensitivity_score",
    text="protein",
    color="log2_fc",
    color_continuous_scale="RdBu_r",
    labels={"log2_fc": "Log2 Impact (KO/WT)", "sensitivity_score": "Cumulative Perturbation (Area)"},
    title="Sensitivity vs. Magnitude: Which genes are most 'fragile'?"
)
fig_scatter.update_traces(textposition='top center')
st.plotly_chart(fig_scatter, use_container_width=True)

# --- The "Delta" Trajectory (Plotly Version) ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")
st.markdown("This shows the **absolute loss of signal** over time ($WT - KO$).")

# 1. Calculate the Delta
delta_df = wt_dfr.copy()
delta_df['delta'] = wt_dfr['pred_fc'] - ko_dfr['pred_fc']

# 2. Add a 'Significance' flag for better visualization
# We calculate the max deviation for each protein to highlight the most impacted ones
max_deltas = delta_df.groupby('protein')['delta'].transform(lambda x: x.abs().max())
delta_df['is_significant'] = max_deltas > 0.1
delta_df['Impact Group'] = delta_df['is_significant'].map({True: "Significant Effect", False: "Minimal/No Effect"})

# 3. Generate the Interactive Plotly Line Chart
fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Alphabet,
    labels={"time": "Time (min)", "delta": "Δ Signal (WT - KO)", "protein": "Protein"},
    title="Propagation of Signal Loss Over Time"
)

# 4. Styling for Clarity
fig_delta.update_layout(
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white"
)

# Reduce opacity for non-significant lines to reduce clutter
for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df['protein'] == p_name]['delta'].abs().max()
    if max_d < 0.1:
        trace.line.width = 1
        trace.opacity = 0.2
        trace.showlegend = False # Hide insignificant proteins from legend to save space
    else:
        trace.line.width = 3
        trace.opacity = 0.9

st.plotly_chart(fig_delta, use_container_width=True)

# --- Delta Logic: Cross-Modality Toggle ---
st.header("⏱️ Signal Loss Propagation (Δ-Trajectories)")

# Toggle for mRNA vs Protein
modality_choice = st.radio("Select Modality for Delta Analysis:", ["mRNA", "Protein"], horizontal=True)
active_wt = wt_dfr if modality_choice == "mRNA" else wt_dfp
active_ko = ko_dfr if modality_choice == "mRNA" else ko_dfp

# 1. Calculate the Delta
delta_df = active_wt.copy()
delta_df['delta'] = active_wt['pred_fc'] - active_ko['pred_fc']

# 2. Add Significance Metadata
max_deltas = delta_df.groupby('protein')['delta'].transform(lambda x: x.abs().max())
delta_df['is_significant'] = max_deltas > 0.05  # Lowered threshold slightly for protein
delta_df['Impact Group'] = delta_df['is_significant'].map({True: "Significant", False: "Minimal"})

# 3. Interactive Plotly Line Chart
fig_delta = px.line(
    delta_df,
    x="time",
    y="delta",
    color="protein",
    line_group="protein",
    hover_name="protein",
    color_discrete_sequence=px.colors.qualitative.Safe,
    labels={"time": "Time (min)", "delta": f"Δ {modality_choice} Signal (WT - KO)", "protein": "Protein"},
    title=f"Propagation of {modality_choice} Signal Loss"
)

# Style the chart: focus on significant proteins
for trace in fig_delta.data:
    p_name = trace.name
    max_d = delta_df[delta_df['protein'] == p_name]['delta'].abs().max()
    if max_d < 0.05:
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
# Extract top 10 impacted (highest absolute Log2 FC)
top_impact = comparison.copy()
top_impact['absolute_impact'] = top_impact['log2_fc'].abs()
top_impact = top_impact.sort_values('absolute_impact', ascending=False).head(15)

st.dataframe(
    top_impact[['protein', 'log2_fc', 'sensitivity_score']],
    column_config={
        "log2_fc": st.column_config.NumberColumn("Log2 Impact", format="%.2f"),
        "sensitivity_score": st.column_config.NumberColumn("Total Area Delta", format="%.2f")
    },
    use_container_width=True
)

# 4. Download Section for Plot Data
st.subheader("📥 Export Plot Data")
# Pivot the data so it's easier to use in Excel/R (Times as rows, Proteins as columns)
pivot_delta = delta_df.pivot(index='time', columns='protein', values='delta')

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

# 1. Filter Data for selected protein
wt_p_data = wt_dfp[wt_dfp['protein'] == selected_p]
ko_p_data = ko_dfp[ko_dfp['protein'] == selected_p]
wt_r_data = wt_dfr[wt_dfr['protein'] == selected_p]
ko_r_data = ko_dfr[ko_dfr['protein'] == selected_p]

col1, col2, col3 = st.columns(3)

with col1:
    # --- mRNA PLOT ---
    fig_insp_r = go.Figure()
    # WT Curve (Predicted)
    fig_insp_r.add_trace(go.Scatter(x=wt_r_data['time'], y=wt_r_data['pred_fc'],
                                    name="Wild Type", line=dict(dash='dash', color='black', width=2)))
    # KO Curve (Predicted)
    fig_insp_r.add_trace(go.Scatter(x=ko_r_data['time'], y=ko_r_data['pred_fc'],
                                    name="Knockout", line=dict(color='red', width=3)))

    fig_insp_r.update_layout(title=f"{selected_p} mRNA Response", xaxis_title="Time (min)", yaxis_title="Fold Change",
                             template="plotly_white")
    st.plotly_chart(fig_insp_r, use_container_width=True)

with col2:
    # --- PROTEIN PLOT ---
    fig_insp_p = go.Figure()
    # WT Curve (Predicted)
    fig_insp_p.add_trace(go.Scatter(x=wt_p_data['time'], y=wt_p_data['pred_fc'],
                                    name="Wild Type", line=dict(dash='dash', color='black', width=2)))
    # KO Curve (Predicted)
    fig_insp_p.add_trace(go.Scatter(x=ko_p_data['time'], y=ko_p_data['pred_fc'],
                                    name="Knockout", line=dict(color='blue', width=3)))

    fig_insp_p.update_layout(title=f"{selected_p} Protein Abundance", xaxis_title="Time (min)",
                             yaxis_title="Fold Change", template="plotly_white")
    st.plotly_chart(fig_insp_p, use_container_width=True)

with col3:
    # --- PANEL 3: Signaling Drive (S-rates) ---
    # We need to access the S_cache or the S_rates_picked.csv you loaded
    st.subheader(f"⚡ Signaling Input for {selected_p}")
    p_s = s_rates[s_rates["protein"] == selected_p]

    fig_s = go.Figure()
    for site in p_s["psite"].unique():
        site_data = p_s[p_s["psite"] == site]
        # Note: Knocking out a kinase would change these S-values in a real simulation
        fig_s.add_trace(go.Scatter(x=site_data["time"], y=site_data["S"],
                                   name=f"Site: {site}", mode='lines'))

    fig_s.update_layout(xaxis_title="Time (min)", yaxis_title="S (Signaling Drive)", template="plotly_white")
    st.plotly_chart(fig_s, use_container_width=True)

# 1. Add this to your Sidebar Section
st.sidebar.divider()
st.sidebar.header("🕸️ Graph Options")
depth = st.sidebar.slider("Cascade Depth", min_value=1, max_value=3, value=1,
                          help="1: Direct targets only. 2+: Includes targets of targets.")

# --- Functional Hierarchy Map with Alpha, Beta, and S-flux ---
if ko_type != "None" and df_tf is not None:
    st.divider()
    st.header(f"🕸️ Functional Influence Map: {target}")

    # 1. Detection of TF Columns
    cols_tf = df_tf.columns.tolist()
    src_col = 'Source' if 'Source' in cols_tf else ('tf' if 'tf' in cols_tf else cols_tf[0])
    tgt_col = 'Target' if 'Target' in cols_tf else ('target' if 'target' in cols_tf else cols_tf[1])

    # 2. Recursive Search Logic with Weights
    edges_list = []
    current_layer_nodes = {target}
    processed_nodes = set()

    # Pre-calculate S-cache for the final timepoint to show current "Flux"
    # S = W * (Kinase_Activity * c_k)
    final_kin_act = sys.kin.Kmat[:, -1] * sys.c_k
    S_final = sys.W_global.dot(final_kin_act)

    for d in range(depth):
        next_layer_nodes = set()
        if not current_layer_nodes:
            break

        # --- LAYER A: Signaling (Kinase -> P-Site -> Protein) ---
        if ko_type == "Kinase (Activity)" or d > 0:
            for k_name in current_layer_nodes:
                if k_name in idx.kinases:
                    k_idx = idx.k2i[k_name]
                    W_csc = sys.W_global.tocsc()
                    col = W_csc.getcol(k_idx)
                    rows, _ = col.nonzero()

                    for site_idx in rows:
                        prot_idx = np.searchsorted(idx.offset_s, site_idx, side='right') - 1
                        target_prot = idx.proteins[prot_idx]

                        # Weight (Beta): The optimized connection strength in W
                        beta_val = W_csc[site_idx, k_idx]
                        # Phosphorylation Flux (S): Dynamic value for this specific site
                        s_flux = S_final[site_idx]

                        edges_list.append({
                            'src': k_name, 'tgt': target_prot,
                            'type': 'signaling', 'weight': beta_val,
                            'label': f"β (Kin-Site): {beta_val:.3f}, S-Flux: {s_flux:.3f}"
                        })
                        next_layer_nodes.add(target_prot)

        # --- LAYER B: Transcriptional (TF -> mRNA) ---
        tf_hits = df_tf[df_tf[src_col].isin(current_layer_nodes)]
        for _, row in tf_hits.iterrows():
            s, t = row[src_col], row[tgt_col]

            # Weight (Alpha): The optimized Transcriptional Efficacy E_i
            t_idx = idx.p2i[t]
            alpha_val = sys.E_i[t_idx]

            edges_list.append({
                'src': s, 'tgt': t,
                'type': 'transcription', 'weight': alpha_val,
                'label': f"α (TF Efficacy): {alpha_val:.3f}"
            })
            next_layer_nodes.add(t)

        processed_nodes.update(current_layer_nodes)
        current_layer_nodes = next_layer_nodes - processed_nodes

        # 3. Enhanced Visualization with Robust Hover Targets
        if not edges_list:
            st.warning("No functional connections found.")
        else:
            plot_df = pd.DataFrame(edges_list).drop_duplicates()
            G = nx.from_pandas_edgelist(plot_df, 'src', 'tgt', create_using=nx.DiGraph())
            pos = nx.spring_layout(G, k=1.0, seed=42)

            edge_traces = []
            midpoint_node_x = []
            midpoint_node_y = []
            midpoint_hover_text = []

            for _, row in plot_df.iterrows():
                x0, y0 = pos[row['src']]
                x1, y1 = pos[row['tgt']]

                # Edge thickness proportional to Alpha/Beta weight
                width = np.clip(row['weight'] * 5, 1.0, 7)
                color = '#3498db' if row['type'] == 'signaling' else '#e67e22'

                # 1. Create the visible line
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=width, color=color),
                    hoverinfo='none',  # Disable hover on the line itself
                    mode='lines',
                    opacity=0.6
                ))

                # 2. Create a hover target at the midpoint of the edge
                mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                midpoint_node_x.append(mid_x)
                midpoint_node_y.append(mid_y)
                midpoint_hover_text.append(row['label'])

            # Create the Hover-Target Trace (invisible markers in the middle of edges)
            edge_hover_trace = go.Scatter(
                x=midpoint_node_x, y=midpoint_node_y,
                mode='markers',
                marker=dict(size=10, color='rgba(0,0,0,0)'),  # Invisible
                hoverinfo='text',
                text=midpoint_hover_text,
                name='Edge Data'
            )

            # 3. Node Trace (Impact colored)
            node_x, node_y, node_color, node_text = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # Pull impact from comparison dataframe
                impact = comparison[comparison['protein'] == node]['log2_fc'].values[0] if node in comparison[
                    'protein'].values else 0

                # Get Internal Alpha (C_i)
                n_idx = idx.p2i[node]
                c_val = sys.C_i[n_idx]

                node_color.append(impact)
                node_text.append(f"<b>{node}</b><br>Log2 Impact: {impact:.2f}<br>Internal α (C_i): {c_val:.3f}")

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                text=[f"<b>{n}</b>" for n in G.nodes()],
                textposition="top center",
                marker=dict(
                    showscale=True, colorscale='RdBu_r', size=35,
                    colorbar=dict(title="Log2 FC", thickness=15),
                    color=node_color, cmin=-2, cmax=2,
                    line=dict(width=2, color='white')
                ),
                hovertext=node_text,
                hoverinfo='text'
            )

            fig = go.Figure(data=edge_traces + [edge_hover_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                plot_bgcolor='rgba(0,0,0,0)'
                            ))

            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "🔵 **Blue Edges**: Signaling ($S$-flux & $\\beta$) | 🟠 **Orange Edges**: Transcription ($\\alpha$)")