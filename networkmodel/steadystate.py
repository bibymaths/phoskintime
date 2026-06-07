"""
Initial Condition and Steady-State Logic Module.

This module is responsible for defining the starting state ($y_0$) of the ODE system.
It supports two primary modes:
1.  **Data-Driven Initialization (`build_y0_from_data`):** Uses experimental data (Protein/RNA abundance
    at t=0) to set the initial conditions directly. This is crucial for matching the
    absolute scale of the measurements.
2.  **Analytical Steady-State (`steady_state_*`):** Solves the algebraic equilibrium equations
    ($dy/dt = 0$) for a system where all kinetic parameters are set to 1.0. This is useful for
    testing structural consistency or initializing systems without data.


"""

import os

import numpy as np
import pandas as pd

from networkmodel.config import MODEL, RESULTS_DIR
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


def _dump_y0(sys, out_dir, max_sites=200):
    """
    Exports the computed initial condition vector $y_0$ to a CSV file for inspection.
    Useful for debugging which biological entity corresponds to which numerical value.
    """
    idx = sys.idx
    y0 = sys.y0()  # uses custom_y0 if set
    rows = []

    for i, gene in enumerate(idx.proteins):
        st = int(idx.offset_y[i])

        # mRNA State
        R0 = float(y0[st])

        if MODEL == 2:
            # Combinatorial Model: States are bitmasks
            nst = int(idx.n_states[i])
            Pm0 = float(y0[st + 1])  # mask 0 (Unphosphorylated)

            # Sum total protein mass across all 2^n states
            Ptot = float(y0[st + 1: st + 1 + nst].sum())

            rows.append(dict(entity=gene, kind="mRNA", substate="R", value=R0))
            rows.append(dict(entity=gene, kind="protein_total", substate="P_total_masksum", value=Ptot))
            rows.append(dict(entity=gene, kind="protein_state", substate="mask0", value=Pm0))

            # Optional: show single-bit masks by site name for clarity
            ns = int(idx.n_sites[i])
            for j, psite in enumerate(idx.sites[i][:max_sites]):
                mask = 1 << j
                val = float(y0[st + 1 + mask])
                rows.append(dict(entity=gene, kind="phospho_state", substate=f"mask_{psite}", value=val))

        else:
            # Distributive/Sequential: States are linear P0, P1, ...
            P0 = float(y0[st + 1])  # unphosphorylated pool
            rows.append(dict(entity=gene, kind="mRNA", substate="R", value=R0))
            rows.append(dict(entity=gene, kind="protein_state", substate="P0", value=P0))

            ns = int(idx.n_sites[i])
            base = st + 2
            for j, psite in enumerate(idx.sites[i][:max_sites]):
                val = float(y0[base + j])
                rows.append(dict(entity=gene, kind="phospho_state", substate=str(psite), value=val))

            if ns > max_sites:
                rows.append(dict(entity=gene, kind="note", substate="truncated_sites",
                                 value=float(ns - max_sites)))

    df_y0 = pd.DataFrame(rows)

    # Print a compact summary to log
    logger.info(f"[IC] y0 rows: {len(df_y0)} | entities: {df_y0['entity'].nunique()}")

    for _, r in df_y0.iterrows():
        # label = fully qualified state name (recommended)
        # entity = protein / gene
        # kind = RNA / PROT / PHOS
        label = f"{r['entity']}_{r['kind']}_{r['substate']}"
        logger.info(
            "[IC] %-6s | %-30s | %-40s = %.6g",
            r.get("kind", "?"),
            r.get("entity", "?"),
            label,
            float(r["value"]),
        )

    # Save full table
    out_path = os.path.join(out_dir, "initial_conditions_y0.csv")
    df_y0.to_csv(out_path, index=False)
    logger.info(f"[IC] Saved y0 table: {out_path}")

    return df_y0


def _dict_at_time(df, key_cols, t0, value_col="fc", time_col="time", tol=1e-8):
    """
    Helper to extract a dictionary of {Entity -> Value} at a specific time t0.
    Handles floating point tolerance for time matching.
    """
    if df is None or df.empty:
        return {}

    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[time_col, value_col])

    # tolerance filter for float times
    m = np.isclose(d[time_col].to_numpy(dtype=float), float(t0), atol=tol, rtol=0.0)
    d = d.loc[m, key_cols + [value_col]]
    if d.empty:
        return {}

    # average replicates if they exist
    g = d.groupby(key_cols, as_index=False)[value_col].mean()

    if len(key_cols) == 1:
        return dict(zip(g[key_cols[0]].astype(str), g[value_col].astype(float)))
    else:
        return {tuple(row[key_cols].astype(str)): float(row[value_col]) for _, row in g.iterrows()}


def build_y0_from_data(
        idx,
        df_prot,
        df_rna,
        df_pho,
        *,
        t_init=0.0,
        t0_pho=0.0,
        eps=1e-9,
        time_tol=1e-8,
        max_pho_frac=0.3,  # at most 30% of protein initially phosphorylated
):
    """
    Constructs the initial state vector $y_0$ strictly from experimental data.

    **Physics-Compliant Mass Balance:**
    1.  **Protein Total ($P_{tot}$):** Taken from `df_prot` at `t_init`.
    2.  **Phospho Mass:** Taken from `df_pho` at `t0_pho`. Because phospho signals are often
        relative intensities, we scale them such that the total phosphorylated mass does not
        exceed `max_pho_frac` (e.g., 30%) of $P_{tot}$.
    3.  **Unphosphorylated Mass ($P_0$):** Calculated by conservation: $P_0 = P_{tot} - \sum P_{phos}$.



    Args:
        idx: System index object.
        df_prot, df_rna, df_pho: Tidy dataframes of observations.
        t_init (float): The simulation start time (usually 0.0).
        max_pho_frac (float): Cap on the initial fraction of phosphorylated protein.

    Returns:
        np.ndarray: The assembled initial condition vector.
    """

    # ------------------------------------------------------------------
    # Protein ICs at t_init
    # ------------------------------------------------------------------
    prot_init = _dict_at_time(df_prot, ["protein"], t_init, tol=time_tol)

    # ------------------------------------------------------------------
    # RNA ICs = FIRST observed value per gene
    # ------------------------------------------------------------------
    rna_init = {}
    if df_rna is not None and not df_rna.empty:
        d = df_rna.copy()
        d["time"] = pd.to_numeric(d["time"], errors="coerce")
        d["fc"] = pd.to_numeric(d["fc"], errors="coerce")
        d = d.dropna(subset=["time", "fc"])

        d0 = (
            d.sort_values("time")
            .groupby("protein", as_index=False)
            .first()
        )
        rna_init = dict(zip(d0["protein"], d0["fc"]))

    # ------------------------------------------------------------------
    # Phospho ICs at t0_pho (direct data lookup)
    # ------------------------------------------------------------------
    pho_init = _dict_at_time(
        df_pho, ["protein", "psite"], t0_pho, tol=time_tol
    )

    # ------------------------------------------------------------------
    # Allocate y0
    # ------------------------------------------------------------------
    y0 = np.zeros(int(idx.state_dim), dtype=np.float64)

    for i, gene in enumerate(idx.proteins):
        st = int(idx.offset_y[i])

        # -------------------
        # mRNA
        # -------------------
        R0 = float(rna_init.get(gene, 1.0))
        y0[st] = max(R0, eps)

        # -------------------
        # Protein total mass
        # -------------------
        P_tot = float(prot_init.get(gene, 1.0))
        P_tot = max(P_tot, eps)

        sites = idx.sites[i]
        raw_pho = np.array(
            [float(pho_init.get((gene, s), 0.0)) for s in sites],
            dtype=np.float64
        )

        # Scale phospho signals into a bounded fraction of protein
        # If sum(raw_pho) is huge (arbitrary units), scale it down so sum <= 0.3 * P_tot
        if raw_pho.sum() > 0:
            scale = min(max_pho_frac, max_pho_frac / raw_pho.sum())
            site_mass = raw_pho * scale * P_tot
        else:
            site_mass = np.zeros_like(raw_pho)

        site_mass = np.maximum(site_mass, 0.0)
        pho_sum = site_mass.sum()

        # -------------------
        # Assign states
        # -------------------
        if MODEL == 2:
            # combinatorial: P0 is mask 0
            y0[st + 1] = max(P_tot - pho_sum, eps)

            for j, mass in enumerate(site_mass):
                if mass > 0:
                    mask = 1 << j
                    y0[st + 1 + mask] = max(mass, eps)

        else:
            # distributive / sequential: P0 is first protein state
            y0[st + 1] = max(P_tot - pho_sum, eps)

            for j, mass in enumerate(site_mass):
                y0[st + 2 + j] = max(mass, eps)

    return y0
