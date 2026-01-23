import os

import numpy as np
import pandas as pd

from global_model.config import MODEL, RESULTS_DIR
from global_model.models import sequential_rhs, combinatorial_rhs, build_random_transitions, distributive_rhs
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)

def _dump_y0(sys, out_dir, max_sites=200):
    idx = sys.idx
    y0 = sys.y0()  # uses custom_y0 if set
    rows = []

    for i, gene in enumerate(idx.proteins):
        st = int(idx.offset_y[i])

        # mRNA
        R0 = float(y0[st])

        if MODEL == 2:
            nst = int(idx.n_states[i])
            Pm0 = float(y0[st + 1])  # mask 0

            # total protein mass across masks
            Ptot = float(y0[st + 1: st + 1 + nst].sum())

            rows.append(dict(entity=gene, kind="mRNA", substate="R", value=R0))
            rows.append(dict(entity=gene, kind="protein_total", substate="P_total_masksum", value=Ptot))
            rows.append(dict(entity=gene, kind="protein_state", substate="mask0", value=Pm0))

            # optional: show single-bit masks by site name
            ns = int(idx.n_sites[i])
            for j, psite in enumerate(idx.sites[i][:max_sites]):
                mask = 1 << j
                val = float(y0[st + 1 + mask])
                rows.append(dict(entity=gene, kind="phospho_state", substate=f"mask_{psite}", value=val))

        else:
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
    Return dict: key -> mean(value_col) at time ~= t0 (tolerance-based).
    key_cols can be ["protein"] or ["protein","psite"].
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


# def build_y0_from_data(
#         idx,
#         df_prot,
#         df_rna,
#         df_pho,
#         *,
#         t_init=None,  # integration start time (usually 0.0)
#         t0_prot=0.0,  # where protein baseline is defined in your loss
#         t0_rna=4.0,  # where RNA baseline is defined in your loss
#         t0_pho=0.0,  # where phospho baseline is defined in your loss
#         pho_frac=0.01,  # initial phospho mass as fraction of P0
#         eps=1e-9,
#         time_tol=1e-8
# ):
#     """
#     Build y0 aligned with your state layout:
#       MODEL 0/1: [R, P0, Psite...]
#       MODEL 2:   [R, P(mask0), P(mask1)...]
#     Note: with fold-change data, absolute scaling is arbitrary. This picks safe, positive ICs.
#     """
#     if t_init is None:
#         # you integrate from 0.0 in your code; keep that default
#         t_init = 0.0
#
#     # values at init time (what y0 represents)
#     prot_init = _dict_at_time(df_prot, ["protein"], t_init, tol=time_tol)
#     rna_init = _dict_at_time(df_rna, ["protein"], t_init, tol=time_tol)
#
#     # if RNA not measured at t_init (common), fall back to RNA baseline time (often 4.0)
#     if not rna_init:
#         rna_init = _dict_at_time(df_rna, ["protein"], t0_rna, tol=time_tol)
#
#     # phospho "baseline fc" at t0_pho: used only as a *relative* scaler
#     pho_base = _dict_at_time(df_pho, ["protein", "psite"], t0_pho, tol=time_tol)
#
#     y0 = np.zeros(int(idx.state_dim), dtype=np.float64)
#
#     for i, gene in enumerate(idx.proteins):
#         st = int(idx.offset_y[i])
#
#         # mRNA
#         R0 = float(rna_init.get(gene, 1.0))
#         y0[st] = max(R0, eps)
#
#         # protein / states
#         P0 = float(prot_init.get(gene, 1.0))
#         P0 = max(P0, eps)
#
#         if MODEL == 2:
#             # combinatorial: mask0 is st+1
#             y0[st + 1] = P0
#
#             ns = int(idx.n_sites[i])
#             # put tiny mass into single-site masks so phospho denominators aren't zero
#             # (only meaningful if ns is small; MODEL==2 is infeasible for large ns anyway)
#             total_added = 0.0
#             for j, psite in enumerate(idx.sites[i]):
#                 base_fc = float(pho_base.get((gene, psite), 1.0))
#                 mass = max(eps, pho_frac * P0 * base_fc)
#                 mask = 1 << j
#                 y0[st + 1 + mask] = mass
#                 total_added += mass
#
#             # conserve total mass roughly (optional but helps stability)
#             y0[st + 1] = max(eps, y0[st + 1] - total_added)
#
#         else:
#             # distributive/sequential: P0 is st+1, sites are st+2...
#             y0[st + 1] = P0
#             ns = int(idx.n_sites[i])
#             for j, psite in enumerate(idx.sites[i]):
#                 base_fc = float(pho_base.get((gene, psite), 1.0))
#                 # IMPORTANT: site state is an amount, not an FC -> keep it small
#                 y0[st + 2 + j] = max(eps, pho_frac * P0 * base_fc)
#
#     return y0

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
    max_pho_frac=0.3,   # at most 30% of protein initially phosphorylated
):
    """
    Build y0 strictly from data with physically valid mass balance.

    - RNA: first observed value per gene (earliest time in df_rna)
    - Protein: value at t_init
    - Phosphosite: data-scaled small fractions of protein (NOT normalized to 1)
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
            # combinatorial
            y0[st + 1] = max(P_tot - pho_sum, eps)

            for j, mass in enumerate(site_mass):
                if mass > 0:
                    mask = 1 << j
                    y0[st + 1 + mask] = max(mass, eps)

        else:
            # distributive / sequential
            y0[st + 1] = max(P_tot - pho_sum, eps)

            for j, mass in enumerate(site_mass):
                y0[st + 2 + j] = max(mass, eps)

    return y0

# -----------------------------
# Helpers
# -----------------------------
def _tf_squash(u: float) -> float:
    # maps R -> (-1, 1)
    return u / (1.0 + abs(u))


def _infer_total_s(offset_s: np.ndarray, n_sites: np.ndarray) -> int:
    i = int(len(n_sites) - 1)
    return int(offset_s[i] + n_sites[i])


def _infer_total_y_distrib_seq(offset_y: np.ndarray, n_sites: np.ndarray) -> int:
    i = int(len(n_sites) - 1)
    return int(offset_y[i] + 2 + n_sites[i])


def _infer_total_y_comb(offset_y: np.ndarray, n_states: np.ndarray) -> int:
    i = int(len(n_states) - 1)
    return int(offset_y[i] + 1 + n_states[i])  # R + (mask states)


def _bit_index_from_lsb_py(lsb: int) -> int:
    j = 0
    while lsb > 1:
        lsb >>= 1
        j += 1
    return j


def _thomas_tridiag(a, b, c, d):
    """
    Solve tridiagonal system with Thomas algorithm.
    a: lower diag (a[0]=0)
    b: main diag
    c: upper diag (c[-1]=0)
    d: rhs
    """
    n = len(b)
    cp = np.zeros(n, dtype=np.float64)
    dp = np.zeros(n, dtype=np.float64)

    denom = b[0]
    if denom == 0.0:
        denom = np.finfo(np.float64).tiny
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if denom == 0.0:
            denom = np.finfo(np.float64).tiny
        cp[i] = (c[i] / denom) if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.zeros(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# -----------------------------
# Steady-state for Distributive
# (sets ALL params to 1 internally)
# -----------------------------
def steady_state_distributive(idx, TF_inputs=None, tf_scale=1.0, verify_with_rhs=False):
    """
    Returns y_ss (and optionally dy(y_ss)) for distributive model with ALL params = 1.
    Needs idx fields: N, offset_y, offset_s, n_sites
    Uses your distributive_rhs for optional verification.
    """
    N = int(getattr(idx, "N", len(idx.n_sites)))
    offset_y = np.asarray(idx.offset_y, dtype=np.int32)
    offset_s = np.asarray(idx.offset_s, dtype=np.int32)
    n_sites = np.asarray(idx.n_sites, dtype=np.int32)

    if TF_inputs is None:
        TF_inputs = np.zeros(N, dtype=np.float64)
    else:
        TF_inputs = np.asarray(TF_inputs, dtype=np.float64)

    total_s = _infer_total_s(offset_s, n_sites)
    total_y = _infer_total_y_distrib_seq(offset_y, n_sites)

    # ALL params = 1
    A_i = np.ones(N, dtype=np.float64)
    B_i = np.ones(N, dtype=np.float64)
    C_i = np.ones(N, dtype=np.float64)
    D_i = np.ones(N, dtype=np.float64)
    E_i = np.ones(N, dtype=np.float64)
    S_all = np.ones(total_s, dtype=np.float64)
    Dp_i = np.ones(total_s, dtype=np.float64)

    y = np.zeros(total_y, dtype=np.float64)

    for i in range(N):
        y0 = int(offset_y[i])
        s0 = int(offset_s[i])
        ns = int(n_sites[i])

        u = _tf_squash(float(TF_inputs[i]))
        synth = A_i[i] * (1.0 + tf_scale * u)  # params=1 => in (0,2) for tf_scale=1
        if synth < 0.0:
            synth = 0.0

        R = synth / B_i[i]  # params=1 => R=synth
        y[y0 + 0] = R

        if ns == 0:
            P = (C_i[i] * R) / D_i[i]  # params=1 => P=R
            y[y0 + 1] = P
            continue

        # General linear steady-state (still params=1 here)
        # ps_j = (s_j * P) / (E + Dp_j)
        # 0 = C R - (D + sum_s) P + sum(E * ps_j)
        # => P = (C R) / (D + sum_s - sum(E*s_j/(E+Dp_j)))
        sum_s = 0.0
        sum_frac = 0.0
        for j in range(ns):
            si = s0 + j
            s = S_all[si]
            sum_s += s
            sum_frac += (E_i[i] * s) / (E_i[i] + Dp_i[si])

        denom = D_i[i] + sum_s - sum_frac
        if denom <= 0.0:
            denom = np.finfo(np.float64).tiny

        P = (C_i[i] * R) / denom
        if P < 0.0:
            P = 0.0
        y[y0 + 1] = P

        base = y0 + 2
        for j in range(ns):
            si = s0 + j
            ps = (S_all[si] * P) / (E_i[i] + Dp_i[si])
            if ps < 0.0:
                ps = 0.0
            y[base + j] = ps

    # enforce non-negativity globally
    np.maximum(y, 0.0, out=y)

    if not verify_with_rhs:
        return y

    dy = np.zeros_like(y)
    distributive_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                     offset_y, offset_s, n_sites)
    return y, dy


# -----------------------------
# Steady-state for Sequential
# (sets ALL params to 1 internally)
# -----------------------------
def steady_state_sequential(idx, TF_inputs=None, tf_scale=1.0, verify_with_rhs=False):
    """
    Returns y_ss (and optionally dy(y_ss)) for sequential model with ALL params = 1.
    Needs idx fields: N, offset_y, offset_s, n_sites
    Uses your sequential_rhs for optional verification.
    """
    N = int(getattr(idx, "N", len(idx.n_sites)))
    offset_y = np.asarray(idx.offset_y, dtype=np.int32)
    offset_s = np.asarray(idx.offset_s, dtype=np.int32)
    n_sites = np.asarray(idx.n_sites, dtype=np.int32)

    if TF_inputs is None:
        TF_inputs = np.zeros(N, dtype=np.float64)
    else:
        TF_inputs = np.asarray(TF_inputs, dtype=np.float64)

    total_s = _infer_total_s(offset_s, n_sites)
    total_y = _infer_total_y_distrib_seq(offset_y, n_sites)

    # ALL params = 1
    A_i = np.ones(N, dtype=np.float64)
    B_i = np.ones(N, dtype=np.float64)
    C_i = np.ones(N, dtype=np.float64)
    D_i = np.ones(N, dtype=np.float64)
    E_i = np.ones(N, dtype=np.float64)
    S_all = np.ones(total_s, dtype=np.float64)  # k_j
    Dp_i = np.ones(total_s, dtype=np.float64)  # dp_j

    y = np.zeros(total_y, dtype=np.float64)

    for i in range(N):
        y0 = int(offset_y[i])
        s0 = int(offset_s[i])
        ns = int(n_sites[i])

        u = _tf_squash(float(TF_inputs[i]))
        synth = A_i[i] * (1.0 + tf_scale * u)
        if synth < 0.0:
            synth = 0.0

        R = synth / B_i[i]
        y[y0 + 0] = R

        if ns == 0:
            P0 = (C_i[i] * R) / D_i[i]
            y[y0 + 1] = max(P0, 0.0)
            continue

        # Solve tridiagonal for x = [P0, P1, ..., Pns], size ns+1
        n = ns + 1
        a = np.zeros(n, dtype=np.float64)  # lower (a[0]=0)
        b = np.zeros(n, dtype=np.float64)  # main
        c = np.zeros(n, dtype=np.float64)  # upper
        d = np.zeros(n, dtype=np.float64)  # rhs

        E = float(E_i[i])
        D = float(D_i[i])

        # P0 eq: (D + k0) P0 - E P1 = C R
        k0 = float(S_all[s0 + 0])
        b[0] = D + k0
        c[0] = -E
        d[0] = float(C_i[i]) * R

        # Middle eqs j=1..ns-1:
        # (k_j + E + dp_{j-1}) Pj - k_{j-1} P_{j-1} - E P_{j+1} = 0
        for j in range(1, ns):
            k_prev = float(S_all[s0 + (j - 1)])
            k_j = float(S_all[s0 + j])
            dp_jm1 = float(Dp_i[s0 + (j - 1)])

            a[j] = -k_prev
            b[j] = (k_j + E + dp_jm1)
            c[j] = -E
            d[j] = 0.0

        # Last eq j=ns:
        # (E + dp_{ns-1}) Pns - k_{ns-1} P_{ns-1} = 0
        k_last = float(S_all[s0 + (ns - 1)])
        dp_last = float(Dp_i[s0 + (ns - 1)])
        a[ns] = -k_last
        b[ns] = (E + dp_last)
        c[ns] = 0.0
        d[ns] = 0.0

        x = _thomas_tridiag(a, b, c, d)

        # Fill y
        y[y0 + 1] = max(x[0], 0.0)  # P0
        base = y0 + 2
        for j in range(ns):
            y[base + j] = max(x[j + 1], 0.0)

    np.maximum(y, 0.0, out=y)

    if not verify_with_rhs:
        return y

    dy = np.zeros_like(y)
    sequential_rhs(y, dy, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale, TF_inputs, S_all,
                   offset_y, offset_s, n_sites)
    return y, dy


# -----------------------------
# Steady-state for Combinatorial
# (sets ALL params to 1 internally)
# -----------------------------
def steady_state_combinatorial(
        idx,
        TF_inputs=None,
        jb=0,
        tf_scale=1.0,
        max_states_per_protein=4096,
        trans_from=None, trans_to=None, trans_site=None, trans_off=None, trans_n=None,
        verify_with_rhs=False,
):
    """
    Returns y_ss (and optionally dy(y_ss)) for combinatorial model with ALL params = 1.

    Needs idx fields:
      N, offset_y, offset_s, n_sites, n_states
    For rhs verification you also need transition arrays (or we build them if not provided):
      trans_from, trans_to, trans_site, trans_off, trans_n
    """
    N = int(getattr(idx, "N", len(idx.n_sites)))
    offset_y = np.asarray(idx.offset_y, dtype=np.int32)
    offset_s = np.asarray(idx.offset_s, dtype=np.int32)
    n_sites = np.asarray(idx.n_sites, dtype=np.int32)

    # n_states must exist (don’t assume 2**ns if ns is large)
    if hasattr(idx, "n_states"):
        n_states = np.asarray(idx.n_states, dtype=np.int32)
    else:
        # fallback ONLY if you truly use full 2**ns
        n_states = (1 << n_sites.astype(np.int64)).astype(np.int32)

    if TF_inputs is None:
        TF_inputs = np.zeros(N, dtype=np.float64)
    else:
        TF_inputs = np.asarray(TF_inputs, dtype=np.float64)

    total_s = _infer_total_s(offset_s, n_sites)
    total_y = _infer_total_y_comb(offset_y, n_states)

    # ALL params = 1
    A_i = np.ones(N, dtype=np.float64)
    B_i = np.ones(N, dtype=np.float64)
    C_i = np.ones(N, dtype=np.float64)
    D_i = np.ones(N, dtype=np.float64)
    E_i = np.ones(N, dtype=np.float64)
    Dp_i = np.ones(total_s, dtype=np.float64)

    # S_cache[site, jb] accessed; with params=1 it’s all ones.
    # Make at least jb+1 columns.
    S_cache = np.ones((total_s, int(jb) + 1), dtype=np.float64)

    y = np.zeros(total_y, dtype=np.float64)

    for i in range(N):
        y0 = int(offset_y[i])
        s0 = int(offset_s[i])
        ns = int(n_sites[i])
        nst = int(n_states[i])

        if nst > max_states_per_protein:
            raise ValueError(
                f"Protein {i}: n_states={nst} exceeds max_states_per_protein={max_states_per_protein}. "
                f"Increase the cap or don’t request combinatorial SS for huge state spaces."
            )

        u = _tf_squash(float(TF_inputs[i]))
        synth = A_i[i] * (1.0 + tf_scale * u)
        if synth < 0.0:
            synth = 0.0

        R = synth / B_i[i]
        y[y0 + 0] = R

        # Solve A P + b = 0  for P over masks (size nst)
        # b[0] = C*R, others 0
        b = np.zeros(nst, dtype=np.float64)
        b[0] = float(C_i[i]) * R

        A = np.zeros((nst, nst), dtype=np.float64)

        Di = float(D_i[i])
        Ei = float(E_i[i])

        # Preload per-site rates for this protein (all ones here, but keep structure)
        # phosphorylation rate for site j is S_cache[s0+j, jb]
        S = S_cache[s0: s0 + ns, int(jb)] if ns > 0 else np.zeros(0, dtype=np.float64)
        Dp = Dp_i[s0: s0 + ns] if ns > 0 else np.zeros(0, dtype=np.float64)

        for frm in range(nst):
            # decay on mask 0 only (as in rhs)
            if frm == 0:
                A[frm, frm] -= Di

            # dephosph + per-site decay for set bits
            mm = frm
            while mm != 0:
                lsb = mm & -mm
                mm -= lsb
                j = _bit_index_from_lsb_py(int(lsb))
                to = frm ^ lsb

                # dephosph transition frm -> to at rate Ei
                A[frm, frm] -= Ei
                A[to, frm] += Ei

                # per-site decay sink for that set bit
                A[frm, frm] -= float(Dp[j])

            # phosphorylation transitions for unset bits
            for j in range(ns):
                bit = 1 << j
                if (frm & bit) == 0:
                    to = frm | bit
                    rate = float(S[j])
                    A[frm, frm] -= rate
                    A[to, frm] += rate

        # Solve A P = -b
        rhs = -b
        try:
            P = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # fallback
            P, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        P = np.maximum(P, 0.0)

        # Fill y: R at y0, then masks at y0+1..y0+nst
        y[y0 + 1: y0 + 1 + nst] = P

    np.maximum(y, 0.0, out=y)

    if not verify_with_rhs:
        return y

    # For verification we need transitions arrays in rhs call
    if trans_from is None:
        trans_from, trans_to, trans_site, trans_off, trans_n = build_random_transitions(idx)
    else:
        trans_from = np.asarray(trans_from, dtype=np.int32)
        trans_to = np.asarray(trans_to, dtype=np.int32)
        trans_site = np.asarray(trans_site, dtype=np.int32)
        trans_off = np.asarray(trans_off, dtype=np.int32)
        trans_n = np.asarray(trans_n, dtype=np.int32)

    dy = np.zeros_like(y)
    combinatorial_rhs(
        y, dy,
        A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale,
        TF_inputs, S_cache, int(jb),
        offset_y, offset_s,
        n_sites, n_states,
        trans_from, trans_to, trans_site, trans_off, trans_n
    )
    return y, dy