import numpy as np
import pandas as pd
from alive_progress.animations.spinner_compiler import sequential

from phoskintime_global.buildmat import site_key
from phoskintime_global.config import TIME_POINTS_PROTEIN, MODEL
from phoskintime_global.models import distributive_rhs, build_random_transitions, sequential_rhs, combinatorial_rhs


class Index:
    def __init__(self, interactions: pd.DataFrame):
        self.proteins = sorted(interactions["protein"].unique().tolist())
        self.p2i = {p: i for i, p in enumerate(self.proteins)}
        self.sites = [
            sorted(
                interactions.loc[interactions["protein"] == p, "psite"]
                .dropna()
                .unique()
                .tolist(),
                key=site_key
            )
            for p in self.proteins
        ]
        if MODEL == 2:
            self.n_states = np.array([1 << int(ns) for ns in self.n_sites], dtype=np.int32)
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        self.k2i = {k: i for i, k in enumerate(self.kinases)}
        self.N = len(self.proteins)

        self.n_sites = np.array([len(s) for s in self.sites], dtype=np.int32)
        self.offset_y = np.zeros(self.N, dtype=np.int32)
        self.offset_s = np.zeros(self.N, dtype=np.int32)

        curr_y = 0
        curr_s = 0
        for i in range(self.N):
            self.offset_y[i] = curr_y
            self.offset_s[i] = curr_s
            if MODEL == 2:
                curr_y += 1 + self.n_sites[i]
            else:
                curr_y += 2 + self.n_sites[i]
            curr_s += self.n_sites[i]

        self.state_dim = curr_y
        print(f"[Model] {self.N} proteins, {len(self.kinases)} kinases, {self.state_dim} state variables.")

    def block(self, i: int) -> slice:
        start = self.offset_y[i]
        end = start + 2 + self.n_sites[i]
        return slice(start, end)


class KinaseInput:
    def __init__(self, kinases, df_fc):
        self.grid = TIME_POINTS_PROTEIN
        self.Kmat = np.ones((len(kinases), len(self.grid)), float)
        if not df_fc.empty:
            for i, k in enumerate(kinases):
                sub = df_fc[df_fc["protein"] == k]
                if not sub.empty:
                    mp_fc = dict(zip(sub["time"], sub["fc"]))
                    for j, t in enumerate(self.grid):
                        if t in mp_fc:
                            self.Kmat[i, j] = max(mp_fc[t], 1e-6)

    def eval(self, t):
        if t <= self.grid[0]:
            return self.Kmat[:, 0]
        if t >= self.grid[-1]:
            return self.Kmat[:, -1]
        j = int(np.searchsorted(self.grid, t, side="right") - 1)
        return self.Kmat[:, j]


class System:
    def __init__(self, idx, W_global, tf_mat, kin_input, defaults, tf_deg):
        self.idx = idx
        self.W_global = W_global
        self.tf_mat = tf_mat
        self.kin = kin_input

        if MODEL != 2:
            self.p_indices = self.idx.offset_y + 1

        self.c_k = defaults["c_k"]
        self.A_i = defaults["A_i"]
        self.B_i = defaults["B_i"]
        self.C_i = defaults["C_i"]
        self.D_i = defaults["D_i"]
        self.E_i = defaults["E_i"]
        self.tf_scale = defaults["tf_scale"]
        # --- CSR buffers for njit RHS ---
        W = self.W_global.tocsr()
        self.W_indptr = W.indptr.astype(np.int32)
        self.W_indices = W.indices.astype(np.int32)
        self.W_data = W.data.astype(np.float64)
        self.n_W_rows = W.shape[0]

        TF = self.tf_mat.tocsr()
        self.TF_indptr = TF.indptr.astype(np.int32)
        self.TF_indices = TF.indices.astype(np.int32)
        self.TF_data = TF.data.astype(np.float64)
        self.n_TF_rows = TF.shape[0]  # == idx.N

        # Kinase input arrays
        self.kin_grid = np.asarray(self.kin.grid, dtype=np.float64)
        self.kin_Kmat = np.asarray(self.kin.Kmat, dtype=np.float64)

        # p_indices must be int32 for njit indexing
        if MODEL != 2:
            self.p_indices = self.p_indices.astype(np.int32)
        # Degree of target TFs
        self.tf_deg = tf_deg

        # Random transitions
        if MODEL == 2:
            (self.trans_from,
             self.trans_to,
             self.trans_site,
             self.trans_off,
             self.trans_n) = build_random_transitions(idx)

    def update(self, c_k, A_i, B_i, C_i, D_i, E_i, tf_scale):
        self.c_k = c_k
        self.A_i = A_i
        self.B_i = B_i
        self.C_i = C_i
        self.D_i = D_i
        self.E_i = E_i
        self.tf_scale = tf_scale

    def rhs(self, t, y):
        dy = np.zeros_like(y)

        Kt = self.kin.eval(t) * self.c_k
        S_all = self.W_global.dot(Kt)
        P_vec = np.zeros(self.idx.N, dtype=np.float64)

        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            if MODEL == 2:
                ns = int(self.idx.n_sites[i])
                tot = 0.0
                for m in range(ns):
                    tot += y[st + 1 + m]
            else:
                ns = self.idx.n_sites[i]
                tot = y[st + 1]
                if ns > 0:
                    tot += y[st + 2: st + 2 + ns].sum()
            P_vec[i] = tot

        TF_inputs = self.tf_mat.dot(P_vec)
        TF_inputs = TF_inputs / self.tf_deg

        if MODEL == 0:
            distributive_rhs(
                y, dy,
                self.A_i, self.B_i, self.C_i, self.D_i, self.E_i, self.tf_scale,
                TF_inputs, S_all,
                self.idx.offset_y, self.idx.offset_s, self.idx.n_sites
            )
        elif MODEL == 1:
            sequential_rhs(
                y, dy,
                self.A_i, self.B_i, self.C_i, self.D_i, self.E_i, self.tf_scale,
                TF_inputs, S_all,
                self.idx.offset_y, self.idx.offset_s, self.idx.n_sites
            )
        elif MODEL == 2:
            combinatorial_rhs(
                y, dy,
                self.A_i, self.B_i, self.C_i, self.D_i, self.E_i, self.tf_scale,
                TF_inputs, S_all,
                self.idx.offset_y, self.idx.offset_s, self.idx.n_sites,
                self.trans_from, self.trans_to, self.trans_site, self.trans_off, self.trans_n
            )
        return dy

    def y0(self):
        y = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            y[st] = 1.0
            if MODEL != 2:
                y[st + 1 + 0] = 1.0
                ns = int(self.idx.n_sites[i])
                if ns > 1:
                    y[st + 1 + 1: st + 1 + ns] = 0.01
            else:
                y[st + 1] = 1.0
                ns = self.idx.n_sites[i]
                if ns > 0:
                    y[st + 2: st + 2 + ns] = 0.01
        return y

    def odeint_args(self, S_cache=None, jb=None):
        """
        Prepare arguments for njit-compiled ODE integrator.

        For MODEL != 2: returns args for rhs_nb_distributive / rhs_nb_sequential style.
        For MODEL == 2: returns args for combinatorial_rhs (expects S_cache and jb).
        """
        if MODEL == 2:
            if S_cache is None:
                raise ValueError("MODEL==2 requires S_cache (e.g., precomputed site-rate cache).")
            if jb is None:
                raise ValueError("MODEL==2 requires jb (time-bin index into S_cache).")

            # For Combinatorial model
            return (
                self.c_k.astype(np.float64),
                self.A_i.astype(np.float64),
                self.B_i.astype(np.float64),
                self.C_i.astype(np.float64),
                self.D_i.astype(np.float64),
                self.E_i.astype(np.float64),
                float(self.tf_scale),

                self.kin_grid,
                np.asarray(S_cache, dtype=np.float64),
                int(jb),

                self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),

                self.idx.offset_y.astype(np.int32),
                self.idx.offset_s.astype(np.int32),
                self.idx.n_sites.astype(np.int32),
                self.idx.n_states.astype(np.int32),

                self.trans_from,
                self.trans_to,
                self.trans_site,
                self.trans_off,
                self.trans_n,

                self.tf_deg,
            )

        # For Distributive and Sequential model
        return (
            self.c_k.astype(np.float64),
            self.A_i.astype(np.float64),
            self.B_i.astype(np.float64),
            self.C_i.astype(np.float64),
            self.D_i.astype(np.float64),
            self.E_i.astype(np.float64),
            float(self.tf_scale),

            self.kin_grid,
            self.kin_Kmat,

            self.W_indptr, self.W_indices, self.W_data, int(self.n_W_rows),

            self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),

            self.p_indices,

            self.idx.offset_y.astype(np.int32),
            self.idx.offset_s.astype(np.int32),
            self.idx.n_sites.astype(np.int32),

            self.tf_deg,
        )
