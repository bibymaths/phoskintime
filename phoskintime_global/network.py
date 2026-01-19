import numpy as np
import pandas as pd

from phoskintime_global.buildmat import site_key
from phoskintime_global.config import TIME_POINTS_PROTEIN, MODEL
from phoskintime_global.models import distributive_rhs, build_random_transitions, sequential_rhs, combinatorial_rhs
from phoskintime_global.steadystate import build_y0_from_data


class Index:
    def __init__(self, interactions: pd.DataFrame, tf_interactions: pd.DataFrame = None):
        # Start with proteins from Kinase interactions (MS data/Kinase net)
        prots = set(interactions["protein"].unique())

        # --- Add TFs and Targets from TF Network ---
        if tf_interactions is not None:
            # Add Source TFs
            if "tf" in tf_interactions.columns:
                prots.update(tf_interactions["tf"].unique())
            # Add Target Genes (mRNA)
            if "target" in tf_interactions.columns:
                prots.update(tf_interactions["target"].unique())

        # Sort and build map
        self.proteins = sorted(list(prots))
        self.p2i = {p: i for i, p in enumerate(self.proteins)}

        # Build sites list (robustly handle proteins with no sites/kinases)
        self.sites = []
        for p in self.proteins:
            # Check if protein exists in kinase interaction df
            sub = interactions.loc[interactions["protein"] == p, "psite"]
            if not sub.empty:
                s_list = sub.dropna().unique().tolist()
                self.sites.append(sorted(s_list, key=site_key))
            else:
                self.sites.append([])  # No phosphorylation sites for this TF/Protein

        if MODEL == 2:
            self.n_sites = np.array([len(s) for s in self.sites], dtype=np.int32)
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
                curr_y += 1 + self.n_states[i]
            else:
                curr_y += 2 + self.n_sites[i]
            curr_s += self.n_sites[i]

        self.state_dim = curr_y
        self.total_sites = int(curr_s)
        print(f"[Model] {self.N} proteins, {len(self.kinases)} kinases, {self.state_dim} state variables.")

    def block(self, i: int) -> slice:
        start = self.offset_y[i]
        if MODEL == 2:
            end = start + 1 + self.n_states[i]
        else:
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
        self._ic_data = None
        self.idx = idx
        self.W_global = W_global
        self.tf_mat = tf_mat
        self.kin = kin_input
        self.S_cache = None
        self.custom_y0 = None
        # ------------------------------------------------------------
        # Parameters (force dtype/contiguity ONCE)
        # ------------------------------------------------------------
        self.c_k = np.ascontiguousarray(defaults["c_k"], dtype=np.float64)
        self.A_i = np.ascontiguousarray(defaults["A_i"], dtype=np.float64)
        self.B_i = np.ascontiguousarray(defaults["B_i"], dtype=np.float64)
        self.C_i = np.ascontiguousarray(defaults["C_i"], dtype=np.float64)
        self.D_i = np.ascontiguousarray(defaults["D_i"], dtype=np.float64)
        self.Dp_i = np.ascontiguousarray(defaults["Dp_i"], dtype=np.float64)
        self.E_i = np.ascontiguousarray(defaults["E_i"], dtype=np.float64)
        self.tf_scale = float(defaults["tf_scale"])

        # ------------------------------------------------------------
        # Offsets / sizes (force int32 ONCE)
        # ------------------------------------------------------------
        self.idx.offset_y = np.ascontiguousarray(self.idx.offset_y, dtype=np.int32)
        self.idx.offset_s = np.ascontiguousarray(self.idx.offset_s, dtype=np.int32)
        self.idx.n_sites = np.ascontiguousarray(self.idx.n_sites, dtype=np.int32)

        if MODEL == 2:
            self.idx.n_states = np.ascontiguousarray(self.idx.n_states, dtype=np.int32)
        else:
            # used only in non-combinatorial models
            self.p_indices = np.ascontiguousarray(self.idx.offset_y + 1, dtype=np.int32)

        # ------------------------------------------------------------
        # CSR buffers for njit RHS (W matrix)
        # ------------------------------------------------------------
        W = self.W_global.tocsr()
        self.W_indptr = W.indptr.astype(np.int32, copy=False)
        self.W_indices = W.indices.astype(np.int32, copy=False)
        self.W_data = W.data.astype(np.float64, copy=False)
        self.n_W_rows = W.shape[0]

        # ------------------------------------------------------------
        # CSR buffers for TF matrix
        # ------------------------------------------------------------
        TF = self.tf_mat.tocsr()
        self.TF_indptr = TF.indptr.astype(np.int32, copy=False)
        self.TF_indices = TF.indices.astype(np.int32, copy=False)
        self.TF_data = TF.data.astype(np.float64, copy=False)
        self.n_TF_rows = TF.shape[0]  # == idx.N

        # ------------------------------------------------------------
        # TF degree (force dtype once)
        # ------------------------------------------------------------
        self.tf_deg = np.ascontiguousarray(tf_deg, dtype=np.float64)

        # ------------------------------------------------------------
        # Kinase input arrays (read-only, contiguous)
        # ------------------------------------------------------------
        self.kin_grid = np.ascontiguousarray(self.kin.grid, dtype=np.float64)
        self.kin_Kmat = np.ascontiguousarray(self.kin.Kmat, dtype=np.float64)

        # ------------------------------------------------------------
        # MODEL == 2 specific setup
        # ------------------------------------------------------------
        if MODEL == 2:
            # reusable work buffers (NO allocs in RHS)
            self.P_vec_work = np.zeros(self.n_TF_rows, dtype=np.float64)
            self.TF_in_work = np.zeros(self.n_TF_rows, dtype=np.float64)
            self.S_cache = np.zeros((self.n_W_rows, self.kin_Kmat.shape[1]), dtype=np.float64)

            # precomputed transition lists
            (
                self.trans_from,
                self.trans_to,
                self.trans_site,
                self.trans_off,
                self.trans_n,
            ) = build_random_transitions(idx)

    def update(self, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale):
        self.c_k[:] = c_k
        self.A_i[:] = A_i
        self.B_i[:] = B_i
        self.C_i[:] = C_i
        self.D_i[:] = D_i
        self.Dp_i[:] = Dp_i
        self.E_i[:] = E_i
        self.tf_scale = float(tf_scale)

    def attach_initial_condition_data(self, df_prot, df_rna, df_pho):
        if self._ic_data is None:
            raise RuntimeError("Initial-condition data already attached to System")
        self._ic_data = dict(
            df_prot=df_prot,
            df_rna=df_rna,
            df_pho=df_pho
        )

    def set_initial_conditions(self):
        if self._ic_data is None:
            raise RuntimeError(
                "Initial-condition data not attached. Call sys.attach_initial_condition_data(df_prot, df_rna, df_pho) before optimize."
            )

        self.custom_y0 = build_y0_from_data(
            self.idx,
            self._ic_data["df_prot"],
            self._ic_data["df_rna"],
            self._ic_data["df_pho"],
            t_init=0.0,
            t0_prot=0.0,
            t0_rna=4.0,
            t0_pho=0.0,
        )

    def rhs(self, t, y):
        dy = np.zeros_like(y)

        Kt = self.kin.eval(t) * self.c_k
        S_all = self.W_global.dot(Kt)
        P_vec = np.zeros(self.idx.N, dtype=np.float64)

        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            if MODEL == 2:
                ns = int(self.idx.n_states[i])
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
                self.A_i, self.B_i, self.C_i, self.D_i, self.Dp_i, self.E_i, self.tf_scale,
                TF_inputs, S_all,
                self.idx.offset_y, self.idx.offset_s, self.idx.n_sites
            )
        elif MODEL == 1:
            sequential_rhs(
                y, dy,
                self.A_i, self.B_i, self.C_i, self.D_i, self.Dp_i, self.E_i, self.tf_scale,
                TF_inputs, S_all,
                self.idx.offset_y, self.idx.offset_s, self.idx.n_sites
            )
        elif MODEL == 2:

            if self.S_cache is None:
                raise ValueError("MODEL==2: System.S_cache is None. simulate_odeint must set it.")

            jb = int(np.searchsorted(self.kin_grid, t, side="right") - 1)
            if jb < 0:
                jb = 0
            elif jb >= self.kin_grid.size:
                jb = self.kin_grid.size - 1

            combinatorial_rhs(
                y, dy,
                self.A_i, self.B_i, self.C_i, self.D_i, self.Dp_i, self.E_i, self.tf_scale,
                TF_inputs,
                self.S_cache, jb,
                self.idx.offset_y, self.idx.offset_s,
                self.idx.n_sites, self.idx.n_states,
                self.trans_from, self.trans_to, self.trans_site, self.trans_off, self.trans_n
            )
        return dy

    def y0(self) -> np.ndarray:

        if getattr(self, "custom_y0", None) is not None:
            return np.array(self.custom_y0, dtype=np.float64, copy=True)

        y = np.zeros(self.idx.state_dim, float)
        for i in range(self.idx.N):
            st = self.idx.offset_y[i]
            y[st] = 1.0
            if MODEL == 2:
                y[st + 1 + 0] = 1.0
                ns = int(self.idx.n_states[i])
                if ns > 1:
                    y[st + 1 + 1: st + 1 + ns] = 0.01
            else:
                y[st + 1] = 1.0
                ns = self.idx.n_sites[i]
                if ns > 0:
                    y[st + 2: st + 2 + ns] = 0.01
        return y

    def odeint_args(self, S_cache=None):
        if MODEL == 2:
            if S_cache is None:
                raise ValueError("MODEL==2 requires S_cache (total_sites x n_timebins).")

            return (
                self.c_k,
                self.A_i,
                self.B_i,
                self.C_i,
                self.D_i,
                self.Dp_i,
                self.E_i,
                float(self.tf_scale),

                self.kin_grid,
                S_cache,

                self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),

                self.idx.offset_y,
                self.idx.offset_s,
                self.idx.n_sites,
                self.idx.n_states,

                self.trans_from,
                self.trans_to,
                self.trans_site,
                self.trans_off,
                self.trans_n,

                self.tf_deg,
                self.P_vec_work,
                self.TF_in_work
            )

        # For Distributive and Sequential model
        return (
            self.c_k,
            self.A_i,
            self.B_i,
            self.C_i,
            self.D_i,
            self.Dp_i,
            self.E_i,
            float(self.tf_scale),
            self.kin_grid,
            self.kin_Kmat,
            self.W_indptr, self.W_indices, self.W_data, int(self.n_W_rows),
            self.TF_indptr, self.TF_indices, self.TF_data, int(self.n_TF_rows),
            self.idx.offset_y,
            self.idx.offset_s,
            self.idx.n_sites,
            self.tf_deg,
        )
