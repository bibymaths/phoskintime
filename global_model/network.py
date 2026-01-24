"""
System State and Topology Management Module.

This module defines the core data structures that represent the biological system:
1.  **Index:** Manages the mapping between biological names (Proteins, Sites) and
    numerical indices in the state vector. It handles the complex logic of
    "Proxy Redirection" for orphan Transcription Factors.
2.  **KinaseInput:** Interpolates experimental kinase data onto the simulation time grid.
3.  **System:** The central container holding all parameters, sparse matrices (CSR),
    and logic required to evaluate the differential equations. It acts as the bridge
    between high-level Python objects and the low-level Numba JIT kernels.


"""

import numpy as np
import pandas as pd

from global_model.buildmat import site_key
from global_model.config import TIME_POINTS_PROTEIN, MODEL, RESULTS_DIR
from global_model.models import distributive_rhs, build_random_transitions, sequential_rhs, combinatorial_rhs
from global_model.steadystate import build_y0_from_data
from config.config import setup_logger

logger = setup_logger(log_dir=RESULTS_DIR)


class Index:
    """
    Manages the indexing of the state vector Y and network topology.

    The state vector Y is a flattened array containing all species.
    For $N$ proteins, the layout depends on the MODEL type:



    - **Distributive/Sequential:** [mRNA_i, Prot_i, Site_i_1, Site_i_2, ...]
    - **Combinatorial:** [mRNA_i, Prot_state_0, Prot_state_1, ..., Prot_state_2^n]
    """

    def __init__(self,
                 interactions: pd.DataFrame,
                 tf_interactions: pd.DataFrame = None,
                 kin_beta_map: dict = None,
                 tf_beta_map: dict = None):
        """
        Index manager for the PhoskinTime ODE system.
        Handles protein/kinase mapping and redirects Orphan TFs to Kinase Proxies.

        Args:
            interactions: Cleaned Kinase-Substrate network (columns: protein, psite, kinase).
            tf_interactions: Cleaned TF-Target network (columns: tf, target).
            kin_beta_map: Optional dictionary of optimized kinase priors {name: beta_val}.
            tf_beta_map: Optional dictionary of optimized TF priors {name: beta_val}.
        """
        # --- 1. Basic Protein/Target Discovery ---
        prots = set(interactions["protein"].unique())

        if tf_interactions is not None:
            if "tf" in tf_interactions.columns:
                prots.update(tf_interactions["tf"].unique())
            if "target" in tf_interactions.columns:
                prots.update(tf_interactions["target"].unique())

        # Alphabetical sort ensures consistent indexing across runs
        self.proteins = sorted(list(prots))

        # Initial mapping: Name -> Unique Integer Index
        self.p2i = {p: i for i, p in enumerate(self.proteins)}

        # List of all unique kinases in the signaling layer
        self.kinases = sorted(interactions["kinase"].unique().tolist())
        self.k2i = {k: i for i, k in enumerate(self.kinases)}

        # --- 2. Orphan TF Redirection (The Proxy Update) ---
        # 
        # "Orphan TFs" are regulators with no upstream kinase in the data.
        # To avoid them being static, we assume feedback: if Orphan A regulates Kinase B,
        # we treat A's activity as proportional to B (A is a "Proxy" for B).
        proxy_map = {}
        if tf_interactions is not None:
            # Orphans = TFs in the regulatory network that have NO sites in the signaling data
            proteins_with_sites = set(interactions["protein"].unique())
            all_tfs = set(tf_interactions["tf"].unique())
            orphan_tfs = all_tfs - proteins_with_sites

            for orphan in orphan_tfs:
                # Find targets of this orphan that are also known Kinases (Feedback Targets)
                targets = tf_interactions.loc[tf_interactions["tf"] == orphan, "target"]
                feedback_kinases = [t for t in targets if t in self.kinases]

                if feedback_kinases:
                    # Choose best proxy based on optimized Beta weights if available
                    best_proxy = feedback_kinases[0]
                    max_weight = -1.0

                    for k in feedback_kinases:
                        # Weighting: Prioritize TF-specific activity or overall Kinase activity
                        weight = tf_beta_map.get(orphan, 0.0) if tf_beta_map else 0.0
                        if kin_beta_map and k in kin_beta_map:
                            weight += kin_beta_map[k]

                        if weight > max_weight:
                            max_weight = weight
                            best_proxy = k

                    # REDIRECTION: Hijack the index mapping.
                    # This orphan now 'shares' the signaling state of its proxy kinase.
                    self.p2i[orphan] = self.p2i[best_proxy]
                    proxy_map[orphan] = best_proxy
                    logger.info(f"[Proxy] Redirected Orphan {orphan} -> {best_proxy} (Beta Weight: {max_weight:.2f})")

        self.proxy_map = proxy_map

        # --- 3. Build Site and State Offsets ---
        self.N = len(self.proteins)
        self.sites = []

        for p in self.proteins:
            # Build sites list for each protein (orphans will naturally have empty lists here)
            sub = interactions.loc[interactions["protein"] == p, "psite"]
            if not sub.empty:
                s_list = sub.dropna().unique().tolist()
                self.sites.append(sorted(s_list, key=site_key))
            else:
                self.sites.append([])

        # State and site counters
        self.n_sites = np.array([len(s) for s in self.sites], dtype=np.int32)

        if MODEL == 2:
            self.n_states = np.array([1 << int(ns) for ns in self.n_sites], dtype=np.int32)

        # Offsets map standard indices to the flattened y-vector
        self.offset_y = np.zeros(self.N, dtype=np.int32)
        self.offset_s = np.zeros(self.N, dtype=np.int32)

        curr_y = 0
        curr_s = 0
        for i in range(self.N):
            self.offset_y[i] = curr_y
            self.offset_s[i] = curr_s

            if MODEL == 2:
                # Combinatorial model states: 1 (mRNA) + 2^n (Protein states)
                curr_y += 1 + self.n_states[i]
            else:
                # Distributive/Sequential: 1 (mRNA) + 1 (Total Protein) + n (Phospho-sites)
                curr_y += 2 + self.n_sites[i]
            curr_s += self.n_sites[i]

        self.state_dim = curr_y
        self.total_sites = int(curr_s)
        self.kinase_indices_in_P = [self.p2i[k] for k in self.kinases if k in self.p2i]
        self.p2k = {k: i for i, k in enumerate(self.kinases)}

        logger.info(f"[Model] {self.N} proteins ({len(proxy_map)} orphans rewired), "
                    f"{len(self.kinases)} kinases, {self.state_dim} state variables.")

    def block(self, i: int) -> slice:
        """Helper to get the range in the state vector for protein i."""
        start = self.offset_y[i]
        if MODEL == 2:
            end = start + 1 + self.n_states[i]
        else:
            end = start + 2 + self.n_sites[i]
        return slice(start, end)


class KinaseInput:
    """
    Manages the external Kinase signal inputs $K(t)$.
    Interpolates sparse experimental observations onto the solver's dense time grid.
    """

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
                            # Clamp to small epsilon to prevent zeros
                            self.Kmat[i, j] = max(mp_fc[t], 1e-6)

    def eval(self, t):
        """Returns the kinase activity vector at time t (using step interpolation)."""
        if t <= self.grid[0]:
            return self.Kmat[:, 0]
        if t >= self.grid[-1]:
            return self.Kmat[:, -1]
        j = int(np.searchsorted(self.grid, t, side="right") - 1)
        return self.Kmat[:, j]


class System:
    """
    The central Simulation Object.

    Holds:
    1.  Parameters (Arrays $A_i, B_i, \dots$).
    2.  Network Topology Matrices (Sparse CSR format).
    3.  Initial Conditions logic.
    4.  The `rhs` method (Python-side derivative calculation).
    5.  Argument packing logic for Numba JIT kernels.
    """

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
        # Numba is faster with C-contiguous float64 arrays
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
        # CSR buffers for njit RHS (W matrix: Kinase -> Site)
        # ------------------------------------------------------------
        # 
        W = self.W_global.tocsr()
        self.W_indptr = W.indptr.astype(np.int32, copy=False)
        self.W_indices = W.indices.astype(np.int32, copy=False)
        self.W_data = W.data.astype(np.float64, copy=False)
        self.n_W_rows = W.shape[0]

        # ------------------------------------------------------------
        # CSR buffers for TF matrix (Regulator -> Gene)
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

            # precomputed transition lists for the combinatorial hypercube graph
            (
                self.trans_from,
                self.trans_to,
                self.trans_site,
                self.trans_off,
                self.trans_n,
            ) = build_random_transitions(idx)

    def update(self, c_k, A_i, B_i, C_i, D_i, Dp_i, E_i, tf_scale):
        """Updates the system parameters from the optimizer."""
        self.c_k[:] = c_k
        self.A_i[:] = A_i
        self.B_i[:] = B_i
        self.C_i[:] = C_i
        self.D_i[:] = D_i
        self.Dp_i[:] = Dp_i
        self.E_i[:] = E_i
        self.tf_scale = float(tf_scale)

    def attach_initial_condition_data(self, df_prot, df_rna, df_pho):
        """Attaches experimental data used to calculate t=0 state."""
        if self._ic_data is not None:
            raise RuntimeError("Initial-condition data already attached to System")
        self._ic_data = dict(
            df_prot=df_prot,
            df_rna=df_rna,
            df_pho=df_pho
        )
        logger.info("[Model] Initial condition data attached successfully.")

    def set_initial_conditions(self):
        """Derives the y0 vector from attached experimental data."""
        if self._ic_data is None:
            raise RuntimeError(
                "Initial-condition data not attached. Call sys.attach_initial_condition_data(df_prot, df_rna, df_pho) before optimize."
            )

        self.custom_y0 = build_y0_from_data(
            self.idx,
            self._ic_data["df_prot"],
            self._ic_data["df_rna"],
            self._ic_data["df_pho"]
        )

    def rhs(self, t, y):
        """
        Python-side Right-Hand Side evaluation.
        Used primarily for testing or when live interactivity/debugging is needed.

        Logic Flow:
        1.  **Live-Drive:** Calculate Kinase Activity $Kt = K_{data}(t) \times c_k$.
        2.  **Signaling:** Calculate Phospho-Drive $S = W \cdot Kt$.
        3.  **Protein Aggregation:** Sum phospho-states to get total protein $P$.
            *Crucially*, if a protein is a Kinase (or Proxy), overwrite its value with $Kt$.
        4.  **Regulation:** Calculate TF inputs $TF_{in} = TF_{mat} \cdot P$.
        5.  **Dynamics:** Call model-specific `_rhs` kernel.
        """
        dy = np.zeros_like(y)

        # 1. Observed Kinase Activity (The Driver)
        Kt = self.kin.eval(t) * self.c_k
        S_all = self.W_global.dot(Kt)

        P_vec = np.zeros(self.idx.N, dtype=np.float64)

        for i in range(self.idx.N):
            # --- Live-Drive for Kinases ---
            # If the protein is a kinase OR an orphan redirected to one,
            # we drive the mRNA using the high-fidelity Kinase trajectory.
            prot_name = self.idx.proteins[i]

            # Check if this name is in the proxy map or is a kinase itself
            if prot_name in self.idx.kinases or (hasattr(self.idx, 'proxy_map') and prot_name in self.idx.proxy_map):
                # Resolve the kinase name (either the protein itself or its proxy)
                k_name = prot_name if prot_name in self.idx.kinases else self.idx.proxy_map[prot_name]
                k_idx = self.idx.k2i[k_name]
                # Use the observed fold-change scaled by optimized multiplier c_k
                P_vec[i] = Kt[k_idx]
            else:
                # Standard logic: Use the predicted phosphorylation state
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

        # 2. Transcription Logic
        TF_inputs = self.tf_mat.dot(P_vec)
        # Squash inputs for stability (u / 1+|u|)
        for i in range(len(TF_inputs)):
            val = TF_inputs[i] / self.tf_deg[i]
            TF_inputs[i] = val / (1.0 + abs(val))

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
        """Returns the initial state vector y0."""
        if getattr(self, "custom_y0", None) is not None:
            return np.array(self.custom_y0, dtype=np.float64, copy=True)

        # Default fallback (rarely used if data attached)
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
        """
        Packs all system arrays into a tuple for the Numba JIT solver.

        This method constructs the `driver_map` array, which tells the low-level solver
        which proteins are actually Kinases (or Proxies) and should be "Driven" by
        experimental data rather than simulated.
        """
        # Create Driver Map for Numba
        # Maps Protein Index -> Kinase Index in Kt.
        # -1 means "Not driven" (use simulation).
        driver_map = np.full(self.idx.N, -1, dtype=np.int32)

        # 1. Map actual Kinases
        for k_name in self.idx.kinases:
            if k_name in self.idx.p2i:
                p_idx = self.idx.p2i[k_name]
                k_idx = self.idx.k2i[k_name]
                driver_map[p_idx] = k_idx

        # 2. Map Proxies (The Orphan Fix)
        if hasattr(self.idx, 'proxy_map'):
            for orphan, proxy in self.idx.proxy_map.items():
                if orphan in self.idx.p2i:
                    p_idx = self.idx.p2i[orphan]
                    k_idx = self.idx.k2i[proxy]
                    driver_map[p_idx] = k_idx

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
                driver_map,
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
            driver_map
        )