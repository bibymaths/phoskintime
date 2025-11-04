#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global ODE dual-fit (FC targets) with effects from 'Alpha Values' and 'Beta Values'.

Usage:
    python global_model.py \
      --interaction input2.csv \
      --kinopt kinopt_results.xlsx \
      --tfopt tfopt_results.xlsx \
      --stage 3 \
      --early-focus 1.0 \
      --l1-alpha 2e-3 --lambda-prior 1e-2 --lambda-c 1e-2 \
      --lambda-rna 1.0 \
      --output-dir out_dual_fc
"""

import argparse  # Import library for parsing command-line arguments
import json  # Import library for JSON encoding and decoding
import os  # Import library for interacting with the operating system (e.g., creating directories)
import re  # Import library for regular expression operations
import time  # Import library for time-related functions
from typing import Dict, List, Tuple, Optional  # Import type hinting for clarity

import numpy as np  # Import numerical computing library
import pandas as pd  # Import data analysis and manipulation library
from scipy.integrate import solve_ivp  # Import ODE solver from SciPy
from scipy import sparse  # Import sparse matrix functions from SciPy
from scipy.optimize import minimize  # Import optimization function from SciPy
from tqdm import tqdm  # Import library for progress bars

# Define global constant for protein time points
TIME_POINTS = np.array([0.0, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0], dtype=float)
# Define global constant for RNA time points
TIME_POINTS_RNA = np.array([4.0, 8.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0], dtype=float)


# ------------------------------
# Utils
# ------------------------------

def _normcols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes DataFrame column names to lowercase and replaces spaces with underscores."""
    df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    # Normalize column names: strip whitespace, convert to lowercase, replace spaces with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df  # Return the DataFrame with normalized column names


def _ensure_dir(p: str):
    """Ensures that a directory exists, creating it if necessary."""
    os.makedirs(p, exist_ok=True)  # Create the directory path; 'exist_ok=True' prevents an error if it already exists


def softplus(x: np.ndarray) -> np.ndarray:
    """Computes the softplus function, a smooth approximation of ReLU, ensuring non-negativity."""
    # Use 'np.where' for numerical stability: for large x, softplus(x) ≈ x
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def inv_softplus(y: np.ndarray) -> np.ndarray:
    """Computes the inverse of the softplus function, used to un-transform parameters."""
    # Ensure y is positive with a small floor (1e-12) for numerical stability
    return np.log(np.expm1(np.maximum(y, 1e-12)))


def _find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    """Finds the first matching column name from a list of candidates."""
    for c in cands:  # Iterate through the candidate column names
        if c in df.columns:  # If the candidate exists in the DataFrame's columns
            return c  # Return the matching column name
    return None  # If no match is found, return None


def _time_cols(df: pd.DataFrame) -> List[str]:
    """Identifies and sorts columns that represent numerical time points."""
    times = []  # Initialize a list to store (column_name, time_value) tuples
    for col in df.columns:  # Iterate through all column names
        m = re.findall(r"[-+]?\d*\.?\d+", str(col))  # Find all numbers in the column name
        if len(m) == 1:  # If exactly one number is found
            try:
                v = float(m[0])  # Try to convert the found number to a float
                times.append((col, v))  # Store the (column_name, float_value) pair
            except ValueError:
                pass  # Ignore columns where the number isn't a valid float
    times = sorted(times, key=lambda x: x[1])  # Sort the list based on the numerical time value
    return [c for c, _ in times]  # Return just the sorted list of column names


# ------------------------------
# Load interaction network
# ------------------------------

def load_interactions(path: str) -> pd.DataFrame:
    """Loads the kinase-substrate interaction network from a CSV file."""
    print(f"Loading interaction network from: {path}")
    df = pd.read_csv(path)  # Read the CSV file into a pandas DataFrame
    df = _normcols(df)  # Normalize the column names
    # Find the correct column names for protein, psite, and kinase using candidate lists
    pcol = _find_col(df, ["protein", "gene", "geneid", "prot"])
    scol = _find_col(df, ["psite", "site", "phosphosite", "residue"])
    kcol = _find_col(df, ["kinase", "kinases", "k", "enzyme"])
    if not (pcol and scol and kcol):  # Check if all required columns were found
        # Raise an error if any essential column is missing
        raise ValueError(f"Interaction file must have Protein, Psite, Kinase. Found: {df.columns.tolist()}")
    # Create a new DataFrame with standardized column names and data types
    out = pd.DataFrame({
        "protein": df[pcol].astype(str).str.strip(),  # Standardize 'protein' column
        "psite": df[scol].astype(str).str.strip(),  # Standardize 'psite' column
        "kinase": df[kcol].astype(str).str.strip(),  # Standardize 'kinase' column
    }).drop_duplicates()  # Remove any duplicate interaction entries
    print("... Interaction network loaded.")
    return out.reset_index(drop=True)  # Return the cleaned DataFrame with a reset index


# ------------------------------
# Effects loaders
# ------------------------------

def load_kinopt_effects(path: str):
    """Loads kinase alpha (interaction) and beta (activity) priors from the kinopt Excel file."""
    print(f"Loading kinopt effects from: {path}")
    xl = pd.ExcelFile(path)  # Open the Excel file
    # --- Alpha Values ---
    print("  -> Reading 'Alpha Values' sheet (kinopt)")
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")  # Read the 'Alpha Values' sheet
    dfA = _normcols(dfA)  # Normalize column names
    # Find columns for gene, psite, kinase, and alpha value
    gcol = _find_col(dfA, ["gene", "protein", "geneid"])
    scol = _find_col(dfA, ["psite", "site"])
    kcol = _find_col(dfA, ["kinase", "kinases", "k"])
    aval = _find_col(dfA, ["alpha", "value", "effect", "score", "weight"])
    if not (gcol and scol and kcol and aval):  # Check if all required columns are present
        raise ValueError("kinopt::Alpha Values must have (Gene, Psite, Kinase, Alpha).")
    # Select, rename, and clean the alpha data
    kin_alpha = dfA[[gcol, scol, kcol, aval]].rename(
        columns={gcol: "protein", scol: "psite", kcol: "kinase", aval: "alpha"})
    kin_alpha["protein"] = kin_alpha["protein"].astype(str).str.strip()
    kin_alpha["psite"] = kin_alpha["psite"].astype(str).str.strip()
    kin_alpha["kinase"] = kin_alpha["kinase"].astype(str).str.strip()
    kin_alpha["alpha"] = pd.to_numeric(kin_alpha["alpha"], errors="coerce").fillna(
        0.0)  # Convert alpha to numeric, fill NaNs with 0

    # --- Beta Values ---
    print("  -> Reading 'Beta Values' sheet (kinopt)")
    dfB = pd.read_excel(xl, sheet_name="Beta Values")  # Read the 'Beta Values' sheet
    dfB = _normcols(dfB)  # Normalize column names
    # Find columns for kinase, psite (optional), and beta value
    kcolB = _find_col(dfB, ["kinase", "kinases", "k"])
    scolB = _find_col(dfB, ["psite", "site"])
    bval = _find_col(dfB, ["beta", "value", "effect", "score", "weight"])
    if not (kcolB and bval):  # Check if required columns are present
        raise ValueError("kinopt::Beta Values must have (Kinase, [Psite], Beta).")
    # Select and rename columns; include 'psite' if it was found
    kin_beta = dfB[[kcolB, bval] + ([scolB] if scolB else [])].rename(
        columns={kcolB: "kinase", bval: "beta", (scolB or "psite"): "psite"})
    kin_beta["kinase"] = kin_beta["kinase"].astype(str).str.strip()  # Clean kinase names
    if "psite" in kin_beta.columns:  # If 'psite' column exists
        kin_beta["psite"] = kin_beta["psite"].astype(str).str.strip()  # Clean psite names
    else:
        kin_beta["psite"] = ""  # If 'psite' column doesn't exist, create it with empty strings
    # Identify rows with general (non-site-specific) kinase effects
    kin_beta["is_unknown_site"] = kin_beta["psite"].isna() | (kin_beta["psite"].astype(str).str.len() == 0)
    kin_beta["beta"] = pd.to_numeric(kin_beta["beta"], errors="coerce").fillna(
        0.0)  # Convert beta to numeric, fill NaNs with 0
    print("... kinopt effects loaded.")
    return kin_alpha, kin_beta  # Return the processed alpha and beta DataFrames


def load_tfopt_effects(path: str):
    """Loads TF alpha (synthesis) and beta (TF activity) priors from the tfopt Excel file."""
    print(f"Loading tfopt effects from: {path}")
    xl = pd.ExcelFile(path)  # Open the Excel file
    # --- Alpha Values ---
    print("  -> Reading 'Alpha Values' sheet (tfopt)")
    dfA = pd.read_excel(xl, sheet_name="Alpha Values")  # Read the 'Alpha Values' sheet
    dfA = _normcols(dfA)  # Normalize column names
    # Find columns for mRNA (protein), TF (optional), and value
    mcol = _find_col(dfA, ["mrna", "gene", "protein", "geneid"])
    tcol = _find_col(dfA, ["tf"])  # TF column is optional for this sheet
    vcol = _find_col(dfA, ["value", "alpha", "effect", "score", "weight"])
    if not (mcol and vcol):  # Check if required columns are present
        raise ValueError("tfopt::Alpha Values must have (mRNA, Value).")
    # Select, rename, and clean the alpha data (synthesis rates)
    tf_alpha = dfA[[mcol, vcol]].rename(columns={mcol: "protein", vcol: "value"})
    tf_alpha["protein"] = tf_alpha["protein"].astype(str).str.strip()
    tf_alpha["value"] = pd.to_numeric(tf_alpha["value"], errors="coerce").fillna(0.0)

    # --- Beta Values --- (Parsed for future use, but not used in this model)
    print("  -> Reading 'Beta Values' sheet (tfopt)")
    try:
        dfB = pd.read_excel(xl, sheet_name="Beta Values")  # Try to read the 'Beta Values' sheet
        dfB = _normcols(dfB)  # Normalize column names
        # Find columns for TF, psite (optional), and value
        tcolB = _find_col(dfB, ["tf"])
        scolB = _find_col(dfB, ["psite", "site"])
        vcolB = _find_col(dfB, ["value", "beta", "effect", "score", "weight"])
        if tcolB and vcolB:  # If the required columns are found
            # Select, rename, and clean the beta data
            tf_beta = dfB[[tcolB, vcolB] + ([scolB] if scolB else [])].rename(
                columns={tcolB: "tf", vcolB: "value", (scolB or "psite"): "psite"})
            tf_beta["psite"] = tf_beta["psite"].astype(str).fillna(
                "").str.strip()  # Clean psite, fill NaNs with empty string
            tf_beta["value"] = pd.to_numeric(tf_beta["value"], errors="coerce").fillna(0.0)  # Convert value to numeric
        else:
            tf_beta = pd.DataFrame(columns=["tf", "psite", "value"])  # Create empty placeholder if columns are missing
    except Exception:
        tf_beta = pd.DataFrame(columns=["tf", "psite", "value"])  # Create empty placeholder if sheet is missing
    print("... tfopt effects loaded.")
    return tf_alpha, tf_beta  # Return the processed alpha and beta DataFrames


# ------------------------------
# Estimated targets (FC, not log2FC)
# ------------------------------

def load_estimated_protein_FC(path: str) -> pd.DataFrame:
    """Loads the estimated protein fold-change (FC) targets from the kinopt Excel file."""
    print(f"Loading estimated protein targets (FC) from: {path}")
    df = pd.read_excel(path, sheet_name="Estimated")  # Read the 'Estimated' sheet
    df = _normcols(df)  # Normalize column names
    namecol = _find_col(df, ["protein", "gene", "geneid", "target"])  # Find the protein identifier column
    if namecol is None:  # Check if the identifier column was found
        raise ValueError("kinopt::Estimated must include a protein/gene identifier column.")
    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))  # Find all time-point columns
    # Convert from wide format (one row per protein) to long format (one row per protein-time)
    tidy = df[[namecol] + tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(
        float)  # Extract numerical time from column name
    tidy = tidy.drop(columns=["time_col"])  # Drop the original time column name
    tidy = tidy.rename(columns={namecol: "protein"})  # Standardize the protein column name
    tidy["protein"] = tidy["protein"].astype(str).str.strip()  # Clean protein names
    tidy = tidy[tidy["time"].isin(TIME_POINTS)].copy()  # Filter to keep only the time points used in the model
    tidy = tidy.sort_values(["protein", "time"]).reset_index(drop=True)  # Sort by protein and time
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")  # Convert FC values to numeric
    print("... Protein targets loaded.")
    return tidy  # Return the long-format DataFrame of protein targets


def load_estimated_rna_FC(path: str) -> pd.DataFrame:
    """Loads the estimated RNA fold-change (FC) targets from the tfopt Excel file."""
    print(f"Loading estimated RNA targets (FC) from: {path}")
    df = pd.read_excel(path, sheet_name="Estimated")  # Read the 'Estimated' sheet
    df = _normcols(df)  # Normalize column names
    namecol = _find_col(df, ["mrna", "protein", "gene", "geneid", "target"])  # Find the mRNA/protein identifier column
    if namecol is None:  # Check if the identifier column was found
        raise ValueError("tfopt::Estimated must include an mRNA/protein identifier column.")
    tcols = _time_cols(df.drop(columns=[namecol], errors="ignore"))  # Find all time-point columns
    # Convert from wide format to long format
    tidy = df[[namecol] + tcols].melt(id_vars=[namecol], var_name="time_col", value_name="fc")
    tidy["time"] = tidy["time_col"].str.extract(r"([-+]?\d*\.?\d+)").astype(float)  # Extract numerical time
    tidy = tidy.drop(columns=["time_col"])  # Drop the original time column name
    tidy = tidy.rename(columns={namecol: "protein"})  # Standardize the identifier column name to 'protein'
    tidy["protein"] = tidy["protein"].astype(str).str.strip()  # Clean protein names
    tidy = tidy[tidy["time"].isin(TIME_POINTS_RNA)].copy()  # Filter to keep only the time points used for RNA
    tidy = tidy.sort_values(["protein", "time"]).reset_index(drop=True)  # Sort by protein and time
    tidy["fc"] = pd.to_numeric(tidy["fc"], errors="coerce")  # Convert FC values to numeric
    print("... RNA targets loaded.")
    return tidy  # Return the long-format DataFrame of RNA targets


# ------------------------------
# Index and sparse maps
# ------------------------------

class Index:
    """Creates and holds mappings for proteins, sites, and kinases to/from integer indices."""

    def __init__(self, interactions: pd.DataFrame):
        print("Building model index (proteins, sites, kinases)...")
        # --- Protein Index ---
        self.proteins = sorted(interactions["protein"].unique().tolist())  # Get sorted list of unique proteins
        self.p2i = {p: i for i, p in enumerate(self.proteins)}  # Create protein-to-index mapping
        # --- Site Index (per protein) ---
        self.sites = []  # List of lists, where self.sites[i] is the list of sites for protein i
        for p in self.proteins:  # Iterate through each protein
            # Find unique sites for this protein
            ps = interactions.loc[interactions["protein"] == p, "psite"].unique().tolist()
            self.sites.append(ps)  # Add the list of sites
        # --- Kinase Index ---
        self.kinases = sorted(interactions["kinase"].unique().tolist())  # Get sorted list of unique kinases
        self.k2i = {k: i for i, k in enumerate(self.kinases)}  # Create kinase-to-index mapping
        # --- Dimensions ---
        self.N = len(self.proteins)  # Total number of proteins
        self.n_sites_per_protein = [len(s) for s in self.sites]  # List of site counts for each protein
        # --- State Vector Offsets ---
        self.offset = []  # Stores the starting index for each protein's state block
        curr = 0  # Current offset
        for i in range(self.N):  # Iterate through proteins
            self.offset.append(curr)  # Store the starting index for protein i
            # Each block has 1 (RNA) + 1 (Unphospho Protein) + n_sites (Phospho Sites)
            curr += 2 + self.n_sites_per_protein[i]
        self.state_dim = curr  # The total dimension of the ODE state vector
        print(f"... Index built. N_Proteins={self.N}, N_Kinases={len(self.kinases)}, State_Dim={self.state_dim}")

    def block(self, i: int) -> slice:
        """Returns the slice corresponding to the state vector for protein i."""
        s = self.offset[i];
        e = s + 2 + self.n_sites_per_protein[i]  # Calculate start and end indices
        return slice(s, e)  # Return the slice object


def build_W(interactions: pd.DataFrame, idx: Index) -> List[sparse.csr_matrix]:
    """Builds a list of sparse matrices (W), one for each protein, mapping (site, kinase) interactions."""
    print("Building interaction (W) sparse matrices...")
    W = []  # Initialize the list to hold sparse matrices
    # Iterate through each protein with a progress bar
    for i, p in tqdm(enumerate(idx.proteins), desc="Building W matrices", total=idx.N, unit="protein"):
        sub = interactions[interactions["protein"] == p][["psite", "kinase"]]  # Get interactions for this protein
        # Create a mapping from site name to row index for this protein's matrix
        site_order = {s: r for r, s in enumerate(idx.sites[i])}
        rows, cols = [], []  # Initialize lists for sparse matrix coordinates
        for _, r in sub.iterrows():  # Iterate through the interactions
            s = r["psite"];
            k = r["kinase"]  # Get site and kinase names
            if s in site_order and k in idx.k2i:  # If both are in our index
                rows.append(site_order[s]);
                cols.append(idx.k2i[k])  # Add (row, col) indices
        data = np.ones(len(rows), float)  # Data is just 1.0 for each existing interaction
        # Create the sparse matrix for this protein
        W.append(sparse.csr_matrix((data, (rows, cols)), shape=(idx.n_sites_per_protein[i], len(idx.kinases))))
    print("... W matrices built.")
    return W  # Return the list of sparse interaction matrices


# ------------------------------
# Initialize parameters from effects
# ------------------------------

def init_params_from_effects(idx: Index, W_list: List[sparse.csr_matrix],
                             kin_alpha: pd.DataFrame, kin_beta: pd.DataFrame,
                             tf_alpha: pd.DataFrame, alpha_scale=0.2):
    """Initializes all model parameters (c_k, A_i, B_i, etc.) using priors from effects files."""
    print("Initializing parameters from effects priors...")
    # --- c_k (Kinase baseline activity) ---
    # Use 'unknown site' entries from kin_beta as the primary source
    base = kin_beta[kin_beta["is_unknown_site"]].groupby("kinase")["beta"].mean()
    # Reindex to match the model's kinase order, filling missing values with the mean beta (all sites) or 0.0
    c_k = base.reindex(idx.kinases).fillna(kin_beta.groupby("kinase")["beta"].mean()).fillna(0.0).values.astype(float)
    c_k = softplus(c_k);
    c_k /= (np.mean(c_k) + 1e-12)  # Ensure non-negative and normalize

    # --- A_i (RNA synthesis rate) ---
    if tf_alpha is not None and len(tf_alpha):  # If tf_alpha data is available
        tf_alpha = tf_alpha.copy()  # Make a copy
        tf_alpha["protein"] = tf_alpha["protein"].astype(str).str.strip()  # Clean protein names
        tf_alpha["value"] = pd.to_numeric(tf_alpha["value"], errors="coerce")  # Convert values to numeric
        # Aggregate multiple entries per protein (if any) by taking the mean
        tf_alpha_series = (
            tf_alpha.groupby("protein", as_index=True)["value"]
            .mean()
            .sort_index()
        )
        # Set a default value (mean of all) for proteins not in the tf_alpha file
        default_A = float(tf_alpha_series.mean()) if len(tf_alpha_series) else 1.0
        A_i = (
            tf_alpha_series
            .reindex(idx.proteins)  # Align to the model's protein order
            .fillna(default_A)  # Fill missing proteins with the default
            .to_numpy(dtype=float)  # Convert to a NumPy array
        )
    else:
        # If no TF alpha info, use a neutral prior of 1.0 for all
        A_i = np.full(idx.N, 1.0, dtype=float)

    A_i = softplus(A_i);
    A_i /= (np.median(A_i) + 1e-12)  # Ensure non-negative and normalize by median

    # --- Other parameters (B_i, C_i, D_i, r_site) ---
    N = idx.N  # Number of proteins
    B_i = np.full(N, 0.2, float)  # RNA degradation rate (uniform prior)
    C_i = np.full(N, 0.5, float)  # Protein synthesis rate (uniform prior)
    D_i = np.full(N, 0.05, float)  # Protein degradation rate (uniform prior)
    r_site = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(N)]  # Site-specific degradation (prior=0)

    # --- alpha (Kinase-site interaction strength) ---
    alpha_init = []  # List to store sparse alpha matrices (priors)
    kin_alpha_keyed = kin_alpha.copy()  # Copy the kin_alpha data
    # Create a (protein, psite, kinase) tuple key for fast lookup
    kin_alpha_keyed["key"] = list(zip(kin_alpha_keyed["protein"].astype(str), kin_alpha_keyed["psite"].astype(str),
                                      kin_alpha_keyed["kinase"].astype(str)))
    # Create a dictionary for {key: alpha_value}
    alpha_lookup = {k: v for k, v in zip(kin_alpha_keyed["key"], kin_alpha_keyed["alpha"])}

    # Iterate through each protein with a progress bar
    print("  -> Building alpha (interaction) priors...")
    for i, p in tqdm(enumerate(idx.proteins), desc="Initializing alpha priors", total=idx.N, unit="protein"):
        Wi = W_list[i].tocsr()  # Get the interaction matrix (W) for protein i
        if Wi.nnz == 0:  # If no interactions for this protein
            alpha_init.append(Wi.copy());
            continue  # Append an empty matrix
        rows, cols = Wi.nonzero()  # Get the (row, col) indices of existing interactions
        data = np.zeros_like(rows, dtype=float)  # Initialize data (alpha values) to zero
        # Assign alpha values from the lookup table
        for n, (r, c) in enumerate(zip(rows, cols)):  # Iterate through each interaction
            s_name = idx.sites[i][r]  # Get site name from row index
            k_name = idx.kinases[c]  # Get kinase name from col index
            val = alpha_lookup.get((p, s_name, k_name), np.nan)  # Look up the alpha value
            if np.isnan(val):  # If no value was found
                data[n] = alpha_scale  # Use the default fallback value
            else:
                data[n] = max(0.0, float(val))  # Use the value from the file (ensuring non-negative)

        # --- Optional: Scale by site-specific beta ---
        # Get site-specific beta values (where psite is known)
        site_beta = kin_beta[~kin_beta["is_unknown_site"]].copy()
        if not site_beta.empty:  # If there are any site-specific betas
            # Create a (kinase, psite) key for lookup
            site_beta["key"] = list(zip(site_beta["kinase"].astype(str), site_beta["psite"].astype(str)))
            # Create the {key: beta_value} dictionary
            beta_map = {k: v for k, v in zip(site_beta["key"], site_beta["beta"])}
            for n, (r, c) in enumerate(zip(rows, cols)):  # Iterate through interactions again
                s_name = idx.sites[i][r];
                k_name = idx.kinases[c]  # Get names
                b = beta_map.get((k_name, s_name), np.nan)  # Look up the site-specific beta
                if not np.isnan(b):  # If found
                    data[n] *= softplus(b)  # Scale the alpha value by softplus(beta)

        # Create the sparse alpha matrix for protein i
        A = sparse.csr_matrix((data, (rows, cols)), shape=Wi.shape)
        alpha_init.append(A)  # Add it to the list

    # Collect all parameters into a dictionary
    params = {"c_k": c_k, "A_i": A_i, "B_i": B_i, "C_i": C_i, "D_i": D_i, "r_site": r_site, "alpha_list": alpha_init}
    print("... Parameters initialized.")
    return params  # Return the dictionary of initial parameters


# ------------------------------
# ODE system
# ------------------------------

class KinaseInput:
    """A simple class to define the kinase activity over time (here, constant)."""

    def __init__(self, kinases: List[str], const_levels: np.ndarray):
        self.kinases = kinases  # Store the list of kinase names
        self.const = np.asarray(const_levels, float)  # Store the constant activity levels

    def eval(self, t: float) -> np.ndarray:
        """Returns the kinase activity at time t."""
        return self.const  # In this model, activity is constant


class System:
    """Defines the complete ODE system, its parameters, and the RHS function."""

    def __init__(self, idx: Index, W_list: List[sparse.csr_matrix], params: Dict[str, object], kin_input: KinaseInput):
        self.idx = idx;
        self.W_list = [W.tocsr() for W in W_list]  # Store index and W matrices
        # Unpack and store all parameters from the 'params' dictionary
        self.c_k = params["c_k"];
        self.A_i = params["A_i"];
        self.B_i = params["B_i"]
        self.C_i = params["C_i"];
        self.D_i = params["D_i"];
        self.r_site = params["r_site"]
        self.alpha_list = params["alpha_list"];
        self.kin = kin_input  # Store kinase input and alpha matrices
        # Pre-allocate buffers for site rates to avoid re-allocation in 'rhs'
        self._buf_S = [np.zeros(idx.n_sites_per_protein[i], float) for i in range(idx.N)]

    # --- Parameter setter methods (used during optimization) ---
    def set_c_k(self, c):
        self.c_k = c

    def set_D_i(self, D):
        self.D_i = D

    def set_alpha_from_vals(self, alpha_vals_list: List[np.ndarray]):
        """Rebuilds the sparse alpha matrices from a flat list of nonzero values."""
        new = []  # List to hold new sparse alpha matrices
        for i, W in enumerate(self.W_list):  # Iterate through each protein's W matrix
            if W.nnz == 0: new.append(W.copy()); continue  # If no interactions, append empty matrix
            rows, cols = W.nonzero()  # Get the (row, col) indices from W
            # Create new sparse matrix using the *new* data values
            A = sparse.csr_matrix((alpha_vals_list[i], (rows, cols)), shape=W.shape)
            new.append(A)  # Append the new matrix
        self.alpha_list = new  # Overwrite the old alpha list

    def site_rates(self, t: float) -> List[np.ndarray]:
        """Calculates the phosphorylation rate (S_i) for each site at time t."""
        Kt = self.kin.eval(t) * self.c_k  # Total kinase activity = Input * baseline (c_k)
        out = self._buf_S  # Use the pre-allocated buffer
        for i in range(self.idx.N):  # For each protein
            if self.W_list[i].nnz == 0:  # If no sites/interactions
                out[i].fill(0.0);
                continue  # Rates are zero
            # Site rate S = alpha * Kt (matrix dot product)
            out[i][:] = self.alpha_list[i].dot(Kt) if self.alpha_list[i].nnz > 0 else 0.0
        return out  # Return the list of site rate vectors

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """The main ODE function, f(t, y), which computes the derivatives dy/dt."""
        dy = np.zeros_like(y);
        S_list = self.site_rates(t)  # Initialize dy and get site rates
        for i in range(self.idx.N):  # Iterate through each protein's state block
            sl = self.idx.block(i);
            block = y[sl];
            R, P = block[0], block[1];
            Ps = block[2:]  # Unpack state
            Ai, Bi, Ci, Di = self.A_i[i], self.B_i[i], self.C_i[i], self.D_i[i]  # Unpack parameters
            if self.idx.n_sites_per_protein[i] > 0:  # If this protein has phospho-sites
                rsi = self.r_site[i];
                kdeg_s = (1.0 + rsi) * Di;
                S_i = S_list[i]  # Get site-specific params
            else:
                kdeg_s = np.array([], float);
                S_i = np.array([], float)  # Use empty arrays if no sites

            # --- Define the ODEs for protein i ---
            dR = Ai - Bi * R  # d(RNA)/dt = Synthesis - Degradation
            sumS = S_i.sum() if S_i.size else 0.0  # Total phosphorylation rate for this protein
            # d(Unphospho Protein)/dt = Synthesis - Degradation - Total Phosphorylation + Sum(Dephosphorylation)
            dP = Ci * R - (Di + sumS) * P + Ps.sum()
            # d(Phospho Site 's')/dt = Phosphorylation - (Degradation + Dephosphorylation)
            dPs = S_i * P - (1.0 + kdeg_s) * Ps

            # --- Assign derivatives back to the 'dy' vector ---
            dy[sl.start + 0] = dR  # Assign dR/dt
            dy[sl.start + 1] = dP  # Assign dP/dt
            if dPs.size: dy[sl.start + 2: sl.start + 2 + dPs.size] = dPs  # Assign dPs/dt for all sites
        return dy  # Return the complete derivative vector

    def y0(self, R0=1.0, P0=1.0, Psite0=0.01) -> np.ndarray:
        """Generates the initial state vector (y0) at t=0."""
        y0 = np.zeros(self.idx.state_dim, float)  # Initialize state vector to zeros
        for i in range(self.idx.N):  # For each protein
            sl = self.idx.block(i)  # Get its state block slice
            y0[sl.start + 0] = R0;
            y0[sl.start + 1] = P0  # Set initial RNA (R0) and Protein (P0)
            nsi = self.idx.n_sites_per_protein[i]  # Number of sites for this protein
            if nsi > 0: y0[sl.start + 2: sl.start + 2 + nsi] = Psite0  # Set initial phospho-site levels
        return y0  # Return the complete initial state vector


# ------------------------------
# Simulation + observables (FC)
# ------------------------------

def simulate_union(sys: System, times_union: np.ndarray, atol=1e-8, rtol=1e-6, method="BDF"):
    """Runs the ODE simulation using solve_ivp."""
    t0, t1 = float(times_union.min()), float(times_union.max())  # Get start and end times
    # Call the ODE solver
    sol = solve_ivp(sys.rhs, (t0, t1), sys.y0(), t_eval=np.asarray(times_union, float), atol=atol, rtol=rtol,
                    method=method)
    if not sol.success:  # Check if the integration was successful
        raise RuntimeError(f"ODE integration failed: {sol.message}")  # Raise error if not
    return sol.t, sol.y.T  # Return time points and the transposed solution (rows=time, cols=state)


def protein_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    """Calculates total protein fold-change (FC) from the simulation trajectory Y."""
    rows = []  # Initialize list to store DataFrames
    # Iterate through each protein with a progress bar
    for i, prot in tqdm(enumerate(idx.proteins), desc="Calculating protein FC", total=idx.N, unit="protein"):
        sl = idx.block(i)  # Get the state block for protein i
        P = Y[:, sl.start + 1]  # Get trajectory for unphospho protein P
        nsi = idx.n_sites_per_protein[i]  # Number of sites
        # Sum the trajectories of all phospho-sites (Ps)
        Ps = Y[:, sl.start + 2: sl.start + 2 + nsi].sum(axis=1) if nsi > 0 else 0.0
        total = np.maximum(P + Ps, 1e-12)  # Total protein = P + Ps (with a floor)
        fc = total / total[0]  # Fold-change = Total(t) / Total(t=0)
        df = pd.DataFrame({"time": t, "protein": prot, "fc": fc})  # Create a DataFrame
        df = df[df["time"].isin(times_needed)]  # Filter for the requested time points
        rows.append(df)  # Add to the list
    return pd.concat(rows, ignore_index=True)  # Concatenate all protein DataFrames


def rna_FC(idx: Index, t: np.ndarray, Y: np.ndarray, times_needed: np.ndarray) -> pd.DataFrame:
    """Calculates RNA fold-change (FC) from the simulation trajectory Y."""
    rows = []  # Initialize list to store DataFrames
    # Iterate through each protein with a progress bar
    for i, prot in tqdm(enumerate(idx.proteins), desc="Calculating RNA FC", total=idx.N, unit="protein"):
        sl = idx.block(i)  # Get the state block for protein i
        R = np.maximum(Y[:, sl.start + 0], 1e-12)  # Get trajectory for RNA (R) (with a floor)
        fc = R / R[0]  # Fold-change = R(t) / R(t=0)
        df = pd.DataFrame({"time": t, "protein": prot, "fc": fc})  # Create a DataFrame
        df = df[df["time"].isin(times_needed)]  # Filter for the requested time points
        rows.append(df)  # Add to the list
    return pd.concat(rows, ignore_index=True)  # Concatenate all protein DataFrames


# ------------------------------
# Parameter packing
# ------------------------------

def init_raw_for_stage(params: Dict[str, object], alpha_list: List[sparse.csr_matrix], stage: int):
    """Packs parameters into a flat vector (theta) for optimization, based on the stage."""
    parts = {};
    vecs = []  # 'parts' maps name to slice; 'vecs' holds parameter chunks

    # Stage 1 & 3: Optimize c_k (kinase baseline activity)
    if stage in (1, 3):
        rc = inv_softplus(params["c_k"])  # Apply inverse transform
        parts["raw_c"] = slice(0, len(rc));
        vecs.append(rc)  # Store slice and vector

    # Stage 2 & 3: Optimize alpha (interaction strengths)
    if stage in (2, 3):
        chunks = [];
        sls = [];
        start = sum(len(v) for v in vecs)  # 'chunks' holds data arrays, 'sls' holds slices
        for A in alpha_list:  # Iterate through sparse alpha matrices
            data = A.data if A.nnz > 0 else np.array([], float)  # Get the nonzero data
            chunks.append(inv_softplus(np.maximum(data, 1e-12)))  # Apply inverse transform
            sls.append(slice(start, start + len(chunks[-1])));
            start += len(chunks[-1])  # Store slice
        ra = np.concatenate(chunks) if len(chunks) > 0 else np.array([], float)  # Concatenate all alpha data
        if ra.size > 0: parts["raw_alpha"] = sls; vecs.append(ra)  # Store slices and vector

    # Stage 3 only: Optimize D_i (protein degradation)
    if stage == 3:
        rD = inv_softplus(params["D_i"]);
        start = sum(len(v) for v in vecs)  # Get start index
        parts["raw_D"] = slice(start, start + len(rD));
        vecs.append(rD)  # Store slice and vector

    theta0 = np.concatenate(vecs) if len(vecs) > 0 else np.array([], float)  # Concatenate all vectors into theta0
    return theta0, parts  # Return initial parameters and the 'parts' map


def assign_theta(theta: np.ndarray, parts: Dict[str, object], sys: System, alpha_init: List[sparse.csr_matrix]):
    """Unpacks a flat vector (theta) and assigns the (softplus-transformed) values to the System."""

    # Unpack c_k if it was optimized
    if "raw_c" in parts:
        sl = parts["raw_c"];
        sys.set_c_k(softplus(theta[sl]))  # Get slice, transform, and set

    # Unpack alpha if it was optimized
    if "raw_alpha" in parts:
        vals = []  # List to hold new alpha data arrays
        for sl in parts["raw_alpha"]:  # Iterate through the slice for each protein
            raw = theta[sl] if sl.start != sl.stop else np.array([], float)  # Get the raw data from theta
            vals.append(softplus(raw) if raw.size > 0 else np.array([], float))  # Transform and append
        sys.set_alpha_from_vals(vals)  # Rebuild sparse matrices with new values
    else:
        # If alpha wasn't optimized, reset it to the initial state (important if 'fun' is called outside 'minimize')
        sys.set_alpha_from_vals([A.data if A.nnz > 0 else np.array([], float) for A in alpha_init])

    # Unpack D_i if it was optimized
    if "raw_D" in parts:
        sl = parts["raw_D"];
        sys.set_D_i(softplus(theta[sl]))  # Get slice, transform, and set


# ------------------------------
# Loss (FC)
# ------------------------------

def build_weights(times: np.ndarray, early_focus: float) -> Dict[float, float]:
    """Creates a dictionary of weights per timepoint, optionally focusing on early times."""
    tmin, tmax = float(times.min()), float(times.max())  # Get time range
    span = (tmax - tmin) if tmax > tmin else 1.0  # Get time span
    # Weight = 1.0 + (focus * normalized_distance_from_end)
    return {float(t): 1.0 + early_focus * (tmax - float(t)) / span for t in times}


def dual_loss(theta: np.ndarray, parts: Dict[str, object], sys: System, idx: Index,
              alpha_init: List[sparse.csr_matrix], c_k_init: np.ndarray,
              df_prot_obs: pd.DataFrame, df_rna_obs: pd.DataFrame,
              lam: Dict[str, float], atol: float, rtol: float):
    """Calculates the total loss (objective function) for a given parameter vector 'theta'."""
    # 1. Update the system with the new parameters from 'theta'
    assign_theta(theta, parts, sys, alpha_init)

    # 2. Run the ODE simulation
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))  # Get all unique time points
    t, Y = simulate_union(sys, times_union, atol=atol, rtol=rtol)  # Run simulation

    # 3. Calculate predicted observables (FC)
    dfp = protein_FC(idx, t, Y, TIME_POINTS).rename(columns={"fc": "pred_fc"})  # Predicted protein FC
    dfr = rna_FC(idx, t, Y, TIME_POINTS_RNA).rename(columns={"fc": "pred_fc"})  # Predicted RNA FC

    # 4. Merge predictions with observations
    mp = df_prot_obs.merge(dfp, on=["protein", "time"], how="inner")  # Merge protein
    mr = df_rna_obs.merge(dfr, on=["protein", "time"], how="inner")  # Merge RNA

    # 5. Calculate data loss (weighted sum of squared errors)
    prot_loss = np.sum(mp["w"].values * (mp["pred_fc"].values - mp["fc"].values) ** 2)  # Protein loss
    rna_loss = np.sum(mr["w"].values * (mr["pred_fc"].values - mr["fc"].values) ** 2)  # RNA loss

    # 6. Calculate regularization penalties
    reg_alpha_l1 = 0.0;
    reg_alpha_prior = 0.0  # Initialize alpha penalties
    if "raw_alpha" in parts:  # If alpha is being optimized
        # Get current alpha values
        cur_alpha = []
        for A in sys.alpha_list:
            cur_alpha.append(A.data if A.nnz > 0 else np.array([], float))
        cur_alpha = np.concatenate(cur_alpha) if len(cur_alpha) > 0 else np.array([], float)
        # Get initial alpha values
        init_alpha = []
        for A in alpha_init:
            init_alpha.append(A.data if A.nnz > 0 else np.array([], float))
        init_alpha = np.concatenate(init_alpha) if len(init_alpha) > 0 else np.array([], float)

        if cur_alpha.size > 0:
            reg_alpha_l1 = np.sum(cur_alpha)  # L1 penalty (encourages sparsity, pushes alpha towards 0)
            if init_alpha.size == cur_alpha.size:
                # L2 penalty vs prior (keeps alpha close to its initial value)
                reg_alpha_prior = np.sum((cur_alpha - init_alpha) ** 2)

    reg_c_prior = 0.0  # Initialize c_k penalty
    if "raw_c" in parts:  # If c_k is being optimized
        # L2 penalty vs prior (keeps c_k close to its initial value)
        reg_c_prior = np.sum((sys.c_k - c_k_init) ** 2)

    # 7. Calculate total loss
    total = prot_loss + lam.get("lambda_rna", 1.0) * rna_loss \
            + lam.get("l1_alpha", 0.0) * reg_alpha_l1 \
            + lam.get("prior_alpha", 0.0) * reg_alpha_prior \
            + lam.get("prior_c", 0.0) * reg_c_prior

    # 8. Store components for logging
    details = {
        "prot_loss": float(prot_loss),
        "rna_loss": float(rna_loss),
        "reg_alpha_l1": float(reg_alpha_l1),
        "reg_alpha_prior": float(reg_alpha_prior),
        "reg_c_prior": float(reg_c_prior),
        "total": float(total)
    }
    return total, details  # Return total loss and the dictionary of components


# ------------------------------
# Main
# ------------------------------

def main():
    """Main execution function."""
    print("--- Starting Global ODE Dual-Fit (FC) ---")
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser(description="Global ODE dual-fit (FC) with effects from 'Alpha Values'/'Beta Values'.")
    ap.add_argument("--interaction", required=True, help="Path to the interaction network CSV.")
    ap.add_argument("--kinopt", required=True, help="Path to the kinopt results Excel file.")
    ap.add_argument("--tfopt", required=True, help="Path to the tfopt results Excel file.")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=3,
                    help="Optimization stage (1=c_k, 2=alpha, 3=all).")
    ap.add_argument("--early-focus", type=float, default=1.0, help="Weighting factor for early time points.")
    ap.add_argument("--l1-alpha", type=float, default=2e-3, help="Lambda for L1 regularization on alpha.")
    ap.add_argument("--lambda-prior", type=float, default=1e-2, help="Lambda for L2 prior regularization on alpha.")
    ap.add_argument("--lambda-c", type=float, default=1e-2, help="Lambda for L2 prior regularization on c_k.")
    ap.add_argument("--lambda-rna", type=float, default=1.0, help="Weighting factor for the RNA loss component.")
    ap.add_argument("--maxiter", type=int, default=300, help="Maximum iterations for the optimizer.")
    ap.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for the ODE solver.")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for the ODE solver.")
    ap.add_argument("--output-dir", default="./out_dual_fc", help="Directory to save output files.")
    args = ap.parse_args()  # Parse the command-line arguments

    t0 = time.time()  # Record start time
    print(f"Creating output directory: {args.output_dir}")
    _ensure_dir(args.output_dir)  # Ensure the output directory exists

    # --- Load Data ---
    print("\n--- 1. Loading Data ---")
    interactions = load_interactions(args.interaction)  # Load interaction network
    kin_alpha, kin_beta = load_kinopt_effects(args.kinopt)  # Load kinopt priors
    tf_alpha, tf_beta = load_tfopt_effects(args.tfopt)  # Load tfopt priors
    df_prot_obs = load_estimated_protein_FC(args.kinopt)  # Load protein targets
    df_rna_obs = load_estimated_rna_FC(args.tfopt)  # Load RNA targets

    # --- Build Index and Maps ---
    print("\n--- 2. Building Model Index & Maps ---")
    idx = Index(interactions);
    W_list = build_W(interactions, idx)  # Build index and W matrices
    # Filter observation data to include only proteins present in the network index
    df_prot_obs = df_prot_obs[df_prot_obs["protein"].isin(idx.proteins)].reset_index(drop=True)
    df_rna_obs = df_rna_obs[df_rna_obs["protein"].isin(idx.proteins)].reset_index(drop=True)
    print(
        f"Filtered targets: {len(df_prot_obs['protein'].unique())} proteins (prot), {len(df_rna_obs['protein'].unique())} proteins (RNA)")

    # --- Initialize Parameters ---
    print("\n--- 3. Initializing System Parameters ---")
    params = init_params_from_effects(idx, W_list, kin_alpha, kin_beta, tf_alpha, alpha_scale=0.2)

    # --- Set Weights ---
    print("Setting loss weights (early_focus={args.early_focus})...")
    wmap = build_weights(TIME_POINTS, early_focus=args.early_focus)  # Get protein time weights
    df_prot_obs["w"] = df_prot_obs["time"].map(wmap).astype(float)  # Apply weights to protein data
    df_rna_obs["w"] = 1.0  # Use uniform (1.0) weights for RNA data

    # --- Initialize ODE System ---
    print("Initializing ODE system...")
    # Kinase input is constant, proportional to the *initial* c_k estimate
    kin_input = KinaseInput(idx.kinases, params["c_k"].copy())
    sys = System(idx, W_list, params, kin_input)  # Create the System object

    # --- Setup Optimization ---
    print(f"\n--- 4. Setting up Optimization (Stage {args.stage}) ---")
    # Pack the parameters to be optimized into the flat vector 'theta0'
    theta0, parts = init_raw_for_stage(params, params["alpha_list"], stage=args.stage)
    c_k_init = sys.c_k.copy()  # Store initial c_k for regularization
    alpha_init = [A.copy() for A in params["alpha_list"]]  # Store initial alpha for regularization
    print(f"Total parameters to optimize: {theta0.size}")

    # --- Define Objective Function ---
    # Store regularization lambdas in a dictionary
    lam = {"l1_alpha": args.l1_alpha, "prior_alpha": args.lambda_prior, "prior_c": args.lambda_c,
           "lambda_rna": args.lambda_rna}

    # Create the objective function 'fun' to be passed to the optimizer
    def fun(theta):
        # This function only needs to return the total loss scalar
        val, _ = dual_loss(theta, parts, sys, idx, alpha_init, c_k_init, df_prot_obs, df_rna_obs, lam, args.atol,
                           args.rtol)
        return val

    # --- Run Optimization ---
    print(f"\n--- 5. Running Optimization (MaxIter: {args.maxiter}) ---")
    if theta0.size > 0:  # Only run if there are parameters to optimize
        # Initialize tqdm progress bar
        pbar = tqdm(total=args.maxiter, desc="Optimizing", unit="iter")

        # Define a callback function to update the progress bar
        def callback(xk):
            pbar.update(1)  # Increment the progress bar by 1 iteration

        # Call the L-BFGS-B optimizer
        res = minimize(fun, theta0, method="L-BFGS-B", options={"maxiter": args.maxiter}, callback=callback)
        pbar.close()  # Close the progress bar

        if not res.success:  # Check if the optimizer succeeded
            print(f"[WARN] Optimizer: {res.message}")  # Print a warning if not
        theta_opt = res.x  # Get the optimized parameter vector
    else:
        print("No parameters to optimize for this stage.")
        theta_opt = theta0  # Use the empty initial vector

    print("... Optimization finished.")

    # --- Final Calculations and Outputs ---
    print("\n--- 6. Calculating Final Statistics & Saving Outputs ---")
    # Run the loss function one last time with the optimal parameters to get final loss and components
    f_opt, comps = dual_loss(theta_opt, parts, sys, idx, alpha_init, c_k_init, df_prot_obs, df_rna_obs, lam, args.atol,
                             args.rtol)
    print("Simulating final trajectory...")
    # Run the simulation one last time with optimal parameters
    times_union = np.unique(np.concatenate([TIME_POINTS, TIME_POINTS_RNA]))
    t, Y = simulate_union(sys, times_union, atol=args.atol, rtol=args.rtol)

    # Calculate final predicted FCs
    out_prot = protein_FC(idx, t, Y, TIME_POINTS)
    out_rna = rna_FC(idx, t, Y, TIME_POINTS_RNA)

    # Save predicted FCs to CSV
    prot_path = os.path.join(args.output_dir, "predicted_protein_fc.csv")
    rna_path = os.path.join(args.output_dir, "predicted_rna_fc.csv")
    out_prot.to_csv(prot_path, index=False)
    out_rna.to_csv(rna_path, index=False)
    print(f"Saved predicted protein FC to: {prot_path}")
    print(f"Saved predicted RNA FC to: {rna_path}")

    # --- Save Fitted Parameters (JSON) ---
    # Store fitted parameters in a dictionary
    fitted = {
        "proteins": idx.proteins,
        "kinases": idx.kinases,
        "c_k": sys.c_k.tolist(),  # Final fitted c_k
        "D_i": sys.D_i.tolist(),  # Final fitted D_i
        "alpha_nonzeros": [  # Store final alpha matrices in sparse format
            {"protein": idx.proteins[i],
             "rows": sys.alpha_list[i].nonzero()[0].tolist(),
             "cols": sys.alpha_list[i].nonzero()[1].tolist(),
             "data": sys.alpha_list[i].data.tolist()
             } for i in range(idx.N)
        ],
        "stage": args.stage
    }
    params_path = os.path.join(args.output_dir, "fitted_params.json")
    with open(params_path, "w") as f:
        json.dump(fitted, f, indent=2)  # Write to JSON file
    print(f"Saved fitted parameters to: {params_path}")

    # --- Save Summary (JSON) ---
    # Store summary of the run
    summary = {
        "objective": float(f_opt),  # Final total loss
        "components": comps,  # Loss components (data, regularization)
        "stage": args.stage,
        "hyperparams": {  # Store all hyperparameters
            "early_focus": args.early_focus,
            "l1_alpha": args.l1_alpha,
            "lambda_prior": args.lambda_prior,
            "lambda_c": args.lambda_c,
            "lambda_rna": args.lambda_rna,
            "maxiter": args.maxiter,
            "atol": args.atol,
            "rtol": args.rtol
        },
        "timing_sec": float(time.time() - t0)  # Total execution time
    }
    summary_path = os.path.join(args.output_dir, "fit_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)  # Write to JSON file
    print(f"Saved fit summary to: {summary_path}")

    # --- Print Final Summary to Console ---
    print("\n--- Fit Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nTotal time: {summary['timing_sec']:.2f} seconds.")
    print("--- Script Finished ---")


if __name__ == "__main__":
    main()  # Run the main function when the script is executed