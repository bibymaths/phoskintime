from __future__ import annotations
import pandas as pd
import jax.numpy as jnp
from .dataclasses import EncodedNetwork


def _labels(vals): return tuple(sorted({str(v).strip().upper() for v in vals if str(v).strip()}))


def encode_kinase_network(df_kin: pd.DataFrame, phospho_observations=None, protein_observations=None) -> tuple[
    EncodedNetwork, pd.DataFrame]:
    req = {"protein", "psite", "kinase"};
    miss = req - set(df_kin.columns)
    if miss: raise ValueError(f"kinase network missing columns: {sorted(miss)}")
    df = df_kin.copy()
    df["protein"] = df["protein"].astype(str).str.strip().str.upper();
    df["kinase"] = df["kinase"].astype(str).str.strip().str.upper();
    df["psite"] = df["psite"].astype(str).str.strip()
    if "alpha" not in df: df["alpha"] = 1.0
    g = df.groupby(["kinase", "protein", "psite"], as_index=False).agg(alpha=("alpha", "max"),
                                                                       support_count=("alpha", "size"))
    kin = _labels(g["kinase"]);
    sub = _labels(g["protein"]);
    sites = tuple(sorted({f"{p}:{s}" for p, s in zip(g.protein, g.psite)}))
    km = {v: i for i, v in enumerate(kin)};
    sm = {v: i for i, v in enumerate(sub)};
    stm = {v: i for i, v in enumerate(sites)}
    node_labels = tuple(sorted(set(kin) | set(sub)));
    nm = {v: i for i, v in enumerate(node_labels)}
    obs_sites = set();
    obs_kin = set()
    if phospho_observations is not None and not phospho_observations.empty:
        po = phospho_observations.copy();
        po["protein"] = po["protein"].astype(str).str.strip().str.upper();
        po["psite"] = po["psite"].astype(str).str.strip()
        obs_sites = {f"{p}:{s}" for p, s in zip(po.protein, po.psite)}
    if protein_observations is not None and not protein_observations.empty:
        obs_kin = {str(x).strip().upper() for x in protein_observations["protein"].unique()}
    enc = EncodedNetwork(
        kinase_ids=jnp.asarray([km[x] for x in g.kinase], jnp.int32),
        substrate_ids=jnp.asarray([sm[x] for x in g.protein], jnp.int32),
        kinase_node_ids=jnp.asarray([nm[x] for x in g.kinase], jnp.int32),
        substrate_node_ids=jnp.asarray([nm[x] for x in g.protein], jnp.int32),
        site_ids=jnp.asarray([stm[f"{p}:{s}"] for p, s in zip(g.protein, g.psite)], jnp.int32),
        edge_weight=jnp.asarray(g.alpha.to_numpy(), jnp.float64),
        support_count=jnp.asarray(g.support_count.to_numpy(), jnp.int32),
        site_observed=jnp.asarray([f"{p}:{s}" in obs_sites for p, s in zip(g.protein, g.psite)], bool),
        kinase_observed=jnp.asarray([k in obs_kin for k in g.kinase], bool), kinase_labels=kin, substrate_labels=sub,
        site_labels=sites)
    return enc, g
