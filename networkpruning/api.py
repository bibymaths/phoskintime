from __future__ import annotations
from pathlib import Path
import logging
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
from .dataclasses import *
from .io_adapters import encode_kinase_network
from .jax_kernels import score_triplets, pruning_flags, sparse_indices_values


def discover_hyperedges(encoded: EncodedNetwork, config: NetworkPreprocessingConfig) -> TripletTable:
    score = score_triplets(encoded.edge_weight, encoded.support_count, encoded.site_observed, encoded.kinase_observed,
                           encoded.kinase_node_ids, encoded.substrate_node_ids)
    return TripletTable(encoded.kinase_ids, encoded.site_ids, encoded.substrate_ids, score, encoded.support_count,
                        jnp.zeros_like(encoded.support_count, dtype=jnp.uint32))


def prune_triplets(triplets: TripletTable, encoded: EncodedNetwork, config: NetworkPreprocessingConfig) -> TripletTable:
    flags = pruning_flags(triplets.score, triplets.support_count, encoded.site_observed, encoded.kinase_observed,
                          encoded.kinase_node_ids, encoded.substrate_node_ids,
                          float(max(config.discovery_threshold, config.min_triplet_score)),
                          int(config.min_support_count), bool(config.prune_self_loops),
                          bool(config.prune_missing_observations))
    keep = flags == 0
    if config.max_triplets is not None and int(keep.sum()) > config.max_triplets:
        kept_idx = jnp.where(keep, size=triplets.score.size, fill_value=-1)[0]
        order = jnp.argsort(jnp.where(keep, -triplets.score, jnp.inf))[:config.max_triplets]
        mask = jnp.zeros_like(keep).at[order].set(True);
        keep = keep & mask
    return TripletTable(triplets.kinase_ids[keep], triplets.site_ids[keep], triplets.substrate_ids[keep],
                        triplets.score[keep], triplets.support_count[keep], flags[keep])


def build_sparse_theta(triplets: TripletTable, shape: tuple[int, int, int]) -> SparseThetaTensor:
    idx, val = sparse_indices_values(triplets.kinase_ids, triplets.site_ids, triplets.substrate_ids, triplets.score)
    return SparseThetaTensor(idx, val, shape)


def _common_node_edges(triplets: TripletTable, encoded: EncodedNetwork) -> tuple[
    list[int], list[int], list[float], tuple[str, ...]]:
    """Map motif edges from independent kinase/substrate IDs into one protein namespace."""
    node_labels = tuple(sorted(set(encoded.kinase_labels) | set(encoded.substrate_labels)))
    node_id = {label: i for i, label in enumerate(node_labels)}
    src = [node_id[encoded.kinase_labels[int(k)]] for k in jnp.asarray(triplets.kinase_ids)]
    dst = [node_id[encoded.substrate_labels[int(s)]] for s in jnp.asarray(triplets.substrate_ids)]
    score = list(map(float, jnp.asarray(triplets.score)))
    return src, dst, score, node_labels


def detect_motifs(triplets: TripletTable, config: NetworkPreprocessingConfig,
                  encoded: EncodedNetwork | None = None) -> MotifTable:
    # bounded host orchestration, core edge testing array based; no dense n^3 enumeration
    if encoded is None:
        src = list(map(int, jnp.asarray(triplets.kinase_ids)));
        dst = list(map(int, jnp.asarray(triplets.substrate_ids)));
        node_labels = ()
    else:
        src, dst, _, node_labels = _common_node_edges(triplets, encoded)
    sc = list(map(float, jnp.asarray(triplets.score)))
    edge = {(a, b): s for a, b, s in zip(src, dst, sc) if a != b};
    rows = []
    for (a, b), sab in edge.items():
        for (bb, c), sbc in edge.items():
            if bb != b or c in (a, b): continue
            if (a, c) in edge:
                rows.append((1, a, b, c, 0b111, (sab * sbc * edge[(a, c)]) ** (1 / 3)))
                if len(rows) >= config.max_motifs: break
        if len(rows) >= config.max_motifs: break
    if not rows:
        return MotifTable(jnp.empty(0, jnp.int16), jnp.empty(0, jnp.int32), jnp.empty(0, jnp.int32),
                          jnp.empty(0, jnp.int32), jnp.empty(0, jnp.uint8), jnp.empty(0, jnp.float64), node_labels)
    arr = jnp.asarray(rows)
    return MotifTable(arr[:, 0].astype(jnp.int16), arr[:, 1].astype(jnp.int32), arr[:, 2].astype(jnp.int32),
                      arr[:, 3].astype(jnp.int32), arr[:, 4].astype(jnp.uint8), arr[:, 5].astype(jnp.float64),
                      node_labels)


def preprocess_identifiability(theta: SparseThetaTensor,
                               config: NetworkPreprocessingConfig) -> IdentifiabilityDiagnostics:
    from .jax_kernels import identifiability_kernel
    retained, gid, norms, red = identifiability_kernel(theta.indices, theta.values)
    rank = int(jnp.unique(gid, size=gid.size, fill_value=-1).size) if gid.size else 0
    return IdentifiabilityDiagnostics(retained, gid, rank, red, norms)


def preprocess_network(kinase_network: pd.DataFrame, *, phospho_observations=None, protein_observations=None,
                       rna_observations=None, tf_network=None, config=None, output_dir=None, logger=None):
    config = config or NetworkPreprocessingConfig();
    logger = logger or logging.getLogger(__name__)
    encoded, grouped = encode_kinase_network(kinase_network, phospho_observations, protein_observations)
    discovered = discover_hyperedges(encoded, config);
    pruned = prune_triplets(discovered, encoded, config)
    theta = build_sparse_theta(pruned,
                               (len(encoded.kinase_labels), len(encoded.site_labels), len(encoded.substrate_labels)))
    motifs = detect_motifs(pruned, config, encoded) if config.enable_motifs else None
    ident = preprocess_identifiability(theta, config) if config.enable_identifiability else None
    summary = {"n_discovered": int(discovered.score.size), "n_retained": int(pruned.score.size),
               "n_pruned": int(discovered.score.size - pruned.score.size),
               "n_motifs": int(0 if motifs is None else motifs.score.size), "tensor_nnz": int(theta.values.size)}
    res = NetworkPreprocessingResult(encoded, discovered, pruned, theta, motifs, ident, summary)
    if output_dir is not None:
        if config.export_csv or config.export_sparse_tensor:
            from .export import export_preprocessing_result
            export_preprocessing_result(
                res, output_dir, output_subdir=config.output_subdir,
                include_csv=config.export_csv, include_sparse_tensor=config.export_sparse_tensor,
            )
        if config.generate_plots:
            from .plotting import plot_preprocessing_result
            plot_preprocessing_result(res, output_dir, output_subdir=config.output_subdir)
        logger.info("[NetworkPreprocessing] outputs saved under %s", Path(output_dir) / config.output_subdir)
    return res


def _pruned_networkmodel_frame(result: NetworkPreprocessingResult) -> pd.DataFrame:
    enc = result.encoded
    retained = {
        (int(k), int(s), int(sub))
        for k, s, sub in zip(result.pruned.kinase_ids, result.pruned.site_ids, result.pruned.substrate_ids)
    }
    rows = []
    for k, s, sub, alpha, support in zip(
            enc.kinase_ids, enc.site_ids, enc.substrate_ids, enc.edge_weight, enc.support_count
    ):
        key = (int(k), int(s), int(sub))
        if key not in retained:
            continue
        site_label = enc.site_labels[key[1]]
        protein, psite = site_label.split(":", 1)
        rows.append({
            "protein": protein,
            "psite": psite,
            "kinase": enc.kinase_labels[key[0]],
            "alpha": float(alpha),
            "support_count": int(support),
        })
    return pd.DataFrame(rows, columns=["protein", "psite", "kinase", "alpha", "support_count"])


def preprocess_networkmodel_frames(df_kin, df_tf, df_prot, df_pho, df_rna, *, config=None, output_dir=None,
                                   logger=None):
    res = preprocess_network(df_kin, phospho_observations=df_pho, protein_observations=df_prot, rna_observations=df_rna,
                             tf_network=df_tf, config=config, output_dir=output_dir, logger=logger)
    return _pruned_networkmodel_frame(res), res
