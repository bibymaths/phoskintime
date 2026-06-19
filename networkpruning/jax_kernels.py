from __future__ import annotations
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

FLAG_LOW_SCORE = jnp.uint32(1 << 0)
FLAG_LOW_SUPPORT = jnp.uint32(1 << 1)
FLAG_SELF_LOOP = jnp.uint32(1 << 2)
FLAG_MISSING_SITE_OBS = jnp.uint32(1 << 3)
FLAG_MISSING_KINASE_OBS = jnp.uint32(1 << 4)
FLAG_OVER_BUDGET = jnp.uint32(1 << 5)


@jax.jit
def score_triplets(edge_weight, support_count, site_observed, kinase_observed, kinase_ids, substrate_ids):
    score = jnp.log1p(edge_weight.astype(jnp.float64)) + 0.5 * jnp.log1p(support_count.astype(jnp.float64))
    score = score + 0.25 * site_observed.astype(jnp.float64) + 0.25 * kinase_observed.astype(jnp.float64)
    score = score - 0.5 * (kinase_ids == substrate_ids).astype(jnp.float64)
    return score.astype(jnp.float64)


@jax.jit
def pruning_flags(score, support_count, site_observed, kinase_observed, kinase_ids, substrate_ids,
                  min_score, min_support, prune_self_loops, prune_missing_observations):
    flags = jnp.zeros(score.shape, dtype=jnp.uint32)
    flags = jnp.where(score < min_score, flags | FLAG_LOW_SCORE, flags)
    flags = jnp.where(support_count < min_support, flags | FLAG_LOW_SUPPORT, flags)
    flags = jnp.where((kinase_ids == substrate_ids) & prune_self_loops, flags | FLAG_SELF_LOOP, flags)
    flags = jnp.where((~site_observed) & prune_missing_observations, flags | FLAG_MISSING_SITE_OBS, flags)
    flags = jnp.where((~kinase_observed) & prune_missing_observations, flags | FLAG_MISSING_KINASE_OBS, flags)
    return flags


@jax.jit
def sparse_indices_values(kinase_ids, site_ids, substrate_ids, score):
    return jnp.stack([kinase_ids, site_ids, substrate_ids], axis=1).astype(jnp.int32), score.astype(jnp.float64)


@jax.jit
def identifiability_kernel(indices, values):
    norms = jnp.abs(values).astype(jnp.float64)
    retained = norms > jnp.finfo(jnp.float64).eps
    idx64 = indices.astype(jnp.int64)
    order = jnp.lexsort((idx64[:, 2], idx64[:, 1], idx64[:, 0]))
    sorted_idx = idx64[order]
    new_group = jnp.concatenate([
        jnp.array([True]),
        jnp.any(sorted_idx[1:] != sorted_idx[:-1], axis=1),
    ])
    gid_sorted = jnp.cumsum(new_group.astype(jnp.int32)) - 1
    gid = jnp.empty_like(gid_sorted).at[order].set(gid_sorted)
    redundancy = 1.0 - retained.astype(jnp.float64)
    return retained, gid, norms, redundancy
