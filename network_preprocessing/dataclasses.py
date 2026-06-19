from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import jax
import jax.numpy as jnp

@dataclass(frozen=True)
class NetworkPreprocessingConfig:
    min_triplet_score: float = 0.0
    discovery_threshold: float = 0.0
    min_support_count: int = 1
    max_triplets: int | None = None
    batch_size: int = 65536
    dtype: str = "float64"
    enable_motifs: bool = True
    enable_identifiability: bool = True
    prune_self_loops: bool = True
    prune_missing_observations: bool = False
    output_subdir: str = "network_preprocessing"
    export_sparse_tensor: bool = True
    generate_plots: bool = True
    export_csv: bool = True
    dense_motif_node_limit: int = 256
    max_motifs: int = 10000
    score_weights: Mapping[str, float] = field(default_factory=dict)

    @staticmethod
    def from_args(args: Any) -> "NetworkPreprocessingConfig":
        return NetworkPreprocessingConfig(
            min_triplet_score=float(getattr(args, "network_preprocessing_min_score", 0.0)),
            discovery_threshold=float(getattr(args, "network_preprocessing_discovery_threshold", 0.0)),
            min_support_count=int(getattr(args, "network_preprocessing_min_support", 1)),
            max_triplets=(None if getattr(args, "network_preprocessing_max_triplets", None) == 0 else getattr(args, "network_preprocessing_max_triplets", None)),
            batch_size=int(getattr(args, "network_preprocessing_batch_size", 65536)),
            enable_motifs=bool(getattr(args, "network_preprocessing_enable_motifs", True)),
            enable_identifiability=bool(getattr(args, "network_preprocessing_enable_identifiability", True)),
            prune_self_loops=bool(getattr(args, "network_preprocessing_prune_self_loops", True)),
            prune_missing_observations=bool(getattr(args, "network_preprocessing_prune_missing_observations", False)),
            output_subdir=str(getattr(args, "network_preprocessing_output_subdir", "network_preprocessing")),
            export_sparse_tensor=bool(getattr(args, "network_preprocessing_export_sparse_tensor", True)),
            generate_plots=bool(getattr(args, "network_preprocessing_generate_plots", True)),
            export_csv=bool(getattr(args, "network_preprocessing_export_csv", True)),
        )

@dataclass(frozen=True)
class EncodedNetwork:
    kinase_ids: jax.Array
    substrate_ids: jax.Array
    kinase_node_ids: jax.Array
    substrate_node_ids: jax.Array
    site_ids: jax.Array
    edge_weight: jax.Array
    support_count: jax.Array
    site_observed: jax.Array
    kinase_observed: jax.Array
    kinase_labels: tuple[str, ...]
    substrate_labels: tuple[str, ...]
    site_labels: tuple[str, ...]

@dataclass(frozen=True)
class TripletTable:
    kinase_ids: jax.Array; site_ids: jax.Array; substrate_ids: jax.Array
    score: jax.Array; support_count: jax.Array; flags: jax.Array

@dataclass(frozen=True)
class SparseThetaTensor:
    indices: jax.Array; values: jax.Array; shape: tuple[int, int, int]

@dataclass(frozen=True)
class MotifTable:
    motif_type: jax.Array; node_a: jax.Array; node_b: jax.Array; node_c: jax.Array
    edge_mask: jax.Array; score: jax.Array
    node_labels: tuple[str, ...] = ()

@dataclass(frozen=True)
class IdentifiabilityDiagnostics:
    retained_param_mask: jax.Array; group_id: jax.Array; local_rank_estimate: int
    redundancy_score: jax.Array; design_column_norm: jax.Array

@dataclass(frozen=True)
class NetworkPreprocessingResult:
    encoded: EncodedNetwork; discovered: TripletTable; pruned: TripletTable
    theta: SparseThetaTensor; motifs: MotifTable | None
    identifiability: IdentifiabilityDiagnostics | None; summary: Mapping[str, Any]
