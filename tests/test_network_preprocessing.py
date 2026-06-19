from pathlib import Path
import jax
import jax.numpy as jnp
import pandas as pd
from network_preprocessing import NetworkPreprocessingConfig, preprocess_network, preprocess_networkmodel_frames
from network_preprocessing.jax_kernels import identifiability_kernel, score_triplets


def frames():
    kin = pd.DataFrame({"protein": ["B", "C", "C", "D", "A"], "psite": ["S1", "S2", "S2", "S3", "S4"],
                        "kinase": ["A", "B", "A", "D", "A"], "alpha": [1.0, 2.0, 1.5, 1.0, 0.1]})
    pho = pd.DataFrame({"protein": ["B", "C", "D"], "psite": ["S1", "S2", "S3"], "time": [0, 0, 0], "fc": [1, 1, 1]})
    prot = pd.DataFrame({"protein": ["A", "B", "D"], "time": [0, 0, 0], "fc": [1, 1, 1]})
    return kin, pho, prot


def test_hyperedge_discovery_and_float64_jit():
    kin, pho, prot = frames()
    res = preprocess_network(kin, phospho_observations=pho, protein_observations=prot,
                             config=NetworkPreprocessingConfig(prune_self_loops=False))
    assert res.summary["n_discovered"] == 5
    assert res.discovered.score.dtype == jnp.float64
    lowered = score_triplets.lower(res.encoded.edge_weight, res.encoded.support_count, res.encoded.site_observed,
                                   res.encoded.kinase_observed, res.encoded.kinase_node_ids,
                                   res.encoded.substrate_node_ids)
    assert lowered.compile() is not None
    assert jax.config.jax_enable_x64


def test_pruning_behavior_and_sparse_tensor_validity():
    kin, pho, prot = frames()
    res = preprocess_network(kin, phospho_observations=pho, protein_observations=prot,
                             config=NetworkPreprocessingConfig(prune_self_loops=True, prune_missing_observations=True,
                                                               min_support_count=1))
    assert res.summary["n_retained"] < res.summary["n_discovered"]
    assert res.theta.indices.shape == (res.summary["n_retained"], 3)
    assert res.theta.indices.dtype == jnp.int32
    assert res.theta.values.dtype == jnp.float64
    if res.theta.indices.size:
        assert int(res.theta.indices[:, 0].max()) < res.theta.shape[0]
        assert int(res.theta.indices[:, 1].max()) < res.theta.shape[1]
        assert int(res.theta.indices[:, 2].max()) < res.theta.shape[2]


def test_self_loop_penalty_uses_common_node_namespace():
    non_loop = pd.DataFrame({
        "protein": ["B", "C"],
        "psite": ["S1", "S2"],
        "kinase": ["A", "X"],
        "alpha": [1.0, 1.0],
    })
    non_loop_res = preprocess_network(non_loop,
                                      config=NetworkPreprocessingConfig(prune_self_loops=False, enable_motifs=False))
    expected_without_penalty = jnp.log1p(jnp.float64(1.0)) + 0.5 * jnp.log1p(jnp.float64(1.0))
    assert float(non_loop_res.discovered.score[0]) == float(expected_without_penalty)

    true_loop = pd.DataFrame({
        "protein": ["A"],
        "psite": ["S1"],
        "kinase": ["A"],
        "alpha": [1.0],
    })
    loop_res = preprocess_network(true_loop,
                                  config=NetworkPreprocessingConfig(prune_self_loops=False, enable_motifs=False))
    assert float(loop_res.discovered.score[0]) == float(expected_without_penalty - 0.5)


def test_cli_max_triplets_zero_normalizes_to_none():
    from argparse import Namespace

    cfg = NetworkPreprocessingConfig.from_args(Namespace(network_preprocessing_max_triplets=0))
    assert cfg.max_triplets is None

    kin, pho, prot = frames()
    res = preprocess_network(kin, phospho_observations=pho, protein_observations=prot, config=cfg)
    assert res.summary["n_retained"] > 0


def test_motif_detection_correctness():
    kin = pd.DataFrame(
        {"protein": ["B", "C", "C"], "psite": ["S1", "S2", "S3"], "kinase": ["A", "B", "A"], "alpha": [1, 1, 1]})
    res = preprocess_network(kin, config=NetworkPreprocessingConfig(prune_self_loops=False))
    assert res.summary["n_motifs"] >= 1
    assert set(map(int, res.motifs.motif_type.tolist())) == {1}


def test_motif_detection_uses_common_node_namespace_false_positive_regression():
    kin = pd.DataFrame({
        "protein": ["C", "D", "D"],
        "psite": ["S1", "S2", "S3"],
        "kinase": ["A", "X", "A"],
        "alpha": [1.0, 1.0, 1.0],
    })
    res = preprocess_network(kin, config=NetworkPreprocessingConfig(prune_self_loops=False))
    assert res.summary["n_motifs"] == 0


def test_motif_detection_uses_common_node_namespace_positive_control():
    kin = pd.DataFrame({
        "protein": ["X", "D", "D"],
        "psite": ["S1", "S2", "S3"],
        "kinase": ["A", "X", "A"],
        "alpha": [1.0, 1.0, 1.0],
    })
    res = preprocess_network(kin, config=NetworkPreprocessingConfig(prune_self_loops=False))
    assert res.summary["n_motifs"] == 1
    motif = res.motifs
    labels = motif.node_labels
    assert (labels[int(motif.node_a[0])], labels[int(motif.node_b[0])], labels[int(motif.node_c[0])]) == ("A", "X", "D")


def test_identifiability_grouping_avoids_legacy_linear_hash_collision():
    indices = jnp.asarray([[1, 0, 0], [0, 991, 84]], dtype=jnp.int32)
    values = jnp.ones(2, dtype=jnp.float64)

    _, group_id, _, _ = identifiability_kernel(indices, values)

    assert int(group_id[0]) != int(group_id[1])


def test_identifiability_grouping_keeps_duplicate_triplets_together():
    indices = jnp.asarray([[2, 3, 4], [2, 3, 4], [2, 3, 5]], dtype=jnp.int32)
    values = jnp.ones(3, dtype=jnp.float64)

    _, group_id, _, _ = identifiability_kernel(indices, values)

    assert int(group_id[0]) == int(group_id[1])
    assert int(group_id[0]) != int(group_id[2])


def test_identifiability_grouping_large_sparse_tensor_is_deterministic():
    indices = jnp.asarray([[1, 0, 0], [0, 991, 84], [7, 1200, 99], [7, 1200, 99]], dtype=jnp.int32)
    values = jnp.ones(4, dtype=jnp.float64)

    _, group_id_a, _, _ = identifiability_kernel(indices, values)
    _, group_id_b, _, _ = identifiability_kernel(indices, values)

    assert group_id_a.tolist() == group_id_b.tolist()
    assert int(group_id_a[0]) != int(group_id_a[1])
    assert int(group_id_a[2]) == int(group_id_a[3])


def test_identifiability_and_outputs(tmp_path):
    kin, pho, prot = frames()
    res = preprocess_network(kin, phospho_observations=pho, protein_observations=prot, output_dir=tmp_path,
                             config=NetworkPreprocessingConfig(prune_self_loops=False))
    assert res.identifiability.retained_param_mask.shape[0] == res.theta.values.shape[0]
    assert (tmp_path / "tables" / "networkpruning" / "discovered_hyperedges.csv").is_file()
    assert (tmp_path / "tables" / "networkpruning" / "retained_triplets.csv").is_file()
    assert (tmp_path / "tables" / "networkpruning" / "sparse_theta_indices_values.csv").is_file()
    assert (tmp_path / "plots" / "networkpruning" / "hyperedge_score_distribution.png").is_file()
    assert (tmp_path / "plots" / "networkpruning" / "identifiability_diagnostics.png").is_file()


def test_networkmodel_preprocessing_collapses_duplicate_retained_triplets():
    kin = pd.DataFrame({
        "protein": ["B", "B", "C"],
        "psite": ["S1", "S1", "S2"],
        "kinase": ["A", "A", "X"],
        "alpha": [0.2, 0.9, 0.5],
    })
    pruned_df, res = preprocess_networkmodel_frames(
        kin, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        config=NetworkPreprocessingConfig(prune_self_loops=False, enable_motifs=False),
    )

    dup = pruned_df[(pruned_df["kinase"] == "A") & (pruned_df["protein"] == "B") & (pruned_df["psite"] == "S1")]
    assert len(dup) == 1
    assert dup["alpha"].iloc[0] == 0.9
    assert dup["support_count"].iloc[0] == 2
    assert pruned_df[["kinase", "protein", "psite"]].duplicated().sum() == 0
    assert int(res.pruned.support_count.max()) == 2


def test_pruned_worker_csv_matches_retained_triplet_diagnostics(tmp_path):
    kin = pd.DataFrame({
        "protein": ["B", "B", "C"],
        "psite": ["S1", "S1", "S2"],
        "kinase": ["A", "A", "X"],
        "alpha": [0.2, 0.9, 0.5],
    })
    pruned_df, _ = preprocess_networkmodel_frames(
        kin, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        config=NetworkPreprocessingConfig(prune_self_loops=False, enable_motifs=False),
        output_dir=tmp_path,
    )
    worker_csv = tmp_path / "networkpruning" / "pruned_network_for_workers.csv"
    worker_csv.parent.mkdir(parents=True)
    pruned_df.to_csv(worker_csv, index=False)

    retained = pd.read_csv(tmp_path / "tables" / "networkpruning" / "retained_triplets.csv")
    retained_keys = retained.assign(
        protein=retained["site"].str.split(":", n=1).str[0],
        psite=retained["site"].str.split(":", n=1).str[1],
    )[["kinase", "protein", "psite", "alpha"]].sort_values(["kinase", "protein", "psite"]).reset_index(drop=True)
    worker_keys = pd.read_csv(worker_csv)[["kinase", "protein", "psite", "alpha"]].sort_values(
        ["kinase", "protein", "psite"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(worker_keys, retained_keys)


def test_networkmodel_preprocessing_nonduplicate_input_preserves_rows():
    kin = pd.DataFrame({
        "protein": ["B", "C"],
        "psite": ["S1", "S2"],
        "kinase": ["A", "X"],
        "alpha": [0.2, 0.5],
    })
    pruned_df, _ = preprocess_networkmodel_frames(
        kin, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        config=NetworkPreprocessingConfig(prune_self_loops=False, enable_motifs=False),
    )

    expected = kin.assign(support_count=1)
    pd.testing.assert_frame_equal(
        pruned_df.sort_values(["kinase", "protein", "psite"]).reset_index(drop=True),
        expected.sort_values(["kinase", "protein", "psite"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_runner_parser_disabled_by_default():
    from networkmodel.runner import _config_defaults, build_parser
    class C:
        kinase_net = tf_net = ms_data = rna_data = phospho_data = kinopt_results = tfopt_results = "x.csv";
        results_dir = "out";
        cores = 1;
        maximum_iterations = 1;
        seed = 1;
        regularization_lambda = regularization_protein = regularization_rna = regularization_phospho = 0.0;
        normalize_fc_steady = False;
        use_initial_condition_from_data = False;
        hyperparam_scan = False;
        sensitivity_analysis = False;
        time_points_prot = (0,);
        time_points_rna = (0,);
        time_points_phospho = (0,)

    p = build_parser(_config_defaults(C(), Path('config.toml'), None))
    args = p.parse_args([])
    assert args.enable_network_preprocessing is False
    args = p.parse_args(["--enable-network-preprocessing"])
    assert args.enable_network_preprocessing is True
