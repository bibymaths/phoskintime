from __future__ import annotations

from pathlib import Path

import pytest


def _base_network_config(path: Path, hyperedge_section: str = "") -> None:
    path.write_text(
        f'''
[networkmodel]
kinase_net = "kin.csv"
tf_net = "tf.csv"
ms = "ms.csv"
rna = "rna.csv"
output_dir = "out"

{hyperedge_section}

[networkmodel.timepoints]
protein = [0, 1]
rna = [0, 1]
phospho_protein = [0, 1]
''',
        encoding="utf-8",
    )


def test_hyperedge_preprocessing_defaults_are_disabled_for_missing_section(tmp_path):
    pytest.importorskip("numpy")
    from config_loader import load_config_toml

    conf = tmp_path / "network.toml"
    _base_network_config(conf)

    cfg = load_config_toml(conf)

    assert cfg.enable_hyperedge_preprocessing is False
    assert cfg.hyperedge_preprocessing_output_subdir == "network_preprocessing"
    assert cfg.hyperedge_discovery_threshold == pytest.approx(0.0)
    assert cfg.hyperedge_pruning_threshold == pytest.approx(0.0)
    assert cfg.hyperedge_motif_detection is True
    assert cfg.hyperedge_sparse_tensor_export is True
    assert cfg.hyperedge_identifiability_preprocessing is True
    assert cfg.hyperedge_max_triplets is None
    assert cfg.hyperedge_batch_size == 65536
    assert cfg.hyperedge_plot_generation is True
    assert cfg.hyperedge_csv_export is True


def test_hyperedge_preprocessing_explicit_enablement_flows_to_runner_defaults(tmp_path):
    pytest.importorskip("numpy")
    from config_loader import load_config_toml
    from networkmodel import runner

    conf = tmp_path / "network.toml"
    _base_network_config(
        conf,
        '''
[networkmodel.hyperedge_preprocessing]
enable_hyperedge_preprocessing = true
output_subdir = "hyperedge_outputs"
hyperedge_discovery_threshold = 0.2
hyperedge_pruning_threshold = 0.7
motif_detection = false
sparse_tensor_export = false
identifiability_preprocessing = false
max_triplets = 25
batch_size = 128
plot_generation = false
csv_export = false
''',
    )

    cfg = load_config_toml(conf)
    assert cfg.enable_hyperedge_preprocessing is True
    assert cfg.hyperedge_preprocessing_output_subdir == "hyperedge_outputs"
    assert cfg.hyperedge_discovery_threshold == pytest.approx(0.2)
    assert cfg.hyperedge_pruning_threshold == pytest.approx(0.7)
    assert cfg.hyperedge_motif_detection is False
    assert cfg.hyperedge_sparse_tensor_export is False
    assert cfg.hyperedge_identifiability_preprocessing is False
    assert cfg.hyperedge_max_triplets == 25
    assert cfg.hyperedge_batch_size == 128
    assert cfg.hyperedge_plot_generation is False
    assert cfg.hyperedge_csv_export is False

    args, _ = runner.parse_runtime_args(["--conf", str(conf)])
    assert args.enable_network_preprocessing is True
    assert args.network_preprocessing_output_subdir == "hyperedge_outputs"
    assert args.network_preprocessing_min_score == pytest.approx(0.7)
    assert args.network_preprocessing_max_triplets == 25
    assert args.network_preprocessing_batch_size == 128
    assert args.network_preprocessing_enable_motifs is False
    assert args.network_preprocessing_export_sparse_tensor is False
    assert args.network_preprocessing_generate_plots is False
    assert args.network_preprocessing_export_csv is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hyperedge_discovery_threshold", "-0.1", "hyperedge_discovery_threshold must be non-negative"),
        ("hyperedge_pruning_threshold", "-0.1", "hyperedge_pruning_threshold must be non-negative"),
        ("max_triplets", "-1", "max_triplets must be non-negative"),
        ("batch_size", "0", "batch_size must be positive"),
    ],
)
def test_hyperedge_preprocessing_invalid_values_raise_clear_errors(tmp_path, field, value, message):
    pytest.importorskip("numpy")
    from config_loader import load_config_toml

    conf = tmp_path / "network.toml"
    _base_network_config(
        conf,
        f'''
[networkmodel.hyperedge_preprocessing]
{field} = {value}
''',
    )

    with pytest.raises(ValueError, match=message):
        load_config_toml(conf)
