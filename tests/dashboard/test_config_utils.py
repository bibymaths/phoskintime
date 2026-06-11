from __future__ import annotations

import json

from dashboard.config_utils import DashboardSelection, load_preset, parse_config_file, save_preset


def test_parse_json_yaml_and_text_configs(tmp_path):
    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.yaml"
    text_path = tmp_path / "notes.txt"
    json_path.write_text('{"alpha": 1}', encoding="utf-8")
    yaml_path.write_text("alpha: 2\nflag: true\n", encoding="utf-8")
    text_path.write_text("hello", encoding="utf-8")

    assert parse_config_file(json_path) == {"alpha": 1}
    assert parse_config_file(yaml_path)["alpha"] == 2
    assert parse_config_file(text_path) == "hello"


def test_save_and_load_json_preset(tmp_path):
    selection = DashboardSelection(
        workflow_key="networkmodel",
        run_name="run1",
        pixi_environment="dev",
        arguments={"cores": "2"},
        input_assignments={"kinase_network": "kin.csv"},
    )
    path = save_preset(selection, tmp_path / "preset.json")

    loaded = load_preset(path)

    assert json.loads(path.read_text(encoding="utf-8"))["workflow_key"] == "networkmodel"
    assert loaded.workflow_key == selection.workflow_key
    assert loaded.arguments == {"cores": "2"}
