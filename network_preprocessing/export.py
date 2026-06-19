from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

def _base(output_dir, subdir="network_preprocessing"):
    try:
        from common.results import ensure_result_dir
        paths=ensure_result_dir(output_dir); root=paths["tables"] / subdir
    except Exception:
        root=Path(output_dir)/subdir
    root.mkdir(parents=True, exist_ok=True)
    return root

def export_preprocessing_result(result, output_dir, *, labels=None):
    root=_base(output_dir); enc=result.encoded; paths={}
    def trip_df(t):
        return pd.DataFrame({"kinase_id":np.asarray(t.kinase_ids),"site_id":np.asarray(t.site_ids),"substrate_id":np.asarray(t.substrate_ids),"kinase":[enc.kinase_labels[int(i)] for i in np.asarray(t.kinase_ids)],"site":[enc.site_labels[int(i)] for i in np.asarray(t.site_ids)],"substrate":[enc.substrate_labels[int(i)] for i in np.asarray(t.substrate_ids)],"score":np.asarray(t.score),"support_count":np.asarray(t.support_count),"flags":np.asarray(t.flags)})
    for name,t in [("discovered_hyperedges",result.discovered),("retained_triplets",result.pruned)]:
        p=root/f"{name}.csv"; trip_df(t).to_csv(p,index=False); paths[name]=p
    p=root/"sparse_theta_indices_values.csv"; trip_df(result.pruned).assign(theta_value=np.asarray(result.theta.values)).to_csv(p,index=False); paths["theta_csv"]=p
    p=root/"sparse_theta.npz"; np.savez_compressed(p, indices=np.asarray(result.theta.indices), values=np.asarray(result.theta.values), shape=np.asarray(result.theta.shape)); paths["theta_npz"]=p
    if result.motifs is not None:
        m=result.motifs; p=root/"motif_table.csv"; pd.DataFrame({"motif_type":np.asarray(m.motif_type),"node_a":np.asarray(m.node_a),"node_b":np.asarray(m.node_b),"node_c":np.asarray(m.node_c),"edge_mask":np.asarray(m.edge_mask),"score":np.asarray(m.score)}).to_csv(p,index=False); paths["motifs"]=p
    if result.identifiability is not None:
        d=result.identifiability; p=root/"identifiability_diagnostics.csv"; pd.DataFrame({"retained_param_mask":np.asarray(d.retained_param_mask),"group_id":np.asarray(d.group_id),"redundancy_score":np.asarray(d.redundancy_score),"design_column_norm":np.asarray(d.design_column_norm)}).to_csv(p,index=False); paths["identifiability"]=p
    p=root/"summary_statistics.json"; p.write_text(json.dumps(result.summary,indent=2,sort_keys=True)+"\n"); paths["summary"]=p
    return paths
