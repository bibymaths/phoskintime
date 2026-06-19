from __future__ import annotations
from pathlib import Path
import numpy as np

def _plot_dir(output_dir, subdir="network_preprocessing"):
    try:
        from common.results import ensure_result_dir
        root=ensure_result_dir(output_dir)["plots"] / subdir
    except Exception:
        root=Path(output_dir)/subdir/"plots"
    root.mkdir(parents=True, exist_ok=True); return root

def plot_preprocessing_result(result, output_dir, *, style="paper", output_subdir="network_preprocessing"):
    import matplotlib.pyplot as plt
    root=_plot_dir(output_dir, output_subdir); paths={}
    scores=np.asarray(result.discovered.score)
    fig,ax=plt.subplots(); ax.hist(scores, bins=min(30, max(1, len(scores)))); ax.set_title("Hyperedge score distribution"); ax.set_xlabel("score"); ax.set_ylabel("count"); p=root/"hyperedge_score_distribution.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["scores"]=p
    fig,ax=plt.subplots(); ax.bar(["retained","removed"],[result.summary["n_retained"], result.summary["n_pruned"]]); ax.set_title("Retained vs removed triplets"); p=root/"retained_vs_removed_triplets.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["retained_removed"]=p
    deg=np.bincount(np.asarray(result.pruned.kinase_ids), minlength=len(result.encoded.kinase_labels)) if result.pruned.kinase_ids.size else np.array([])
    fig,ax=plt.subplots(); ax.hist(deg, bins=min(20,max(1,len(deg)))); ax.set_title("Kinase degree distribution"); p=root/"degree_distribution.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["degree"]=p
    fig,ax=plt.subplots(); ax.bar(["nnz","dense size"],[result.summary["tensor_nnz"], np.prod(result.theta.shape)]); ax.set_title("Sparse tensor summary"); p=root/"sparse_tensor_summary.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["tensor"]=p
    fig,ax=plt.subplots(); ax.bar(["motifs"],[result.summary["n_motifs"]]); ax.set_title("Motif count summary"); p=root/"motif_count_summary.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["motifs"]=p
    if result.identifiability is not None:
        fig,ax=plt.subplots(); ax.hist(np.asarray(result.identifiability.redundancy_score), bins=10); ax.set_title("Identifiability diagnostics"); p=root/"identifiability_diagnostics.png"; fig.savefig(p,dpi=200,bbox_inches="tight"); plt.close(fig); paths["identifiability"]=p
    return paths
