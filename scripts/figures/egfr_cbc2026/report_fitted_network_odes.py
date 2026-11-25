#!/usr/bin/env python3
"""Export fitted *distributive* networkmodel ODE equations and existing reports.

Run at the PhosKinTime repository root, for example:
    python scripts/report_fitted_network_odes.py --results results_network_distributive --gene EGFR
    python scripts/report_fitted_network_odes.py --results results_network_distributive

The numerical rates come from fitted_params_picked.json (the picked solution),
not posterior_theta0.npy (which may describe a different optimizer selection).
The network topology and regulator proxy map are rebuilt with the repository's
PosteriorObjective module by default. No fitting, MCMC, or ODE integration runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Make the repository root importable when running from scripts/.
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def latex_escape(value: object) -> str:
    table = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
             '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
             '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(table.get(c, c) for c in str(value))


def mm(name: str) -> str:
    return r'\mathrm{' + latex_escape(name) + '}'


def num(x: float, sig: int = 6) -> str:
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f'Nonfinite numerical coefficient: {x}')
    return f'{x:.{sig}g}'


def signed_terms(items: list[tuple[float, str]], sig: int = 6) -> str:
    chunks = []
    for c, symbol in items:
        if abs(c) < 1e-15:
            continue
        a = num(abs(c), sig)
        term = f'{a}\\,{symbol}'
        if not chunks:
            chunks.append(('-' if c < 0 else '') + term)
        else:
            chunks.append(('-' if c < 0 else '+') + term)
    return ' '.join(chunks) if chunks else '0'


def read_csv(path: Path, columns: set[str] | None = None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Missing required file: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        missing = (columns or set()) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        return list(reader)


def topology_from_exports(root: Path) -> dict:
    # Fallback for a portable report from exported files only; exact proxy
    # mappings cannot be reconstructed from the exported TF matrix alone.
    genes_rows = read_csv(root / 'model_parameters_genes.csv', {'Protein_Gene'})
    kinase_rows = read_csv(root / 'model_parameters_kinases.csv', {'Kinase'})
    site_rows = read_csv(root / 'S_rates_picked.csv', {'protein', 'psite', 'time', 'S'})
    w_rows = read_csv(root / 'network_W_global.csv', {'Site', 'Kinase', 'Weight'})
    tf_rows = read_csv(root / 'network_tf_mat.csv', {'Target', 'Source_TF', 'Weight'})
    genes = [str(r['Protein_Gene']) for r in genes_rows]
    kinases = [str(r['Kinase']) for r in kinase_rows]
    sites = {g: [] for g in genes}
    for row in site_rows:
        g, s = str(row['protein']), str(row['psite'])
        if g in sites and s not in sites[g]:
            sites[g].append(s)
    w: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in w_rows:
        full = str(row['Site'])
        found = [(g, s) for g in genes for s in sites[g] if f'{g}_{s}' == full]
        if len(found) != 1:
            raise ValueError(f'Cannot resolve kinase-to-site label {full!r} uniquely')
        w.setdefault(found[0], []).append((str(row['Kinase']), float(row['Weight'])))
    tf = {g: [] for g in genes}
    for row in tf_rows:
        target = str(row['Target'])
        if target in tf:
            tf[target].append((str(row['Source_TF']), float(row['Weight'])))
    return dict(genes=genes, kinases=kinases, sites=sites, w=w, tf=tf,
                proxies={}, authoritative=False)


def topology_from_model(root: Path, report_dir: Path) -> dict:
    # This is the actual production objective-rebuild API. It re-creates the
    # same Index, sparse matrices, kinase input map, and orphan proxy rules.
    from networkmodel.PosteriorObjective import build_networkmodel_posterior_context
    from networkmodel.config import MODEL

    if int(MODEL) != 0:
        raise RuntimeError(
            f'Active networkmodel.config.MODEL={MODEL}. This exporter writes '
            'MODEL=0 (distributive) equations only; refusing a mismatched report.'
        )
    ctx = build_networkmodel_posterior_context(
        run_config_path=root / 'posterior_payload' / 'posterior_run_config.json',
        chain_output_dir=report_dir / 'context_rebuild',
    )
    bound = getattr(ctx.objective_fun, '__self__', None)
    if bound is None or not hasattr(bound, 'sys'):
        raise RuntimeError('Unexpected objective API: cannot recover System from bound objective')
    system = bound.sys
    idx = system.idx
    genes = list(idx.proteins)
    kinases = list(idx.kinases)
    sites = {g: list(idx.sites[i]) for i, g in enumerate(genes)}
    W = system.W_global.tocoo()
    T = system.tf_mat.tocoo()
    site_keys = [(g, s) for g in genes for s in sites[g]]
    if W.shape != (len(site_keys), len(kinases)):
        raise ValueError('Unexpected W matrix shape')
    if T.shape != (len(genes), len(genes)):
        raise ValueError('Unexpected TF matrix shape')
    w = {}
    tf = {g: [] for g in genes}
    for i, j, v in zip(W.row, W.col, W.data):
        w.setdefault(site_keys[int(i)], []).append((kinases[int(j)], float(v)))
    for i, j, v in zip(T.row, T.col, T.data):
        tf[genes[int(i)]].append((genes[int(j)], float(v)))
    return dict(genes=genes, kinases=kinases, sites=sites, w=w, tf=tf,
                proxies=dict(getattr(idx, 'proxy_map', {})), authoritative=True)


def load_parameters(root: Path, topology: dict) -> dict:
    path = root / 'fitted_params_picked.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} is required: do not substitute posterior_theta0.npy because '
            'the posterior initial vector and picked fitted solution can differ.'
        )
    data = json.loads(path.read_text())
    n = len(topology['genes'])
    ns = sum(map(len, topology['sites'].values()))
    nk = len(topology['kinases'])
    shapes = {'A_i': n, 'B_i': n, 'C_i': n, 'D_i': n, 'E_i': n,
              'Dp_i': ns, 'c_k': nk, 'tf_scale': 1}
    for key, size in shapes.items():
        if key not in data:
            raise ValueError(f'{path}: missing parameter {key}')
        raw = data[key]
        if key == 'tf_scale':
            raw = raw if isinstance(raw, list) else [raw]
        if len(raw) != size:
            raise ValueError(f'{key}: got {len(raw)} values, expected {size}')
        values = [float(x) for x in raw]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'{key}: nonfinite values')
        data[key] = values
    kin = dict(zip(topology['kinases'], data['c_k']))
    gene = {}
    site = {}
    for i, g in enumerate(topology['genes']):
        gene[g] = {k: data[k][i] for k in ('A_i', 'B_i', 'C_i', 'D_i', 'E_i')}
    ix = 0
    for g in topology['genes']:
        for s in topology['sites'][g]:
            site[(g, s)] = data['Dp_i'][ix]
            ix += 1
    return dict(gene=gene, site=site, kin=kin, tf_scale=data['tf_scale'][0])


def validate_against_exports(root: Path, topo: dict, pars: dict, tolerance=1e-6) -> None:
    gene_rows = read_csv(root / 'model_parameters_genes.csv', {'Protein_Gene', 'Synthesis_A'})
    site_rows = read_csv(root / 'S_rates_picked.csv', {'protein', 'psite', 'time', 'S'})
    w_rows = read_csv(root / 'network_W_global.csv', {'Site', 'Kinase', 'Weight'})
    tf_rows = read_csv(root / 'network_tf_mat.csv', {'Target', 'Source_TF', 'Weight'})
    gene_columns = {
        'Synthesis_A': 'A_i', 'mRNA_Degradation_B': 'B_i',
        'Translation_C': 'C_i', 'Protein_Degradation_D': 'D_i',
        'De-Phosphorylation_E': 'E_i',
    }
    for row in gene_rows:
        g = row['Protein_Gene']
        if g not in pars['gene']:
            raise ValueError(f'Gene {g} in exported table but not rebuilt topology')
        for column, key in gene_columns.items():
            if column in row and row[column] not in (None, ''):
                if not math.isclose(float(row[column]), pars['gene'][g][key],
                                    rel_tol=tolerance, abs_tol=tolerance):
                    raise ValueError(f'Picked parameter mismatch for {g}: {key}')
        if 'Global_TF_Scale' in row and row['Global_TF_Scale'] not in ('',None):
            if not math.isclose(float(row['Global_TF_Scale']), pars['tf_scale'],
                                rel_tol=tolerance, abs_tol=tolerance):
                raise ValueError(f'Picked tf_scale mismatch for {g}')
    kinase_rows = read_csv(root/'model_parameters_kinases.csv', {'Kinase','Activity_Scale_ck'})
    for row in kinase_rows:
        k = row['Kinase']
        if k not in pars['kin'] or not math.isclose(float(row['Activity_Scale_ck']),pars['kin'][k],
                                                   rel_tol=tolerance,abs_tol=tolerance):
            raise ValueError(f'Picked kinase gain mismatch for {k}')
    gene_site_rows = read_csv(root/'model_parameters_genes_psites.csv',
                              {'Protein_Gene','psite','Phospho_Degradation_Dp'})
    for row in gene_site_rows:
        key=(row['Protein_Gene'],row['psite'])
        if key in pars['site'] and row['Phospho_Degradation_Dp'] not in (None,''):
            if not math.isclose(float(row['Phospho_Degradation_Dp']),pars['site'][key],
                                rel_tol=tolerance,abs_tol=tolerance):
                raise ValueError(f'Picked phosphosite decay mismatch for {key}')
    expected_site_keys = {(g, s) for g in topo['genes'] for s in topo['sites'][g]}
    observed_site_keys = {(r['protein'], r['psite']) for r in site_rows}
    if expected_site_keys != observed_site_keys:
        raise ValueError(f'Site export mismatch: missing={expected_site_keys-observed_site_keys}, '
                         f'extra={observed_site_keys-expected_site_keys}')
    if topo['authoritative']:
        from_model_w = {(g + '_' + s, k): float(v) for (g,s),edges in topo['w'].items() for k,v in edges}
        from_export_w = {(r['Site'],r['Kinase']):float(r['Weight']) for r in w_rows}
        if from_model_w.keys() != from_export_w.keys() or any(
            not math.isclose(v, from_export_w[k], abs_tol=tolerance, rel_tol=tolerance)
            for k,v in from_model_w.items()):
            raise ValueError('Rebuilt W and saved network_W_global.csv differ; refusing mixed network report')
        from_model_tf = {(g,h):float(v) for g,edges in topo['tf'].items() for h,v in edges}
        from_export_tf = {(r['Target'],r['Source_TF']):float(r['Weight']) for r in tf_rows}
        if from_model_tf.keys() != from_export_tf.keys() or any(
            not math.isclose(v, from_export_tf[k], abs_tol=tolerance, rel_tol=tolerance)
            for k,v in from_model_tf.items()):
            raise ValueError('Rebuilt TF and saved network_tf_mat.csv differ; refusing mixed network report')


def drv_tex(g: str, s: str) -> str:
    return 'S_{' + mm(g) + ',' + mm(s) + '}(t)'


def gene_tex(g: str, kind: str) -> str:
    return ('R' if kind == 'rna' else 'P') + '_{' + mm(g) + '}'


def site_tex(g: str, s: str) -> str:
    return 'P_{' + mm(g) + ',' + mm(s) + '}'


def fmt_linear_terms(terms: list[tuple[float, str]]) -> str:
    if not terms:
        return '0'
    pieces=[]
    for i,(coef,name) in enumerate(terms):
        if abs(coef)<1e-15:
            continue
        prefix='-' if coef<0 else ('+' if i else '')
        pieces.append(f'{prefix}{num(abs(coef))}\\,{name}')
    if not pieces:
        return '0'
    return (r' \\ &\quad ').join(' '.join(pieces[i:i+2]) for i in range(0,len(pieces),2))


def fmt_rate_expansion(g: str, s: str, topo: dict, pars: dict) -> str:
    return fmt_linear_terms([
        (weight * pars['kin'][k], 'u_{' + mm(k) + '}(t)')
        for k,weight in topo['w'].get((g,s), [])
    ])


def make_tex(topo: dict, pars: dict, genes: list[str], root: Path) -> str:
    preamble = r'''\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,margin=19mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,longtable,booktabs}
\usepackage[hidelinks]{hyperref}
\allowdisplaybreaks
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\newcommand{\softplus}{\operatorname{softplus}}
\begin{document}
\begin{center}
{\Large\bfseries Fitted Distributive Network ODEs}\\[4pt]
{\small PhosKinTime networkmodel: picked fitted parameters; not posterior summaries}
\end{center}
'''
    out = [preamble, r'\section*{Model definition}']
    out += [r'''The dynamic kinase drive uses observed kinase inputs $u_k(t)$ and fitted gains $c_k$.
The network coefficients $W_{(g,s),k}$ and $T_{g,h}$ are fixed exported
interaction weights; $S_{g,s}(t)$ is time-varying, not a fitted constant.
\begin{align*}
K_k(t) &= c_k u_k(t), \\
S_{g,s}(t) &= \sum_{k} W_{(g,s),k} c_k u_k(t), \\
U_g(t) &= \frac{\sum_h T_{g,h}V_h(t)}{d_g}, \\
q_g(t) &= \frac{U_g(t)}{1+|U_g(t)|},
\end{align*}
\begin{equation*}
F_g(t)=\begin{cases}
1+\dfrac{TF_SCALE_REPLACEq_g(t)}{1+q_g(t)+10^{-6}}, & q_g(t)\geq 0,\\[5pt]
\dfrac{1}{1+TF_SCALE_REPLACE|q_g(t)|}, & q_g(t)<0.
\end{cases}
\end{equation*}
For kinase and proxy-driven TF sources $V_h(t)$ is the corresponding
scaled kinase input $K_k(t)$; otherwise $V_h(t)$ is the sum of the
unphosphorylated and site-phosphorylated states of source protein $h$.
Kinase inputs are held piecewise constant on the model input time grid.
The TF normalization is $d_g=\sum_h |T_{g,h}|$ if nonzero and $d_g=1$ otherwise.
The forward model is the \texttt{networkmodel.backend.make\_networkmodel\_rhs}
JAX/Diffrax distributive branch (MODEL=0).
''']
    out[-1] = out[-1].replace('TF_SCALE_REPLACE',num(pars['tf_scale']))
    out += [r'\section*{Numerical ODEs}']
    ic_path=root/'initial_conditions_y0.csv'
    ic_rows=read_csv(ic_path, {'entity','kind','substate','value'}) if ic_path.exists() else []
    ic_by_gene={}
    for row in ic_rows:
        ic_by_gene.setdefault(row['entity'],[]).append(row)

    for g in genes:
        p = pars['gene'][g]
        R,P = gene_tex(g,'rna'),gene_tex(g,'protein')
        f = 'F_{'+mm(g)+'}(t)'
        sites = topo['sites'][g]
        out.append(r'\subsection*{'+latex_escape(g)+'}')
        out.append(r'\begin{align*}')
        out.append(r'\frac{d '+R+r'}{dt} &= '+num(p['A_i'])+r'\,'+f+' - '+num(p['B_i'])+r'\,'+R+r', \\')
        if sites:
            sum_S = r'\sum_{s} S_{'+mm(g)+r',s}(t)'
            sum_P = r'\sum_{s} P_{'+mm(g)+r',s}(t)'
            out.append(r'\frac{d '+P+r'}{dt} &= '+num(p['C_i'])+r'\,'+R+' - ('+num(p['D_i'])+' + '+sum_S+')'+P+' + '+num(p['E_i'])+r'\,'+sum_P+r', \\')
        else:
            out.append(r'\frac{d '+P+r'}{dt} &= '+num(p['C_i'])+r'\,'+R+' - '+num(p['D_i'])+r'\,'+P+r'. \\')
        out.append(r'\end{align*}')
        if sites:
            out.append(r'{\small\begin{align*}')
            for s in sites:
                Ps=site_tex(g,s)
                S=drv_tex(g,s)
                val=pars['site'][(g,s)]
                total=p['E_i']+p['D_i']+val
                out.append(r'\frac{d '+Ps+r'}{dt} &= '+S+r'\,'+P+' - '+num(total)+r'\,'+Ps+r', \\')
            out.append(r'\end{align*}}')
            out.append(r'\textit{Site-level phosphorylation drives (fitted kinase gains applied to exported weights):}')
            out.append(r'{\small\begin{align*}')
            for s in sites:
                terms=fmt_rate_expansion(g,s,topo,pars)
                out.append(drv_tex(g,s) + ' &= '+terms+r', \\')
            out.append(r'\end{align*}}')
        tf_edges=topo['tf'].get(g, [])
        d=sum(abs(v) for _,v in tf_edges) if tf_edges else 0.0
        d=d if d>1e-12 else 1.0
        if tf_edges:
            terms=[(v/d,'V_{'+mm(h)+'}(t)') for h,v in tf_edges]
            out.append(r'\textit{TF-input expansion (degree-normalized network coefficients):}')
            out.append(r'{\small\begin{align*}')
            out.append('U_{'+mm(g)+'}(t) &= '+fmt_linear_terms(terms)+r', \\')
            out.append(r'\end{align*}}')
            out.append('TF normalization: $d_{'+mm(g)+'}='+num(d)+r'$.')
            proxies=[(h,topo['proxies'][h]) for h,_ in tf_edges if h in topo['proxies']]
            if proxies:
                out.append('TF proxy drivers: '+', '.join(latex_escape(h)+' $\\rightarrow$ '+latex_escape(k) for h,k in proxies)+'.')
        else:
            out.append('No incoming TF edges in the saved network; $U_{'+mm(g)+'}(t)=0$.')
        if ic_by_gene.get(g):
            out.append(r'\textit{Saved initial conditions (from the fitted run):}')
            out.append(r'{\small\begin{align*}')
            for row in ic_by_gene[g]:
                if row['kind']=='mRNA' and row['substate']=='R':
                    symbol=gene_tex(g,'rna')
                elif row['kind']=='protein_state' and row['substate']=='P0':
                    symbol=gene_tex(g,'protein')
                elif row['kind']=='phospho_state' and row['substate'] in topo['sites'][g]:
                    symbol=site_tex(g,row['substate'])
                else:
                    continue
                out.append(symbol+'(0) &= '+num(float(row['value']))+r', \\')
            out.append(r'\end{align*}}')
        out.append('')
    out += [r'''\section*{Interpretation and provenance}
The fitted kinetic constants are loaded from \texttt{fitted\_params\_picked.json}.
The topology, ordered parameter indices, TF proxy map and sparse interaction
weights are reconstructed using the existing \texttt{PosteriorObjective} module.
The phosphorylation drive $S_{g,s}(t)$ is a time-dependent combination of
kinase inputs; a single observed or fitted rate at one time point should not be
substituted for this function.

This document lists the ODE right-hand sides and coefficients, but it does not
contain an explicit solution of the system. All parameter values shown are
rounded for reporting only; calculations use their full saved precision.
\end{document}
''']
    return '\n'.join(out)


def save_tables(out_dir: Path, topo: dict, pars: dict, genes: list[str]) -> None:
    with (out_dir/'kinase_to_site_coefficients.csv').open('w', newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['target_gene','psite','kinase','network_weight_W','fitted_gain_c_k','effective_coefficient_W_times_c_k'])
        for g in genes:
            for s in topo['sites'][g]:
                for k,w in topo['w'].get((g,s),[]):
                    wr.writerow([g,s,k,w,pars['kin'][k],w*pars['kin'][k]])
    with (out_dir/'tf_to_gene_coefficients.csv').open('w', newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['target_gene','source_tf','network_weight_T','driver_proxy_if_any'])
        for g in genes:
            for h,w in topo['tf'].get(g,[]):
                wr.writerow([g,h,w,topo['proxies'].get(h,'')])
    with (out_dir/'fitted_rate_parameters.csv').open('w',newline='') as f:
        wr=csv.writer(f)
        wr.writerow(['gene','psite','A','B','C','D','E','Dp','tf_scale'])
        for g in genes:
            p=pars['gene'][g]
            site_names=topo['sites'][g] or ['']
            for s in site_names:
                wr.writerow([g,s,p['A_i'],p['B_i'],p['C_i'],p['D_i'],p['E_i'],
                             pars['site'].get((g,s),''),pars['tf_scale']])


def call_existing_reports(root: Path, out_dir: Path, genes: list[str]) -> list[str]:
    import pandas as pd
    from networkmodel.export import plot_s_rates_report, save_gene_timeseries_plots
    outputs=[]
    rates=pd.read_csv(root/'S_rates_picked.csv')
    traj=pd.read_csv(root/'model_trajectories.csv')
    for gene in genes:
        name=re.sub(r'[^a-zA-Z0-9_.-]+','_',gene)
        subset=rates[rates['protein'].astype(str)==gene]
        if not subset.empty:
            rate_csv=out_dir/f'S_rates_{name}.csv'
            subset.to_csv(rate_csv,index=False)
            result=out_dir/f'S_rates_{name}.pdf'
            plot_s_rates_report(rate_csv,result,log_x=False,
                                top_k_sites_per_protein=24,max_sites_per_page=12,
                                ncols=3,heatmap_per_protein=True)
            outputs.append(str(result))
        blocks={}
        for typ in ('Protein','RNA','Phosphorylation'):
            part=traj[(traj['protein'].astype(str)==gene)&(traj['type']==typ)]
            cols=['protein','time']+(['psite'] if typ=='Phosphorylation' else [])
            blocks[typ]=(part[cols+['fc_obs']].rename(columns={'fc_obs':'fc'}).copy(),
                         part[cols+['fc_pred']].copy())
        if any(not b[0].empty or not b[1].empty for b in blocks.values()):
            plot=save_gene_timeseries_plots(
                gene=gene,
                df_prot_obs=blocks['Protein'][0],df_prot_pred=blocks['Protein'][1],
                df_rna_obs=blocks['RNA'][0],df_rna_pred=blocks['RNA'][1],
                df_phos_obs=blocks['Phosphorylation'][0],df_phos_pred=blocks['Phosphorylation'][1],
                output_dir=str(out_dir), filename_prefix='fit')
            if plot:
                outputs.append(str(plot))
    return outputs


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path('results_network_distributive'),
                    help='Existing fitted results directory')
    ap.add_argument('--gene',action='append',default=[],help='Gene to include, repeatable. Default: all genes')
    ap.add_argument('--output',type=Path,help='Output path (default: <results>/fitted_ode_report)')
    ap.add_argument('--offline',action='store_true',help='Use saved exports only (cannot reconstruct orphan-TF proxy map)')
    ap.add_argument('--skip-native-reports',action='store_true',help='Skip module-produced rate PDF and time-series plots')
    ap.add_argument('--no-compile',action='store_true',help='Write LaTeX only; do not compile the equations PDF')
    args=ap.parse_args(argv)
    root=args.results.expanduser().resolve()
    out_dir=(args.output or root/'fitted_ode_report').expanduser().resolve()
    if not root.is_dir():
        ap.error(f'No results directory: {root}')
    out_dir.mkdir(parents=True,exist_ok=True)
    if args.offline:
        topo=topology_from_exports(root)
    else:
        topo=topology_from_model(root,out_dir)
    pars=load_parameters(root,topo)
    validate_against_exports(root,topo,pars)
    genes=args.gene or topo['genes']
    missing=set(genes)-set(topo['genes'])
    if missing:
        ap.error('Unknown genes: '+', '.join(sorted(missing)))
    genes=list(dict.fromkeys(genes))
    save_tables(out_dir,topo,pars,genes)
    tex=make_tex(topo,pars,genes,root)
    tex_path=out_dir/'fitted_network_equations.tex'
    tex_path.write_text(tex,encoding='utf-8')
    pdf_status='not requested'
    if not args.no_compile:
        exe=shutil.which('pdflatex')
        if exe:
            run=subprocess.run([exe,'-interaction=nonstopmode','-halt-on-error','-output-directory',str(out_dir),str(tex_path)],
                               cwd=out_dir,capture_output=True,text=True)
            if run.returncode:
                pdf_status=f'LaTeX failed; inspect {tex_path.with_suffix(".log")}'
                print(run.stdout[-2500:],file=sys.stderr)
            else:
                pdf_status=str(tex_path.with_suffix('.pdf'))
        else:
            pdf_status='pdflatex unavailable; LaTeX source generated'
    native=[]
    if not args.skip_native_reports:
        try:
            native=call_existing_reports(root,out_dir,genes)
        except Exception as exc:
            # Equations still useful; error preserved in report and surfaced.
            native=[f'Existing report functions failed: {type(exc).__name__}: {exc}']
            print(native[0], file=sys.stderr)
    provenance={
        'results_directory':str(root), 'genes':genes,'model':'distributive (MODEL=0)',
        'topology_reconstructed_from_code':topo['authoritative'],
        'fit_parameter_source':str(root/'fitted_params_picked.json'),
        'equation_implementation':'networkmodel.backend.make_networkmodel_rhs',
        'native_report_functions':['networkmodel.export.plot_s_rates_report','networkmodel.export.save_gene_timeseries_plots'],
        'equations_pdf':pdf_status,
        'native_reports':native,
    }
    (out_dir/'report_manifest.json').write_text(json.dumps(provenance,indent=2))
    readme=[f'# Fitted network ODE report', '', f'Genes: {", ".join(genes)}',
            '', '## Outputs', '',
            '- `fitted_network_equations.tex`: fitted numerical ODEs, regulatory/input definitions.',
            '- `fitted_rate_parameters.csv`: picked kinetic parameters by gene/site.',
            '- `kinase_to_site_coefficients.csv`: full W × fitted kinase gains.',
            '- `tf_to_gene_coefficients.csv`: TF edges and proxy information.',
            f'- Equation PDF: {pdf_status}',
            *[f'- Native output: `{x}`' for x in native], '',
            '## Provenance', '',
            'Kinetic parameters from `fitted_params_picked.json` (picked solution).',
            'The fitted network equations are derived from the actual MODEL=0 JAX RHS.',
            'Site drives retain measured time-varying kinase inputs.',
            'No optimization or sampling was repeated.',
            '', 'Caution: equations reflect model assumptions; the gene-level fit is not a causal validation.',
            '']
    if not topo['authoritative']:
        readme += ['OFFLINE MODE: TF orphan proxy mappings are not recoverable from saved exported edges.',
                   'For an exact complete report rerun without `--offline`.', '']
    (out_dir/'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print('Report:',out_dir)
    print('Gene count:',len(genes),'site count:',sum(len(topo['sites'][g]) for g in genes))
    print('Equations:',tex_path)
    print('PDF:',pdf_status)
    print('Existing module reports:',len(native))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
