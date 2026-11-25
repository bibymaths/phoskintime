#!/usr/bin/env python3
"""72-hour PhosKinTime distributive-ODE phase-space analysis, without MCMC.

Re-simulates the *full* network for the fitted model and EGFR-specific kinetic
parameter draws. Reads an existing ``parameter_ensemble.csv`` (preferred) or
samples a truncated fitted-centered lognormal ensemble. Parameter ensembles are
NOT automatically posterior samples. Saves nominal 70-state trajectories,
phase-space coordinates, individual ensemble trajectories (NPZ), pointwise
ensemble summaries, and publication-style PDF/PNG two-panel figures.

Forcing is piecewise constant on the original kinase time grid. Beyond the
final recorded kinase input, the final input is held constant: this is a model
extrapolation scenario, not an experimental constraint.

Dependencies: numpy, scipy, pandas, matplotlib and the original repository
``scripts/phoskintime_dynestyx_smoothing.py`` (for load_run/reference_rhs).

Example:
  python scripts/phoskintime_phase_portraits_72h.py \\
    --source . --results results_network_distributive \\
    --parameter-csv results_network_distributive/egfr_forward_72h/parameter_ensemble.csv \\
    --out results_network_distributive/egfr_phase_72h \\
    --protein EGFR --hours 72 --points 10000 --draws 100 --dpi 400
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

EPS = 1e-12
TEAL = '#126b81'
PALE = '#a8cbd4'
RUST = '#a6532e'
INK = '#29343b'
GRID = '#e1e5e8'


def load_original(source: Path):
    # Current layout first; the pre-reorganisation layout keeps --source usable for older checkouts.
    candidates = [source / 'scripts' / 'figures' / 'egfr_cbc2026' / 'phoskintime_dynestyx_smoothing.py',
                  source / 'scripts' / 'phoskintime_dynestyx_smoothing.py']
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            'Missing original PhosKinTime script; tried: ' + ', '.join(str(c) for c in candidates) + '. '
            'This script requires its validated load_run/reference_rhs functions.'
        )
    spec = importlib.util.spec_from_file_location('_phoskintime_phase_reference', path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    spec.loader.exec_module(module)
    return module


def parameter_specs(m, protein: str):
    if protein not in m.proteins or protein not in m.kinases:
        raise ValueError(f'{protein} must be both a model protein and a kinase input')
    gi, ki = m.proteins.index(protein), m.kinases.index(protein)
    specs = [(f'{key}_{protein}', key, gi)
             for key in ('A_i', 'B_i', 'C_i', 'D_i', 'E_i')]
    specs.append((f'c_k_{protein}', 'c_k', ki))
    for i, (gene, site) in enumerate(m.sites):
        if gene == protein:
            specs.append((f'Dp_i_{gene}_{site}', 'Dp_i', i))
    reference = np.array([m.fitted[k][i] for _, k, i in specs], dtype=float)
    if np.any(~np.isfinite(reference)) or np.any(reference <= 0):
        raise ValueError('Sampled fitted EGFR parameters must be positive and finite')
    return specs, reference


def fitted_with_target(m, specs, values):
    params = {k: np.asarray(v, dtype=float).copy() for k, v in m.fitted.items()}
    for (_, key, idx), value in zip(specs, values):
        params[key][idx] = float(value)
    return params


class ModelEngine:
    """Vectorized NumPy RHS; verified against the original reference RHS."""

    def __init__(self, m, reference_rhs):
        self.m = m
        self.reference_rhs = reference_rhs
        self.n = len(m.proteins)
        self.ns = len(m.sites)
        self.rna_index = np.array([m.off_y[p] for p in m.proteins])
        self.p0_index = self.rna_index + 1
        owners = []
        state_index = []
        for gi, gene in enumerate(m.proteins):
            count = sum(gene == g for g, _ in m.sites)
            owners.extend([gi] * count)
            state_index.extend(range(m.off_y[gene] + 2,
                                     m.off_y[gene] + 2 + count))
        self.owners = np.asarray(owners, dtype=np.intp)
        self.site_index = np.asarray(state_index, dtype=np.intp)
        expected = [m.proteins[i] for i in self.owners]
        actual = [gene for gene, _ in m.sites]
        if expected != actual or len(self.site_index) != self.ns:
            raise ValueError('Phosphosite order must agree with full distributive state layout')
        self.driver = np.asarray(m.driver, dtype=np.intp)
        self.driver_mask = self.driver >= 0

    def _rhs(self, y, par, K, S):
        R = y[self.rna_index]
        P0 = y[self.p0_index]
        PS = y[self.site_index]
        site_sum = np.bincount(self.owners, weights=PS, minlength=self.n)
        P = P0 + site_sum
        P[self.driver_mask] = K[self.driver[self.driver_mask]]
        U = (self.m.TF @ P) / self.m.tf_deg
        u = U / (1 + np.abs(U))
        tf = float(par['tf_scale'][0])
        synthesis = np.where(
            u >= 0,
            par['A_i'] * (1 + tf * u / (1 + u + 1e-6)),
            par['A_i'] / (1 + tf * np.abs(u)),
        )
        forward_by_site = S * P0[self.owners]
        recycled_by_site = par['E_i'][self.owners] * PS
        F = np.bincount(self.owners, weights=forward_by_site, minlength=self.n)
        E = np.bincount(self.owners, weights=recycled_by_site, minlength=self.n)
        dy = np.zeros_like(y)
        dy[self.rna_index] = synthesis - par['B_i'] * R
        dy[self.p0_index] = par['C_i'] * R - par['D_i'] * P0 - F + E
        dy[self.site_index] = (forward_by_site -
            (par['E_i'][self.owners] + par['Dp_i'] + par['D_i'][self.owners]) * PS)
        return dy

    def verify(self, par, rng=None):
        """The speed-optimized RHS must agree with the model's reference algebra."""
        rng = np.random.default_rng(0) if rng is None else rng
        check_steps = np.unique([0, max(0, len(self.m.ktime)//2), len(self.m.ktime)-1])
        pseudo = SimpleNamespace(**{**vars(self.m), 'fitted': par})
        maxerr = 0.0
        for step in check_steps:
            K = self.m.K[:, step] * par['c_k']
            S = self.m.W @ K
            for _ in range(2):
                y = self.m.y0 * rng.uniform(0.9, 1.1, size=len(self.m.y0))
                a = self._rhs(y, par, K, S)
                b = self.reference_rhs(pseudo, y, int(step))
                if not np.allclose(a, b, atol=1e-9, rtol=1e-9):
                    raise RuntimeError('Optimized RHS does not reproduce reference PhosKinTime RHS')
                maxerr = max(maxerr, float(np.max(np.abs(a - b))))
        return maxerr

    def simulate(self, par, times, method='BDF', rtol=1e-6, atol=1e-8):
        """Integrate around every forcing jump with adaptive internal steps."""
        m = self.m
        if times[0] != 0 or not np.all(np.diff(times) > 0):
            raise ValueError('Time grid must start at zero and be strictly increasing')
        stop_time = float(times[-1])
        cuts = np.unique(np.r_[0., m.ktime[(m.ktime > 0) &
                                                 (m.ktime < stop_time)], stop_time])
        out = np.empty((len(times), len(m.y0)), dtype=np.float64)
        out[0] = m.y0
        state = m.y0.copy()
        for start, stop in zip(cuts[:-1], cuts[1:]):
            step = int(np.clip(np.searchsorted(m.ktime, start, side='right') - 1,
                               0, len(m.ktime)-1))
            K = m.K[:, step] * par['c_k']
            S = m.W @ K
            solver = solve_ivp(
                lambda t, y: self._rhs(y, par, K, S),
                (float(start), float(stop)), state,
                method=method, dense_output=True, rtol=rtol, atol=atol,
            )
            if not solver.success:
                raise RuntimeError(f'{method} failed at {start:g}-{stop:g} min: {solver.message}')
            ix = np.flatnonzero((times > start) & (times <= stop))
            if ix.size:
                out[ix] = solver.sol(times[ix]).T
            state = np.asarray(solver.y[:, -1])
            if not np.all(np.isfinite(state)):
                raise FloatingPointError('Nonfinite ODE state encountered')
        if not np.all(np.isfinite(out)):
            raise FloatingPointError('Nonfinite ODE outputs')
        return out


def dense_grid(m, hours, points, first_positive=0.05):
    """Exactly `points` times, log-dense early, including forcing knots and observations."""
    end = 60. * hours
    if end <= 4. or first_positive <= 0 or first_positive >= end:
        raise ValueError('Horizon must exceed four minutes and first-positive must lie within it')
    observed = m.traj.time.to_numpy(dtype=float)
    mandatory = np.unique(np.r_[0., 4., end, 60.,
        m.ktime[(m.ktime >= 0) & (m.ktime <= end)],
        observed[np.isfinite(observed) & (observed >= 0) & (observed <= end)]])
    mandatory = mandatory[mandatory <= end]
    if len(mandatory) >= points:
        raise ValueError(f'{len(mandatory)} required times exceed --points {points}')
    needed = points - len(mandatory)
    candidates = np.geomspace(first_positive, end * (1 - 1e-10), points * 3)
    candidates = candidates[~np.isin(candidates, mandatory)]
    indices = np.linspace(0, len(candidates)-1, needed).astype(np.intp)
    times = np.unique(np.r_[mandatory, candidates[indices]])
    if len(times) != points or times[0] != 0 or abs(times[-1]-end)>1e-9:
        raise RuntimeError(f'Time-grid construction failed: {len(times)} instead of {points}')
    return times


def sample_bounded(rng, reference, sigma, fold):
    limit = math.log(fold)
    z = rng.normal(size=len(reference)) * sigma
    while (bad := np.abs(z) > limit).any():
        z[bad] = rng.normal(size=int(bad.sum())) * sigma
    return reference * np.exp(z)


def read_parameter_vectors(path, specs, max_count, seed):
    df = pd.read_csv(path)
    cols = [name for name, _, _ in specs]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'Missing parameter columns in {path}: {sorted(missing)}')
    mat = df[cols].to_numpy(dtype=float)
    if np.any(~np.isfinite(mat)) or np.any(mat <= 0):
        raise ValueError('Saved parameter draws must be positive and finite')
    if max_count < 0:
        raise ValueError('--draws cannot be negative')
    sel = np.arange(len(df))
    if max_count and max_count < len(sel):
        sel = np.sort(np.random.default_rng(seed).choice(sel, max_count, replace=False))
    ll = (df['log_likelihood_without_constants'].to_numpy(dtype=float)[sel]
          if 'log_likelihood_without_constants' in df.columns else None)
    return mat[sel], sel, ll


def features_for_states(m, engine, par, states, times, protein, selected_sites):
    """Mechanistic (state and flux) features, not fit or posterior diagnostics."""
    gi = m.proteins.index(protein)
    site_global = np.flatnonzero(engine.owners == gi)
    if site_global.size == 0:
        raise ValueError(f'{protein} has no phosphosites')
    R = states[:, engine.rna_index[gi]]
    P0 = states[:, engine.p0_index[gi]]
    PS = states[:, engine.site_index[site_global]]
    phosphototal = PS.sum(axis=1)
    total_protein = P0 + phosphototal
    baseline_p0 = float(m.y0[engine.p0_index[gi]])
    baseline_ps = float(m.y0[engine.site_index[site_global]].sum())
    baseline_total = baseline_p0 + baseline_ps
    at4 = np.flatnonzero(np.isclose(times,4.,rtol=0.,atol=1e-10))
    if len(at4) != 1:
        raise ValueError('Time grid must contain t=4 min for RNA normalization')
    if min(baseline_p0,baseline_ps,baseline_total,abs(R[at4[0]])) <= EPS:
        raise ValueError('Zero/tiny EGFR state denominator: cannot define requested fold changes')

    # Forcing changes discontinuously at experimental times: use the value for
    # the interval to the right of each knot, and hold final forcing thereafter.
    step = np.clip(np.searchsorted(m.ktime, times, side='right')-1,
                   0, len(m.ktime)-1)
    S = (m.W[site_global] @ (m.K * par['c_k'][:,None]))[:,step].T
    forward = (S * P0[:,None]).sum(axis=1)
    recycled = par['E_i'][gi] * phosphototal
    irreversible = (PS * (par['Dp_i'][site_global][None,:]
                          +par['D_i'][gi])).sum(axis=1)
    removal = recycled + irreversible
    dphospho = forward - removal
    feat = {
        'P0': P0, 'Phospho_total': phosphototal,
        'RNA': R, 'Protein_total': total_protein,
        'P0_FC': P0 / baseline_p0,
        'Phospho_FC': phosphototal / baseline_ps,
        'RNA_FC': R / float(R[at4[0]]),
        'Protein_FC': total_protein / baseline_total,
        'Forward_flux': forward, 'Recycling_flux': recycled,
        'Irreversible_loss_flux': irreversible,
        'Removal_flux': removal, 'Net_phospho_flux': dphospho,
    }
    for site in selected_sites:
        ix = next((j for j, (g,s) in enumerate(m.sites)
                   if g == protein and s == site), None)
        if ix is None:
            continue
        initial = float(m.y0[engine.site_index[ix]])
        feat[f'Site_{site}_FC'] = (states[:, engine.site_index[ix]] /
                                  max(initial, EPS))
    return feat


def build_pca(nominal, times, components=3):
    """Fit ONE PCA to standardized nominal 70-state dynamics, then project draws."""
    # Log-spaced grid: sample indices uniformly to represent log-time regimes.
    idx = np.linspace(0, len(times)-1, min(len(times), 1600)).astype(int)
    data = nominal[idx]
    center = data.mean(axis=0)
    scales = data.std(axis=0)
    magnitude = np.maximum(np.max(np.abs(data), axis=0), EPS)
    variable = scales > np.maximum(1e-10, 1e-6 * magnitude)
    std = np.where(variable, scales, 1.0)
    z = (data-center) / std
    z[:, ~variable] = 0.0
    if not variable.any():
        raise ValueError('No sufficiently variable molecular states for network PCA')
    _, singular, vt = np.linalg.svd(z, full_matrices=False)
    basis = np.zeros((len(center), components), dtype=float)
    rank = min(components, vt.shape[0], int(variable.sum()))
    basis[:, :rank] = vt[:rank].T
    # Fix PCA axis sign for deterministic plots.
    for j in range(rank):
        pivot = np.argmax(np.abs(basis[:, j]))
        if basis[pivot,j] < 0:
            basis[:,j] *= -1
    explained = (singular**2) / max(float((singular**2).sum()), EPS)
    ratios = np.zeros(components)
    ratios[:min(components,len(explained))] = explained[:components]
    return center, std, variable, basis, ratios


def project(states, center, scales, variable, basis):
    z = (states - center) / scales
    z[:,~variable] = 0.0
    return z @ basis


def observed_pair(m, protein, type_a, site_a, type_b, site_b):
    data = m.traj.copy()
    data['psite'] = data['psite'].fillna('')
    data = data[data.protein == protein]
    data = data[np.isfinite(data['fc_obs'])]
    x = (data[(data.type == type_a) & (data.psite == site_a)]
         .groupby('time',as_index=False).fc_obs.mean().rename(columns={'fc_obs':'x'}))
    y = (data[(data.type == type_b) & (data.psite == site_b)]
         .groupby('time',as_index=False).fc_obs.mean().rename(columns={'fc_obs':'y'}))
    return x.merge(y,on='time',how='inner').sort_values('time')


def set_plot_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10.5,
        'axes.titlesize': 15.0,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#4c555c',
        'axes.linewidth': 0.85,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'grid.color': GRID,
        'grid.linewidth': 0.65,
        'axes.axisbelow': True,
        'legend.frameon': False,
        'legend.fontsize': 8.8,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def panels():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(6.2, 8.0),
                            constrained_layout=True)
    return fig, axs


def save_figure(fig, out, name, dpi):
    for extension in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{extension}', dpi=dpi,
                    bbox_inches='tight', pad_inches=0.16)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'Figure: {name}.pdf / .png', flush=True)


def phase_axes(ax, nominal, arrays, times, cutoff, xkey, ykey,
               title, xlabel, ylabel, plotted, obs=None, diagonal=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rng = np.random.default_rng(19)
    ndraws = len(arrays.get(xkey, []))
    indices = (np.sort(rng.choice(ndraws, min(ndraws, plotted), replace=False))
               if ndraws else [])
    for i in indices:
        ax.plot(arrays[xkey][i], arrays[ykey][i], color=PALE,
                lw=0.72, alpha=0.24, zorder=1, rasterized=True)
    before = times <= cutoff
    after = times >= cutoff
    ax.plot(nominal[xkey][before], nominal[ykey][before],
            color=TEAL, lw=2.1, zorder=3)
    if after.any():
        ax.plot(nominal[xkey][after], nominal[ykey][after],
                color=TEAL, lw=2.0, ls='--', dashes=(5,2.5), zorder=3)
    if obs is not None and len(obs):
        ax.scatter(obs.x, obs.y, s=19, color=RUST, edgecolors='none', zorder=4)
    ax.scatter([nominal[xkey][0]], [nominal[ykey][0]], s=44, marker='s',
               color=TEAL, edgecolors='white',linewidth=0.55,zorder=5)
    ax.scatter([nominal[xkey][-1]], [nominal[ykey][-1]], s=50, marker='D',
               color=RUST, edgecolors='white',linewidth=0.7,zorder=5)
    ax.set_title(title,loc='left',pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y',alpha=0.8)
    if diagonal:
        lo = min(ax.get_xlim()[0],ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1],ax.get_ylim()[1])
        ax.plot([lo,hi],[lo,hi],color=INK,ls=':',lw=1.1,alpha=.65,zorder=0)
    handles = [
        Line2D([0],[0],color=TEAL,lw=2.1,label='Fitted trajectory'),
        Line2D([0],[0],color=PALE,lw=2,label='Parameter ensemble'),
        Line2D([0],[0],color=TEAL,lw=2,ls='--',label='Held-input extrapolation'),
    ]
    if obs is not None and len(obs):
        handles.append(Line2D([0],[0],ls='none',marker='o',ms=4,
                              color=RUST,label='Observed pairs'))
    handles.append(Line2D([0],[0],ls='none',marker='D',ms=5,
                          color=RUST,label='72 h endpoint'))
    ax.legend(handles=handles,loc='best',fontsize=7.6,
              ncol=2,handlelength=1.8,columnspacing=.8)


def quantile_table(nominal, arrays, times):
    table = pd.DataFrame({'time_min': times, 'time_h':times/60.})
    for name, curve in nominal.items():
        table[f'{name}__fitted'] = curve
        if name in arrays and len(arrays[name]):
            q = np.quantile(arrays[name],[.025,.5,.975],axis=0)
            for suffix, vals in zip(('low_95','median','high_95'),q):
                table[f'{name}__{suffix}'] = vals
    return table


def trajectory_metrics(f, times, cutoff, label):
    """Mechanistic descriptors; not evidence of equilibria or attractors."""
    peak = int(np.argmax(f['Phospho_FC']))
    during = times <= cutoff
    return {
        'trajectory':label,
        'peak_phospho_fold_change':float(f['Phospho_FC'][peak]),
        'time_to_peak_phospho_min':float(times[peak]),
        'phospho_fold_change_at_72h':float(f['Phospho_FC'][-1]),
        'P0_fold_change_at_72h':float(f['P0_FC'][-1]),
        'RNA_fold_change_at_72h':float(f['RNA_FC'][-1]),
        'protein_fold_change_at_72h':float(f['Protein_FC'][-1]),
        'net_phosphorylation_flux_at_72h':float(f['Net_phospho_flux'][-1]),
        'integrated_net_phosphorylation_flux':float(np.trapezoid(
            f['Net_phospho_flux'],times)),
        'integrated_net_flux_before_last_input':float(np.trapezoid(
            f['Net_phospho_flux'][during],times[during])),
        'full_network_PCA_path_length':float(np.linalg.norm(
            np.diff(np.column_stack([f['PC1'],f['PC2'],f['PC3']]),axis=0),
            axis=1).sum()),
    }


def time_axes(ax, names, title, times, nominal, arrays, cutoff):
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D
    for j, (feature, label) in enumerate(names):
        col = TEAL if j == 0 else RUST
        if feature in arrays and len(arrays[feature]):
            q = np.quantile(arrays[feature],[.025,.5,.975],axis=0)
            ax.fill_between(times,q[0],q[2],color=col,alpha=.15,zorder=1)
            ax.plot(times,q[1],lw=1.9,color=col,label=f'{label} median',zorder=3)
        else:
            ax.plot(times,nominal[feature],lw=1.9,color=col,label=label,zorder=3)
    if cutoff < times[-1]:
        ax.axvspan(cutoff,times[-1],color=PALE,alpha=.11,zorder=0)
        ax.axvline(cutoff,color=INK,ls=':',lw=.9,alpha=.55)
    ax.set_xscale('symlog',linthresh=.5,linscale=.85)
    ticks = np.asarray([0.,.5,1,4,16,60,240,960,4320],dtype=float)
    ticks = ticks[ticks <= times[-1]+1e-8]
    if times[-1] not in ticks:
        ticks = np.r_[ticks,times[-1]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value,pos: '0*' if value==0 else f'{value:g}'))
    ax.set_xlim(0,times[-1]*1.003)
    ax.set_xlabel('Time (min, log scale)  ·  0* marks baseline')
    ax.set_ylabel('Model flux (state units/min)')
    ax.set_title(title,loc='left',pad=8)
    ax.grid(axis='y',alpha=.85)
    ax.legend(loc='best',fontsize=8,frameon=False)


def state_column_names(m):
    columns=[]
    for gene in m.proteins:
        columns += [f'RNA|{gene}',f'P0|{gene}']
        columns += [f'Phospho|{gene}|{site}' for g,site in m.sites if g==gene]
    if len(columns) != len(m.y0):
        raise ValueError('State CSV names do not match model layout')
    return columns


def produce_plots(out,dpi,nominal,arrays,times,cutoff,obs_rna_protein,
                  obs_sites,sites,explained,plot_draws):
    set_plot_style()
    import matplotlib.pyplot as plt

    fig,axs=panels()
    phase_axes(axs[0],nominal,arrays,times,cutoff,'P0_FC','Phospho_FC',
               'EGFR phosphorylation-state portrait',
               'Unphosphorylated EGFR (relative to 0 min)',
               'Phosphorylated EGFR pool (relative to 0 min)',plot_draws)
    phase_axes(axs[1],nominal,arrays,times,cutoff,'RNA_FC','Protein_FC',
               'EGFR RNA-protein phase portrait',
               'RNA fold change (4-min reference)',
               'Total protein fold change (0-min reference)',plot_draws,obs_rna_protein)
    save_figure(fig,out,'EGFR_state_phase_portraits',dpi)

    fig,axs=panels()
    phase_axes(axs[0],nominal,arrays,times,cutoff,'Forward_flux','Removal_flux',
               'EGFR phosphorylation flux balance',
               'Forward phosphorylation flux (state units/min)',
               'Recycling + loss flux (state units/min)',plot_draws,diagonal=True)
    if len(sites)==2:
        x,y=[f'Site_{site}_FC' for site in sites]
        phase_axes(axs[1],nominal,arrays,times,cutoff,x,y,
                   f'EGFR {sites[0]} vs {sites[1]}',
                   f'{sites[0]} fold change',f'{sites[1]} fold change',
                   plot_draws,obs_sites)
    else:
        axs[1].axis('off')
        axs[1].text(.02,.55,'At least two sites are required for a site-pair portrait',
                    transform=axs[1].transAxes)
    save_figure(fig,out,'EGFR_flux_and_site_phase',dpi)

    fig,axs=panels()
    for axis,which in zip(axs,('PC2','PC3')):
        phase_axes(axis,nominal,arrays,times,cutoff,'PC1',which,
                   f'Full-network phase portrait: PC1 vs {which}',
                   f'PC1 ({100*explained[0]:.1f}% nominal variance)',
                   f'{which} ({100*explained[1 if which=="PC2" else 2]:.1f}% nominal variance)',
                   plot_draws)
    save_figure(fig,out,'Network_PCA_phase_portraits',dpi)

    fig,axs=panels()
    time_axes(axs[0], [('Forward_flux','Forward'),('Removal_flux','Recycling + loss')],
              'EGFR phosphorylation fluxes',times,nominal,arrays,cutoff)
    time_axes(axs[1],[('Net_phospho_flux','Net phosphorylation')],
              'EGFR net phosphorylation flux',times,nominal,arrays,cutoff)
    save_figure(fig,out,'EGFR_phosphorylation_flux_timecourse',dpi)


def main():
    p=argparse.ArgumentParser(description=__doc__,
              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--source',type=Path,default=Path('.'))
    p.add_argument('--results',type=Path,default=Path('results_network_distributive'))
    p.add_argument('--out',type=Path,default=Path('results_network_distributive/egfr_phase_72h'))
    p.add_argument('--protein',default='EGFR')
    p.add_argument('--parameter-csv',type=Path,default=None,
                   help='Reuse parameter_ensemble.csv produced by egfr_forward_ensemble_72h.py')
    p.add_argument('--hours',type=float,default=72.)
    p.add_argument('--points',type=int,default=10000,
                   help='Exact output count; log-dense plus recorded observation/forcing times')
    p.add_argument('--draws',type=int,default=100,
                   help='Successful ensemble draws; use 0 to process all CSV rows')
    p.add_argument('--max-attempt-factor',type=int,default=3,
                   help='Maximum attempts for newly sampled parameters')
    p.add_argument('--prior-log-sd',type=float,default=.2)
    p.add_argument('--prior-fold',type=float,default=2.)
    p.add_argument('--solver',choices=['BDF','Radau','LSODA','RK45'],default='BDF')
    p.add_argument('--rtol',type=float,default=1e-6)
    p.add_argument('--atol',type=float,default=1e-8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--plot-draws',type=int,default=35)
    p.add_argument('--dpi',type=int,default=400)
    p.add_argument('--nominal-only',action='store_true',help='Only fitted ODE; no uncertainty ensemble')
    p.add_argument('--save-full-network-draws',action='store_true',
                   help='Also export all 70 states for all draws (very large NPZ)')
    args=p.parse_args()
    if (args.points<20 or args.hours<=0 or args.draws<0 or args.plot_draws<0 or
        args.rtol<=0 or args.atol<=0 or args.prior_log_sd<=0 or
        args.prior_fold<=1 or args.max_attempt_factor<1 or args.dpi<50):
        p.error('Invalid simulation/plot configuration')
    if args.parameter_csv is None and args.draws==0 and not args.nominal_only:
        p.error('--draws 0 requires --parameter-csv, or use --nominal-only')
    args.out.mkdir(parents=True,exist_ok=True)
    h=load_original(args.source.resolve())
    m=h.load_run(args.results)
    specs,reference=parameter_specs(m,args.protein)
    names=[s[0] for s in specs]
    engine=ModelEngine(m,h.reference_rhs)
    rhs_error=engine.verify(m.fitted)
    times=dense_grid(m,args.hours,args.points)
    cutoff=float(m.ktime[-1])
    samplesites = [site for gene,site in m.sites if gene==args.protein]
    sites = ([s for s in ('S991','S1166') if s in samplesites]
             if 'S991' in samplesites and 'S1166' in samplesites
             else samplesites[:2])
    print(f'Network: {len(m.proteins)} proteins, {len(m.sites)} phosphosites, '
          f'{len(m.y0)} states; optimized RHS max error={rhs_error:.3g}',flush=True)
    print(f'EGFR-specific parameters ({len(names)}): {names}',flush=True)
    print(f'Grid: {len(times)} points, horizon={times[-1]/60:g} h; '
          f'hold final kinase input after {cutoff:g} min',flush=True)
    nominal_states=engine.simulate(m.fitted,times,args.solver,args.rtol,args.atol)
    center,scales,variable,basis,explained=build_pca(nominal_states,times)
    nominal=features_for_states(m,engine,m.fitted,nominal_states,times,
                                args.protein,sites)
    pca0=project(nominal_states,center,scales,variable,basis)
    for i in range(3):
        nominal[f'PC{i+1}']=pca0[:,i]
    del pca0
    pd.DataFrame({
        'state':state_column_names(m),
        'reference_mean':center,
        'reference_std':scales,
        'variable_for_PCA':variable,
        'PC1_loading':basis[:,0],
        'PC2_loading':basis[:,1],
        'PC3_loading':basis[:,2],
    }).to_csv(args.out/'network_PCA_loadings.csv',index=False)

    # Preserve exact nominal model states for further mechanistic analysis.
    states_df=pd.DataFrame(nominal_states,columns=state_column_names(m))
    states_df.insert(0,'time_min',times)
    states_df.insert(1,'time_h',times/60.)
    states_df.to_csv(args.out/'nominal_full_network_states.csv',index=False)
    del states_df

    frame_cols=names+['source_row','log_likelihood_without_constants']
    accepted=[]
    score=[]
    source_rows=[]
    errors=[]
    features_by_draw={key:[] for key in nominal}
    full_network=[]
    rng=np.random.default_rng(args.seed)
    if args.nominal_only:
        candidates=[]
        target=0
    elif args.parameter_csv:
        params,sel,likelihood=read_parameter_vectors(args.parameter_csv,specs,args.draws,args.seed)
        candidates=list(zip(params,sel,likelihood if likelihood is not None
                            else [np.nan]*len(sel)))
        target=len(candidates)
    else:
        target=args.draws
        candidates=None
    attempt=0
    while len(accepted)<target:
        if candidates is None:
            if attempt>=target*args.max_attempt_factor:
                break
            values=sample_bounded(rng,reference,args.prior_log_sd,args.prior_fold)
            source_row=-1
            likelihood=np.nan
        else:
            if attempt>=len(candidates):
                break
            values,source_row,likelihood=candidates[attempt]
        attempt+=1
        par=fitted_with_target(m,specs,values)
        try:
            if len(accepted)==0:
                engine.verify(par)
            states=engine.simulate(par,times,args.solver,args.rtol,args.atol)
            this=features_for_states(m,engine,par,states,times,args.protein,sites)
            pca=project(states,center,scales,variable,basis)
            for i in range(3):
                this[f'PC{i+1}']=pca[:,i]
            if any(np.any(~np.isfinite(v)) for v in this.values()):
                raise FloatingPointError('Nonfinite phase feature')
            for name in nominal:
                features_by_draw[name].append(this[name].astype('float32'))
            if args.save_full_network_draws:
                full_network.append(states.astype('float32'))
            accepted.append(values)
            score.append(float(likelihood))
            source_rows.append(int(source_row))
            if len(accepted)%10==0 or len(accepted)==target:
                print(f'Forward phase trajectories {len(accepted)}/{target}, '
                      f'attempted {attempt}',flush=True)
        except (RuntimeError,ValueError,FloatingPointError,OverflowError) as exc:
            errors.append({'attempt':attempt,'reason':str(exc)[:300]})
            print(f'WARNING: draw {attempt} skipped: {exc}',flush=True)
    if target and not accepted:
        raise RuntimeError(f'No usable EGFR parameter draws; first errors: {errors[:3]}')
    arrays={k:np.stack(v) if len(v) else np.empty((0,len(times)),dtype=np.float32)
            for k,v in features_by_draw.items()}
    summary=quantile_table(nominal,arrays,times)
    summary.to_csv(args.out/'egfr_phase_ensemble_quantiles.csv',index=False)
    coords=pd.DataFrame({'time_min':times,'time_h':times/60.})
    for name,val in nominal.items():
        coords[name]=val
    coords.to_csv(args.out/'egfr_nominal_phase_coordinates.csv',index=False)
    np.savez_compressed(args.out/'egfr_phase_draws.npz',time_min=times,**arrays)
    if args.save_full_network_draws and full_network:
        np.savez_compressed(args.out/'full_network_draw_states.npz',
                            time_min=times,states=np.stack(full_network),
                            state_names=np.asarray(state_column_names(m)))
    par_df=pd.DataFrame(accepted,columns=names)
    par_df['source_row']=source_rows
    par_df['log_likelihood_without_constants']=score
    par_df.to_csv(args.out/'used_parameter_draws.csv',index=False)
    metrics=[trajectory_metrics(nominal,times,cutoff,'fitted')]
    for j in range(len(accepted)):
        feature={key:arrays[key][j] for key in nominal}
        metrics.append(trajectory_metrics(feature,times,cutoff,f'ensemble_{j:04d}'))
    pd.DataFrame(metrics).to_csv(args.out/'egfr_dynamical_metrics.csv',index=False)
    obs_rna_protein=observed_pair(m,args.protein,'RNA','','Protein','')
    obs_rna_protein.to_csv(args.out/'egfr_observed_RNA_protein_pairs.csv',index=False)
    obs_sites=(observed_pair(m,args.protein,'Phosphorylation',sites[0],
                             'Phosphorylation',sites[1])
               if len(sites)==2 else pd.DataFrame(columns=['time','x','y']))
    obs_sites.to_csv(args.out/'egfr_observed_site_pairs.csv',index=False)
    produce_plots(args.out,args.dpi,nominal,arrays,times,cutoff,obs_rna_protein,
                  obs_sites,sites,explained,args.plot_draws)
    report={
        'method':'Full-network distributive ODE forward simulation, no MCMC',
        'model_reference':'Original phoskintime_dynestyx_smoothing.py RHS',
        'optimized_rhs_verification_max_abs_error':rhs_error,
        'proteins':len(m.proteins),'phosphosites':len(m.sites),'states':len(m.y0),
        'parameter_source':str(args.parameter_csv) if args.parameter_csv else
                           'fitted-centered bounded lognormal (not a posterior)',
        'sampled_target':args.protein,'sampled_parameter_names':names,
        'fixed_parameters':'All non-target fitted parameters and initial conditions',
        'horizon_hours':args.hours,'points':len(times),
        'time_grid':'log-dense plus all recorded observation and kinase-input times',
        'solver':args.solver,'rtol':args.rtol,'atol':args.atol,
        'number_of_parameter_draws':len(accepted),'number_of_failed_draws':len(errors),
        'failed_draws':errors[:20],
        'pca_fit':'standardized nominal model trajectory (common basis across draws)',
        'pca_nominal_explained_variance_first_3':explained.tolist(),
        'pca_variable_state_count':int(variable.sum()),
        'PCA_warning':'Projection only; crossings or loops do not establish multistability or oscillations',
        'kinase_last_input_time_min':cutoff,
        'forcing_after_last_input':'Hold final supplied kinase-input column constant',
        'experimental_extrapolation_warning':'Dynamics beyond the measured forcing window are hypothetical',
        'confidence_intervals':'Pointwise 95% forward parameter-ensemble quantiles, NOT posterior credible intervals',
        'observed_pairs_note':'Only matched observations; no measured P0 or flux values assumed',
        'no_ODE_smoothing':'10,000 dense output points; adaptive solver controls numerical accuracy independently',
        'site_pair':sites,
    }
    (args.out/'methods_and_diagnostics.json').write_text(json.dumps(report,indent=2))
    print(f'COMPLETE: {args.out.resolve()} ({len(accepted)} ensemble draws)',flush=True)


if __name__=='__main__':
    main()
