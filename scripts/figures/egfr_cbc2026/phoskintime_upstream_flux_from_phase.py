#!/usr/bin/env python3
"""Repository-consistent EGFR site/kinase forward-flux ensembles for CBC poster.

Source alignment (PhosKinTime MODEL=distributive):
    K_eff[k,t] = K_input[k,t] * c_k[k]
    S_s[t]     = sum_k W[s,k] * K_eff[k,t]          [min^-1]
    J_s,k[t]   = W[s,k] * K_eff[k,t] * P0_EGFR[t] [model-state/min]

K_input is piecewise constant at recorded protein time knots and held after
last knot. Posterior claims are NOT made. The interval is pointwise [2.5, 97.5]
percentiles from the EXACT SAME forward-state draw arrays in phase NPZ.

This avoids inventing upstream c_k uncertainty. In --quantity drive mode, the
lack of uncertainty is shown explicitly if no relevant c_k was sampled.

Requires files from scripts/phoskintime_phase_portraits_72h.py:
  egfr_phase_draws.npz (time_min, P0, Forward_flux)
  egfr_nominal_phase_coordinates.csv (time_min, P0, Forward_flux)
  used_parameter_draws.csv (draw-order-aligned c_k_... where applicable)

If --quantity flux (default), the original poster drive figures are REPLACED
with *flux figures*; update the LaTeX caption and units accordingly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEAL = '#126d83'
RUST = '#ac522e'
PALE = '#a7ccd6'
GRID = '#e0e5e8'
COLORS = {'CDK1': '#b9902e', 'GSK3B': '#a6532e', 'MAPK1': '#796a96',
          'MAPK3': '#146b80', 'MAPK8': '#48887d', 'GRK2': '#a6532e'}


def _load_model(source, results):
    # Current layout first; the pre-reorganisation layout keeps --source usable for older checkouts.
    root = Path(source).resolve()
    candidates = [root / 'scripts' / 'figures' / 'egfr_cbc2026' / 'phoskintime_dynestyx_smoothing.py',
                  root / 'scripts' / 'phoskintime_dynestyx_smoothing.py']
    file = next((c for c in candidates if c.is_file()), None)
    if file is None:
        raise FileNotFoundError('Validated original loader is missing; tried: '
                                + ', '.join(str(c) for c in candidates))
    spec = importlib.util.spec_from_file_location('_source_loader', file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {file}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.load_run(results)
    return model


def _canon(s):
    return re.sub(r'[^A-Za-z0-9]', '', str(s)).upper()


def _site_idx(m, protein, requested):
    matches = [(i, s) for i, (p, s) in enumerate(m.sites)
               if p == protein and _canon(s) == _canon(requested)]
    if len(matches) != 1:
        raise ValueError(f'Site {protein}/{requested} not unique in fitted state layout: {matches}')
    return matches[0]


def _load_phase(phase_dir, m, protein, points, hours):
    phase_dir = Path(phase_dir)
    with np.load(phase_dir / 'egfr_phase_draws.npz') as z:
        times = np.asarray(z['time_min'], dtype=np.float64)
        P0 = np.asarray(z['P0'], dtype=np.float64)
        fwd_total = np.asarray(z['Forward_flux'], dtype=np.float64) if 'Forward_flux' in z else None
    nominal = pd.read_csv(phase_dir / 'egfr_nominal_phase_coordinates.csv')
    params = pd.read_csv(phase_dir / 'used_parameter_draws.csv')
    if P0.ndim != 2 or P0.shape != (len(params), len(times)) or len(times) != points:
        raise ValueError('NPZ P0 shape, parameter row count or --points mismatch')
    if times[0] != 0 or np.any(np.diff(times) <= 0) or not np.isclose(times[-1], 60 * hours):
        raise ValueError('Phase time grid is not strictly increasing, or --hours mismatched')
    if not np.allclose(nominal.time_min.to_numpy(), times, rtol=0, atol=1e-7):
        raise ValueError('Nominal and ensemble time grids differ')
    if not np.all(np.isfinite(P0)) or (P0 < -1e-8).any():
        raise ValueError('Nonfinite or materially negative EGFR P0 state')
    nominal_p0 = nominal['P0'].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(nominal_p0)):
        raise ValueError('Nonfinite nominal P0')
    return times, nominal_p0, P0, params, fwd_total, nominal


def _kinase_gains(m, params):
    gains = np.broadcast_to(np.asarray(m.fitted['c_k'], dtype=float),
                            (len(params), len(m.kinases))).copy()
    found = []
    for i, k in enumerate(m.kinases):
        col = f'c_k_{k}'
        if col in params.columns:
            gains[:, i] = params[col].to_numpy(dtype=np.float64)
            found.append(col)
    if not np.isfinite(gains).all() or np.min(gains) < 0:
        raise ValueError('Nonfinite or negative kinase multiplier in phase draw parameters')
    return gains, found


def _inputs(m, times):
    t = np.asarray(m.ktime, dtype=float)
    if len(t) != m.K.shape[1] or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Inconsistent exported kinase-input grid')
    ix = np.clip(np.searchsorted(t, times, side='right')-1, 0, len(t)-1)
    return np.asarray(m.K, dtype=float)[:, ix], ix


def _calcs(m, site_index, Kt, gains, nominal_p0, P0, kind):
    weights = np.asarray(m.W[site_index], dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(weights) > 1e-12)
    if not len(nonzero):
        raise RuntimeError(f'No upstream kinases for {m.sites[site_index]}')
    base_c = np.asarray(m.fitted['c_k'], dtype=float)
    # Non-normalized physical rate: W * c_k * measured protein FC (per min).
    nominal_rate = weights[nonzero,None] * base_c[nonzero,None] * Kt[nonzero]
    # Broadcasting: draws x contributing kinases x time points.
    sampled_rates = (weights[None,nonzero,None] * gains[:,nonzero,None] *
                     Kt[None,nonzero,:])
    if kind == 'flux':
        nominal_series = nominal_rate * nominal_p0[None,:]
        sampled_series = sampled_rates * P0[:,None,:]
    else:
        nominal_series, sampled_series = nominal_rate, sampled_rates
    nominal_total = nominal_series.sum(axis=0)
    sampled_total = sampled_series.sum(axis=1)
    # We report a total over ALL nonzero drivers, including those not in the chart legend.
    return nonzero, nominal_series, sampled_series, nominal_total, sampled_total


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':10.7,
        'axes.titlesize':12.2,'axes.titleweight':'bold',
        'axes.labelsize':11.1,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#65717a','axes.linewidth':.75,
        'xtick.labelsize':9.1,'ytick.labelsize':9.3,
        'grid.color':GRID,'grid.linewidth':.7,'grid.alpha':.65,
        'legend.frameon':False,'legend.fontsize':8.4,
        'pdf.fonttype':42,'ps.fonttype':42,'figure.facecolor':'white',
        'savefig.facecolor':'white',
    })
    return plt, FuncFormatter


def _draw_plots(m, obj, args):
    plt, FuncFormatter = _style()
    times = obj['times']
    cutoff = float(m.ktime[-1])
    kind = args.quantity
    ylab = ('Forward flux (state units/min)' if kind == 'flux'
            else 'Modelled kinase drive (min$^{-1}$)')
    for plot_kind in ['total','per_kinase']:
        fig, axs = plt.subplots(2, 1, figsize=(6,7), sharex=True,
            gridspec_kw={'hspace':.24, 'top':.962,'bottom':.105,'left':.165,'right':.975})
        for ax, site in zip(axs, obj['sites']):
            label = f'{args.protein} {_canon(site["name"])}'
            ax.set_title(label, loc='left', pad=6)
            ax.grid(axis='y')
            if times[-1] > cutoff:
                ax.axvspan(cutoff, times[-1], color=PALE, alpha=.10, zorder=0)
                ax.axvline(cutoff,color='#8799a4',lw=.8,ls=':',zorder=2)
            if plot_kind == 'total':
                series = [('Total',TEAL,site['nominal_total'],site['draw_total'])]
            else:
                series = [(m.kinases[ki], COLORS.get(m.kinases[ki], TEAL),
                           site['nominal_by_kin'][j],site['draw_by_kin'][:,j,:])
                          for j,ki in enumerate(site['indices'])]
            for name, color, median_nom, ensembles in series:
                q = np.percentile(ensembles,[2.5,50,97.5],axis=0)
                variability = np.max(np.ptp(ensembles, axis=0))
                if variability > 1e-9 * max(1.,float(np.max(np.abs(median_nom)))):
                    ax.fill_between(times,q[0],q[2],color=color,
                                    alpha=.19 if plot_kind=='total' else .10,
                                    lw=0,zorder=1,
                                    label='95% forward ensemble' if plot_kind=='total' else None)
                ax.plot(times,median_nom,color=color,lw=2.05 if plot_kind=='total' else 1.55,
                        label='Fitted flux' if plot_kind=='total' and kind=='flux' else
                              ('Fitted drive' if plot_kind=='total' else name),zorder=3)
            ax.set_ylabel(ylab)
            ax.set_xlim(0,times[-1]); ax.set_xscale('symlog',linthresh=.5,linscale=.68)
            ax.legend(loc='upper right' if plot_kind=='total' else 'upper center',
                      ncol=1 if plot_kind=='total' else 3,fontsize=8,
                      columnspacing=.8,handlelength=1.3)
        ticks=[x for x in (0,.5,1,4,16,60,240,960,4320) if x<=times[-1]]
        for ax in axs:
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x,pos: '0*' if abs(x)<1e-10 else f'{x:g}'))
            ax.tick_params(axis='x',labelbottom=True)
            ax.set_xlabel('Time (min, log scale)  ·  0* marks baseline')
        basename = ('egfr_phosphorylation_drive_log_time' if plot_kind=='total'
                    else 'egfr_kinase_contributions')
        for extension in ('pdf','png'):
            fig.savefig(args.out/f'{basename}.{extension}',dpi=args.dpi,
                        facecolor='white',bbox_inches=None)
        plt.close(fig)
        print(f'Created {basename}.pdf and .png ({kind})', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('.'))
    p.add_argument('--results',type=Path,default=Path('results_network_distributive'))
    p.add_argument('--phase-dir',type=Path,required=True,
                   help='Output folder created by phoskintime_phase_portraits_72h.py')
    p.add_argument('--out',type=Path,default=Path('results_network_distributive/egfr_upstream_flux_poster'))
    p.add_argument('--protein',default='EGFR')
    p.add_argument('--sites',nargs=2,default=['S991','S1166'])
    p.add_argument('--quantity',choices=['flux','drive'],default='flux',
                   help='flux uses evolving unphosphorylated EGFR and produces ensemble variation;'
                        'drive plots exogenous pseudo-first-order rates')
    p.add_argument('--hours',type=float,default=72.)
    p.add_argument('--points',type=int,default=10000)
    p.add_argument('--dpi',type=int,default=400)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    m=_load_model(args.source,args.results)
    if args.protein not in m.proteins:
        raise ValueError(f'{args.protein} absent from model')
    times,nominal_p0,P0,params,ref_fwd,nominal=_load_phase(
        args.phase_dir,m,args.protein,args.points,args.hours)
    Kt,ix=_inputs(m,times)
    gains, variable_gains=_kinase_gains(m,params)
    summaries=[];sites=[]
    for name in args.sites:
        si, actual = _site_idx(m,args.protein,name)
        indices,nominal_kin,draw_kin,nominal_tot,draw_tot=_calcs(
            m,si,Kt,gains,nominal_p0,P0,args.quantity)
        sites.append({'index':si,'name':actual,'indices':indices,
                      'nominal_by_kin':nominal_kin,'draw_by_kin':draw_kin,
                      'nominal_total':nominal_tot,'draw_total':draw_tot})
        for key,nom,dra in [('TOTAL',nominal_tot,draw_tot),
                            *[(m.kinases[ki],nominal_kin[j],draw_kin[:,j,:])
                              for j,ki in enumerate(indices)]]:
            low, med, hi=np.percentile(dra,[2.5,50,97.5],axis=0)
            summaries.append(pd.DataFrame({
                'time_min':times,'time_h':times/60,
                'protein':args.protein,'site':actual,'kinase':key,
                'quantity':args.quantity,
                'nominal':nom,'ensemble_median':med,
                'ensemble_low_95':low,'ensemble_high_95':hi,
                'input_knot_min':np.asarray(m.ktime)[ix],
            }))
    pd.concat(summaries,ignore_index=True).to_csv(
        args.out/'egfr_upstream_flux_or_drive_10000_points.csv',index=False)
    # Validate entire target-protein forward flux against same saved ODE ensemble.
    if args.quantity == 'flux':
        all_sites=[si for si,(p,_) in enumerate(m.sites) if p==args.protein]
        W=np.asarray(m.W[all_sites],dtype=float)
        base=np.asarray(m.fitted['c_k'],dtype=float)
        nom_rate=(W @ (Kt*base[:,None])).sum(axis=0)
        arr_rate=np.zeros_like(P0)
        for ki in np.flatnonzero(np.any(W!=0,axis=0)):
            arr_rate += W[:,ki].sum()*gains[:,ki,None]*Kt[ki,None,:]
        reconstructed=arr_rate*P0
        nominal_reconstructed=nom_rate*nominal_p0
        if ref_fwd is not None:
            maxerr=float(np.max(np.abs(reconstructed-ref_fwd)))
            tol=1e-4 * max(1.,float(np.max(np.abs(ref_fwd))))
            if maxerr>tol:
                raise RuntimeError(f'Joint draw/kinase mapping mismatch: forward-flux error={maxerr:g} > {tol:g}. '
                                   'Do not use figures until the ensemble pairing is corrected.')
        else:
            maxerr=None
        if 'Forward_flux' in nominal:
            nom_error=float(np.max(np.abs(nominal_reconstructed-
                                          nominal['Forward_flux'].to_numpy(dtype=float))))
            if nom_error>1e-7*max(1.,float(np.max(np.abs(nominal_reconstructed)))):
                raise RuntimeError('Nominal forward flux does not reconstruct source ODE')
        else:
            nom_error=None
    else:
        maxerr=None;nom_error=None
    report={
        'model':'PhosKinTime distributive RHS, W @ (K_input(t)*c_k), and flux times P0_EGFR',
        'quantity':args.quantity,'units':'model-state/min' if args.quantity=='flux' else 'min^-1',
        'draws':len(P0),'points':len(times),'hours':args.hours,
        'sampled_kinase_gains_from_phase_draws':variable_gains,
        'uncertainty':'Pointwise 95% forward parameter-ensemble range (NOT posterior CI)',
        'input_last_measured_min':float(m.ktime[-1]),
        'input_after_last_knot':'constant, as implemented in KinaseInput.eval',
        'flux_reconstruction_max_abs_error':maxerr,
        'nominal_flux_reconstruction_max_abs_error':nom_error,
        'outputs':'poster-compatible figure names; if quantity=flux, UPDATE AXIS/CAPTION in LaTeX',
        'note':'Forcing is exogenous. P0-dependent flux is dynamical; kinase input rates remain stepwise.',
        'series':{s['name']:[m.kinases[int(i)] for i in s['indices']] for s in sites},
    }
    (args.out/'methods_and_diagnostics.json').write_text(json.dumps(report,indent=2))
    _draw_plots(m,{'times':times,'sites':sites},args)
    print(f'OK: {len(times)} points, {len(P0)} paired draws. '
          f'Whole-EGFR flux check max error: {maxerr}',flush=True)


if __name__=='__main__':
    main()
