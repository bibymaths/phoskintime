#!/usr/bin/env python3
"""Non-MCMC EGFR kinetic-rate ensemble for the full PhosKinTime distributive ODE.

Generates high-resolution 72-hour ODE forward curves with the source-validated
reference RHS used in phoskintime_dynestyx_smoothing.py. The unweighted quantiles
are PRIOR-PREDICTIVE (not posterior). Optional likelihood importance weights
produce an approximate empirical-Bayes posterior only if ESS is adequate.

Forcing beyond its last recorded knot is held constant at the final kinase input.
The input extrapolation is a MODEL SCENARIO, not an observation-backed claim.
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
from scipy.special import logsumexp


def load_helpers(source):
    # Current layout first; the pre-reorganisation layout keeps --source usable for older checkouts.
    candidates = [source / 'scripts' / 'figures' / 'egfr_cbc2026' / 'phoskintime_dynestyx_smoothing.py',
                  source / 'scripts' / 'phoskintime_dynestyx_smoothing.py']
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise FileNotFoundError('Required original reference script missing; tried: '
                                + ', '.join(str(c) for c in candidates))
    spec = importlib.util.spec_from_file_location('_phoskintime_ode_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_specs(m, protein):
    if protein not in m.proteins or protein not in m.kinases:
        raise ValueError(f'{protein} must appear in proteins and kinase inputs')
    gi = m.proteins.index(protein)
    ki = m.kinases.index(protein)
    specs = [(f'{key}_{protein}', key, gi) for key in ('A_i','B_i','C_i','D_i','E_i')]
    specs.append((f'c_k_{protein}', 'c_k', ki))
    for idx, (gene, site) in enumerate(m.sites):
        if gene == protein:
            specs.append((f'Dp_i_{gene}_{site}', 'Dp_i', idx))
    vals = np.array([float(m.fitted[k][idx]) for _,k,idx in specs])
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError('All target fitted rates must be finite and positive; do not perturb zero rates on a log scale')
    return specs, vals


def make_draw(m, specs, values):
    fitted = {k: np.asarray(v).copy() for k,v in m.fitted.items()}
    for (_, key, idx), value in zip(specs, values):
        fitted[key][idx] = float(value)
    return SimpleNamespace(**{**vars(m), 'fitted': fitted})


def simulate_piecewise(m, reference_rhs, evaluation_times, rtol, atol):
    """BDF solves on discontinuity-separated intervals; constant final forcing after last knot."""
    times = np.asarray(evaluation_times, dtype=np.float64)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('evaluation times must start at zero and increase')
    final = float(times[-1])
    cuts = np.unique(np.r_[0., m.ktime[(m.ktime > 0) & (m.ktime < final)], final])
    y = m.y0.copy()
    out = np.empty((len(times), len(y)), dtype=np.float64)
    out[0] = y
    for start, stop in zip(cuts[:-1], cuts[1:]):
        step = int(np.searchsorted(m.ktime, start, side='right') - 1)
        # For t > last recorded grid point, this uses the final kinase-input column.
        step = min(step, len(m.ktime)-1)
        sol = solve_ivp(lambda t, z: reference_rhs(m,z,step), (float(start),float(stop)),
                        y, method='BDF', rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:
            raise RuntimeError(f'BDF failed in [{start:g}, {stop:g}]: {sol.message}')
        select = np.where((times > start) & (times <= stop))[0]
        if len(select):
            out[select] = sol.sol(times[select]).T
        y = np.asarray(sol.y[:,-1])
        if not np.all(np.isfinite(y)):
            raise RuntimeError('Nonfinite model trajectory')
    if not np.all(np.isfinite(out)):
        raise RuntimeError('Nonfinite evaluation outputs')
    return out


def extract_observables(m, states, times, proteins):
    """Same 0-minute protein/phosphosite and 4-minute RNA FC references as source."""
    idx4 = np.where(np.isclose(times, 4.0, atol=1e-10, rtol=0))[0]
    if len(idx4) != 1:
        raise ValueError('Grid must contain 4.0 min exactly')
    i4 = int(idx4[0])
    result = {}
    for protein in proteins:
        off = m.off_y[protein]
        sites = [site for gene, site in m.sites if gene == protein]
        n = len(sites)
        r = states[:,off]
        result[('RNA',protein,'')] = r / max(float(r[i4]),1e-12)
        p = states[:,off+1:off+2+n].sum(axis=1)
        result[('Protein',protein,'')] = p / max(float(m.y0[off+1:off+2+n].sum()),1e-12)
        for j, site in enumerate(sites):
            result[('Phosphorylation',protein,site)] = states[:,off+2+j] / max(float(m.y0[off+2+j]),1e-12)
    return result


def observed_rows(m, protein, keys):
    frame = m.traj[m.traj.protein == protein].copy()
    frame.psite = frame.psite.fillna('')
    frame = frame.groupby(['type','protein','psite','time'],as_index=False).fc_obs.mean()
    return [(str(r.type),str(r.protein),str(r.psite),float(r.time),float(r.fc_obs))
            for r in frame.itertuples(index=False)
            if (r.type,r.protein,r.psite) in keys and np.isfinite(r.fc_obs)]


def weighted_quantile(x, weights, quantiles):
    """Discrete importance-sampling weighted inverse CDF for each time point."""
    ix = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, ix, axis=0)
    ws = np.take_along_axis(np.broadcast_to(weights[:,None],x.shape),ix,axis=0)
    cw = np.cumsum(ws, axis=0)
    return np.stack([xs[np.argmax(cw>=q,axis=0),np.arange(x.shape[1])]
                     for q in quantiles])


def sample_log_bounded(rng, fitted, sigma, fold):
    """Independent fitted-centered, truncated lognormal draws; no MCMC."""
    spread = math.log(fold)
    z = rng.normal(0,sigma,size=len(fitted))
    while np.any(np.abs(z)>spread):
        mask = np.abs(z)>spread
        z[mask] = rng.normal(0,sigma,size=int(mask.sum()))
    return fitted * np.exp(z)



def plot_ensemble(df, protein, destination, weighted=False):
    """Plot latent forward FC ensemble with observed points, without a residual smoother."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    sub = df[df.protein == protein]
    kinds = {'RNA':0, 'Protein':1, 'Phosphorylation':2}
    groups = sorted(sub.groupby(['type','psite']),
                    key=lambda item: (kinds.get(item[0][0],9),str(item[0][1])))
    ncols = 3
    nrows = int(np.ceil(len(groups)/ncols))
    fig, axs = plt.subplots(nrows,ncols,figsize=(15,3.5*nrows),squeeze=False,
                            constrained_layout=True)
    prefix = 'importance' if weighted else 'ensemble'
    for ax, ((kind,site), frame) in zip(axs.flat, groups):
        observed=frame[frame.observed_fc.notna()]
        frame = frame[frame.observed_fc.isna()].sort_values('time_min')
        t = frame.time_min.to_numpy()
        ax.fill_between(t,frame[f'{prefix}_lower_95'],frame[f'{prefix}_upper_95'],
                        color='#1676ad',alpha=.2,label='95% ensemble interval')
        ax.plot(t,frame[f'{prefix}_median'],color='#1676ad',linewidth=1.7,
                label='Ensemble median')
        if len(observed):
            ax.scatter(observed.time_min,observed.observed_fc,
                       color='#283845',facecolors='white',edgecolors='#283845',
                       s=20,zorder=4,label='Observed')
        ax.set_xscale('symlog',linthresh=1.)
        ax.set_xlabel('Time (min; symlog)')
        ax.set_ylabel('Fold change')
        ax.set_title(f'{kind}: {protein}' if not site else f'{protein}: {site}',loc='left')
        ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(groups):]:
        ax.set_visible(False)
    if groups:
        axs.flat[0].legend(frameon=False,fontsize=8)
    label = 'importance-weighted approximate posterior' if weighted else 'prior-predictive forward ensemble'
    fig.suptitle(f'{protein} | 72-hour {label}',fontsize=14)
    for ext in ('png','pdf'):
        fig.savefig(destination/f'{protein}_{prefix}_72h.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('.'))
    p.add_argument('--results',type=Path,default=Path('results_network_distributive'))
    p.add_argument('--out',type=Path,default=Path('results_network_distributive/egfr_forward_72h'))
    p.add_argument('--protein',default='EGFR')
    p.add_argument('--output-proteins',nargs='+',default=None,help='Default EGFR; use ALL for 17 proteins')
    p.add_argument('--hours',type=float,default=72)
    p.add_argument('--points',type=int,default=10000,help='Exactly this many equally spaced output points')
    p.add_argument('--draws',type=int,default=100,help='Number of successful forward trajectories')
    p.add_argument('--max-attempt-factor',type=int,default=3)
    p.add_argument('--prior-log-sd',type=float,default=0.2)
    p.add_argument('--prior-fold',type=float,default=2.)
    p.add_argument('--weight-by-observations',action='store_true',
                   help='Importance weighting by EGFR observed FC; optional approximate posterior')
    p.add_argument('--min-effective-samples',type=float,default=20.)
    p.add_argument('--sigma-rna',type=float,default=None)
    p.add_argument('--sigma-protein',type=float,default=None)
    p.add_argument('--sigma-phospho',type=float,default=None)
    p.add_argument('--rtol',type=float,default=1e-6)
    p.add_argument('--atol',type=float,default=1e-8)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-raw-draws',action='store_true')
    args = p.parse_args()
    if args.points < 5 or args.draws < 2 or args.hours <= 0 or args.prior_log_sd<=0 or args.prior_fold<=1 or args.max_attempt_factor<1:
        p.error('Invalid points/draws/hours/prior scale/attempts')
    args.out.mkdir(parents=True,exist_ok=True)
    source = args.source.resolve()
    h = load_helpers(source)
    m = h.load_run(args.results)
    specs, fitted = target_specs(m,args.protein)
    names = [spec[0] for spec in specs]
    proteins = (m.proteins if args.output_proteins==['ALL'] else
                [args.protein] if args.output_proteins is None else args.output_proteins)
    if any(pr not in m.proteins for pr in proteins):
        p.error('Invalid --output-proteins choice')
    # Evaluations have exactly args.points main output points, plus 4 min and all obs times.
    grid = np.linspace(0.,args.hours*60.,args.points)
    if grid[-1] < 4.:
        p.error('Simulation horizon must include 4 minutes for RNA reference')
    all_obs_times = m.traj.time.to_numpy(dtype=float)
    obs_times = all_obs_times[(all_obs_times>=0)&(all_obs_times<=grid[-1])]
    time_eval = np.unique(np.r_[grid,4.,obs_times])
    output_indices = np.searchsorted(time_eval,grid)
    exact_base = np.where(np.isclose(time_eval,4.,rtol=0,atol=1e-10))[0]
    if len(exact_base)!=1:
        raise RuntimeError('Failed to include t=4 min')
    print(f'Network: {len(m.proteins)} proteins, {len(m.sites)} sites, {len(m.y0)} states',flush=True)
    print(f'Sampling {len(specs)} parameters: {names}',flush=True)
    print(f'72-hour forcing assumption: hold last kinase column after {m.ktime[-1]:g} min',flush=True)
    print(f'Output: {len(grid)} points, {len(proteins)} proteins',flush=True)
    sigma = h.get_assumed_sigmas(m,args.protein,args)
    obs_keys = {(kind,args.protein,site) for kind,protein,site in extract_observables(m,np.tile(m.y0,(len(time_eval),1)),time_eval,[args.protein])}
    obs = observed_rows(m,args.protein,obs_keys)
    exact_indices = [int(np.searchsorted(time_eval,t)) for _,_,_,t,_ in obs]
    rng = np.random.default_rng(args.seed)
    batches = {}
    draws_kept = []
    score = []
    attempts = 0
    errors = []
    while len(draws_kept)<args.draws and attempts<args.draws*args.max_attempt_factor:
        attempts+=1
        theta_values = sample_log_bounded(rng,fitted,args.prior_log_sd,args.prior_fold)
        draw_m = make_draw(m,specs,theta_values)
        try:
            states = simulate_piecewise(draw_m,h.reference_rhs,time_eval,args.rtol,args.atol)
            predicted = extract_observables(draw_m,states,time_eval,proteins)
            egfr_prediction = predicted if args.protein in proteins else extract_observables(draw_m,states,time_eval,[args.protein])
            ll = 0.
            for (kind,protein,site,t,y),index in zip(obs,exact_indices):
                mu = float(egfr_prediction[(kind,protein,site)][index])
                if not np.isfinite(mu):
                    raise RuntimeError('Nonfinite FC')
                sd = sigma[kind]
                ll -= .5*((y-mu)/sd)**2
            for key, v in predicted.items():
                output = np.asarray(v[output_indices],dtype=np.float32)
                if not np.all(np.isfinite(output)):
                    raise RuntimeError('Nonfinite evaluated fold changes')
                batches.setdefault(key,[]).append(output)
            draws_kept.append(theta_values)
            score.append(ll)
            if len(draws_kept)%10==0 or len(draws_kept)==args.draws:
                print(f'Forward trajectories {len(draws_kept)}/{args.draws} (attempts={attempts})',flush=True)
        except (RuntimeError,ValueError,OverflowError,FloatingPointError) as exc:
            errors.append(str(exc)[:300])
    if len(draws_kept)<2:
        raise RuntimeError(f'Only {len(draws_kept)} successful draws in {attempts} attempts; inspect ODE stability.')
    rates = pd.DataFrame(np.asarray(draws_kept),columns=names)
    rates['log_likelihood_without_constants'] = score
    rates.to_csv(args.out/'parameter_ensemble.csv',index=False)
    weights = np.exp(np.asarray(score)-logsumexp(score))
    ess = float(1/np.sum(weights**2))
    posterior_ok = args.weight_by_observations and ess >= args.min_effective_samples
    if args.weight_by_observations:
        print(f'Importance sampling ESS: {ess:.2f}/{len(weights)}; approximate posterior usable: {posterior_ok}',flush=True)
    data=[]
    raw={}
    qs=[0.025,0.5,0.975]
    for (kind,protein,site),rows in batches.items():
        mat=np.stack(rows)
        unweighted=np.quantile(mat,qs,axis=0)
        weighted = weighted_quantile(mat,weights,qs) if posterior_ok else None
        if args.save_raw_draws:
            raw[f'{kind}|{protein}|{site or "total"}']=mat
        frame=pd.DataFrame({'type':kind,'protein':protein,'psite':site,'time_min':grid,
                            'ensemble_lower_95':unweighted[0],
                            'ensemble_median':unweighted[1],
                            'ensemble_upper_95':unweighted[2]})
        if weighted is not None:
            frame['importance_lower_95']=weighted[0]
            frame['importance_median']=weighted[1]
            frame['importance_upper_95']=weighted[2]
        data.append(frame)
    output = pd.concat(data,ignore_index=True)
    # Preserve measured observations separately from interpolated/evaluated curves.
    measured = m.traj[['type','protein','psite','time','fc_obs']].copy()
    measured['psite'] = measured.psite.fillna('')
    measured = measured.groupby(['type','protein','psite','time'],as_index=False).fc_obs.mean()
    measured.to_csv(args.out/'observed_data.csv',index=False)
    output.to_csv(args.out/'forward_ensemble_10000_points.csv',index=False)
    # Overlay actual observations from the original data; no interpolation of measurements.
    overlay = output.copy()
    overlay['observed_fc'] = np.nan
    for protein in proteins:
        # Plot measured points on matching curve by taking the prediction grid's nearest
        # time index only for plotting; retain their ORIGINAL measurement time.
        fig_rows=[]
        for (kind,prot,site), chunk in overlay[overlay.protein==protein].groupby(['type','protein','psite']):
            orig = measured[(measured.type==kind)&(measured.protein==prot)&(measured.psite==site)]
            c = chunk.copy()
            for r in orig.itertuples(index=False):
                if 0 <= r.time <= grid[-1]:
                    addition = c.iloc[[0]].copy()
                    addition['time_min'] = r.time
                    addition['observed_fc'] = r.fc_obs
                    c = pd.concat([c,addition],ignore_index=True)
            fig_rows.append(c)
        to_plot = pd.concat(fig_rows,ignore_index=True)
        plot_ensemble(to_plot,protein,args.out,weighted=False)
        if posterior_ok:
            plot_ensemble(to_plot,protein,args.out,weighted=True)
    if args.save_raw_draws:
        np.savez_compressed(args.out/'all_forward_draws.npz',time_min=grid,**raw)
    report={'method':'Prior-predictive truncated-lognormal forward ensemble (no MCMC)',
            'horizon_hours':args.hours,'resolution_points':len(grid),'successful_draws':len(draws_kept),
            'attempts':attempts,'failed_attempts':attempts-len(draws_kept),
            'example_errors':errors[:10],'sampled_parameters':names,
            'fixed_parameters':'All other network parameters and all 70 initial states',
            'prior_log_sd':args.prior_log_sd,'prior_fold':args.prior_fold,
            'forcing_extrapolation':'final measured kinase forcing held constant until simulation end',
            'observations_used_for_importance_weighting':len(obs),
            'observation_SD_assumed_from_saved_fit_or_explicit':sigma,
            'importance_weighting_requested':args.weight_by_observations,
            'importance_ESS':ess if args.weight_by_observations else None,
            'importance_min_ESS':args.min_effective_samples,
            'approximate_posterior_intervals_exported':bool(posterior_ok),
            'caveat':'Unweighted ensemble is NOT a posterior; importance-weighted intervals are an empirical-Bayes approximation only with adequate ESS; curve values are latent-state fold changes, without measurement noise.'}
    (args.out/'methods_and_diagnostics.json').write_text(json.dumps(report,indent=2))
    print(f'Complete: {len(draws_kept)} successes, {attempts} attempts, outputs: {args.out.resolve()}',flush=True)

if __name__=='__main__':
    main()
