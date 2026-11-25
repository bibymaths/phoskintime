#!/usr/bin/env python3
"""Dynestyx ODE posterior CIs and dynestyx discrete residual smoothing.

Full 70-state PhosKinTime distributive ODE from the source code; inference
conditional on all exported fitted parameters except two target EGFR rates.
Posterior trajectories use 2.5/50/97.5 percentiles at every requested time.
An independent discrete-time linear-Gaussian residual smoother uses all
available measurements at each observation index (not a nonlinear-state RTS
smoother). Its intervals are separate and do not include parameter uncertainty.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

EPS = 1e-12


def load_run(root):
    root = Path(root).resolve()
    data = root / 'tables' if not (root / 'initial_conditions_y0.csv').exists() else root
    ic = pd.read_csv(data / 'initial_conditions_y0.csv')
    kin = pd.read_csv(data / 'network_kinase_inputs.csv')
    proteins = ic.entity.drop_duplicates().tolist()
    kinases = kin.iloc[:, 0].tolist()
    sites = [(row.entity, row.substate) for row in ic[ic.kind == 'phospho_state'].itertuples()]
    fitted = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in
              json.loads((data / 'fitted_params_picked.json').read_text()).items()}
    N, ns, nk = len(proteins), len(sites), len(kinases)
    expected = dict(A_i=N, B_i=N, C_i=N, D_i=N, E_i=N, Dp_i=ns, c_k=nk, tf_scale=1)
    for key, size in expected.items():
        if fitted[key].size != size:
            raise ValueError(f'Parameter {key}: expected {size}, got {fitted[key].size}')
    off_y, off_s, y0, site_cursor = {}, {}, [], 0
    for gene in proteins:
        part = ic[ic.entity == gene]
        off_y[gene], off_s[gene] = len(y0), site_cursor
        r = part[part.kind == 'mRNA'].value.to_numpy(dtype=float)
        p = part[(part.kind == 'protein_state') & (part.substate == 'P0')].value.to_numpy(dtype=float)
        ph = part[part.kind == 'phospho_state'].value.to_numpy(dtype=float)
        if len(r) != 1 or len(p) != 1:
            raise ValueError(f'Invalid distributive state layout: {gene}')
        y0.extend([r[0], p[0], *ph])
        site_cursor += len(ph)
    site_map = {f'{gene}_{site}': i for i, (gene, site) in enumerate(sites)}
    kinase_map = {k: i for i, k in enumerate(kinases)}
    pmap = {p: i for i, p in enumerate(proteins)}
    W = np.zeros((ns, nk))
    for r in pd.read_csv(data / 'network_W_global.csv').itertuples(index=False):
        W[site_map[r.Site], kinase_map[r.Kinase]] += r.Weight
    TF = np.zeros((N, N))
    for r in pd.read_csv(data / 'network_tf_mat.csv').itertuples(index=False):
        TF[pmap[r.Target], pmap[r.Source_TF]] += r.Weight
    deg = np.abs(TF).sum(axis=1)
    deg[deg < 1e-12] = 1.0  # Exactly as PosteriorObjective/runner
    ktime = np.array([float(c[2:]) for c in kin.columns[1:]], dtype=float)
    if not np.all(np.diff(ktime) > 0) or ktime[0] != 0:
        raise ValueError('Unsupported kinase time grid')
    if len(y0) != 2 * N + ns or site_cursor != ns:
        raise ValueError('Incorrect number of model states')
    meta = json.loads((root / 'metadata.json').read_text())
    if meta.get('command_arguments', {}).get('model') != 'distributive':
        raise ValueError('This integration is restricted to the distributive ODE model')
    traj = pd.read_csv(root / 'model_trajectories.csv')
    return SimpleNamespace(root=root, proteins=proteins, kinases=kinases, sites=sites,
         off_y=off_y, off_s=off_s, y0=np.asarray(y0), W=W, TF=TF, tf_deg=deg,
         ktime=ktime, K=kin.iloc[:, 1:].to_numpy(dtype=float), fitted=fitted,
         driver=np.array([kinase_map.get(p, -1) for p in proteins]), traj=traj)


def reference_rhs(m, y, step):
    """Source-equivalent distributive RHS; kinase forcing fixed over one interval."""
    p = m.fitted
    K = m.K[:, step] * p['c_k']
    S = m.W @ K
    P = np.empty(len(m.proteins))
    for i, gene in enumerate(m.proteins):
        off = m.off_y[gene]
        ns = sum(gene == g for g, _ in m.sites)
        P[i] = K[m.driver[i]] if m.driver[i] >= 0 else y[off+1:off+2+ns].sum()
    U = (m.TF @ P) / m.tf_deg
    dy = np.zeros_like(y)
    for i, gene in enumerate(m.proteins):
        o, so = m.off_y[gene], m.off_s[gene]
        ns = sum(gene == g for g, _ in m.sites)
        R, P0 = y[o:o+2]
        u = U[i] / (1 + abs(U[i]))
        if u >= 0:
            synth = p['A_i'][i] * (1 + p['tf_scale'][0] * u / (1 + u + 1e-6))
        else:
            synth = p['A_i'][i] / (1 + p['tf_scale'][0] * abs(u))
        dy[o] = synth - p['B_i'][i] * R
        ph = y[o+2:o+2+ns]
        fwd = S[so:so+ns] * P0
        back = p['E_i'][i] * ph
        dy[o+1] = p['C_i'][i]*R - p['D_i'][i]*P0 - fwd.sum() + back.sum()
        dy[o+2:o+2+ns] = fwd - (p['E_i'][i]+p['Dp_i'][so:so+ns]+p['D_i'][i])*ph
    return dy


def integrate_reference(m, times):
    """Separate BDF solves at each kinase-input jump to avoid boundary error."""
    times = np.asarray(times)
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Simulation times must start at zero and increase strictly')
    cuts = np.unique(np.r_[m.ktime[m.ktime < times[-1]], times[-1]])
    Y = np.full((len(times), len(m.y0)), np.nan)
    Y[0] = m.y0
    state = m.y0.copy()
    for start, stop in zip(cuts[:-1], cuts[1:]):
        step = int(np.searchsorted(m.ktime, start, side='right')-1)
        req = times[(times > start) & (times <= stop)]
        save = np.unique(np.r_[req, stop])
        sol = solve_ivp(lambda t, y: reference_rhs(m, y, step),
                        (start, stop), state, t_eval=save, method='BDF',
                        rtol=1e-8, atol=1e-9)
        if not sol.success:
            raise RuntimeError(sol.message)
        for j, t in enumerate(save):
            idx = np.where(np.isclose(times, t, atol=1e-10, rtol=0))[0]
            Y[idx] = sol.y[:, j]
        state = sol.y[:, -1]
    if not np.all(np.isfinite(Y)):
        raise RuntimeError('Nonfinite simulation output')
    return Y


def validate_export(m, output, tolerance):
    """Compare all saved RNA/protein/phospho predictions, no dynestyx needed."""
    times = np.unique(np.r_[0., 4., m.traj.time.unique()])
    Y = integrate_reference(m, times)
    t0 = int(np.where(times == 0.0)[0][0])
    t4 = int(np.where(times == 4.0)[0][0])
    rows = []
    for gene in m.proteins:
        o = m.off_y[gene]
        n = sum(gene == g for g, _ in m.sites)
        rna = np.maximum(Y[:, o], EPS)
        tot = np.maximum(Y[:, o+1:o+2+n].sum(axis=1), EPS)
        rows.extend([('RNA',gene,'',t,v) for t,v in zip(times, rna / rna[t4])])
        rows.extend([('Protein',gene,'',t,v) for t,v in zip(times, tot / tot[t0])])
        sites = [s for g,s in m.sites if g == gene]
        for j,s in enumerate(sites):
            ph = np.maximum(Y[:, o+2+j], EPS)
            rows.extend([('Phosphorylation',gene,s,t,v) for t,v in zip(times, ph / ph[t0])])
    pred = pd.DataFrame(rows,columns=['type','protein','psite','time','prediction'])
    exported = m.traj[['type','protein','psite','time','fc_pred']].copy()
    exported.psite = exported.psite.fillna('')
    matched = exported.merge(pred, how='left', on=['type','protein','psite','time'], validate='one_to_one')
    if matched.prediction.isna().any():
        raise RuntimeError('Could not map every exported prediction to the source-derived layout')
    matched['abs_error'] = np.abs(matched.fc_pred-matched.prediction)
    report = {'proteins':len(m.proteins),'phosphosites':len(m.sites),
              'kinases':len(m.kinases),'states':len(m.y0),
              'max_absolute_error':float(matched.abs_error.max()),
              'by_layer':matched.groupby('type').abs_error.agg(['mean','max']).to_dict('index'),
              'tolerance':tolerance, 'pass':bool(matched.abs_error.max() <= tolerance)}
    output.mkdir(parents=True,exist_ok=True)
    matched.to_csv(output/'saved_fit_comparison.csv',index=False)
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not report['pass']:
        raise RuntimeError('Export reproduction failed tolerance; inference blocked')
    return report


def observation_matrix(m, protein, include_protein=True):
    """Use the exact zero-minute FC denominators of simulate_and_measure."""
    if protein not in m.off_y:
        raise ValueError(f'{protein} absent from the network')
    x = m.traj[m.traj.protein == protein]
    species_sites = [site for gene, site in m.sites if gene == protein]
    times = np.sort(x[x.type == 'Phosphorylation'].time.unique())
    names = [f'{protein}_{site}' for site in species_sites]
    if include_protein:
        names.append(f'{protein}_protein')
    Y = np.zeros((len(times), len(names)))
    H = np.zeros((len(names), len(m.y0)))
    off = m.off_y[protein]
    for j, site in enumerate(species_sites):
        sub = x[(x.type == 'Phosphorylation') & (x.psite == site)].set_index('time')
        Y[:, j] = sub.loc[times, 'fc_obs'].to_numpy(dtype=float)
        H[j, off+2+j] = 1 / max(m.y0[off+2+j], EPS)
    if include_protein:
        sub = x[x.type == 'Protein'].set_index('time')
        Y[:, -1] = sub.loc[times, 'fc_obs'].to_numpy(dtype=float)
        H[-1, off+1:off+2+len(species_sites)] = 1 / max(
            m.y0[off+1:off+2+len(species_sites)].sum(), EPS)
    if not np.all(np.isfinite(Y)) or times[0] != 0:
        raise ValueError('Expected finite observations including t=0')
    return times, Y, H, names




def make_source_context(m, source):
    """Load *real* PhosKinTime JAX RHS using the source's model configuration."""
    import os
    source = Path(source).expanduser().resolve()
    if not (source / 'networkmodel' / 'backend.py').is_file():
        raise ValueError(f'--source is not the PhosKinTime repository: {source}')
    os.environ['PHOSKINTIME_NETWORKMODEL_CONFIG'] = str(source / 'config.toml')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from networkmodel.config import MODEL
    from networkmodel.backend import make_networkmodel_rhs
    if MODEL != 0:
        raise RuntimeError('Expected distributive (distmod) model; check config.toml')
    idx = SimpleNamespace(
        N=len(m.proteins), proteins=m.proteins, kinases=m.kinases,
        n_sites=np.array([sum(g == p for g, _ in m.sites) for p in m.proteins], dtype=np.int32),
        offset_y=np.array([m.off_y[p] for p in m.proteins], dtype=np.int32),
        offset_s=np.array([m.off_s[p] for p in m.proteins], dtype=np.int32),
        p2i={p: i for i, p in enumerate(m.proteins)},
        k2i={k: i for i, k in enumerate(m.kinases)}, proxy_map={},
    )
    sysobj = SimpleNamespace(
        idx=idx, W_global=csr_matrix(m.W), tf_mat=csr_matrix(m.TF),
        tf_deg=np.asarray(m.tf_deg),
        kin=SimpleNamespace(grid=m.ktime, Kmat=m.K),
    )
    rhs = make_networkmodel_rhs(sysobj)
    params = {
        k: jnp.asarray(float(v[0]) if k == 'tf_scale' else v, dtype=jnp.float64)
        for k, v in m.fitted.items()
    }
    y0 = jnp.asarray(m.y0, dtype=jnp.float64)
    # Check source-model and independently reconstructed RHS at an input interval.
    err = float(np.max(np.abs(
        np.asarray(rhs(jnp.asarray(0.25), y0, params)) - reference_rhs(m, m.y0, 0)
    )))
    if not np.isfinite(err) or err > 1e-9:
        raise RuntimeError(f'PhosKinTime source RHS verification failed: max_err={err}')
    print(f'Original PhosKinTime RHS verified, max absolute error {err:.3g}', flush=True)
    return rhs, params, y0, err


def get_assumed_sigmas(m, protein, args):
    """Fixed Gaussian observation SDs; plug-in residual RMS if not user-supplied."""
    df = m.traj[m.traj.protein == protein]
    result = {}
    for key, label in [('Phosphorylation', 'sigma_phospho'),
                       ('Protein', 'sigma_protein'), ('RNA', 'sigma_rna')]:
        explicit = getattr(args, label)
        chunk = df[df.type == key]
        if explicit is not None:
            result[key] = explicit
        elif len(chunk):
            rmse = float(np.sqrt(np.mean((chunk.fc_obs - chunk.fc_pred) ** 2)))
            result[key] = max(rmse, 0.02)
        else:
            result[key] = 0.1
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'Invalid assumed observation sigma for {key}')
    return result


def run_ode_posterior(m, args, rhs, fitted, y0, config):
    """NUTS via dynestyx.LatentPathBuilder, or load completed prior NUTS draws."""
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value
    import dynestyx as dsx
    from dynestyx import ContinuousTimeStateEvolution, DynamicalModel, LinearGaussianObservation

    if args.protein not in m.kinases:
        raise ValueError(f'{args.protein} is not a kinase input; set --protein appropriately')
    gene_i = m.proteins.index(args.protein)
    kin_i = m.kinases.index(args.protein)
    obs_times, obs, H, names = observation_matrix(m, args.protein, include_protein=True)
    sigmas = get_assumed_sigmas(m, args.protein, args)
    scales = [sigmas['Phosphorylation']] * (len(names)-1) + [sigmas['Protein']]
    H = jnp.asarray(H, dtype=jnp.float64)
    R = jnp.diag(jnp.asarray(scales, dtype=jnp.float64)**2)
    init = {
        'c_k_target': jnp.asarray(float(m.fitted['c_k'][kin_i])),
        'E_i_target': jnp.asarray(float(m.fitted['E_i'][gene_i])),
    }
    for name, bounds in [('c_k_target', args.ck_bounds), ('E_i_target', args.e_bounds)]:
        if not bounds[0] < float(init[name]) < bounds[1]:
            raise ValueError(f'Fitted {name}={init[name]} outside prior {bounds}')

    if args.posterior_file is not None:
        with np.load(args.posterior_file) as z:
            posterior = {k: np.asarray(z[k]).reshape(-1)
                         for k in ('c_k_target', 'E_i_target')}
        if len(posterior['c_k_target']) < 5 or (
            len(posterior['c_k_target']) != len(posterior['E_i_target'])
        ) or not all(np.all(np.isfinite(v)) for v in posterior.values()):
            raise ValueError('Posterior file must contain at least 5 valid paired draws')
        print(f'Using {len(posterior["c_k_target"])} saved posterior draws.',flush=True)
        return posterior, sigmas, {'posterior_source':str(args.posterior_file),
                                   'sampler_status':'loaded_existing_draws'}

    # All other fitted parameters are conditioned on their exported values;
    # the 70-dimensional initial state is fixed at its exported estimate.
    def generative_model(obs_times=None, obs_values=None, predict_times=None):
        ck = numpyro.sample('c_k_target', dist.Uniform(*args.ck_bounds))
        ee = numpyro.sample('E_i_target', dist.Uniform(*args.e_bounds))
        theta = dict(fitted)
        theta['c_k'] = fitted['c_k'].at[kin_i].set(ck)
        theta['E_i'] = fitted['E_i'].at[gene_i].set(ee)
        dynamics = DynamicalModel(
            initial_condition=dist.Normal(y0, jnp.ones_like(y0)).to_event(1),
            state_evolution=ContinuousTimeStateEvolution(
                drift=lambda state, control, t: rhs(t, state, theta)),
            observation_model=LinearGaussianObservation(H=H, R=R),
        )
        return dsx.sample('f', dynamics,
                          obs_times=obs_times, obs_values=obs_values,
                          predict_times=predict_times)

    fixed_initial = numpyro.handlers.condition(
        generative_model, data={'f_state_path_params': y0[None, :]})
    print(f'Running dynestyx ODE NUTS: {args.chains} chains × '
          f'{args.warmup} warmup + {args.samples} posterior samples.', flush=True)
    with dsx.LatentPathBuilder(ode_simulator_config=config):
        mcmc = MCMC(
            NUTS(fixed_initial,
                 init_strategy=init_to_value(values=init),
                 target_accept_prob=args.target_accept, dense_mass=True),
            num_warmup=args.warmup, num_samples=args.samples,
            num_chains=args.chains, chain_method='sequential', progress_bar=True, jit_model_args=True)
        mcmc.run(jr.PRNGKey(args.seed),
                 obs_times=jnp.asarray(obs_times), obs_values=jnp.asarray(obs))
    posterior = {k: np.asarray(v).reshape(-1) for k, v in
                 mcmc.get_samples(group_by_chain=True).items()
                 if k in ('c_k_target', 'E_i_target')}
    if set(posterior) != {'c_k_target','E_i_target'}:
        raise RuntimeError('Missing posterior parameter draws')
    extra = mcmc.get_extra_fields(group_by_chain=True)
    divergences = int(np.asarray(extra.get('diverging', np.zeros(0))).sum())
    info = {'posterior_source':'dynestyx.LatentPathBuilder+NumPyro.NUTS',
            'sampler_status':'completed', 'chains':args.chains,
            'warmup':args.warmup, 'samples_per_chain':args.samples,
            'divergences':divergences,
            'note':'Check R-hat, ESS and divergences before interpreting 95% intervals'}
    if divergences:
        print(f'WARNING: NUTS reported {divergences} divergences. '
              'Do not treat intervals as reliable without resolving them.', flush=True)
    try:
        mcmc.print_summary()
    except Exception as e:
        print(f'Sampler summary unavailable: {e}', flush=True)
    return posterior, sigmas, info


def simulate_posterior(m, args, rhs, fitted, y0, posterior, ode_config, times):
    """Solve the *full* JAX ODE for posterior draws on the dense time grid."""
    import jax
    import jax.numpy as jnp
    import diffrax as dfx
    gene_i = m.proteins.index(args.protein)
    kin_i = m.kinases.index(args.protein)
    t0 = float(times[0]); t1 = float(times[-1])
    assert t0 == 0.0
    save_ts = jnp.asarray(times, dtype=jnp.float64)

    @jax.jit
    def solve_draw(ck, ee):
        theta = dict(fitted)
        theta['c_k'] = fitted['c_k'].at[kin_i].set(ck)
        theta['E_i'] = fitted['E_i'].at[gene_i].set(ee)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            solver=ode_config.solver, t0=t0, t1=t1, dt0=ode_config.dt0,
            y0=y0, args=theta, saveat=dfx.SaveAt(ts=save_ts),
            stepsize_controller=ode_config.stepsize_controller,
            adjoint=ode_config.adjoint, max_steps=ode_config.max_steps,
        )
        return sol.ys

    total = len(posterior['c_k_target'])
    n = min(args.predictive_draws, total)
    rng = np.random.default_rng(args.seed + 17)
    sel = np.sort(rng.choice(total, size=n, replace=False))
    result = []
    for i, k in enumerate(sel):
        ck = float(posterior['c_k_target'][k])
        ee = float(posterior['E_i_target'][k])
        state = np.asarray(solve_draw(jnp.asarray(ck), jnp.asarray(ee)))
        if state.shape != (len(times), len(m.y0)) or not np.all(np.isfinite(state)):
            raise RuntimeError(f'Nonfinite/invalid ODE solution for draw {k}')
        result.append(state)
        if (i+1) % 10 == 0 or i+1 == n:
            print(f'ODE posterior predictions {i+1}/{n}',flush=True)
    return np.stack(result), sel


def observable_series(m, states, times, target_proteins):
    """Return {observable_id: draws × times} with experiment-specific baselines."""
    if 4.0 not in times:
        raise RuntimeError('Time grid must include exactly 4.0 min (RNA reference)')
    at4 = int(np.where(times == 4.0)[0][0])
    eps = 1e-12
    res = {}
    for protein in target_proteins:
        off = m.off_y[protein]
        sites = [s for g, s in m.sites if g == protein]
        n = len(sites)
        # Correct fold-change normalizers: protein & phosphosites t=0,
        # RNA t=4. Preserve normalization for each posterior trajectory.
        r = states[:, :, off]
        res[('RNA', protein, '')] = r / np.maximum(r[:, at4:at4+1], eps)
        total = states[:, :, off+1:off+2+n].sum(axis=-1)
        res[('Protein', protein, '')] = total / max(m.y0[off+1:off+2+n].sum(),eps)
        for j, site in enumerate(sites):
            ps = states[:, :, off+2+j]
            res[('Phosphorylation', protein, site)] = ps / max(m.y0[off+2+j],eps)
    return res


def dense_time_grid(m, args):
    """Every experimental time plus log-spaced dense times and 4-min RNA ref."""
    max_t = float(np.max(m.traj.time.to_numpy()))
    if max_t <= 4:
        raise ValueError('Unexpected experiment maximum time')
    dense = np.geomspace(args.min_positive_time, max_t, args.dense_points)
    return np.unique(np.r_[0.0, 4.0, dense, m.ktime[m.ktime<=max_t],
                             m.traj.time.to_numpy()])


def smoothing_residuals_dynestyx(residuals, sigma, q_sd, seed):
    """Exact 1D Kalman RTS smoothing of a *residual* random walk, not ODE states.

    z[k] = z[k-1] + N(0,q_sd²); residual[k] = z[k] + N(0,sigma²).
    k is the observation INDEX, not elapsed minutes. Irregular assay times
    therefore have the same process variance per acquired sample.
    """
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import dynestyx as dsx
    from numpyro.infer import Predictive
    from dynestyx import Smoother
    from dynestyx.inference.configs.smoother import KFSmootherConfig

    def residual_model(obs_times=None, obs_values=None, predict_times=None):
        dynamics = dsx.LTI_discrete(
            A=jnp.array([[1.0]]), Q=jnp.array([[q_sd**2]]),
            H=jnp.array([[1.0]]), R=jnp.array([[sigma**2]]),
            initial_mean=jnp.array([0.0]),
            initial_cov=jnp.array([[max(sigma**2, q_sd**2)]]),
        )
        return dsx.sample('f', dynamics,
                          obs_times=obs_times, obs_values=obs_values,
                          predict_times=predict_times)

    # Discrete LTI transitions correspond to one observation increment,
    # regardless of nonuniform physical times in the experimental series.
    discrete_index = jnp.arange(len(residuals), dtype=jnp.float64)
    with Smoother(smoother_config=KFSmootherConfig(
        filter_source='cd_dynamax',
        record_smoothed_states_mean=True,
        record_smoothed_states_cov_diag=True)):
        result = Predictive(residual_model, params={}, num_samples=1,
                            condition_deterministic=False)(
            jr.PRNGKey(seed), obs_times=discrete_index,
            obs_values=jnp.asarray(residuals)[:, None])
    z_mean = np.asarray(result['f_smoothed_states_mean'])[0, :, 0]
    z_var = np.asarray(result['f_smoothed_states_cov_diag'])[0, :, 0]
    if not np.all(np.isfinite(z_mean)) or not np.all(np.isfinite(z_var)):
        raise RuntimeError('Dynestyx smoother gave nonfinite residual estimates')
    return z_mean, np.sqrt(np.maximum(z_var,0.0))


def construct_tables(m, args, times, observables, sigmas_by_protein):
    """Build dense ODE conditional posterior table with null smoothing until measured."""
    records = []
    for (kind, protein, site), values in observables.items():
        # Pointwise 95% equal-tail credible interval across parameter draws.
        q = np.percentile(values, [2.5, 50, 97.5], axis=0)
        for t, low, median, high in zip(times, *q):
            records.append({'type':kind, 'protein':protein,'psite':site,
                            'time_min':float(t),'ode_ci_lower_95':float(low),
                            'ode_median':float(median),'ode_ci_upper_95':float(high),
                            'observed_fc':np.nan, 'smoother_mean':np.nan,
                            'smoother_ci_lower_95':np.nan,
                            'smoother_ci_upper_95':np.nan})
    table = pd.DataFrame(records)
    obs = m.traj[['type','protein','psite','time','fc_obs']].copy()
    obs['psite'] = obs.psite.fillna('')
    obs.rename(columns={'time':'time_min','fc_obs':'observed_fc'},inplace=True)
    obs = obs.groupby(['type','protein','psite','time_min'],as_index=False).observed_fc.mean()
    table.drop(columns=['observed_fc'],inplace=True)
    table = table.merge(obs, how='left',
                        on=['type','protein','psite','time_min'],validate='one_to_one')
    table['is_measured'] = table.observed_fc.notna()
    smooth_records = []
    relevant = table.loc[table.is_measured].groupby(['type','protein','psite'],sort=False)
    for n, ((kind, protein, site), chunk) in enumerate(relevant):
        chunk = chunk.sort_values('time_min')
        ts = chunk.time_min.to_numpy()
        med = chunk.ode_median.to_numpy()
        observed = chunk.observed_fc.to_numpy()
        sd = sigmas_by_protein[protein][kind]
        proc_sd = args.smoothing_process_factor * sd
        if len(ts) < 2:
            continue
        z_mean, z_sd = smoothing_residuals_dynestyx(
            observed-med, sd, proc_sd, seed=args.seed+n+100)
        smoother_mean = med + z_mean
        lo = smoother_mean - 1.95996398454 * z_sd
        hi = smoother_mean + 1.95996398454 * z_sd
        for t, mean, left, right in zip(ts, smoother_mean, lo, hi):
            smooth_records.append({'type':kind,'protein':protein,'psite':site,
                                   'time_min':float(t), 'smoother_mean':float(mean),
                                   'smoother_ci_lower_95':float(left),
                                   'smoother_ci_upper_95':float(right)})
    if smooth_records:
        smoothing = pd.DataFrame(smooth_records)
        table.drop(columns=['smoother_mean','smoother_ci_lower_95','smoother_ci_upper_95'],
                   inplace=True)
        table = table.merge(smoothing,how='left',
                            on=['type','protein','psite','time_min'],validate='one_to_one')
    else:
        smoothing = pd.DataFrame(columns=['type','protein','psite','time_min',
             'smoother_mean','smoother_ci_lower_95','smoother_ci_upper_95'])
    table['smoother_method'] = 'dynestyx_KF_RTS_residual_random_walk'
    table['ode_ci_method'] = 'dynestyx_NUTS_conditional_parameter_posterior'
    table['measurement_sd_assumed'] = [sigmas_by_protein[p][k]
        for p,k in zip(table.protein,table.type)]
    table.sort_values(['protein','type','psite','time_min'],inplace=True,ignore_index=True)
    return table, smoothing


def plot_uncertainty_overlay(df, protein, output, dpi=320):
    """Publication-style small multiples, ODE CI band + smoother point intervals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12.3,
        'axes.labelsize':11.0,'axes.titleweight':'semibold',
        'axes.edgecolor':'#9aabb7','axes.linewidth':.75,
        'axes.spines.top':False,'axes.spines.right':False,
        'grid.color':'#dbe2e8','grid.linewidth':.55,'grid.alpha':.7,
        'figure.facecolor':'white','savefig.facecolor':'white',
        'legend.frameon':False,'legend.fontsize':9.0,
    })
    sub = df[df.protein==protein]
    kinds = {'RNA':0,'Protein':1,'Phosphorylation':2}
    parts = sorted(sub.groupby(['type','psite']),
                   key=lambda g:(kinds.get(g[0][0],9),str(g[0][1])))
    ncols = 3
    nrows = int(np.ceil(len(parts)/ncols))
    fig,axs=plt.subplots(nrows,ncols,figsize=(16,4.1*nrows),squeeze=False,
                         constrained_layout=True)
    for ax, ((typ,site),part) in zip(axs.flat,parts):
        part=part.sort_values('time_min')
        t=part.time_min.to_numpy()
        ax.fill_between(t,part.ode_ci_lower_95.to_numpy(),part.ode_ci_upper_95.to_numpy(),
                        color='#1676ad',alpha=.18,label='ODE 95% parameter CI')
        ax.plot(t,part.ode_median.to_numpy(),color='#1676ad',lw=2.1,label='ODE median')
        sm=part[part.smoother_mean.notna()].sort_values('time_min')
        if len(sm):
            smt=sm.time_min.to_numpy()
            smm=sm.smoother_mean.to_numpy()
            ax.errorbar(smt,smm,
                yerr=np.vstack([smm-sm.smoother_ci_lower_95.to_numpy(),
                                sm.smoother_ci_upper_95.to_numpy()-smm]),
                fmt='s--',ms=3.7,lw=1.2,elinewidth=.8,capsize=2,
                color='#d47728',alpha=.95,label='Discrete RTS smoother ±95%')
            ax.scatter(smt,sm.observed_fc.to_numpy(),s=24,
                       marker='o',facecolor='white',edgecolor='#243b53',
                       linewidth=1.05,zorder=4,label='Measured')
        ax.set_xscale('symlog',linthresh=1.0,linscale=.65)
        xticks=np.asarray([0.,1.,4.,16.,60.,240.,960.])
        ax.set_xticks(xticks[xticks<=t[-1]+1e-8])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:g}'))
        ax.set_xlim(0.,t[-1]*1.03)
        ax.set_xlabel('Time (min; log above 1 min)')
        ax.set_ylabel('Fold change')
        title=f'{typ}: {protein}' if not site else f'{protein}: {site}'
        ax.set_title(title,loc='left')
        ax.grid(True,alpha=.55)
    for ax in list(axs.flat)[len(parts):]:
        ax.set_visible(False)
    if len(parts):
        axs.flat[0].legend(loc='best',fontsize=8)
    fig.suptitle(f'{protein} | ODE posterior 95% CI and discrete-time residual smoothing',
                 fontsize=15,fontweight='semibold')
    out=Path(output)/f'{protein}_dynestyx_ODE_CI_discrete_smoothing.png'
    fig.savefig(out,dpi=dpi,bbox_inches='tight',pad_inches=.17)
    plt.close(fig)
    print(f'Saved figure: {out}',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--results',type=Path,default=Path('results_network_distributive'),
                   help='Unzipped run root containing tables/ and model_trajectories.csv')
    p.add_argument('--source',type=Path,default=Path('.'),
                   help='Unzipped PhosKinTime git root, with networkmodel/ and config.toml')
    p.add_argument('--out',type=Path,default=Path('dynestyx_ode_smoothing'))
    p.add_argument('--protein',default='EGFR',help='Target kinase/gene with inferred rates')
    p.add_argument('--output-proteins',nargs='+',default=None,
                   help='Report/plot specified proteins; default inference target. Use ALL for entire model.')
    p.add_argument('--posterior-file',type=Path,default=None,
                   help='Skip NUTS and use previously COMPLETED posterior_samples.npz')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--warmup',type=int,default=250)
    p.add_argument('--samples',type=int,default=500)
    p.add_argument('--chains',type=int,default=2)
    p.add_argument('--predictive-draws',type=int,default=100)
    p.add_argument('--dense-points',type=int,default=250,
                   help='Additional geometrically spaced points to predict (all observed times included)')
    p.add_argument('--min-positive-time',type=float,default=0.05)
    p.add_argument('--ck-bounds',nargs=2,type=float,default=(0.001,4.0))
    p.add_argument('--e-bounds',nargs=2,type=float,default=(0.0001,10.0))
    p.add_argument('--sigma-phospho',type=float,default=None)
    p.add_argument('--sigma-protein',type=float,default=None)
    p.add_argument('--sigma-rna',type=float,default=None,
                   help='Used in RNA smoothing, not in the ODE NUTS likelihood')
    p.add_argument('--smoothing-process-factor',type=float,default=.3,
                   help='Discrete residual process SD as a fraction of observation SD')
    p.add_argument('--rtol',type=float,default=1e-6)
    p.add_argument('--atol',type=float,default=1e-8)
    p.add_argument('--max-steps',type=int,default=200000)
    p.add_argument('--target-accept',type=float,default=.9)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dpi',type=int,default=320)
    p.add_argument('--validation-tol',type=float,default=.001)
    args=p.parse_args()
    if any(v<=0 for v in [args.predictive_draws,args.dense_points,args.min_positive_time,
                          args.rtol,args.atol,args.max_steps,args.dpi,
                          args.smoothing_process_factor]) or args.samples<5 or args.chains<1 or args.warmup<0:
        p.error('Invalid run configuration; counts and tolerances must be positive.')
    args.out.mkdir(parents=True,exist_ok=True)
    m=load_run(args.results)
    validate_export(m,args.out,args.validation_tol)
    if args.validate_only:
        print('Deterministic export reproduction PASS. No posterior CI produced.',flush=True)
        return
    target_proteins = ([args.protein] if args.output_proteins is None else
                       m.proteins if args.output_proteins==['ALL'] else args.output_proteins)
    for protein in target_proteins:
        if protein not in m.proteins:
            p.error(f'Invalid output protein {protein}. Valid: {m.proteins}')

    # Initialize the actual JAX ODE, not a toy surrogate.
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import diffrax as dfx
    from dynestyx import ODESimulatorConfig
    rhs,fitted,y0,rhs_err=make_source_context(m,args.source)
    ode_config=ODESimulatorConfig(
        solver=dfx.Tsit5(),
        # solver=dfx.Kvaerno4(root_finder=dfx.VeryChord(
        #     rtol=args.rtol,atol=args.atol,kappa=.01),root_find_max_steps=20),
        stepsize_controller=dfx.PIDController(
            rtol=args.rtol,atol=args.atol,
            jump_ts=jnp.asarray(m.ktime[1:-1])),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
        dt0=jnp.asarray(.01),max_steps=args.max_steps)
    posterior,_likelihood_sigmas,inference_info=run_ode_posterior(
        m,args,rhs,fitted,y0,ode_config)
    np.savez_compressed(args.out/'posterior_samples.npz',**posterior)
    times=dense_time_grid(m,args)
    states,selected=simulate_posterior(m,args,rhs,fitted,y0,posterior,ode_config,times)
    values=observable_series(m,states,times,target_proteins)
    del states  # limit RAM before smoothing
    sigmas_by_protein={p:get_assumed_sigmas(m,p,args) for p in target_proteins}
    table,smooth=construct_tables(m,args,times,values,sigmas_by_protein)
    table.to_csv(args.out/'timepoint_ode_ci_and_smoothing.csv',index=False)
    table[['type','protein','psite','time_min','ode_ci_lower_95','ode_median',
           'ode_ci_upper_95','is_measured','observed_fc']].to_csv(
               args.out/'dense_ode_credible_intervals.csv',index=False)
    smooth.to_csv(args.out/'discrete_smoothing_at_observed_times.csv',index=False)
    for protein in target_proteins:
        plot_uncertainty_overlay(table,protein,args.out,args.dpi)
    report={**inference_info, 'source_model':'PhosKinTime distributive RHS',
            'rhs_validation_max_abs_error':rhs_err,
            'inference_target':args.protein,
            'inferred_parameters':['c_k_target','E_i_target'],
            'fixed_parameters':'All other fitted params and all 70 initial conditions',
            'likelihood_channels':'Target gene phosphosites and total protein',
            'RNA':'Posterior forward prediction only; not in inferred likelihood (t=4 reference)',
            'selected_output_proteins':target_proteins,
            'pointwise_interval':'Equal-tail 95% credible interval conditional on two inferred rates',
            'posterior_forward_draws':len(selected),
            'dense_time_points':len(times),
            'discrete_smoother':'dynestyx exact 1D Gaussian RTS smoother of each channel residual',
            'smoother_process_SD_per_observation_step_factor':args.smoothing_process_factor,
            'smoothing_intervals':'95% conditional residual-state intervals, exclude ODE parameter uncertainty',
            'estimated_measurement_SD_by_protein_and_type':sigmas_by_protein,
            'measurement_SD_source':'Explicit --sigma or plug-in RMS of saved fit residuals (not experimental replicates)',
            'time_units':'minutes',
            'table_rows':len(table),
            'note':'Keep ODE credible interval distinct from residual smoothing interval; not a joint posterior.'}
    (args.out/'methods_and_diagnostics.json').write_text(json.dumps(report,indent=2))
    print(f'\nCOMPLETE: {len(table)} timepoint rows in {args.out.resolve()}',flush=True)

if __name__=='__main__':
    main()
