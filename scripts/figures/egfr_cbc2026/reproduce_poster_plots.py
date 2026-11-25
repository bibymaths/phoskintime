#!/usr/bin/env python3
"""Reproduce the latest CBC PhosKinTime figures from numerical result exports.

Usage:
    python scripts/figures/egfr_cbc2026/reproduce_poster_plots.py RESULTS_DIRECTORY --output poster_figures

Requires numpy, pandas, scipy, matplotlib. No PhosKinTime installation required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

BLUE, RUST = "#14667d", "#a34c23"
SITES = ["S_991", "S_1166"]


class PosterPlots:
    def __init__(self, root, output, preview=False):
        self.root, self.output, self.preview = root, output, preview
        output.mkdir(parents=True, exist_ok=True)
        self.inputs, self.figures = {}, []
        self.x = self.read("tables/model_trajectories.csv")
        self.x["residual"] = self.x.fc_pred - self.x.fc_obs
        self.x["e2"] = self.x.residual ** 2
        self.s = self.read("tables/S_rates_picked.csv")
        self.s["site"] = self.s.protein + "_" + self.s.psite
        self.m = self.s.pivot(index="site", columns="time", values="S").sort_index()
        self.w = self.read("tables/network_W_global.csv").pivot(
            index="Site", columns="Kinase", values="Weight").fillna(0)
        u = self.read("tables/network_kinase_inputs.csv").set_index("Unnamed: 0")
        k = self.read("tables/model_parameters_kinases.csv").set_index("Kinase").Activity_Scale_ck
        self.activity = u.mul(k, axis=0).reindex(self.w.columns)
        self.times = np.array([float(c[2:]) for c in u.columns])
        self.contrib = self.w.to_numpy()[:, :, None] * self.activity.to_numpy()[None, :, :]
        np.testing.assert_allclose(self.contrib.sum(axis=1), self.m.reindex(self.w.index))
        if not np.isfinite(self.x[["fc_obs", "fc_pred"]]).all().all():
            raise ValueError("Nonfinite observations or predictions.")
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11,
            "axes.spines.top": False, "axes.spines.right": False,
            "pdf.fonttype": 42, "path.simplify": False,
        })

    def path(self, relative):
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required source export missing: {path}")
        self.inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(self, relative):
        return pd.read_csv(self.path(relative))

    def save(self, fig, name):
        fig.savefig(self.output / f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
        if self.preview:
            fig.savefig(self.output / f"{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        self.figures.append(name + ".pdf")

    @staticmethod
    def display_time(t):
        return np.where(np.asarray(t) == 0, .22, t)

    @staticmethod
    def time_axis(ax, bottom=True):
        ax.set_xscale("log")
        ax.set_xlim(.17, 1120)
        ax.set_xticks([.22, .5, 1, 4, 16, 60, 240, 960],
                      ["0*", "0.5", "1", "4", "16", "60", "240", "960"])
        ax.tick_params(axis="x", labelbottom=bottom)
        ax.minorticks_off()
        ax.axvline(.34, color=".75", ls=":", lw=.7)
        ax.grid(axis="y", alpha=.2)
        if bottom:
            ax.set_xlabel("Time (min, log scale) · 0* marks the separate baseline")

    @staticmethod
    def site_labels(index):
        return [v.replace("_", " ", 1).replace("_", "") for v in index]

    def egfr_timeseries(self):
        for dense, name in [(False, "egfr_fit_log_time"),
                            (True, "egfr_high_res_fit_log_time")]:
            fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True, layout="constrained")
            for j, site in enumerate(SITES):
                q = self.x[(self.x.protein == "EGFR") & (self.x.psite == site)].sort_values("time")
                t, y = q.time.to_numpy(), q.fc_pred.to_numpy()
                ax = axs[j]
                if dense:
                    tt = np.geomspace(t[1], t[-1], 10000)
                    curve = PchipInterpolator(t, y)
                    np.testing.assert_allclose(curve(t), y)
                    ax.plot(tt, curve(tt), color=BLUE, label="Fitted (cubic interpolation)")
                else:
                    ax.plot(t[1:], y[1:], color=BLUE, marker=".", label="Fitted")
                ax.scatter(.22, y[0], color=BLUE, marker="s", s=24)
                ax.scatter(self.display_time(t), q.fc_obs, color=RUST, s=20,
                           label="Observed", zorder=4)
                ax.set_title("EGFR " + site.replace("_", ""), loc="left", weight="bold")
                ax.set_ylabel("Fold change")
                self.time_axis(ax, j == 1)
            axs[0].legend(frameon=False, fontsize=9)
            self.save(fig, name)
        fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True, layout="constrained")
        for j, site in enumerate(SITES):
            q = self.s[(self.s.protein == "EGFR") & (self.s.psite == site)].sort_values("time")
            t, y = q.time.to_numpy(), q.S.to_numpy()
            tt = np.geomspace(t[1], t[-1], 10000)
            curve = PchipInterpolator(t, y)
            np.testing.assert_allclose(curve(t), y)
            axs[j].plot(tt, curve(tt), color=BLUE)
            axs[j].scatter(self.display_time(t), y, s=15, color=BLUE)
            axs[j].set_ylim(0, .53)
            axs[j].set_ylabel(r"Modelled drive (min$^{-1}$)")
            axs[j].set_title("EGFR " + site.replace("_", ""), loc="left", weight="bold")
            self.time_axis(axs[j], j == 1)
        self.save(fig, "egfr_phosphorylation_drive_log_time")

    def goodness_of_fit(self):
        fig, ax = plt.subplots(figsize=(6.3, 6.1), layout="constrained")
        for name, marker, color in [("Phosphorylation", "o", RUST),
                                    ("Protein", "s", BLUE), ("RNA", "^", "#795897")]:
            q = self.x[self.x.type == name]
            ax.scatter(q.fc_obs, q.fc_pred, c=color, marker=marker, s=16, alpha=.55,
                       linewidths=0, label=f"{name} ({len(q)})")
        limit = max(4.35, self.x[["fc_obs", "fc_pred"]].max().max() * 1.05)
        ax.plot([0, limit], [0, limit], "--", color=BLUE, lw=1, zorder=0)
        positions = {"EGFR": (3.75, .65), "EPHA2": (.68, 2.75),
                     "MAP2K2": (2.45, .52), "GRK5": (1.55, 1.39)}
        for gene, pos in positions.items():
            q = self.x[self.x.protein == gene]
            row = q.loc[q.residual.abs().idxmax()]
            ax.annotate(gene, (row.fc_obs, row.fc_pred), pos, ha="center", fontsize=9,
                        arrowprops={"arrowstyle": "->", "lw": .8})
        ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Observed fold change",
               ylabel="Fitted fold change")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(axis="y", alpha=.18)
        self.save(fig, "goodness_of_fit_all_modalities")
        fig, axs = plt.subplots(2, 1, figsize=(6, 7.35), layout="constrained")
        for ax, site in zip(axs, SITES):
            q = self.x[(self.x.protein == "EGFR") & (self.x.psite == site)].sort_values("time")
            low = min(q.fc_obs.min(), q.fc_pred.min())
            high = max(q.fc_obs.max(), q.fc_pred.max())
            pad = .13 * (high - low)
            lim = (low - pad, high + pad)
            ax.plot(lim, lim, "--", color=BLUE, label="Perfect agreement")
            ax.scatter(q.fc_obs, q.fc_pred, color=RUST, s=22)
            for (_, row), offset in zip(q.nlargest(2, "e2").iterrows(), [(-10, 16), (8, -17)]):
                ax.annotate(f"{row.time:g} min", (row.fc_obs, row.fc_pred),
                            xytext=offset, textcoords="offset points", fontsize=9,
                            ha="right" if offset[0] < 0 else "left",
                            arrowprops={"arrowstyle": "->", "lw": .85})
            ax.set(xlim=lim, ylim=lim, ylabel="Fitted fold change")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(axis="y", alpha=.22)
            ax.set_title("EGFR " + site.replace("_", ""), loc="left", weight="bold")
        axs[0].legend(frameon=False, fontsize=9)
        axs[-1].set_xlabel("Observed fold change")
        self.save(fig, "egfr_goodness_of_fit")

    def residuals(self):
        fig, axs = plt.subplots(1, 3, figsize=(13, 8.4), layout="constrained")
        times = sorted(self.x.time.unique())
        limit = self.x.residual.abs().max()
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#d5dadc")
        for kind, ax in zip(["RNA", "Protein", "Phosphorylation"], axs):
            q = self.x[self.x.type == kind].copy()
            q["row"] = q.protein
            if kind == "Phosphorylation":
                q["row"] += " " + q.psite.str.replace("_", "")
            z = q.pivot(index="row", columns="time", values="residual").reindex(columns=times)
            assert z.count().sum() == len(q)
            im = ax.pcolormesh(np.arange(len(times)+1), np.arange(len(z)+1),
                               np.ma.masked_invalid(z), cmap=cmap, norm=norm,
                               edgecolors="white", linewidth=.2)
            ax.invert_yaxis()
            ax.set_yticks(np.arange(len(z))+.5, z.index, fontsize=8)
            ax.set_xticks(np.arange(len(times))+.5, [f"{t:g}" for t in times],
                          rotation=60, ha="right", fontsize=8)
            ax.set_title(kind, loc="left", weight="bold")
            ax.set_xlabel("Sample time (min)")
            ax.tick_params(length=0)
        fig.colorbar(im, ax=axs, shrink=.75, label="Residual (fitted − observed fold change)")
        self.save(fig, "residuals_by_time_and_protein")
        q = self.x[self.x.type == "Phosphorylation"].groupby("time").e2.mean()
        fig, ax = plt.subplots(figsize=(6, 3.65), layout="constrained")
        ax.plot(q.index, np.sqrt(q), marker="o", color=BLUE, ms=4)
        ax.set_xscale("symlog", linthresh=.5)
        ax.set_xticks([0, 1, 8, 60, 960], ["0", "1", "8", "60", "960"])
        ax.set(xlabel="Time (min, symlog axis)", ylabel="RMSE (fold change)", ylim=(0, .75))
        ax.set_title("All 36 fitted phosphosites", loc="left", weight="bold")
        ax.grid(axis="y", alpha=.22)
        self.save(fig, "phosphosite_rmse")

    def multistart(self):
        q = self.read("optimization/multistart_summary.csv")
        selected = q.selected_best.astype(str).str.lower() == "true"
        if selected.sum() != 1:
            raise ValueError("Expected one selected multistart solution.")
        best = q[selected].iloc[0]
        np.testing.assert_allclose(best.final_objective, q.final_objective.min())
        fig, ax = plt.subplots(figsize=(6, 3.7), layout="constrained")
        ax.scatter(q.loc[~selected, "start_id"], q.loc[~selected, "final_objective"], color=BLUE, s=39)
        ax.scatter(best.start_id, best.final_objective, color=RUST, s=75)
        ax.annotate(f"Selected: {best.final_objective:.4f}",
                    (best.start_id, best.final_objective), xytext=(4, 44),
                    textcoords="offset points", ha="center", fontsize=10,
                    arrowprops={"arrowstyle": "->", "lw": .85})
        ax.set_xticks(q.start_id)
        ax.set(xlim=(-.5, 9.5), ylim=(.21, .44),
               xlabel="Start ID", ylabel="Final calibration objective")
        ax.grid(axis="y", alpha=.22)
        self.save(fig, "multistart_objective")

    def drive_heatmap(self):
        m = self.m.loc[self.m.median(axis=1).sort_values(ascending=False).index]
        times = m.columns.to_numpy()
        lp = np.log(times[1:])
        edges = np.exp(np.r_[lp[0]-(lp[1]-lp[0])/2, (lp[:-1]+lp[1:])/2,
                             lp[-1]+(lp[-1]-lp[-2])/2])
        fig, (b, ax) = plt.subplots(1, 2, figsize=(7.1, 8.4), sharey=True,
                                    width_ratios=[.065, 1], layout="constrained")
        norm = PowerNorm(.30, vmin=0, vmax=m.to_numpy().max())
        for panel, xe, values in [(b, [0, 1], m.iloc[:, :1]), (ax, edges, m.iloc[:, 1:])]:
            im = panel.pcolormesh(xe, np.arange(len(m)+1), values,
                                 cmap="viridis", norm=norm, edgecolors="white", linewidth=.22)
            panel.tick_params(length=0)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(times[1:], [f"{t:g}" for t in times[1:]],
                      rotation=55, ha="right", fontsize=8)
        ax.set_xlabel("Sample time (min, log scale)")
        ax.set_title("Network-wide phosphorylation drive", loc="left", weight="bold")
        b.set_xticks([.5], ["0"])
        b.set_xlabel("Baseline", fontsize=8)
        b.set_yticks(np.arange(len(m))+.5, self.site_labels(m.index), fontsize=8)
        b.invert_yaxis()
        for label, site in zip(b.get_yticklabels(), m.index):
            if site in ["EGFR_S_991", "EGFR_S_1166"]:
                label.set_color(RUST)
                label.set_weight("bold")
        fig.colorbar(im, ax=[b, ax], pad=.02, label=r"Modelled drive (min$^{-1}$)", shrink=.8)
        self.save(fig, "network_phosphorylation_drive_log_time")

    def kinase_scatter(self):
        q = self.read("steady_state_summary/steady_state_kinase_drive.csv")
        summed = self.w.sum().reindex(q.kinase).to_numpy() * q.Kt
        np.testing.assert_allclose(summed, q.phospho_drive_sum)
        fig, ax = plt.subplots(figsize=(6.5, 5.2), layout="constrained")
        ax.scatter(q.Kt, q.phospho_drive_sum, c=BLUE, s=32)
        label = pd.concat([q.nlargest(3, "Kt"), q.nlargest(3, "phospho_drive_sum")]).drop_duplicates("kinase")
        for row in label.itertuples():
            offset = {"GRK2": (-10, 16), "MET": (18, 7)}.get(row.kinase, (-5, 8))
            ax.annotate(row.kinase, (row.Kt, row.phospho_drive_sum), xytext=offset,
                        textcoords="offset points", ha="right" if offset[0] < 0 else "left",
                        fontsize=9, arrowprops={"arrowstyle": "-", "lw": .6})
        ax.margins(x=.1, y=.12)
        ax.set_xlabel(r"Steady-state kinase activity, $K_k$")
        ax.set_ylabel(r"Summed site drive, $\sum_{i,s}W_{(i,s),k}K_k$ (min$^{-1}$)")
        ax.grid(axis="y", alpha=.2)
        self.save(fig, "kinase_activity_vs_phosphorylation_drive")

    def contributions(self):
        colors = {"CDK1": "#9a7104", "GSK3B": RUST, "MAPK1": "#7a5799",
                  "MAPK3": BLUE, "MAPK8": "#287962", "GRK2": "#b76a3b"}
        fig, axs = plt.subplots(2, 1, figsize=(6, 7.1), sharex=True, layout="constrained")
        for j, (ax, site) in enumerate(zip(axs, SITES)):
            weights = self.w.loc["EGFR_" + site]
            values = self.activity.mul(weights, axis=0).loc[weights > 0]
            expected = self.m.loc["EGFR_" + site]
            np.testing.assert_allclose(values.sum(), expected)
            for name, v in values.iterrows():
                ax.step(self.times[1:], v.iloc[1:], where="post", lw=1.75,
                        color=colors[name], label=name)
                ax.scatter(self.times[1:], v.iloc[1:], color=colors[name], s=10)
                ax.scatter(.22, v.iloc[0], color=colors[name], marker="s", s=15)
            ax.set_ylim((0, .060) if j == 0 else (0, .54))
            ax.set_ylabel(r"Kinase contribution (min$^{-1}$)")
            ax.set_title("EGFR " + site.replace("_", ""), loc="left", weight="bold")
            ax.legend(frameon=False, ncol=5, fontsize=8, handlelength=1.3, columnspacing=.9)
            self.time_axis(ax, j == 1)
        self.save(fig, "egfr_kinase_contributions")
        ref = self.read("steady_state_summary/steady_state_dominant_kinase_per_site.csv").set_index("site")
        ss = self.read("steady_state_summary/steady_state_kinase_drive.csv").set_index("kinase").Kt
        c = self.w.mul(ss.reindex(self.w.columns), axis=1).reindex(ref.index)
        np.testing.assert_allclose(c.max(axis=1), ref.dominant_contrib)
        np.testing.assert_array_equal(c.idxmax(axis=1), ref.dominant_kinase)
        ref["share"] = 100 * c.max(axis=1) / c.sum(axis=1)
        ref = ref.sort_index()
        fig, (ax, b) = plt.subplots(1, 2, figsize=(6, 8.8), sharey=True,
                                    width_ratios=[4, 1.25], layout="constrained")
        highlight = ref.index.isin(["EGFR_S_991", "EGFR_S_1166"])
        colors = np.where(highlight, RUST, BLUE)
        ax.barh(np.arange(len(ref)), ref.share, color=colors, height=.57)
        ax.set(xlim=(0, 103), ylim=(len(ref)-.5, -.5),
               xlabel="Share of modelled site drive (%)")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks(np.arange(len(ref)), self.site_labels(ref.index), fontsize=8)
        ax.set_title("Phosphosite", loc="left", fontsize=10, weight="bold")
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        b.set_axis_off()
        b.set_xlim(0, 1)
        b.set_title("Dominant kinase", loc="left", fontsize=10, weight="bold")
        for j, (kinase, color) in enumerate(zip(ref.dominant_kinase, colors)):
            b.text(.025, j, kinase, color=color, va="center", fontsize=8)
        for tick, marked in zip(ax.get_yticklabels(), highlight):
            if marked:
                tick.set_color(RUST)
                tick.set_weight("bold")
        self.save(fig, "steady_state_dominant_kinase_map")

    def network_biology(self):
        m = self.m.reindex(self.w.index)
        share = 100 * self.contrib / self.contrib.sum(axis=1, keepdims=True)
        lead = share.argmax(axis=1)
        used = np.unique(lead)
        mapping = {k: j for j, k in enumerate(used)}
        palette = list(plt.get_cmap("tab20").colors) + [
            "#3c3558", "#8c5137", "#007777", "#d0a600", "#333333", "#a62e63", "#224d26"]
        colors = {self.w.columns[k]: palette[j] for k, j in mapping.items()}
        fig = plt.figure(figsize=(11.7, 8.5), layout="constrained")
        gs = fig.add_gridspec(5, 2, width_ratios=[1.48, 1], height_ratios=[1, 1, 1, .46, .32])
        ax = fig.add_subplot(gs[:4, 0])
        ax.pcolormesh(np.arange(15), np.arange(len(m)+1), np.vectorize(mapping.get)(lead),
                      cmap=ListedColormap(palette[:len(used)]), vmin=-.5, vmax=len(used)-.5,
                      edgecolors="white", linewidth=.22)
        ordered = np.sort(share, axis=1)
        row, col = np.where(ordered[:, -1] - ordered[:, -2] < 5)
        ax.scatter(col+.5, row+.5, facecolors="white", edgecolors="#222222", s=16, linewidth=.65)
        self.sample_heat_axis(ax, m.index, "Leading kinase by site and time")
        switches = [j for j, values in enumerate(lead) if len(set(values)) > 1]
        if len(switches) != 3:
            raise ValueError("This poster layout expects the three switches in the original results.")
        for j, idx in enumerate(switches):
            ax = fig.add_subplot(gs[j, 1])
            for k in np.unique(lead[idx]):
                name = self.w.columns[k]
                ax.plot(range(14), share[idx, k], marker="o", ms=3, color=colors[name], label=name)
            ax.set_title(self.site_labels([m.index[idx]])[0], loc="left", weight="bold")
            ax.set_ylabel("Drive share (%)")
            ax.grid(axis="y", alpha=.2)
            ax.set_xticks(range(14), [f"{t:g}" for t in m.columns],
                          rotation=55, ha="right", fontsize=7)
            if j < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Sampled time points (min)", fontsize=9)
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
        ax = fig.add_subplot(gs[3:, 1])
        ax.axis("off")
        handles = [Patch(facecolor=colors[self.w.columns[k]], label=self.w.columns[k]) for k in used]
        handles.append(plt.Line2D([], [], marker="o", mfc="white", mec="#222222",
                                  ls="", label="Lead margin < 5 pp"))
        ax.legend(handles=handles, ncol=4, loc="center", frameon=False, fontsize=8,
                  columnspacing=.9, handlelength=1.1)
        self.save(fig, "time_resolved_kinase_leadership")
        if (m[0] <= 0).any():
            raise ValueError("Baseline drive must be positive for log ratios.")
        order = m.median(axis=1).sort_values(ascending=False).index
        relative = np.log2(m.div(m[0], axis=0)).loc[order]
        fig, (ax, b) = plt.subplots(1, 2, figsize=(8.3, 8.2), sharey=True,
                                    width_ratios=[4, 1.17], layout="constrained")
        limit = np.abs(relative).to_numpy().max()
        im = ax.pcolormesh(np.arange(15), np.arange(len(m)+1), relative,
                           cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                           edgecolors="white", linewidth=.18)
        self.sample_heat_axis(ax, order, "Change in site drive relative to baseline")
        fig.colorbar(im, ax=ax, orientation="horizontal", shrink=.83, pad=.12,
                     label="Log₂(drive / baseline drive)")
        b.scatter(m.median(axis=1).loc[order], np.arange(len(m))+.5, c=BLUE, s=19)
        b.set_xscale("log")
        b.set_title("Median drive", fontsize=10, weight="bold")
        b.set_xlabel(r"min$^{-1}$ (log scale)")
        b.tick_params(axis="y", left=False, labelleft=False)
        b.grid(axis="x", alpha=.2)
        for tick in ax.get_yticklabels():
            if tick.get_text() in ["CLK3 S157", "EPHA2 Y575", "GRK2 S670"]:
                tick.set_color(RUST)
                tick.set_weight("bold")
        self.save(fig, "network_relative_phosphorylation_drive")
        q = self.x[self.x.type == "Phosphorylation"].copy()
        q["window"] = np.where(q.time <= 2, "0–2 min", "After 2 min")
        errors = q.pivot_table(index="protein", columns="window", values="e2", aggfunc="sum", fill_value=0)
        np.testing.assert_allclose(errors.to_numpy().sum(), q.e2.sum())
        errors = errors.loc[errors.sum(axis=1).sort_values(ascending=False).index] / q.e2.sum() * 100
        fig, ax = plt.subplots(figsize=(6.5, 5.6), layout="constrained")
        early, late = errors["0–2 min"], errors["After 2 min"]
        y = np.arange(len(errors))
        ax.barh(y, early, color=RUST, label="0–2 min")
        ax.barh(y, late, left=early, color=BLUE, label="After 2 min")
        ax.invert_yaxis()
        ax.set_yticks(y, errors.index)
        ax.set_xlim(0, 53)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlabel("Share of total phosphosite squared error (%)")
        ax.set_title("Where phosphosite calibration error accumulates", loc="left", weight="bold")
        for j in range(3):
            total = early.iloc[j] + late.iloc[j]
            ax.text(total+.7, j, f"{total:.1f}%", va="center", fontsize=9)
        ax.legend(loc="lower right", frameon=False)
        self.save(fig, "phosphosite_error_by_protein")

    def sample_heat_axis(self, ax, index, title):
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold")
        ax.set_yticks(np.arange(len(index))+.5, self.site_labels(index), fontsize=8)
        ax.set_xticks(np.arange(len(self.m.columns))+.5,
                      [f"{t:g}" for t in self.m.columns], rotation=55, ha="right", fontsize=8)
        ax.set_xlabel("Sampled time points (min)")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def posterior(self):
        folder = "egfr_smoke_posterior/"
        summary = self.read(folder + "egfr_posterior_summary.csv")
        draws = self.read(folder + "egfr_posterior_samples.csv")
        extra = json.loads(self.path(folder + "posterior_chains/chain_000/posterior/posterior_extra_fields.json").read_text())
        divergences = sum(extra["diverging"])
        chains = draws.chain.nunique()
        warnings.warn(f"Posterior uses {chains} chain(s), {len(draws)} draws and {divergences} divergences.")
        fig, ax = plt.subplots(figsize=(7.3, 6.1), layout="constrained")
        for i, row in enumerate(summary.itertuples()):
            lo, median, hi = np.quantile(draws[row.parameter], [.05, .5, .95])
            np.testing.assert_allclose([lo, median, hi], [row.q05, row.median, row.q95],
                                       rtol=1e-6, atol=1e-14)
            ax.hlines(i, lo, hi, color=BLUE, lw=2)
            ax.scatter(median, i, color=BLUE, s=24,
                       label="Posterior median and 90% credible interval" if i == 0 else None)
            ax.scatter(row.fitted_physical, i, color=RUST, marker="D", s=24,
                       label="Calibration estimate" if i == 0 else None)
        labels = summary.parameter.str.replace("[EGFR", "[", regex=False).str.replace(
            "[_", "[", regex=False).str.replace("[]", "", regex=False)
        ax.set_yticks(range(len(summary)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Parameter value (physical scale, log axis)")
        ax.set_title(f"EGFR posterior\n{chains} chain · {len(draws)} draws · {divergences} divergences",
                     loc="left", weight="bold")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, fontsize=8)
        ax.grid(axis="x", alpha=.17)
        self.save(fig, "egfr_posterior")

    def steady_state(self):
        # Reconstruct the distributive RHS from exported physical parameters.
        g = self.read("tables/model_parameters_genes.csv").set_index("Protein_Gene")
        p = self.read("tables/model_parameters_genes_psites.csv").dropna(subset=["psite"]).copy()
        p["site"] = p.Protein_Gene + "_" + p.psite
        index = g.index.get_indexer(p.Protein_Gene)
        initial = self.read("tables/initial_conditions_y0.csv").set_index(["entity", "substate"]).value
        u = self.read("tables/network_kinase_inputs.csv").set_index("Unnamed: 0")
        k = self.read("tables/model_parameters_kinases.csv").set_index("Kinase").Activity_Scale_ck
        activity = u.mul(k, axis=0).to_numpy()
        w = self.w.reindex(index=p.site, columns=u.index).fillna(0).to_numpy()
        tf = self.read("tables/network_tf_mat.csv").pivot(
            index="Target", columns="Source_TF", values="Weight").reindex(
                index=g.index, columns=g.index).fillna(0).to_numpy()
        degree = np.abs(tf).sum(axis=1)
        degree[degree < 1e-12] = 1
        driver = u.index.get_indexer(g.index)
        A, B, C, D, E = [g[c].to_numpy() for c in [
            "Synthesis_A", "mRNA_Degradation_B", "Translation_C",
            "Protein_Degradation_D", "De-Phosphorylation_E"]]
        scale = g.Global_TF_Scale.iloc[0]
        decay = p.Phospho_Degradation_Dp.to_numpy()
        n = len(g)
        y = np.array([initial.loc[(gene, sub)] for sub in ["R", "P0"] for gene in g.index]
                     + [initial.loc[(row.Protein_Gene, row.psite)] for row in p.itertuples()])
        grid = np.r_[0, np.geomspace(.001, 20160, 999)]
        Y = np.zeros((len(y), len(grid)))
        for j, (lo, hi) in enumerate(zip(self.times, np.r_[self.times[1:], 20160])):
            kinase = activity[:, j]
            drive = w @ kinase

            def rhs(t, state):
                R, P, Q = np.split(state, [n, 2*n])
                total = P + np.bincount(index, weights=Q, minlength=n)
                total[driver >= 0] = kinase[driver[driver >= 0]]
                raw = (tf @ total) / degree
                z = raw / (1 + np.abs(raw))
                factor = np.where(z >= 0, 1 + scale*z/(1+z+1e-6), 1/(1+scale*np.abs(z)))
                flux = drive*P[index] - E[index]*Q
                return np.r_[A*factor-B*R, C*R-D*P-np.bincount(index, weights=flux, minlength=n),
                             flux-(D[index]+decay)*Q]

            solution = solve_ivp(rhs, (lo, hi), y, method="BDF",
                                 rtol=1e-8, atol=1e-10, dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            selected = (grid >= lo) & (grid <= hi)
            Y[:, selected] = solution.sol(grid[selected])
            y = solution.y[:, -1]
        idx = g.index.get_loc("EGFR")
        Q = Y[2*n:][index == idx]
        R, P = Y[idx], Y[n+idx] + Q.sum(axis=0)
        ref = self.read("steady_state_summary/steady_state_protein_summary.csv").set_index("protein").loc["EGFR"]
        np.testing.assert_allclose([R[-1], P[-1], Q.sum(axis=0)[-1]],
                                  ref[["R_ss", "P_total_ss", "P_phos_ss"]].astype(float), rtol=4e-5)
        fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True, layout="constrained")
        for ax, title, values in zip(axs, ["EGFR mRNA", "EGFR total protein",
                                          "EGFR phosphorylated states"], [R, P, Q.sum(axis=0)]):
            ax.plot(grid[1:], values[1:], color=BLUE, lw=1.7, label="Total phosphorylated")
            ax.scatter(.00045, values[0], color=BLUE, marker="s", s=15)
            ax.set_title(title, loc="left", weight="bold")
            ax.set_ylabel("Concentration (a.u.)")
            ax.set_xscale("log")
            ax.set_xlim(.00034, 25000)
            ax.axvline(.00067, ls=":", lw=.65, color=".7")
            ax.grid(axis="y", alpha=.2)
            ax.minorticks_off()
        for site, values in zip(p.loc[index == idx, "psite"], Q):
            if site in SITES:
                color = RUST if site == "S_1166" else "#795897"
                axs[-1].plot(grid[1:], values[1:], color=color, lw=1.25, ls="--",
                             label=site.replace("_", ""))
                axs[-1].scatter(.00045, values[0], s=10, color=color)
        axs[-1].legend(frameon=False, fontsize=8, loc="center right")
        axs[-1].set_xticks([.00045, .001, .01, .1, 1, 10, 100, 1000, 10000],
                          ["0*", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1", "10",
                           "$10^2$", "$10^3$", "$10^4$"])
        axs[-1].set_xlabel("Time (min, log scale) · 0* marks the separate baseline")
        self.save(fig, "egfr_steady_state_dynamics")

    def run(self):
        for operation in [self.egfr_timeseries, self.goodness_of_fit, self.residuals,
                          self.multistart, self.drive_heatmap, self.kinase_scatter,
                          self.contributions, self.network_biology, self.posterior, self.steady_state]:
            print(f"Generating {operation.__name__}...", flush=True)
            operation()
        manifest = {"source_directory": str(self.root.resolve()),
                    "source_sha256": self.inputs, "figures": self.figures,
                    "versions": {}}
        import scipy
        for module in [np, pd, matplotlib, scipy]:
            manifest["versions"][module.__name__] = module.__version__
        (self.output / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Created {len(self.figures)} PDFs in {self.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Extracted results_network_distributive directory")
    parser.add_argument("--output", type=Path, default=Path("poster_figures"))
    parser.add_argument("--preview", action="store_true", help="Also export PNG previews")
    args = parser.parse_args()
    PosterPlots(args.results, args.output, args.preview).run()


if __name__ == "__main__":
    main()
