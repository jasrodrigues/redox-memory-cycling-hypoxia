"""Figures 1-6 of the paper, built from results/*.npz (run experiments.py first).

    python figures.py            # writes figures/fig1_scheme.png ... fig6_sensitivity_fast.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import model as mdl
from model import N, H, Q, params, equilibrium
from pdmp import simulate
from protocols import run_schedule, equal_burden

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
RES, OUT = "results", "figures"
CN, CH, CQ = "#4C78A8", "#E45756", "#F2A541"
LAB = ["G (GSH)", "R (ROS)", "A (ATP)", "M (memory)"]
T = 240.0


def load(name):
    return np.load(os.path.join(RES, name + ".npz"), allow_pickle=True)


def fig1():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    a = ax[0]; a.set_xlim(0, 10); a.set_ylim(-0.6, 8); a.set_aspect("equal"); a.axis("off")
    pos = {"N": (5, 6.6), "H": (8.2, 1.8), "Q": (1.8, 1.8)}
    col = {"N": CN, "H": CH, "Q": CQ}
    txt = {"N": "N\nnormoxia", "H": "H\nhypoxia", "Q": "Q\nreoxy-\ngenation"}
    for k, (x, y) in pos.items():
        a.add_patch(Circle((x, y), 1.15, color=col[k], alpha=.9))
        a.text(x, y, txt[k], ha="center", va="center", color="w", fontsize=8.5, weight="bold")

    def arr(p, q, s, off):
        a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color="0.25",
                                    connectionstyle="arc3,rad=-0.15"))
        a.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], s, ha="center", fontsize=8)
    arr((6.0, 5.8), (7.6, 2.9), r"$c\,\lambda_{NH}^{0}$", (0.9, 0.2))
    arr((7.0, 1.4), (3.0, 1.4), r"$c\,\lambda_{HQ}^{0}$" + "\nROS burst, amplitude\n" + r"$1-e^{-d_H/\tau_S}$", (0, -1.55))
    arr((2.3, 2.9), (4.0, 5.8), r"$c\,\lambda_{QN}^{0}$", (-1.0, 0.2))
    a.set_title("(a) Environmental switching $E(t)$", fontsize=9, loc="left")

    b = ax[1]; b.set_xlim(0, 10); b.set_ylim(0, 8); b.axis("off")
    nodes = {"R": (2, 6), "G": (8, 6), "M": (5, 1.6), "A": (8, 1.6)}
    nl = {"R": "R\nROS", "G": "G\nGSH", "M": "M\nredox\nmemory", "A": "A\nATP"}
    for k, (x, y) in nodes.items():
        b.add_patch(FancyBboxPatch((x - 1, y - 0.8), 2, 1.6, boxstyle="round,pad=0.1", fc="0.93", ec="0.3"))
        b.text(x, y, nl[k], ha="center", va="center", fontsize=8.5)

    def e(p, q, c, rad=0.0):
        b.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.3, color=c,
                                    connectionstyle=f"arc3,rad={rad}"))
    red, green = "#B22", "#262"
    e((3.1, 6.4), (6.9, 6.4), red); e((6.9, 5.7), (3.1, 5.7), red)
    b.text(5, 6.65, r"scavenging $-k_{GR}GR$", fontsize=7.5, ha="center", color=red)
    e((2.4, 5.1), (4.3, 2.5), red); b.text(2.2, 3.8, r"Hill $R^n$", fontsize=7.5, color=red)
    e((5.9, 2.4), (7.6, 5.1), red); b.text(5.3, 4.2, r"$s_G/(1+\gamma_M M)$", fontsize=7.5, color=red)
    e((8, 5.1), (8, 2.5), green); b.text(8.2, 3.7, r"$G/(K_G+G)$", fontsize=7.5, color=green)
    e((3.0, 5.2), (7.1, 2.0), red, 0.25); b.text(4.3, 3.15, r"$K_R/(K_R+R)$", fontsize=7.5, color=red, rotation=-28)
    b.text(0.2, 7.6, r"$E(t)$ sets $p_R$ (incl. burst) and $p_A$", fontsize=8)
    b.set_title("(b) Intracellular core", fontsize=9, loc="left")
    fig.savefig(os.path.join(OUT, "fig1_scheme.png")); plt.close(fig)


def fig2():
    P = params()
    runs = [(run_schedule([(N, 96)], P), CN, "-", "normoxia"),
            (run_schedule([(N, 12), (H, 84)], P), CH, "-", "chronic hypoxia"),
            (run_schedule([(N, 12), (H, 36), (Q, 2), (N, 46)], P), CQ, "--", "36 h hypoxia, then reoxygenation")]
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.6))
    for i in range(4):
        for (t, X, _), c, ls, lb in runs:
            ax[i].plot(t, X[i], color=c, ls=ls, label=lb)
        ax[i].axvspan(12, 48, color=CH, alpha=.06); ax[i].set_title(LAB[i], fontsize=9); ax[i].set_xlabel("time (h)")
    ax[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_deterministic.png")); plt.close(fig)


def fig3():
    fig, ax = plt.subplots(2, 3, figsize=(11, 4.8), sharex=True)
    cols = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    for row, gM in enumerate((1.0, 0.0)):
        P = params(gM=gM)
        for j, n in enumerate((1, 4, 12, 24)):
            t, X, _ = run_schedule(equal_burden(n), P)
            for k, var in enumerate((3, 0, 1)):
                ax[row, k].plot(t, X[var], color=cols[j], lw=1.1, label=f"{n}×{12 / n:g} h")
        for k, var in enumerate((3, 0, 1)):
            ax[row, k].set_title(f"{LAB[var]}   ($\\gamma_M$={gM:g})", fontsize=9)
            ax[row, k].axvspan(12, 60, color="0.5", alpha=.06)
    for k in range(3):
        ax[1, k].set_xlabel("time (h)")
    ax[0, 0].legend(fontsize=7, frameon=False, title="hypoxic episodes", title_fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_equal_burden.png")); plt.close(fig)


def fig4():
    d = load("main_sweep"); cs = d["cs"]; P = params()
    fig = plt.figure(figsize=(11, 5.2)); gs = fig.add_gridspec(2, 3)
    for j, c in enumerate((0.2, 1.26, 7.9)):
        o = simulate(c, 6, T=T, P=P, seed=21 + j, record_every=0.1)
        a = fig.add_subplot(gs[0, j])
        a.plot(o["t_rec"], o["rec"][:, 0, :], lw=.8)
        a.axhline(mdl.THRESHOLDS["Gc"], color="k", ls=":", lw=.8); a.set_ylim(0, 1.1)
        a.set_title(f"G(t), c = {c:g}  (mean cycle {7 / c:.1f} h)", fontsize=9); a.set_xlabel("time (h)")
    sel = [(0.5, "#7570b3"), (1.26, "#d95f02"), (3.16, "#1b9e77")]
    a = fig.add_subplot(gs[1, :2]); bins = np.linspace(0, T, 25)
    for c_, col in sel:
        i = np.argmin(abs(cs - c_)); tau = d["tau"][i]; tf = tau[np.isfinite(tau)]
        a.hist(tf, bins=bins, histtype="step", lw=1.5, color=col,
               label=f"c={cs[i]:.2f}: P(τc≤240 h)={len(tf) / len(tau):.2f}")
    a.set_xlabel("collapse time τc (h)"); a.set_ylabel(f"count (of {d['tau'].shape[1]})")
    a.legend(fontsize=7, frameon=False)
    a = fig.add_subplot(gs[1, 2]); tg = np.linspace(0, T, 200)
    for c_, col in sel:
        i = np.argmin(abs(cs - c_)); tau = d["tau"][i]
        a.plot(tg, [(tau <= x).mean() for x in tg], color=col)
    a.set_xlabel("t (h)"); a.set_ylabel("P(τc ≤ t)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_stochastic.png")); plt.close(fig)


def fig5():
    d, rf, h, ctl = load("main_sweep"), load("refine"), load("heat"), load("control")
    from pdmp import wilson
    cs, n = d["cs"], d["tau"].shape[1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    a = ax[0]
    for src, col, lab, ms in ((d, "k", "1000 paths", 3), (rf, "#d95f02", "3000 paths", 3)):
        npth = src["tau"].shape[1]; k = (src["tau"] <= T).sum(1); lo, hi = wilson(k, npth); p = k / npth
        a.errorbar(src["cs"], p, yerr=[p - lo, hi - p], color=col, marker="o", ms=ms, lw=1, ls="-" if col == "k" else "none",
                   label=f"$P_{{col}}$, {lab} (Wilson 95%)")
    a.plot(ctl["cs"], ctl["Pcol"], color="0.6", marker="s", ms=3, ls="--", label="constant burst amplitude")
    a.set_xscale("log"); a.set_xlabel("switching speed c"); a.set_ylabel("collapse probability (240 h)")
    a.legend(fontsize=6.5, frameon=False, loc="center right"); a.set_title("(a) Threshold-based", fontsize=9, loc="left")
    a = ax[1]; P = params(); cstar = mdl.c_star(P)
    curves = [(d["Bload"].mean(1), CQ, "burst load L"), (d["DG"].mean(1), "#1b9e77", r"deficit $\bar D_G$"),
              (d["Mavg"].mean(1), "#7570b3", r"memory $\bar M$"), (1 - d["Gmin"].mean(1), "#E45756", r"$1-G_{min}$")]
    for y, col, lab in curves:
        a.plot(cs, (y - y.min()) / (y.max() - y.min()), "-o", ms=2.5, color=col, label=lab)
    a.axvline(cstar, color="0.4", ls=":"); a.text(cstar * 1.1, 0.02, "$c^*_L$", fontsize=8)
    a.set_xscale("log"); a.set_xlabel("switching speed c"); a.set_ylabel("min-max normalised")
    a.legend(fontsize=6.5, frameon=False, loc="upper left"); a.set_title("(b) Threshold-free metrics", fontsize=9, loc="left")
    for a, key, lab, vmax, tt in ((ax[2], "Pcol", r"$P_{col}$(240 h)", 1, "(c) Collapse"), (ax[3], "DG", r"$\bar D_G$", None, "(d) Deficit")):
        im = a.pcolormesh(np.log10(h["cs"]), h["gs"], h[key].T, cmap="magma", vmin=0, vmax=vmax, shading="nearest")
        a.set_xlabel(r"$\log_{10}$ c"); a.set_ylabel(r"memory feedback $\gamma_M$"); fig.colorbar(im, ax=a, label=lab)
        a.set_title(tt, fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_resonance.png")); plt.close(fig)


def fig6():
    co, g = load("collapse"), load("global")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
    cmap = plt.cm.viridis; lr = np.log10(co["rho"]); nr = (lr - lr.min()) / (lr.max() - lr.min())
    x = co["x"]
    for i in range(len(co["rho"])):
        ax[0].plot(x * co["cstar"][i], co["curves"][i], color=cmap(nr[i]), lw=.6, alpha=.7)
        ax[1].plot(x, co["curves"][i], color=cmap(nr[i]), lw=.6, alpha=.7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("switching speed c"); ax[0].set_ylabel(r"$L/L(c^*_L)$")
    ax[0].set_title("(a) 100 random $(m_H,m_Q,\\tau_S,\\tau_Q)$", fontsize=9, loc="left")
    ax[1].set_xscale("log"); ax[1].axvline(1, color="k", ls=":", lw=.8); ax[1].set_xlabel(r"$\Pi=c/c^*_L$")
    ax[1].set_title("(b) Same curves against $\\Pi$", fontsize=9, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lr.min(), lr.max())); fig.colorbar(sm, ax=ax[1], label=r"$\log_{10}\rho$")
    X = g["X_GRID"]; DG = g["DG"]; Dn = DG / DG.max(1, keepdims=True)
    q = np.percentile(Dn, [10, 50, 90], axis=0)
    ax[2].fill_between(X, q[0], q[2], color="#1b9e77", alpha=.25, label="10-90% of sets")
    ax[2].plot(X, q[1], color="#1b9e77", lw=1.5, label="median"); ax[2].axvline(1, color="k", ls=":", lw=.8)
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"$\Pi=c/c^*_L$"); ax[2].set_ylabel(r"$\bar D_G$ / max")
    ax[2].legend(fontsize=7, frameon=False); ax[2].set_title(f"(c) {len(DG)} admissible parameter sets", fontsize=9, loc="left")
    vals, cnt = np.unique(g["xDG"], return_counts=True); valsM, cntM = np.unique(g["xM"], return_counts=True)
    w = 0.08
    ax[3].bar(np.log10(vals) - w / 2, cnt / len(DG), w, color="#1b9e77", label=r"$\bar D_G$")
    ax[3].bar(np.log10(valsM) + w / 2, cntM / len(DG), w, color="#7570b3", label=r"$\bar M$")
    ax[3].set_xlabel(r"$\log_{10}\Pi$ at the maximum"); ax[3].set_ylabel("fraction of sets"); ax[3].legend(fontsize=7, frameon=False)
    ax[3].set_title("(d) Location of the damage maximum", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_collapse_global.png")); plt.close(fig)


def fig7():
    a_ = load("ablation"); cs = a_["cs"]
    models = [("M0", "0.6", "M0: no burst"), ("M1", "#E45756", "M1: constant burst"),
              ("M1F", "#F2A541", "M1 + feedback"), ("M2", "#4C78A8", "M2: priming, no feedback"), ("M3", "k", "M3: full model")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for key, a, lab in (("Bload", ax[0], "burst load L"), ("DG", ax[1], r"transient deficit $\bar D_G$"),
                        ("RES24", ax[2], "residual deficit after 24 h normoxia"), ("Pcol", ax[3], r"$P_{col}$(240 h)")):
        for m, col, ml in models:
            a.plot(cs, a_[f"{m}_{key}"], "-o", ms=2.5, color=col, label=ml)
        a.set_xscale("log"); a.set_xlabel("switching speed c"); a.set_title(lab, fontsize=9, loc="left")
    ax[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_ablation.png")); plt.close(fig)


def fig8():
    gct, lw = load("gct"), load("laws")
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    cs = gct["cs"]; P = gct["Pcol"]
    cols = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    for g, Gc in enumerate(gct["Gc"]):
        ax[0].plot(cs, P[g, 1], "-o", ms=2.5, color=cols[g], label=f"$G_c$={Gc:.2f}")
    for t, Tt in enumerate(gct["Ts"]):
        ax[1].plot(cs, P[1, t], "-o", ms=2.5, color=cols[t], label=f"T={Tt:.0f} h")
    ax[0].set_title("(a) Threshold, T = 240 h", fontsize=9, loc="left"); ax[1].set_title("(b) Horizon, $G_c$ = 0.25", fontsize=9, loc="left")
    names = ["exponential", "gamma CV 0.1", "gamma CV 0.5", "gamma CV 2", "lognormal CV 1.5"]
    lc = ["k", "#1b9e77", "#7570b3", "#d95f02", "#e7298a"]
    from experiments import LAWS
    cc = np.logspace(-1.5, 1.5, 400)
    for li in range(5):
        D = lw[f"L{li}_DG"]; ax[2].plot(lw["cs"], D / D.max(), "-o", ms=2.5, color=lc[li], label=names[li])
        L = mdl.burst_load_general(cc, params(), law=LAWS[li]); ax[2].axvline(cc[L.argmax()], color=lc[li], ls=":", lw=.8)
        ax[3].plot(lw["cs"], lw[f"L{li}_Pcol"], "-o", ms=2.5, color=lc[li], label=names[li])
    ax[2].set_title(r"(c) $\bar D_G$ / max; dotted: maximiser of L", fontsize=9, loc="left")
    ax[3].set_title(r"(d) $P_{col}$ by sojourn law", fontsize=9, loc="left")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("switching speed c"); a.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig8_robustness.png")); plt.close(fig)


def fig9():
    pr, fs = load("prcc"), load("fast")["table"]
    sym = dict(gM=r"$\gamma_M$", dM=r"$\delta_M$", aM=r"$\alpha_M$", KM=r"$K_M$", BQ=r"$B_Q$",
               tauQ=r"$\tau_Q$", tauS=r"$\tau_S$", pH=r"$p_H$", sG=r"$s_G$", dG=r"$\delta_G$")
    names = [sym[str(x)] for x in pr["names"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    o = np.argsort(-abs(pr["Pcol"])); y = np.arange(len(o))
    for off, key, col, lab in ((-0.27, "Pcol", "k", "$P_{col}$"), (0, "Mavg", "#7570b3", r"$\bar M$"), (0.27, "DG", "#1b9e77", r"$\bar D_G$")):
        v = pr[key][o]
        ax[0].barh(y + off, v, 0.25, color=col, label=lab,
                   xerr=[v - pr[key + "_lo"][o], pr[key + "_hi"][o] - v], error_kw=dict(lw=.6, capsize=1.5))
    ax[0].set_yticks(y); ax[0].set_yticklabels([names[i] for i in o]); ax[0].invert_yaxis()
    ax[0].axvline(0, color="k", lw=.6); ax[0].set_xlabel("PRCC at c = 1.26 (95% bootstrap)"); ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title("(a) Local sensitivity", fontsize=9, loc="left")
    for k, col in enumerate(["#4C78A8", "#E45756", "#54A24B", "#7570b3"]):
        ax[1].loglog(fs[:, 0], fs[:, 1 + k], "-o", ms=3, color=col, label=LAB[k])
    ax[1].loglog(fs[:, 0], 0.5 * (fs[:, 0] / 3) ** -1.0, "k:", lw=.8, label="slope -1")
    ax[1].set_xlabel("switching speed c"); ax[1].set_ylabel(r"E sup$_{t\leq 48h}$|X$^c$ - x̄|"); ax[1].legend(fontsize=7, frameon=False)
    ax[1].set_title("(b) Convergence to the averaged system", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig9_sensitivity_fast.png")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f(); print("wrote", f.__name__)
