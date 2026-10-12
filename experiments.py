"""All numerical experiments of the paper.

Usage:
    python experiments.py                  # every experiment, full size (about 60 min on one core)
    python experiments.py --quick          # reduced sizes, for checking
    python experiments.py main gct laws    # only the named experiments
    python experiments.py global --chunk 3 # one chunk of the global robustness study
    python experiments.py global_merge     # combine the chunks

Results go to results/*.npz; summaries are printed (results_full.log keeps the paper's run).
"""
import argparse, glob, os, time
import numpy as np
from scipy.stats import rankdata
import model as mdl
from model import N, H, params, equilibrium, equilibria, loop_gain, jacobian
from pdmp import simulate, sweep_c, wilson, GC_LIST
from protocols import run_schedule, equal_burden, hypoxic_exposure

OUT = "results"
T = 240.0
C_MAIN = np.logspace(-1.5, 1.5, 16)
C_13 = np.logspace(-1.5, 1.5, 13)


def save(name, **arrays):
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, name + ".npz"), **arrays)


def summary_row(r, i):
    """One line of threshold-free and threshold-based metrics for row i of a sweep."""
    n = r["tau"].shape[1]
    k = int(np.sum(r["tau"][i] <= T)); lo, hi = wilson(k, n)
    return (f"Pcol={k / n:.3f} [{lo:.3f},{hi:.3f}]  DG={r['DG'][i].mean():.3f}  "
            f"Mavg={r['Mavg'][i].mean():.3f}  Gmin={r['Gmin'][i].mean():.3f}  Gavg={r['Gavg'][i].mean():.3f}")


def interior_peak(y, ratio=1.25):
    """True if the maximum of y is at an interior grid point and exceeds both ends by `ratio`."""
    j = int(np.argmax(y))
    return 0 < j < len(y) - 1 and y[j] >= ratio * max(y[0], y[-1]), j


# ======================================================== Section 4 (analytic)
def exp_equilibria(scale=1.0):
    P = params()
    print("Equilibria and loop gain (Proposition 2, Theorem 1)")
    rows = []
    for name, E in (("normoxia", N), ("chronic hypoxia", H)):
        X = equilibrium(P, E)
        ev = np.linalg.eigvals(jacobian(X, E, P))
        print(f"  {name:16s} X* = {np.round(X, 3)}  Lambda = {loop_gain(X, P):.3f}  eig = {np.round(np.sort(ev.real), 3)}")
        rows.append(np.r_[X, loop_gain(X, P)])
    Xa = equilibrium(mdl.averaged_params(P), N)
    print(f"  averaged system  X* = {np.round(Xa, 3)}")
    counts = [len(equilibria(params(gM=g, pN=p), N, grid=20001)) for g in (1, 3, 5, 8) for p in np.linspace(0.1, 1.2, 45)]
    print("  uniqueness scan, max number of equilibria =", max(counts))
    save("equilibria", table=np.array(rows), averaged=Xa)


def exp_symbolic(scale=1.0):
    """Symbolic check of the characteristic polynomial used in Theorem 1 (Appendix B)."""
    import sympy as sp
    sG, dG, k, dR, gM, aM, KM, n, dM, G, R, M, p, lam = sp.symbols(
        "s_G delta_G k_GR delta_R gamma_M alpha_M K_M n delta_M G R M p lambda", positive=True)
    f = sp.Matrix([sG / (1 + gM * M) - dG * G - k * G * R,
                   p - k * G * R - dR * R,
                   aM * R ** n / (KM ** n + R ** n) - dM * M])
    J = f.jacobian([G, R, M])
    sigma = sG * gM / (1 + gM * M) ** 2
    eta = sp.diff(aM * R ** n / (KM ** n + R ** n), R)
    D = dG * k * G + dG * dR + k * R * dR
    a1 = -J.trace()
    a2 = sum(J.extract([i, j], [i, j]).det() for i, j in ((0, 1), (0, 2), (1, 2)))
    a3 = -J.det()
    print("Characteristic polynomial lambda^3 + a1 lambda^2 + a2 lambda + a3 (Appendix B)")
    print("  a1 - (delta_G + k R + k G + delta_R + delta_M) =",
          sp.simplify(a1 - (dG + k * R + k * G + dR + dM)))
    print("  a2 - [D + delta_M (delta_G + k R + k G + delta_R)] =",
          sp.simplify(a2 - (D + dM * (dG + k * R + k * G + dR))))
    print("  a3 - [delta_M D - sigma k R eta] =", sp.simplify(a3 - (dM * D - sigma * k * R * eta)))
    print("  a1 a2 - a3 - [a1 a2 - delta_M D] - sigma k R eta =",
          sp.simplify((a1 * a2 - a3) - (a1 * a2 - dM * D) - sigma * k * R * eta))


def exp_protocols(scale=1.0):
    print("Equal hypoxic burden protocols (Section 5.2)")
    out = []
    for gM in (1.0, 0.0):
        P = params(gM=gM)
        for nep in (1, 4, 12, 24):
            t, X, E = run_schedule(equal_burden(nep), P)
            DG = np.trapezoid(np.maximum(1 - X[0] / X[0, 0], 0), t) / t[-1]
            row = [gM, nep, hypoxic_exposure(t, E), X[3, -1], X[3].max(), X[0].min(), X[0, -1], X[1].max(), DG]
            out.append(row)
            print(f"  gM={gM:g} {nep:2d}x{12 / nep:<4g}h H={row[2]:.1f}h M(96)={row[3]:.3f} maxM={row[4]:.2f} "
                  f"minG={row[5]:.2f} G(96)={row[6]:.2f} maxR={row[7]:.2f} DG={DG:.3f}")
    save("protocols", table=np.array(out))


def exp_collapse(scale=1.0):
    """Parameter-set collapse of the burst load in Pi = c / c*_L (Section 4, Figure 6a)."""
    rng = np.random.default_rng(42)
    nset = 100
    mH, mQ = 10 ** rng.uniform(-0.5, 1.0, nset), 10 ** rng.uniform(-0.7, 0.7, nset)
    tS, tQ = 10 ** rng.uniform(-0.7, 0.7, nset), 10 ** rng.uniform(-1.0, 0.3, nset)
    x = np.logspace(-2, 2, 801)
    curves, cstar, rho, xmax = [], [], [], []
    for i in range(nset):
        P = params(tauS=tS[i], tauQ=tQ[i]); means = np.array([4.0, mH[i], mQ[i]])
        cs = mdl.c_star(P, means)
        L = mdl.burst_load(x * cs, P, means)
        curves.append(L / mdl.burst_load(cs, P, means)); cstar.append(cs)
        rho.append((tQ[i] / mQ[i]) / (tS[i] / mH[i])); xmax.append(x[np.argmax(L)])
        assert np.allclose(curves[-1], mdl.load_shape(x, rho[-1]))
    xmax = np.array(xmax)
    print(f"Parameter-set collapse: {nset} sets, c*_L from {min(cstar):.2f} to {max(cstar):.1f};"
          f" maximiser of L in Pi: min {xmax.min():.4f}, max {xmax.max():.4f}; rho from {min(rho):.3f} to {max(rho):.1f}")
    save("collapse", x=x, curves=np.array(curves), cstar=np.array(cstar), rho=np.array(rho),
         mH=mH, mQ=mQ, tS=tS, tQ=tQ)


def exp_laws_analytic(scale=1.0):
    P = params(); cc = np.logspace(-2, 2, 4001)
    print("Burst load for non-exponential sojourns (Proposition 3')")
    for law in LAWS:
        L = mdl.burst_load_general(cc, P, law=law)
        print(f"  {law_name(law):22s} maximiser c = {cc[L.argmax()]:.2f}, L(c_min)/Lmax = {L[0] / L.max():.3f},"
              f" L(c_max)/Lmax = {L[-1] / L.max():.3f}")


# ======================================================== Section 5
def exp_main(scale=1.0):
    n = max(50, int(1000 * scale))
    print(f"Main sweep (Section 5.3): {len(C_MAIN)} values of c x {n} paths")
    r = sweep_c(C_MAIN, n, params(), T=T, seed=7)
    P = params()
    for i, c in enumerate(C_MAIN):
        tf = r["tau"][i][np.isfinite(r["tau"][i])]
        med = np.median(tf) if len(tf) > 10 else np.nan
        print(f"  c={c:7.3f} {summary_row(r, i)}  causes={np.bincount(r['cause'][i], minlength=4)[1:]}"
              f" med_tau={med:5.1f} load={r['Bload'][i].mean():.4f} (Eq.10 {mdl.burst_load(c, P):.4f})"
              f" Hfrac={r['Hfrac'][i].mean():.3f} nHQ={r['nHQ'][i].mean():.1f}")
    for key in ("DG", "Mavg", "Gavg", "Gmin", "RES24"):
        m = r[key].mean(1); j = np.argmax(m) if key in ("DG", "Mavg", "RES24") else np.argmin(m)
        print(f"  extremum of mean {key} at c = {C_MAIN[j]:.2f}")
    save("main_sweep", **r)


def exp_refine(scale=1.0):
    n = max(200, int(3000 * scale))
    cs = np.array([0.25, 0.32, 0.40, 0.50, 0.63, 0.79, 1.0, 1.26, 1.58, 2.0, 2.5, 3.16, 3.98, 5.01])
    print(f"Refined sweep near the window (Section 5.3): {len(cs)} values x {n} paths")
    r = sweep_c(cs, n, params(), T=T, seed=17)
    for i, c in enumerate(cs):
        print(f"  c={c:5.2f} {summary_row(r, i)}")
    save("refine", **{k: r[k] for k in ("cs", "Pcol", "tau", "DG", "Mavg", "Gmin")})


def exp_control(scale=1.0):
    n = max(50, int(300 * scale))
    print(f"Constant burst amplitude control: {len(C_13)} x {n}")
    r = sweep_c(C_13, n, params(tauS=1e-9), T=T, seed=12)
    print("  Pcol:", np.round(r["Pcol"], 2))
    save("control", **r)


def exp_ablation(scale=1.0):
    """Formal ablation M0-M3 (Section 5.6)."""
    n = max(50, int(300 * scale))
    models = {"M0": params(BQ=0.0, gM=0.0), "M1": params(tauS=1e-9, gM=0.0),
              "M1F": params(tauS=1e-9), "M2": params(gM=0.0), "M3": params()}
    print(f"Ablation (Section 5.6): {len(models)} models x {len(C_13)} values of c x {n} paths")
    res = {"cs": C_13}
    for name, P in models.items():
        r = sweep_c(C_13, n, P, T=T, seed=31)
        for key in ("Bload", "Mavg", "DG", "Gmin", "RES24"):
            res[f"{name}_{key}"] = r[key].mean(1)
        res[f"{name}_Pcol"] = r["Pcol"]
        R24 = res[f"{name}_RES24"]
        L, D, Mv = res[f"{name}_Bload"], res[f"{name}_DG"], res[f"{name}_Mavg"]
        print(f"  {name:3s} load interior peak={interior_peak(L)[0] if L.max() > 0 else False}"
              f"  DG interior peak={interior_peak(D)[0]} (max {D.max():.3f} at c={C_13[D.argmax()]:.2f}, ends {D[0]:.3f}/{D[-1]:.3f})"
              f"  Mavg peak at c={C_13[Mv.argmax()]:.2f}  residual deficit after 24 h max {R24.max():.3f} at c={C_13[R24.argmax()]:.2f}"
              f" (ends {R24[0]:.3f}/{R24[-1]:.3f})  max Pcol={r['Pcol'].max():.2f}")
    save("ablation", **res)


def exp_heat(scale=1.0):
    n = max(30, int(250 * scale))
    gs = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    print(f"(c, gamma_M) map: {len(C_13)} x {len(gs)} x {n}")
    C, Gm = np.meshgrid(C_13, gs, indexing="ij")
    P = params(); P["gM"] = np.repeat(Gm.ravel(), n)
    o = simulate(np.repeat(C.ravel(), n), C.size * n, T=T, P=P, seed=11, X0=equilibrium(params(), N))
    shape = (len(C_13), len(gs), n)
    Pc = (o["tau"] <= T).reshape(shape).mean(2); DG = o["DG"].reshape(shape).mean(2)
    print("  Pcol, rows gamma_M, columns c:"); print(np.round(Pc.T, 2))
    print("  DG, rows gamma_M:"); print(np.round(DG.T, 2))
    save("heat", cs=C_13, gs=gs, Pcol=Pc, DG=DG, Mavg=o["Mavg"].reshape(shape).mean(2))


def exp_scaling(scale=1.0):
    n = max(40, int(200 * scale))
    cs = np.logspace(-1, 1.5, 11)
    print(f"Peak location checks (Section 5.4): {len(cs)} x {n}")
    res = {}
    for label, P in [("f=0.5", params(tauS=0.5, tauQ=0.25)), ("f=1", params()), ("f=2", params(tauS=2.0, tauQ=1.0)),
                     ("tauM=100h", params(dM=0.01, aM=0.05)), ("tauM=20h", params(dM=0.05, aM=0.25))]:
        r = sweep_c(cs, n, P, T=T, seed=1)
        m, d = r["Mavg"].mean(1), r["DG"].mean(1)
        print(f"  {label:10s} c*_L={mdl.c_star(P):.2f}  argmax M at c={cs[m.argmax()]:.2f}  argmax DG at c={cs[d.argmax()]:.2f}"
              f"  argmax Pcol at c={cs[r['Pcol'].argmax()]:.2f}")
        res[label + "_M"] = m; res[label + "_DG"] = d; res[label + "_P"] = r["Pcol"]
    save("scaling", cs=cs, **res)


# ======================================================== Section 6
def exp_gct(scale=1.0):
    """Robustness to the collapse criterion: G_c in GC_LIST, T in {120, 240, 360, 480} h."""
    n = max(60, int(600 * scale)); Tlong = 480.0
    print(f"Collapse-criterion robustness (Section 6.3): {len(C_MAIN)} x {n} paths, horizon {Tlong} h")
    o = simulate(np.repeat(C_MAIN, n), len(C_MAIN) * n, T=Tlong, P=params(), seed=23)
    tg = o["tau_gc"].reshape(len(GC_LIST), len(C_MAIN), n)
    Ts = np.array([120.0, 240.0, 360.0, 480.0])
    P = np.array([[(tg[g] <= Tt).mean(1) for Tt in Ts] for g in range(len(GC_LIST))])   # (Gc, T, c)
    for g, Gc in enumerate(GC_LIST):
        for t, Tt in enumerate(Ts):
            p = P[g, t]
            peak = interior_peak(p + 1e-12, ratio=1.0)
            print(f"  Gc={Gc:.2f} T={Tt:3.0f}  max Pcol={p.max():.2f} at c={C_MAIN[p.argmax()]:.2f}"
                  f"  window (Pcol>=0.05): {C_MAIN[p >= 0.05].min() if (p >= 0.05).any() else np.nan:.2f}"
                  f"-{C_MAIN[p >= 0.05].max() if (p >= 0.05).any() else np.nan:.2f}  zero at both ends: {p[0] < 0.02 and p[-1] < 0.02}")
    save("gct", cs=C_MAIN, Gc=np.array(GC_LIST), Ts=Ts, Pcol=P)


LAWS = [("exp",), ("gamma", 0.1), ("gamma", 0.5), ("gamma", 2.0), ("lognormal", 1.5)]


def law_name(law):
    return {"exp": "exponential (CV 1)", "gamma": f"gamma CV {law[-1]}", "lognormal": f"lognormal CV {law[-1]}"}[law[0]] \
        if law[0] != "gamma" or law[1] != 0.1 else "near-periodic (gamma CV 0.1)"


def exp_laws(scale=1.0):
    n = max(50, int(300 * scale))
    print(f"Non-exponential sojourns (Section 6.4): {len(LAWS)} laws x {len(C_13)} x {n}")
    P = params(); res = {"cs": C_13}
    for li, law in enumerate(LAWS):
        r = sweep_c(C_13, n, P, T=T, seed=41, law=law)
        Lsim, Lth = r["Bload"].mean(1), mdl.burst_load_general(C_13, P, law=law)
        D, Mv = r["DG"].mean(1), r["Mavg"].mean(1)
        print(f"  {law_name(law):28s} Hfrac={r['Hfrac'][3:].mean():.3f}  load sim/theory (c>=0.2) "
              f"{np.round((Lsim / Lth)[C_13 >= 0.2].min(), 3)}-{np.round((Lsim / Lth)[C_13 >= 0.2].max(), 3)}"
              f"  DG peak c={C_13[D.argmax()]:.2f} interior={interior_peak(D)[0]}  M peak c={C_13[Mv.argmax()]:.2f}"
              f"  max Pcol={r['Pcol'].max():.2f} at c={C_13[r['Pcol'].argmax()]:.2f}  ends {r['Pcol'][0]:.2f}/{r['Pcol'][-1]:.2f}")
        for key in ("Bload", "DG", "Mavg", "Gmin"):
            res[f"L{li}_{key}"] = r[key].mean(1)
        res[f"L{li}_Pcol"] = r["Pcol"]; res[f"L{li}_Lth"] = Lth
    save("laws", **res)


VARY = ["sG", "dG", "kGR", "pN", "pH", "dR", "BQ", "tauQ", "tauS", "pAN", "pAH", "KG", "KR", "dA",
        "aM", "KM", "dM", "gM", "mN", "mH", "mQ"]
X_GRID = np.logspace(-1.5, 1.0, 11)        # c / c*_L


def admissible_sets(nsets, seed=2024):
    """Random parameter sets (factor 0.5-2, log-uniform; n in {2..6}) that satisfy the
    qualitative constraints used to build the baseline (Appendix A)."""
    rng = np.random.default_rng(seed); sets = []; tried = 0
    base = dict(params()); base.update(mN=4.0, mH=2.0, mQ=1.0)
    while len(sets) < nsets:
        tried += 1
        f = 2.0 ** rng.uniform(-1, 1, len(VARY))
        P = dict(base)
        for i, k in enumerate(VARY):
            P[k] = base[k] * f[i]
        P["n"] = int(rng.integers(2, 7))
        try:
            XN, XH = equilibrium(P, N), equilibrium(P, H)
        except RuntimeError:
            continue
        ok = (loop_gain(XN, P) < 1 and loop_gain(XH, P) < 1
              and XN[0] > 0.5 and XH[0] > 0.35 and XH[0] > 0.4 * XN[0]
              and XN[2] > 0.4 and XH[2] > 0.25 and XN[1] < 1.0 and XH[1] < 1.0
              and XN[3] < 0.1 * P["aM"] / P["dM"]
              and 1 / P["dM"] > 5 * P["tauQ"])
        if ok:
            sets.append((P, XN, XH))
    return sets, tried


def exp_global(scale=1.0, chunk=None, nchunks=6, per_chunk=100, npath=12):
    """Global robustness of the vulnerability window (Section 6.2)."""
    per_chunk = max(10, int(per_chunk * scale))
    sets, tried = admissible_sets(nchunks * per_chunk)
    chunks = range(nchunks) if chunk is None else [chunk]
    for ch in chunks:
        sub = sets[ch * per_chunk:(ch + 1) * per_chunk]
        t0 = time.time()
        nx = len(X_GRID); L = len(sub) * nx * npath
        P = {k: np.zeros(L) for k in sub[0][0] if k not in ("mN", "mH", "mQ")}
        means = np.zeros((L, 3)); c = np.zeros(L); X0 = np.zeros((4, L))
        for s, (Ps, XN, XH) in enumerate(sub):
            sl = slice(s * nx * npath, (s + 1) * nx * npath)
            for k in P:
                P[k][sl] = Ps[k]
            m = np.array([Ps["mN"], Ps["mH"], Ps["mQ"]]); means[sl] = m
            c[sl] = np.repeat(X_GRID * mdl.c_star(Ps, m), npath); X0[:, sl] = XN[:, None]
        o = simulate(c, L, T=T, P=P, seed=100 + ch, X0=X0, means=means, dt_factor=0.25)
        shape = (len(sub), nx, npath)
        save(f"global_{ch}", DG=o["DG"].reshape(shape).mean(2), Mavg=o["Mavg"].reshape(shape).mean(2),
             Gmin=(o["Gmin"] / X0[0]).reshape(shape).mean(2), Pcol=(o["tau"] <= T).reshape(shape).mean(2),
             Bload=o["Bload"].reshape(shape).mean(2),
             params=np.array([[Ps[k] for k in VARY + ["n"]] for Ps, _, _ in sub]),
             XH=np.array([XH for _, _, XH in sub]), dt=o["dt"])
        print(f"  global chunk {ch}: {len(sub)} sets, dt={o['dt']:.4f}, {time.time() - t0:.0f} s"
              f" (acceptance {len(sets)}/{tried})")


def exp_global_merge(scale=1.0):
    files = sorted(glob.glob(os.path.join(OUT, "global_[0-9]*.npz")))
    d = [np.load(f) for f in files]
    cat = {k: np.concatenate([x[k] for x in d]) for k in ("DG", "Mavg", "Gmin", "Pcol", "Bload", "params", "XH")}
    ns = len(cat["DG"])
    win_DG = np.array([interior_peak(y)[0] for y in cat["DG"]])
    win_M = np.array([interior_peak(y)[0] for y in cat["Mavg"]])
    win_G = np.array([interior_peak(1 - y)[0] for y in cat["Gmin"]])
    xDG = X_GRID[cat["DG"].argmax(1)]; xM = X_GRID[cat["Mavg"].argmax(1)]
    has_col = cat["Pcol"].max(1) >= 0.1
    win_P = has_col & (cat["Pcol"][:, 0] <= 0.5 * cat["Pcol"].max(1)) & (cat["Pcol"][:, -1] <= 0.5 * cat["Pcol"].max(1))
    print(f"Global robustness (Section 6.2): {ns} admissible parameter sets, c / c*_L in "
          f"[{X_GRID[0]:.3f}, {X_GRID[-1]:.0f}]")
    print(f"  interior peak (>=1.25x both ends): DG {win_DG.mean():.1%}, mean M {win_M.mean():.1%},"
          f" 1-Gmin {win_G.mean():.1%}")
    print(f"  location of DG peak in c/c*_L: median {np.median(xDG[win_DG]):.2f}, IQR "
          f"{np.percentile(xDG[win_DG], 25):.2f}-{np.percentile(xDG[win_DG], 75):.2f}; "
          f"M peak median {np.median(xM[win_M]):.2f}")
    print(f"  sets with any collapse (max Pcol>=0.1): {has_col.mean():.1%}; of these with an interior collapse window:"
          f" {win_P[has_col].mean():.1%}")
    names = VARY + ["n"]
    lp = np.log(cat["params"])
    print("  parameters that differ between sets without and with a DG window (median ratio no/yes):")
    diffs = []
    for j, nm in enumerate(names):
        if (~win_DG).sum() >= 3:
            r = np.exp(np.median(lp[~win_DG, j]) - np.median(lp[win_DG, j]))
            diffs.append((abs(np.log(r)), nm, r))
    for _, nm, r in sorted(diffs, reverse=True)[:6]:
        print(f"    {nm:5s} {r:.2f}")
    # PRCC of peak height and location, with bootstrap
    peak = cat["DG"].max(1); rng = np.random.default_rng(0)
    pr_h, pr_x = prcc(lp, peak), prcc(lp, np.log(xDG))
    boot = np.array([prcc(lp[i], peak[i]) for i in [rng.integers(0, ns, ns) for _ in range(200)]])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print("  PRCC with DG peak height [95% bootstrap] and with log peak location:")
    for j in np.argsort(-abs(pr_h)):
        print(f"    {names[j]:5s} {pr_h[j]:+.2f} [{lo[j]:+.2f},{hi[j]:+.2f}]   loc {pr_x[j]:+.2f}")
    save("global", X_GRID=X_GRID, names=np.array(names), win_DG=win_DG, win_M=win_M, win_G=win_G, win_P=win_P,
         has_col=has_col, xDG=xDG, xM=xM, prcc_h=pr_h, prcc_lo=lo, prcc_hi=hi, prcc_x=pr_x, **cat)


def prcc(Xm, y):
    R = np.column_stack([rankdata(col) for col in Xm.T]); ry = rankdata(y); out = []
    for j in range(R.shape[1]):
        Z = np.column_stack([np.ones(len(ry)), np.delete(R, j, 1)])
        rx = R[:, j] - Z @ np.linalg.lstsq(Z, R[:, j], rcond=None)[0]
        rr = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
        out.append(np.corrcoef(rx, rr)[0, 1])
    return np.array(out)


def exp_prcc(scale=1.0):
    names = ["gM", "dM", "aM", "KM", "BQ", "tauQ", "tauS", "pH", "sG", "dG"]
    ns, npath = max(60, int(300 * scale)), 80
    print(f"Local PRCC at c = 1.26 (Section 6.1): {ns} LHS samples x {npath} paths, 500 bootstrap resamples")
    rng = np.random.default_rng(5)
    U = (np.argsort(rng.random((ns, len(names))), axis=0) + rng.random((ns, len(names)))) / ns
    F = 2.0 ** (2 * U - 1)
    P = params()
    for i, nm in enumerate(names):
        P[nm] = np.repeat(params()[nm] * F[:, i], npath)
    o = simulate(np.full(ns * npath, 1.26), ns * npath, T=T, P=P, seed=9, X0=equilibrium(params(), N))
    out = {}
    for key, y in (("Pcol", (o["tau"] <= T).reshape(ns, npath).mean(1)),
                   ("Mavg", o["Mavg"].reshape(ns, npath).mean(1)), ("DG", o["DG"].reshape(ns, npath).mean(1))):
        pr = prcc(np.log(F), y)
        boot = np.array([prcc(np.log(F)[i], y[i]) for i in [rng.integers(0, ns, ns) for _ in range(500)]])
        out[key], out[key + "_lo"], out[key + "_hi"] = pr, *np.percentile(boot, [2.5, 97.5], axis=0)
        out["mean_" + key] = y.mean()
    for j in np.argsort(-abs(out["Pcol"])):
        print(f"  {names[j]:5s} Pcol {out['Pcol'][j]:+.2f} [{out['Pcol_lo'][j]:+.2f},{out['Pcol_hi'][j]:+.2f}]"
              f"  M {out['Mavg'][j]:+.2f} [{out['Mavg_lo'][j]:+.2f},{out['Mavg_hi'][j]:+.2f}]"
              f"  DG {out['DG'][j]:+.2f} [{out['DG_lo'][j]:+.2f},{out['DG_hi'][j]:+.2f}]")
    print(f"  mean Pcol over the sample = {out['mean_Pcol']:.2f}")
    save("prcc", names=np.array(names), **out)


def exp_fast(scale=1.0):
    n = max(30, int(100 * scale))
    P = params(); X0 = equilibrium(P, N); Tf = 48.0
    t_ref, x_ref = mdl.integrate_fixed(mdl.averaged_params(P), N, X0, Tf, dt=0.002)
    print(f"Fast-switching convergence (Section 6.5): {n} paths, horizon {Tf} h")
    rows = []
    for c in (3, 10, 30, 100):
        o = simulate(c, n, T=Tf, P=P, seed=3, X0=X0, dt=min(0.002, 0.05 / c), record_every=0.1)
        ref = np.array([np.interp(o["t_rec"], t_ref, x_ref[k]) for k in range(4)])
        err = np.abs(o["rec"] - ref.T[:, :, None]).max(axis=0).mean(axis=1)
        rows.append(np.r_[c, err])
        print(f"  c={c:4d}  E sup|X - xbar| (G, R, A, M) = {np.round(err, 4)}")
    save("fast", table=np.array(rows))


def exp_dt(scale=1.0):
    n = max(200, int(1000 * scale))
    cs = [0.32, 1.26, 7.9, 20.0]
    print(f"Step-size convergence in the slow, vulnerable and fast regimes (Section 6.6): {n} paths per point")
    rows = []
    for dt in (0.02, 0.01, 0.005, 0.0025):
        r = sweep_c(cs, n, params(), T=T, seed=55, dt=dt)
        for i, c in enumerate(cs):
            rows.append([dt, c, r["Pcol"][i], r["DG"][i].mean(), r["Mavg"][i].mean()])
            print(f"  dt={dt:<6g} c={c:5.2f}  {summary_row(r, i)}")
    save("dt", table=np.array(rows))


EXPERIMENTS = dict(equilibria=exp_equilibria, symbolic=exp_symbolic, protocols=exp_protocols, collapse=exp_collapse,
                   laws_analytic=exp_laws_analytic, main=exp_main, refine=exp_refine, control=exp_control,
                   ablation=exp_ablation, heat=exp_heat, scaling=exp_scaling, gct=exp_gct, laws=exp_laws,
                   global_run=exp_global, global_merge=exp_global_merge, prcc=exp_prcc, fast=exp_fast, dt=exp_dt)

# ------------------------------------------------- validation of the global study
N_VALID, VALID_PATHS, VALID_CHUNK = 50, 200, 10


def _run_sets(sub, npath, seed, gM_override=None, dt_factor=0.25):
    """Simulate a list of (P, XN, XH) sets on X_GRID; returns per-set means (and RES24)."""
    from pdmp import residual_deficit
    nx = len(X_GRID); L = len(sub) * nx * npath
    P = {k: np.zeros(L) for k in sub[0][0] if k not in ("mN", "mH", "mQ")}
    means = np.zeros((L, 3)); c = np.zeros(L); X0 = np.zeros((4, L))
    for s, (Ps, XN, XH) in enumerate(sub):
        sl = slice(s * nx * npath, (s + 1) * nx * npath)
        for k in P:
            P[k][sl] = Ps[k]
        m = np.array([Ps["mN"], Ps["mH"], Ps["mQ"]]); means[sl] = m
        c[sl] = np.repeat(X_GRID * mdl.c_star(Ps, m), npath); X0[:, sl] = XN[:, None]
    if gM_override is not None:
        P["gM"] = np.full(L, gM_override)
        # the normoxic equilibrium hardly depends on gM (M_N* is tiny); recompute it exactly
        for s, (Ps, _, _) in enumerate(sub):
            sl = slice(s * nx * npath, (s + 1) * nx * npath)
            X0[:, sl] = equilibrium(dict(Ps, gM=gM_override), N)[:, None]
    o = simulate(c, L, T=T, P=P, seed=seed, X0=X0, means=means, dt_factor=dt_factor)
    res24 = residual_deficit(o["X"], P, X0[0])
    shape = (len(sub), nx, npath)
    return dict(DG=o["DG"].reshape(shape).mean(2), Mavg=o["Mavg"].reshape(shape).mean(2),
                Pcol=(o["tau"] <= T).reshape(shape).mean(2), RES24=res24.reshape(shape).mean(2), dt=o["dt"])


def _valid_indices():
    return np.sort(np.random.default_rng(777).choice(600, N_VALID, replace=False))


def exp_validate(scale=1.0, chunk=0):
    """Repeat 50 randomly chosen sets of the global study with 200 paths (chunks of 10 sets)."""
    sets, _ = admissible_sets(600); idx = _valid_indices()[chunk * VALID_CHUNK:(chunk + 1) * VALID_CHUNK]
    t0 = time.time()
    r = _run_sets([sets[i] for i in idx], VALID_PATHS, seed=500 + chunk)
    save(f"validate_{chunk}", idx=idx, **r)
    print(f"  validation chunk {chunk}: sets {list(idx)}, dt={r['dt']:.4f}, {time.time() - t0:.0f} s")


def exp_validate_ablation(scale=1.0, chunk=0):
    """Same 50 sets with gamma_M = 0 (model M2), 50 paths, chunks of 25 sets."""
    sets, _ = admissible_sets(600); idx = _valid_indices()[chunk * 25:(chunk + 1) * 25]
    t0 = time.time()
    r = _run_sets([sets[i] for i in idx], 50, seed=900 + chunk, gM_override=0.0)
    save(f"validate_abl_{chunk}", idx=idx, **r)
    print(f"  ablation check chunk {chunk}: dt={r['dt']:.4f}, {time.time() - t0:.0f} s")


def exp_validate_merge(scale=1.0):
    g = np.load(os.path.join(OUT, "global.npz"))
    v = [np.load(os.path.join(OUT, f"validate_{k}.npz")) for k in range(N_VALID // VALID_CHUNK)]
    idx = np.concatenate([x["idx"] for x in v]); V = {k: np.concatenate([x[k] for x in v]) for k in ("DG", "Mavg", "Pcol", "RES24")}
    n = len(idx)
    def cls_P(P):
        mx = P.max(1); return (mx >= 0.1) & (P[:, 0] <= 0.5 * mx) & (P[:, -1] <= 0.5 * mx), mx >= 0.1
    w12 = np.array([interior_peak(y)[0] for y in g["DG"][idx]]); w200 = np.array([interior_peak(y)[0] for y in V["DG"]])
    m12 = np.array([interior_peak(y)[0] for y in g["Mavg"][idx]]); m200 = np.array([interior_peak(y)[0] for y in V["Mavg"]])
    loc12, loc200 = g["DG"][idx].argmax(1), V["DG"].argmax(1)
    p12, c12 = cls_P(g["Pcol"][idx]); p200, c200 = cls_P(V["Pcol"])
    print(f"Validation of the global study (Section 6.1): {n} random sets re-run with {VALID_PATHS} paths")
    print(f"  DG window: 12 paths {w12.sum()}/{n}, 200 paths {w200.sum()}/{n}, same classification in {(w12 == w200).sum()}/{n}")
    print(f"  M window: same classification in {(m12 == m200).sum()}/{n} ({m200.sum()}/{n} with 200 paths)")
    print(f"  DG maximum at the same grid point in {(loc12 == loc200).sum()}/{n}, within one grid point in "
          f"{(abs(loc12 - loc200) <= 1).sum()}/{n}; Pi at max (200 paths): "
          f"{dict(zip(*np.unique(np.round(X_GRID[loc200], 2), return_counts=True)))}")
    print(f"  collapse (max Pcol>=0.1): same in {(c12 == c200).sum()}/{n}; interior collapse window: same in {(p12 == p200).sum()}/{n}")
    rel = np.abs(V["DG"] - g["DG"][idx]) / V["DG"]
    print(f"  relative difference of mean DG, 12 vs 200 paths: median {np.median(rel):.3f}, 95th percentile {np.percentile(rel, 95):.3f}")
    out = dict(idx=idx, w12=w12, w200=w200, loc12=loc12, loc200=loc200, p12=p12, p200=p200, **{f"V_{k}": V[k] for k in V})
    fa = [os.path.join(OUT, f"validate_abl_{k}.npz") for k in (0, 1)]
    if all(os.path.exists(f) for f in fa):
        A = [np.load(f) for f in fa]; Ai = np.concatenate([a["idx"] for a in A])
        AV = {k: np.concatenate([a[k] for a in A]) for k in ("DG", "Pcol", "RES24")}
        assert np.array_equal(Ai, idx)
        wA = np.array([interior_peak(y)[0] for y in AV["DG"]])
        r_full, r_abl = V["RES24"].max(1), AV["RES24"].max(1)
        print(f"  M2 check (gamma_M = 0, 50 paths) on the same sets: interior DG maximum in {wA.sum()}/{n};"
              f" max Pcol = {AV['Pcol'].max():.2f}; collapse in the full model in {(V['Pcol'].max(1) >= 0.1).sum()}/{n}")
        print(f"  max residual deficit D24: full model median {np.median(r_full):.3f} (IQR {np.percentile(r_full, 25):.3f}-"
              f"{np.percentile(r_full, 75):.3f}), gamma_M=0 median {np.median(r_abl):.3f} (max {r_abl.max():.3f});"
              f" ratio full/M2 median {np.median(r_full / r_abl):.1f}, min {np.min(r_full / r_abl):.1f}")
        out.update(wA=wA, A_DG=AV["DG"], A_Pcol=AV["Pcol"], A_RES24=AV["RES24"])
    save("validate", **out)


EXPERIMENTS.update(validate_merge=exp_validate_merge)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=list(EXPERIMENTS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--chunk", type=int, default=None)
    a = ap.parse_args()
    s = 0.15 if a.quick else 1.0
    for w in a.which:
        t0 = time.time()
        if w in ("global", "global_run"):
            exp_global(s, chunk=a.chunk)
        elif w == "validate":
            exp_validate(s, chunk=a.chunk or 0)
        elif w == "validate_ablation":
            exp_validate_ablation(s, chunk=a.chunk or 0)
        else:
            EXPERIMENTS[w](s)
        print(f"  [{w}: {time.time() - t0:.0f} s]\n", flush=True)


