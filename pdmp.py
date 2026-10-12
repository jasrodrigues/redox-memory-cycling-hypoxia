"""Monte Carlo simulation of the stochastic hybrid model.

The environment cycles N -> H -> Q -> N. Sojourn times have means MEANS/c and,
by default, are exponential (the PDMP of the paper). Gamma, lognormal and
near-periodic sojourns with the same means are available for the robustness
analysis of Section 6.4. Paths are integrated together with Heun's method; jump
times are drawn exactly and applied at the end of the step in which they fall,
while the burst age and the priming amplitude use the exact jump times.
"""
import numpy as np
from model import N, H, Q, MEANS0, THRESHOLDS, rhs, equilibrium, BASE

NEXT = np.array([H, Q, N])
GC_LIST = (0.20, 0.25, 0.30, 0.35)          # thresholds for the robustness of Section 6.3


def draw_sojourn(rng, mean, law):
    """Draw sojourns with the given means. law = ('exp',) | ('gamma', cv) | ('lognormal', cv)."""
    kind = law[0]
    if kind == "exp":
        return rng.exponential(mean)
    cv = law[1]
    if kind == "gamma":
        k = 1.0 / cv ** 2
        return rng.gamma(k, mean / k)
    if kind == "lognormal":
        s2 = np.log1p(cv ** 2)
        return rng.lognormal(np.log(mean) - 0.5 * s2, np.sqrt(s2))
    raise ValueError(kind)


def default_dt(c, means=MEANS0, factor=0.1):
    """Step: at most 0.01 h and at most `factor` times the shortest mean Q sojourn."""
    mQ = np.min(np.asarray(means)[..., Q] / np.asarray(c))
    return float(min(0.01, factor * mQ))


def simulate(c, npaths, T=240.0, P=None, seed=0, X0=None, dt=None, thresholds=None,
             record_every=None, means=MEANS0, law=("exp",), gc_list=GC_LIST, dt_factor=0.1, Gref=None):
    """Simulate npaths paths.

    c      : scalar or array (npaths,) of switching-speed factors
    P      : parameter dict; values may be arrays (npaths,) for per-path parameters
    means  : mean sojourns at c = 1, shape (3,) or (npaths, 3)
    law    : sojourn distribution, see draw_sojourn
    Returns a dict with, per path,
      tau      first collapse time with the thresholds of the paper (inf if none)
      cause    0 none, 1 R, 2 G, 3 A
      tau_gc   (len(gc_list), npaths) collapse times with G_c taken from gc_list
      Mavg     time-averaged memory            (threshold-free)
      Gmin     minimum of G over [0, T]        (threshold-free)
      Gavg     time-averaged G                 (threshold-free)
      DG       (1/T) * integral of [1 - G/Gref]_+, Gref = normoxic G* (threshold-free deficit)
      Bload    time-averaged burst ROS production
      Hfrac, nHQ, X (final state); and t_rec, rec if record_every is given.
    """
    P = dict(BASE) if P is None else P
    th = dict(THRESHOLDS) if thresholds is None else thresholds
    rng = np.random.default_rng(seed)
    c = np.broadcast_to(np.asarray(c, float), (npaths,))
    means = np.broadcast_to(np.asarray(means, float), (npaths, 3))
    mean_soj = means / c[:, None]
    dt = default_dt(c, means, dt_factor) if dt is None else dt
    if X0 is None:
        X0 = equilibrium(P, N)
    tauS = np.broadcast_to(np.asarray(P["tauS"], float), (npaths,))
    gcs = np.asarray(gc_list, float)[:, None]
    Gref = np.asarray(X0, float)[0] if Gref is None else np.asarray(Gref, float)

    X0 = np.asarray(X0, float)
    X = X0.copy() if X0.ndim == 2 else np.tile(X0[:, None], (1, npaths))
    E = np.full(npaths, N)
    t_left = draw_sojourn(rng, mean_soj[:, N], law)
    age = np.full(npaths, 1e9)
    amp = np.zeros(npaths)
    t_enter_H = np.zeros(npaths)

    tau = np.full(npaths, np.inf); cause = np.zeros(npaths, int)
    tau_gc = np.full((len(gcs), npaths), np.inf)
    M_int = np.zeros(npaths); B_int = np.zeros(npaths); H_int = np.zeros(npaths)
    G_int = np.zeros(npaths); D_int = np.zeros(npaths); Gmin = X[0].copy()
    nHQ = np.zeros(npaths, int)
    nsteps = int(round(T / dt))
    stride = None if record_every is None else max(1, int(round(record_every / dt)))
    rec, t_rec = ([X.copy()], [0.0]) if stride else ([], [])

    for k in range(nsteps):
        t = k * dt
        k1 = rhs(X, E, age, P, amp)
        k2 = rhs(X + dt * k1, E, age + dt, P, amp)
        X = np.maximum(X + 0.5 * dt * (k1 + k2), 0.0)
        age += dt
        t_left -= dt
        G = X[0]
        M_int += X[3] * dt
        G_int += G * dt
        D_int += np.maximum(1.0 - G / Gref, 0.0) * dt
        np.minimum(Gmin, G, out=Gmin)
        B_int += (E == Q) * amp * P["BQ"] * np.exp(-age / P["tauQ"]) * dt
        H_int += (E == H) * dt

        cR, cA = X[1] >= th["Rc"], X[2] <= th["Ac"]
        cG = G <= th["Gc"]
        hit = ~np.isfinite(tau) & (cR | cG | cA)
        if hit.any():
            tau[hit] = t + dt
            cause[hit & cR] = 1
            cause[hit & ~cR & cG] = 2
            cause[hit & ~cR & ~cG & cA] = 3
        hit_gc = ~np.isfinite(tau_gc) & ((G[None, :] <= gcs) | cR[None, :] | cA[None, :])
        if hit_gc.any():
            tau_gc[hit_gc] = t + dt

        jump = t_left <= 0
        while jump.any():
            i = np.where(jump)[0]
            new = NEXT[E[i]]
            t_jump = t + dt + t_left[i]
            nHQ[i] += (E[i] == H)
            t_enter_H[i] = np.where(new == H, t_jump, t_enter_H[i])
            toQ = new == Q
            amp[i] = np.where(toQ, 1.0 - np.exp(-(t_jump - t_enter_H[i]) / tauS[i]), amp[i])
            age[i] = np.where(toQ, -t_left[i], age[i])
            E[i] = new
            t_left[i] += draw_sojourn(rng, mean_soj[i, new], law)
            jump = t_left <= 0

        if stride and (k + 1) % stride == 0:
            rec.append(X.copy()); t_rec.append(t + dt)

    out = dict(tau=tau, cause=cause, tau_gc=tau_gc, X=X, Mavg=M_int / T, Gavg=G_int / T,
               DG=D_int / T, Gmin=Gmin, Bload=B_int / T, Hfrac=H_int / T, nHQ=nHQ, dt=dt)
    if stride:
        out["t_rec"] = np.array(t_rec); out["rec"] = np.array(rec)
    return out


METRICS = ("tau", "cause", "Mavg", "Gavg", "DG", "Gmin", "Bload", "Hfrac", "nHQ")


def sweep_c(cs, npaths, P=None, T=240.0, seed=0, **kw):
    """Run npaths paths for every value in cs; arrays shaped (len(cs), npaths)."""
    cs = np.asarray(cs, float)
    o = simulate(np.repeat(cs, npaths), len(cs) * npaths, T=T, P=P, seed=seed, **kw)
    shape = (len(cs), npaths)
    res = {k: o[k].reshape(shape) for k in METRICS}
    res["tau_gc"] = o["tau_gc"].reshape((-1,) + shape)
    res["G_T"] = o["X"][0].reshape(shape); res["M_T"] = o["X"][3].reshape(shape)
    P_ = dict(BASE) if P is None else P
    X0_ = kw.get("X0", None)
    Gref = (equilibrium(P_, N) if X0_ is None else np.asarray(X0_))[0]
    res["RES24"] = residual_deficit(o["X"], P_, Gref).reshape(shape)
    res["Pcol"] = (res["tau"] <= T).mean(1)
    res["cs"] = cs
    return res


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return centre - half, centre + half


def residual_deficit(X_T, P, Gref, hours=24.0, dt=0.01):
    """[1 - G/Gref]_+ after `hours` of normoxia without burst, starting from the final states X_T
    (4, npaths). Measures how much of the antioxidant deficit persists once switching stops."""
    from model import rhs as _rhs
    X = np.asarray(X_T, float).copy()
    E = np.full(X.shape[1], N); age = np.full(X.shape[1], 1e9)
    for _ in range(int(round(hours / dt))):
        k1 = _rhs(X, E, age, P, 0.0); k2 = _rhs(X + dt * k1, E, age, P, 0.0)
        X = np.maximum(X + 0.5 * dt * (k1 + k2), 0.0)
    return np.maximum(1.0 - X[0] / Gref, 0.0)
