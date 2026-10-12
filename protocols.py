"""Deterministic protocols with a prescribed sequence of environments."""
import numpy as np
from model import N, H, Q, rhs, equilibrium


def run_schedule(schedule, P, dt=0.005, X0=None):
    """schedule: list of (E, duration in h). Returns t, X (4, n), E (n,)."""
    X = np.asarray(equilibrium(P, N) if X0 is None else X0, float).copy()
    ts, Xs, Es = [0.0], [X.copy()], [N]
    t, amp, last_H = 0.0, 0.0, 0.0
    for E, d in schedule:
        if E == H:
            last_H = d
        if E == Q:
            amp, age = 1 - np.exp(-last_H / P["tauS"]), 0.0
        else:
            age = 1e9
        n = max(1, int(round(d / dt))); h = d / n
        for _ in range(n):
            k1 = rhs(X, np.array(E), np.array(age), P, amp)
            k2 = rhs(X + h * k1, np.array(E), np.array(age + h), P, amp)
            X = np.maximum(X + 0.5 * h * (k1 + k2), 0.0)
            t += h; age += h
            ts.append(t); Xs.append(X.copy()); Es.append(E)
    return np.array(ts), np.array(Xs).T, np.array(Es)


def equal_burden(n_episodes, total_H=12.0, reox=1.0, total=96.0, lead=12.0, window=48.0):
    """n episodes of total_H/n hours of hypoxia, each followed by `reox` h of Q,
    evenly spaced in a `window` that starts after `lead` h of normoxia."""
    dH = total_H / n_episodes
    gap = window / n_episodes - dH - reox
    if gap < 0:
        raise ValueError("episodes do not fit in the window")
    s = [(N, lead)]
    for _ in range(n_episodes):
        s += [(H, dH), (Q, reox)] + ([(N, gap)] if gap > 0 else [])
    s.append((N, total - lead - window))
    return s


def hypoxic_exposure(t, E):
    return float(np.sum(np.diff(t) * (E[1:] == H)))
