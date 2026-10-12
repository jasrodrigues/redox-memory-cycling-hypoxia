"""Intracellular core of the redox-memory model.

State X = (G, R, A, M): glutathione, ROS, ATP, redox memory (dimensionless).
Time unit: hours. Environment E in {N, H, Q} = {0, 1, 2}.
All functions accept parameter values that are scalars or NumPy arrays with one
entry per path, so many paths (or many parameter sets) can be integrated at once.
"""
import numpy as np
from scipy.optimize import brentq

N, H, Q = 0, 1, 2
STATE_NAMES = ("G", "R", "A", "M")

# Baseline parameters (Table A1 of the paper).
BASE = dict(
    sG=0.15, dG=0.05, kGR=1.0,          # glutathione
    pN=0.20, pH=0.25, dR=1.0,           # ROS production / removal
    BQ=3.0, tauQ=0.5, tauS=1.0,         # reoxygenation burst and hypoxic priming
    pAN=0.90, pAH=0.45, KG=0.5, KR=0.5, dA=0.5,   # ATP
    aM=0.10, KM=0.5, n=4, dM=0.02, gM=1.0,        # redox memory
)
MEANS0 = np.array([4.0, 2.0, 1.0])      # mean sojourns (h) in N, H, Q at c = 1
THRESHOLDS = dict(Rc=2.0, Gc=0.25, Ac=0.20)
PI = MEANS0 / MEANS0.sum()              # stationary distribution, Eq. (7)


def params(**overrides):
    P = dict(BASE)
    P.update(overrides)
    return P


def rhs(X, E, age, P, amp):
    """Vector field of Eqs. (1)-(5).

    X   : array (4, ...) of states
    E   : environment (int array or scalar)
    age : time since the last H -> Q transition (burst age)
    amp : priming amplitude a_k of the current burst, Eq. (6)
    """
    G, R, A, M = X
    burst = np.where(E == Q, amp * P["BQ"] * np.exp(-age / P["tauQ"]), 0.0)
    pR = np.where(E == H, P["pH"], P["pN"]) + burst
    pA = np.where(E == H, P["pAH"], P["pAN"])
    Rn = R ** P["n"]
    dG = P["sG"] / (1 + P["gM"] * M) - P["dG"] * G - P["kGR"] * G * R
    dR = pR - P["kGR"] * G * R - P["dR"] * R
    dA = pA * G / (P["KG"] + G) * P["KR"] / (P["KR"] + R) - P["dA"] * A
    dM = P["aM"] * Rn / (P["KM"] ** P["n"] + Rn) - P["dM"] * M
    return np.array([dG, dR, dA, dM])


def invariant_box(P):
    """Upper corners of the invariant set K of Proposition 1."""
    return np.array([P["sG"] / P["dG"],
                     (max(P["pN"], P["pH"]) + P["BQ"]) / P["dR"],
                     P["pAN"] / P["dA"],
                     P["aM"] / P["dM"]])


# ---------------------------------------------------------------- equilibria
def _R_of_G(G, p, P):
    return p / (P["kGR"] * G + P["dR"])


def _M_of_R(R, P):
    Rn = R ** P["n"]
    return P["aM"] / P["dM"] * Rn / (P["KM"] ** P["n"] + Rn)


def F_scalar(G, p, P):
    """F(G) of Proposition 2; its zeros are the equilibria."""
    R = _R_of_G(G, p, P)
    return P["sG"] / (1 + P["gM"] * _M_of_R(R, P)) - G * (P["dG"] + P["kGR"] * R)


def equilibria(P, E=N, grid=200_001):
    """All equilibria for a fixed environment without burst (Proposition 2)."""
    p = P["pH"] if E == H else P["pN"]
    pA = P["pAH"] if E == H else P["pAN"]
    Gs = np.linspace(1e-6, P["sG"] / P["dG"], grid)
    f = F_scalar(Gs, p, P)
    idx = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    out = []
    for i in idx:
        G = brentq(F_scalar, Gs[i], Gs[i + 1], args=(p, P), xtol=1e-14)
        R = _R_of_G(G, p, P)
        M = _M_of_R(R, P)
        A = pA * G / (P["KG"] + G) * P["KR"] / (P["KR"] + R) / P["dA"]
        out.append(np.array([G, R, A, M]))
    return out


def equilibrium(P, E=N):
    eqs = equilibria(P, E)
    if len(eqs) != 1:
        raise RuntimeError(f"expected one equilibrium, found {len(eqs)}")
    return eqs[0]


def loop_gain(X, P):
    """Loop gain Lambda of Theorem 1, Eq. (9)."""
    G, R, A, M = X
    k, n, K = P["kGR"], P["n"], P["KM"]
    sigma = P["sG"] * P["gM"] / (1 + P["gM"] * M) ** 2
    eta = P["aM"] * n * K ** n * R ** (n - 1) / (K ** n + R ** n) ** 2
    D = P["dG"] * k * G + P["dG"] * P["dR"] + k * R * P["dR"]
    return sigma * k * R * eta / (P["dM"] * D)


def jacobian(X, E, P, eps=1e-7):
    J = np.zeros((4, 4))
    for j in range(4):
        d = np.zeros(4); d[j] = eps
        J[:, j] = (rhs(X + d, np.array(E), np.array(1e9), P, 0.0)
                   - rhs(X - d, np.array(E), np.array(1e9), P, 0.0)) / (2 * eps)
    return J


# ------------------------------------------------------- analytic results
def burst_load(c, P, means=MEANS0):
    """Long-run burst ROS production L(c), Eq. (10)."""
    mN, mH, mQ = means
    return (c / (mN + mH + mQ) * P["BQ"] * P["tauQ"]
            * mH / (mH + c * P["tauS"]) * mQ / (mQ + c * P["tauQ"]))


def c_star(P, means=MEANS0):
    """Maximiser of L(c), Eq. (11)."""
    return np.sqrt(means[1] * means[2] / (P["tauS"] * P["tauQ"]))


def averaged_params(P, pi=PI):
    """Parameters of the averaged system of Theorem 2 (burst-free, run in state N)."""
    Pa = dict(P)
    Pa["pN"] = (pi[N] + pi[Q]) * P["pN"] + pi[H] * P["pH"]
    Pa["pAN"] = (pi[N] + pi[Q]) * P["pAN"] + pi[H] * P["pAH"]
    return Pa


def integrate_fixed(P, E, X0, T, dt=0.002):
    """Heun integration for a fixed environment without burst."""
    X = np.asarray(X0, float).copy()
    ts = [0.0]; Xs = [X.copy()]
    for k in range(int(round(T / dt))):
        k1 = rhs(X, np.array(E), np.array(1e9), P, 0.0)
        k2 = rhs(X + dt * k1, np.array(E), np.array(1e9), P, 0.0)
        X = X + 0.5 * dt * (k1 + k2)
        ts.append((k + 1) * dt); Xs.append(X.copy())
    return np.array(ts), np.array(Xs).T


# ------------------------------------------- dimensionless form and general sojourns
def Pi_groups(c, P, means=MEANS0):
    """Pi_H = c tau_S / m_H, Pi_Q = c tau_Q / m_Q, Pi = sqrt(Pi_H Pi_Q), rho = Pi_Q / Pi_H."""
    PiH = c * P["tauS"] / means[1]
    PiQ = c * P["tauQ"] / means[2]
    return PiH, PiQ, np.sqrt(PiH * PiQ), PiQ / PiH


def load_shape(Pi, rho):
    """Normalised burst load L/L(c*_L) as a function of Pi = c / c*_L and rho = Pi_Q / Pi_H."""
    s = np.sqrt(rho) + 1 / np.sqrt(rho)
    return Pi * (2 + s) / (1 + s * Pi + Pi ** 2)


def _phi(mean, tau, law, n_gh=80):
    """1 - E[exp(-d/tau)] for a sojourn d with the given mean and law."""
    mean = np.asarray(mean, float)
    kind = law[0]
    if kind == "exp":
        return mean / (mean + tau)
    cv = law[1]
    if kind == "gamma":
        k = 1 / cv ** 2
        return 1 - (1 + mean / (k * tau)) ** (-k)
    if kind == "lognormal":
        s2 = np.log1p(cv ** 2); mu = np.log(mean)[..., None] - 0.5 * s2
        x, w = np.polynomial.hermite.hermgauss(n_gh)
        d = np.exp(mu + np.sqrt(2 * s2) * x)
        return 1 - (w * np.exp(-d / tau)).sum(-1) / np.sqrt(np.pi)
    raise ValueError(kind)


def burst_load_general(c, P, means=MEANS0, law=("exp",)):
    """Burst load for any sojourn law with the given means (renewal-reward, Prop. 3')."""
    c = np.asarray(c, float)
    return (c / np.sum(means) * P["BQ"] * P["tauQ"]
            * _phi(means[1] / c, P["tauS"], law) * _phi(means[2] / c, P["tauQ"], law))
