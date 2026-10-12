"""Fast checks of the analytic results against the code.  Run:  pytest -q tests"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as mdl
from model import N, H, params, equilibrium, equilibria, loop_gain, jacobian
from pdmp import simulate
from protocols import run_schedule, equal_burden, hypoxic_exposure


def test_stationary_fraction_independent_of_c():
    assert np.allclose(mdl.PI, [4 / 7, 2 / 7, 1 / 7])


def test_positivity_and_invariant_box():
    P = params(); box = mdl.invariant_box(P)
    o = simulate(1.26, 200, T=120, P=P, seed=1, record_every=0.5)
    rec = o["rec"]                              # (nt, 4, paths)
    assert rec.min() >= 0
    assert np.all(rec.max(axis=(0, 2)) <= box + 1e-9)


def test_equilibria_unique_and_stable():
    P = params()
    for E in (N, H):
        eqs = equilibria(P, E)
        assert len(eqs) == 1
        X = eqs[0]
        lam = loop_gain(X, P)
        ev = np.linalg.eigvals(jacobian(X, E, P))
        assert lam < 1 and np.all(ev.real < 0)


def test_loop_gain_matches_determinant():
    # det J (3x3 block G,R,M) = -delta_M * D * (1 - Lambda), Theorem 1
    P = params(gM=5.0, pN=0.6)
    X = equilibrium(P, N)
    J = jacobian(X, N, P)[np.ix_([0, 1, 3], [0, 1, 3])]
    G, R, A, M = X
    D = P["dG"] * P["kGR"] * G + P["dG"] * P["dR"] + P["kGR"] * R * P["dR"]
    assert np.isclose(np.linalg.det(J), -P["dM"] * D * (1 - loop_gain(X, P)), rtol=1e-5)


def test_burst_load_formula():
    P = params()
    for c in (0.5, 2.0, 6.0):
        o = simulate(c, 300, T=400, P=P, seed=2)
        assert abs(o["Bload"].mean() / mdl.burst_load(c, P) - 1) < 0.05


def test_c_star_is_maximiser():
    P = params(); cc = np.logspace(-2, 2, 4001)
    assert np.isclose(cc[np.argmax(mdl.burst_load(cc, P))], mdl.c_star(P), rtol=0.01)


def test_constant_amplitude_load_is_monotone():
    P = params(tauS=1e-12); cc = np.logspace(-2, 2, 200)
    assert np.all(np.diff(mdl.burst_load(cc, P)) > 0)


def test_equal_burden_protocols():
    P = params()
    Ms = []
    for n in (1, 4, 12):
        t, X, E = run_schedule(equal_burden(n), P)
        assert np.isclose(hypoxic_exposure(t, E), 12.0, atol=1e-6)
        Ms.append(X[3, -1])
    assert Ms[0] < Ms[1] < Ms[2]


def test_slow_switching_bound():
    P = params(); c, T = 0.03, 240.0
    o = simulate(c, 300, T=T, P=P, seed=4)
    assert np.mean(o["tau"] <= T) <= 1 - np.exp(-c * T / mdl.MEANS0[0])


def test_fast_switching_is_safe():
    o = simulate(20.0, 100, T=240, P=params(), seed=5)
    assert np.mean(o["tau"] <= 240) == 0


def test_load_shape_collapses_at_Pi_one():
    rng = np.random.default_rng(0); x = np.logspace(-2, 2, 4001)
    for _ in range(20):
        P = params(tauS=10 ** rng.uniform(-0.7, 0.7), tauQ=10 ** rng.uniform(-1, 0.3))
        means = np.array([4.0, 10 ** rng.uniform(-0.5, 1), 10 ** rng.uniform(-0.7, 0.7)])
        cs = mdl.c_star(P, means)
        L = mdl.burst_load(x * cs, P, means)
        assert np.isclose(x[np.argmax(L)], 1.0, rtol=0.01)
        rho = (P["tauQ"] / means[2]) / (P["tauS"] / means[1])
        assert np.allclose(L / mdl.burst_load(cs, P, means), mdl.load_shape(x, rho))


def test_general_law_load_has_interior_maximum_and_matches_simulation():
    P = params(); cc = np.logspace(-3, 3, 3001)
    for law in [("gamma", 0.1), ("gamma", 2.0), ("lognormal", 1.5)]:
        L = mdl.burst_load_general(cc, P, law=law)
        j = np.argmax(L); assert 0 < j < len(cc) - 1 and L[0] < 0.05 * L[j] and L[-1] < 0.05 * L[j]
    o = simulate(1.5, 300, T=400, P=P, seed=6, law=("gamma", 0.5))
    assert abs(o["Bload"].mean() / mdl.burst_load_general(1.5, P, law=("gamma", 0.5)) - 1) < 0.05


def test_threshold_free_metrics_are_consistent():
    o = simulate(1.26, 200, T=240, P=params(), seed=8)
    assert np.all(o["Gmin"] <= o["Gavg"] + 1e-12) and np.all(o["DG"] >= 0) and np.all(o["DG"] <= 1)
    # collapse with G_c = 0.25 is the same event in tau and tau_gc
    assert np.array_equal(np.isfinite(o["tau"]), np.isfinite(o["tau_gc"][1]))
