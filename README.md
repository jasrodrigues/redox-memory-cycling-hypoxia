# Redox memory under cycling tumour hypoxia: code

Code for the paper *Deterministic and Stochastic Dynamics of Redox Memory under Cycling Tumor Hypoxia*.
It reproduces every number, table and figure in the manuscript.

## Files

| File | Contents |
| --- | --- |
| `model.py` | Parameters (Table A1), vector field (Eqs. 1–5), equilibria (Prop. 2), loop gain (Thm. 1), burst load and c* (Prop. 3, Eqs. 10–11), averaged system (Thm. 2) |
| `pdmp.py` | Vectorised Monte Carlo simulation of the PDMP (Section 3, Appendix A) |
| `protocols.py` | Deterministic protocols with equal hypoxic burden (Section 5.2) |
| `experiments.py` | Every experiment of Sections 4–6 and Appendix B; writes `results/*.npz` and prints the numbers used in the tables |
| `figures.py` | Figures 1–9 from `results/` into `figures/` |
| `tests/test_model.py` | Thirteen checks of the analytic results against the code |

## Usage

```bash
pip install -r requirements.txt
python experiments.py --quick                  # reduced sizes, for checking
python experiments.py main refine gct laws     # selected experiments
for i in 0 1 2 3 4 5; do python experiments.py global --chunk $i; done
python experiments.py global_merge             # global robustness (600 sets)
for i in 0 1 2 3 4; do python experiments.py validate --chunk $i; done            # 50 sets, 200 paths
for i in 0 1; do python experiments.py validate_ablation --chunk $i; done         # same sets, gamma_M = 0
python experiments.py validate_merge                                              # compare 12 vs 200 paths
python figures.py                # needs results/ from a full or quick run
pytest -q tests                  # about 30 s
```

Experiments: `symbolic` (characteristic polynomial, Appendix B), `equilibria`, `protocols`, `collapse` (load against Π for 100 random sets), `laws_analytic`, `main`, `refine`, `control`, `ablation` (M0–M3), `heat`, `scaling`, `gct` (collapse threshold and horizon), `laws` (gamma, lognormal, near-periodic sojourns), `global` + `global_merge`, `prcc` (with bootstrap), `fast`, `dt` (step size in three regimes).

The full set takes about one hour on one core. The global study is split into six chunks of about 90 s each.

Threshold-free observables returned by `pdmp.simulate`: time-averaged memory `Mavg`, mean antioxidant deficit `DG`, minimum and mean glutathione `Gmin`, `Gavg`; `sweep_c` also returns the residual deficit after 24 h of normoxia, `RES24`.

## Conventions

- Time in hours. G, R, A, M are dimensionless.
- Environment codes: N = 0, H = 1, Q = 2.
- Any parameter may be given as an array with one value per path. This is how the (c, γ_M) map and the PRCC run all parameter sets in a single vectorised call.
- Random seeds are fixed in `experiments.py`, so reruns give identical numbers.

The folders `results/` and `figures/` are created when the scripts are run.
