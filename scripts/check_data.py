"""Validate all initial and weekly datasets and two reconstructed objectives.

Run from anywhere: python scripts/check_data.py
Requires NumPy and standard Python libraries. No notebook execution needed.
"""
from pathlib import Path
from csv import DictReader
from ast import literal_eval
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"
DIMS = [2, 2, 3, 4, 4, 5, 6, 8]
INITIAL_N = [10, 10, 15, 30, 20, 20, 30, 40]

rows = list(DictReader((SUB / "weekly_submissions.csv").open(encoding="utf-8", newline="")))
returned = [literal_eval(line) for line in (SUB / "outputs.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(returned) == 13 and all(len(r) == 8 for r in returned)
assert len(rows) == 104

# Standard six-dimensional Hartmann (maximisation) benchmark coefficients.
A = np.array([
    [10., 3., 17., 3.5, 1.7, 8.],
    [0.05, 10., 17., 0.1, 8., 14.],
    [3., 3.5, 1.7, 10., 17., 8.],
    [17., 8., 0.05, 10., 0.1, 14.],
])
P = np.array([
    [.1312, .1696, .5569, .0124, .8283, .5886],
    [.2329, .4135, .8307, .3736, .1004, .9991],
    [.2348, .1451, .3522, .2883, .3047, .6650],
    [.4047, .8828, .8732, .5743, .1091, .0381],
])
ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

def hartmann6(x):
    x = np.atleast_2d(x)
    return np.sum(ALPHA[None, :] * np.exp(-np.sum(A[None, :, :] * (x[:, None, :] - P[None, :, :])**2, axis=2)), axis=1)

CENTRE = np.array([0.10, 0.15, 0.13, 0.15, 0.80, 0.50, 0.20, 0.60])
WEIGHTS = np.array([2.0, 1.0, 3.0, 1.0, 0.5, 1.0, 2.0, 0.1])
def quadratic8(x):
    x = np.atleast_2d(x)
    return 10 - np.sum(WEIGHTS * (x - CENTRE)**2, axis=1)

all_initial = 0
all_weekly = 0
for i, (d,n) in enumerate(zip(DIMS,INITIAL_N), 1):
    folder = ROOT / f"function_{i}"
    X0 = np.load(folder / "initial_inputs.npy", allow_pickle=False)
    y0 = np.load(folder / "initial_outputs.npy", allow_pickle=False)
    assert X0.shape == (n,d) and y0.shape == (n,)
    assert X0.dtype.kind == 'f' and y0.dtype.kind == 'f'
    assert np.isfinite(X0).all() and np.isfinite(y0).all()
    assert ((0 <= X0)&(X0 <= 1)).all()
    assert len(np.unique(X0, axis=0)) == len(X0)
    subset=sorted((r for r in rows if int(r['function'])==i), key=lambda r:int(r['week']))
    assert len(subset) == 13
    Xi = np.array([[float(r[f'x{c}']) for c in range(1,d+1)] for r in subset])
    yi = np.array([float(r['output']) for r in subset])
    assert np.isfinite(Xi).all() and np.isfinite(yi).all()
    assert ((0<=Xi)&(Xi<=1)).all()
    assert len(np.unique(Xi,axis=0)) == 13
    assert len(np.unique(np.vstack([X0,Xi]),axis=0)) == n+13
    assert all(int(r['week']) == k for k,r in enumerate(subset,1))
    np.testing.assert_allclose(yi, [r[i-1] for r in returned], rtol=0, atol=1e-12)
    allx=np.vstack([X0,Xi]); ally=np.r_[y0,yi]
    winning=int(np.argmax(ally))
    wk=f'Week {winning-n+1}' if winning>=n else 'initial dataset'
    print(f'F{i}: initial {n:2d} + 13 weekly = {len(ally):2d}; initial best {y0.max():.12g}; overall best {ally[winning]:.12g} ({wk})')
    if i in (7,8):
        calc=hartmann6(allx) if i==7 else quadratic8(allx)
        err=np.max(np.abs(calc-ally))
        assert err < 1e-10, (i,err)
        print(f'      F{i} identified formula: max residual {err:.3g} across {len(ally)} observations')
    all_initial+=n;all_weekly+=len(yi)
assert (all_initial,all_weekly)==(175,104)
print(f'Complete: {all_initial} initial + {all_weekly} weekly = {all_initial+all_weekly} observations; all records accounted for.')
