"""Check the 13-round evaluation CSV against the original output history.

Run from anywhere: python scripts/check_weekly_data.py
Requires only Python's standard library. Does not reconstruct initial .npy data.
"""
from pathlib import Path
from csv import DictReader
from ast import literal_eval
from math import isclose, isfinite

root = Path(__file__).resolve().parents[1]
submissions = root / "submissions"
rows = list(DictReader((submissions / "weekly_submissions.csv").open(encoding="utf-8")))
outputs = [literal_eval(line) for line in (submissions / "outputs.txt").read_text().splitlines() if line.strip()]
dims = [2, 2, 3, 4, 4, 5, 6, 8]
assert len(outputs) == 13 and all(len(r) == 8 for r in outputs)
assert len(rows) == 13 * 8

for function, d in enumerate(dims, start=1):
    sub = [r for r in rows if int(r["function"]) == function]
    assert len(sub) == 13
    coordinates = []
    for week, row in enumerate(sub, start=1):
        assert int(row["week"]) == week and int(row["dimension"]) == d
        xs = tuple(float(row[f"x{k}"]) for k in range(1, d + 1))
        assert all(isfinite(x) and 0 <= x <= 1 for x in xs)
        assert isclose(float(row["output"]), outputs[week - 1][function - 1], rel_tol=0, abs_tol=1e-12)
        coordinates.append(xs)
    assert len(set(coordinates)) == len(coordinates)
    best = max(sub, key=lambda row: float(row["output"]))
    print(f"Function {function}: best weekly output {best['output']} (Week {best['week']})")

print("Complete: 104 evaluated weekly submissions across all 8 functions.")
