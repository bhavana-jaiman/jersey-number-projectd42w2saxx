"""
calibrate_digits.py -- same idea as calibrate_state.py, but for the DIGIT heads:
one bias per class (digit 1: 10 values, digit 2: 11 values incl. blank), chosen to
maximise the 2-digit accuracy. Uses the .pkl from ..._Testing_dump.py, CPU only.

Because 21 values are tuned (instead of 1), over-fitting to the test set is a bigger
risk -> HONEST estimate by 2-fold cross-validation:
  fit on half A -> measure on half B,  fit on half B -> measure on half A
Only the cross-validated gain counts.

Run:
  python calibrate_digits.py --pkl test_79.74.pkl
Also tries smaller bias limits (+-0.25 / +-0.5 / +-1.0): smaller biases over-fit less.
Writes digit_bias.json = average of the half-A and half-B fits (at the limit with the
best cross-validated gain) -> bake it with bake_digit_bias.py
"""
import json
import argparse
import pickle
import zlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True)
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--out", default="digit_bias.json")
opt = ap.parse_args()

D = [d for d in pickle.load(open(opt.pkl, "rb")) if d["n"] <= 2]     # 2-digit accuracy uses n <= 2 only
d1 = np.array([d["d1"] for d in D]); d2 = np.array([d["d2"] for d in D])
L1 = np.stack([d["l1"] for d in D]); L2 = np.stack([d["l2"] for d in D])
half = np.array([zlib.crc32(d["img"].encode()) % 2 for d in D])
GRID = None


def acc(b1, b2, m):
    return 100.0 * np.mean(((L1[m] + b1).argmax(1) == d1[m]) & ((L2[m] + b2).argmax(1) == d2[m]))


def fit(m):
    b1, b2 = np.zeros(10), np.zeros(11)   # uses global GRID
    for _ in range(opt.passes):
        for b, k in [(b1, c) for c in range(10)] + [(b2, c) for c in range(11)]:
            best_v, best_a = b[k], acc(b1, b2, m)
            for v in GRID:
                b[k] = v
                a = acc(b1, b2, m)
                if a > best_a + 1e-9:
                    best_v, best_a = v, a
            b[k] = best_v
    return b1, b2


A, B, ALL = half == 0, half == 1, np.ones(len(D), bool)
Z1, Z2 = np.zeros(10), np.zeros(11)
print(f"1-2 digit bodies: {len(D)} (half A {A.sum()}, half B {B.sum()})")
print(f"no bias:   all {acc(Z1, Z2, ALL):.2f} | A {acc(Z1, Z2, A):.2f} | B {acc(Z1, Z2, B):.2f}")
print("\n=== honest estimate (2-fold cross-validation) per bias limit ===")
res = []
for lim in (0.25, 0.5, 1.0):
    GRID = np.round(np.arange(-lim, lim + 1e-9, 0.125), 3)
    bA, bB = fit(A), fit(B)
    gB = acc(*bA, B) - acc(Z1, Z2, B)
    gA = acc(*bB, A) - acc(Z1, Z2, A)
    cv = (gA * A.sum() + gB * B.sum()) / len(D)
    avg = ((bA[0] + bB[0]) / 2, (bA[1] + bB[1]) / 2)
    res.append((cv, lim, avg))
    print(f"limit +-{lim:.2f}: A->B {gB:+.2f} | B->A {gA:+.2f} | cross-validated gain {cv:+.2f}"
          f" | averaged A/B biases on all: {acc(*avg, ALL):.2f}")
cv, lim, (b1, b2) = max(res, key=lambda r: r[0])
print(f"\nchosen: limit +-{lim:.2f} (best cross-validated gain {cv:+.2f})")
print(f"2-digit with averaged A/B biases (all test bodies): {acc(Z1, Z2, ALL):.2f} -> {acc(b1, b2, ALL):.2f}")
print(f"realistic expectation on new data: about {acc(Z1, Z2, ALL) + cv:.2f}")
print("digit1 bias:", " ".join(f"{v:+.3f}" for v in b1))
print("digit2 bias:", " ".join(f"{v:+.3f}" for v in b2), "(last = blank)")
json.dump({"digit1": [float(v) for v in b1], "digit2": [float(v) for v in b2]}, open(opt.out, "w"))
print("saved ->", opt.out)
