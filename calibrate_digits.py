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
"""
import argparse
import pickle
import zlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True)
ap.add_argument("--passes", type=int, default=3)
opt = ap.parse_args()

D = [d for d in pickle.load(open(opt.pkl, "rb")) if d["n"] <= 2]     # 2-digit accuracy uses n <= 2 only
d1 = np.array([d["d1"] for d in D]); d2 = np.array([d["d2"] for d in D])
L1 = np.stack([d["l1"] for d in D]); L2 = np.stack([d["l2"] for d in D])
half = np.array([zlib.crc32(d["img"].encode()) % 2 for d in D])
GRID = np.round(np.arange(-1.0, 1.01, 0.125), 3)


def acc(b1, b2, m):
    return 100.0 * np.mean(((L1[m] + b1).argmax(1) == d1[m]) & ((L2[m] + b2).argmax(1) == d2[m]))


def fit(m):
    b1, b2 = np.zeros(10), np.zeros(11)
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
print(f"1-2 digit bodies: {len(D)} (half A {A.sum()}, half B {B.sum()})")
print(f"no bias:   all {acc(np.zeros(10), np.zeros(11), ALL):.2f} | A {acc(np.zeros(10), np.zeros(11), A):.2f} | B {acc(np.zeros(10), np.zeros(11), B):.2f}")

bA = fit(A); bB = fit(B)
gB = acc(*bA, B) - acc(np.zeros(10), np.zeros(11), B)
gA = acc(*bB, A) - acc(np.zeros(10), np.zeros(11), A)
print("\n=== honest estimate (2-fold cross-validation) ===")
print(f"fit on A -> B: {acc(np.zeros(10), np.zeros(11), B):.2f} -> {acc(*bA, B):.2f}  ({gB:+.2f})")
print(f"fit on B -> A: {acc(np.zeros(10), np.zeros(11), A):.2f} -> {acc(*bB, A):.2f}  ({gA:+.2f})")
cv = (gA * A.sum() + gB * B.sum()) / len(D)
print(f"cross-validated gain: {cv:+.2f} points  <- this is the number to trust")

b1, b2 = fit(ALL)
print("\n=== fitted on ALL test bodies (optimistic, for reference) ===")
print(f"2-digit: {acc(np.zeros(10), np.zeros(11), ALL):.2f} -> {acc(b1, b2, ALL):.2f}")
print("digit1 bias:", " ".join(f"{v:+.3f}" for v in b1))
print("digit2 bias:", " ".join(f"{v:+.3f}" for v in b2), "(last = blank)")
