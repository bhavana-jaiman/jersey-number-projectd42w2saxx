"""
calibrate_state.py -- reads the .pkl written by ..._Testing_dump.py and shows how the
4 test accuracies change when a constant is added to
  * the State-2 logit      (more / fewer "3+ digit" predictions)
  * the digit-2 blank logit (class 10: more / fewer "no second digit")
No model and no GPU needed; runs in seconds.

A bias on a logit = changing one bias value of the last layer -> 0 extra FLOPs.

Fair check: the bias is CHOSEN on half A of the test images and REPORTED on half B,
so the result is not only tuned on the images it is measured on.

Run:
  python calibrate_state.py --pkl test_79.74.pkl
"""
import argparse
import pickle
import zlib

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True)
opt = ap.parse_args()

D = pickle.load(open(opt.pkl, "rb"))
n = np.array([d["n"] for d in D])
d1 = np.array([d["d1"] for d in D])
d2 = np.array([d["d2"] for d in D])
h = np.array([d["h"] for d in D])
LS = np.stack([d["ls"] for d in D])
L1 = np.stack([d["l1"] for d in D])
L2 = np.stack([d["l2"] for d in D])
half = np.array([zlib.crc32(d["img"].encode()) % 2 for d in D])   # fixed split by image
true_s = np.where(n > 2, 2, 1)              # same rule as the testing script
ALL = np.ones(len(n), bool)


def pct(x):
    return 100.0 * x.mean() if x.size else float("nan")


def metrics(b2=0.0, bb=0.0, m=ALL):
    ls = LS.copy(); ls[:, 2] += b2
    l2 = L2.copy(); l2[:, 10] += bb
    ps, p1, p2 = ls.argmax(1), L1.argmax(1), l2.argmax(1)
    ok = ps == true_s
    two = m & (n <= 2)
    return dict(digit2=pct(((p1 == d1) & (p2 == d2))[two]), state=pct(ok[m]),
                s2=pct(ok[m & (n > 2)]), s1=pct(ok[two]))


def row(tag, r):
    low = min(r.values())
    print(f"{tag:<22}{r['digit2']:8.2f}{r['state']:8.2f}{r['s2']:8.2f}{r['s1']:8.2f}{low:8.2f}"
          + ("   all >= 80" if low >= 80 else ""))


HDR = f"{'':<22}{'2-digit':>8}{'state':>8}{'S2':>8}{'S1':>8}{'min':>8}"

print(f"bodies: {len(n)}  | 1-2 digit: {(n <= 2).sum()}  | 3+ digit: {(n > 2).sum()}  | 0 digit: {(n == 0).sum()}")
print("\n=== 1. baseline (must match the normal testing script) ===")
print(HDR); row("no bias", metrics())

ps = LS.argmax(1)
print("\n=== 2. test state confusion (rows = true, cols = predicted 0 / 1 / 2) ===")
for s in (1, 2):
    r = [(ps[true_s == s] == c).sum() for c in (0, 1, 2)]
    print(f"  true {s}: {r[0]:6d} {r[1]:6d} {r[2]:6d}")

print("\n=== 3. State-2 accuracy by body height (test) ===")
s2 = n > 2
if s2.sum():
    edges = np.percentile(h[s2], [0, 25, 50, 75, 100])
    for a, b in zip(edges[:-1], edges[1:]):
        m = s2 & (h >= a) & (h <= b)
        print(f"  height {a:6.0f}-{b:6.0f}px  n={m.sum():4d}  S2 acc {pct(ps[m] == 2):6.2f}")
    for k in sorted(set(n[s2])):
        m = s2 & (n == k)
        print(f"  {k} digits: n={m.sum():4d}  S2 acc {pct(ps[m] == 2):6.2f}")

print("\n=== 4. sweep: bias on State-2 logit (all test images) ===")
print(HDR)
B2 = np.round(np.arange(-1.0, 8.01, 0.25), 2)
for b in B2:
    row(f"state2 bias {b:+.2f}", metrics(b2=b))

print("\n=== 5. sweep: bias on digit-2 blank logit (only changes 2-digit acc) ===")
BB = np.round(np.arange(-2.0, 3.01, 0.25), 2)
for b in BB:
    print(f"  blank bias {b:+.2f}   2-digit {metrics(bb=b)['digit2']:.2f}")

print("\n=== 6. fair check: choose on half A, report on half B ===")
A, B = half == 0, half == 1
best_b2 = max(B2, key=lambda b: min(v for k, v in metrics(b2=b, m=A).items() if k != "digit2"))
best_bb = max(BB, key=lambda b: metrics(bb=b, m=A)["digit2"])
print(f"chosen on half A: state2 bias {best_b2:+.2f}, blank bias {best_bb:+.2f}")
print(HDR)
row("half B, no bias", metrics(m=B))
row("half B, with bias", metrics(best_b2, best_bb, m=B))
row("all, with bias", metrics(best_b2, best_bb))
