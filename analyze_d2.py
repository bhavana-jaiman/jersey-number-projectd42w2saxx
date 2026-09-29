import pandas as pd

CSV = "wrong_predictions.csv"          # path printed by test_New_2.py
N_ONE, N_TWO = 1265, 1776              # from your test output (1-digit n, 2-digit n)

df = pd.read_csv(CSV, dtype={"true_digits": str})
s1 = df[df.true_state == 1].copy()
s1["n"] = s1.true_digits.str.len()

# ---- 2-digit numbers: what does digit2 do wrong? ----
two = s1[s1.n == 2]
d2w = two[two.pred_d2 != two.true_d2]
missed  = d2w.pred_d2 == 10                                           # says "no 2nd digit"
copied  = d2w.pred_d2 == d2w.true_d1                                  # repeats the 1st digit
swapped = (d2w.pred_d1 == d2w.true_d2) & (d2w.pred_d2 == d2w.true_d1) # order reversed
misread = ~(missed | copied)

print(f"2-digit bodies: {N_TWO},  digit2 wrong: {len(d2w)} ({100*len(d2w)/N_TWO:.1f}%)")
print(f"  missed 2nd digit (pred blank) : {missed.sum()}")
print(f"  copied digit1 into digit2     : {copied.sum()}")
print(f"  swapped order                 : {swapped.sum()}")
print(f"  misread (wrong digit)         : {misread.sum()}")

# ---- 1-digit numbers: does digit2 invent a 2nd digit? ----
one = s1[s1.n == 1]
extra = one[one.pred_d2 != 10]
print(f"\n1-digit bodies: {N_ONE},  digit2 invented a digit: {len(extra)} ({100*len(extra)/N_ONE:.1f}%)")
print("  invented digits:", extra.pred_d2.value_counts().to_dict())

# ---- which digits get confused (true -> pred) ----
print("\nMost common digit2 misreads (true -> pred):")
mr = d2w[misread]
print(mr.groupby(["true_d2", "pred_d2"]).size().sort_values(ascending=False).head(10))
