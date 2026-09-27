import sys
import pandas as pd

# usage: python analyze_errors.py output/exp6/wrong_predictions.csv
path = sys.argv[1] if len(sys.argv) > 1 else "output/exp6/wrong_predictions.csv"
df = pd.read_csv(path)
BLANK = 10

print(f"Total wrong rows: {len(df)}\n")

# ---- state errors ----
print("State errors (true -> predicted):")
state_err = df[df.true_state != df.pred_state]
print(state_err.groupby(["true_state", "pred_state"]).size().to_string(), "\n")

# ---- digit errors: only rows where state is 1 and state was predicted right ----
d = df[(df.true_state == 1) & (df.pred_state == 1)]
two = d[d.true_d2 != BLANK]   # true 2-digit numbers
one = d[d.true_d2 == BLANK]   # true 1-digit numbers

swapped = two[(two.pred_d1 == two.true_d2) & (two.pred_d2 == two.true_d1)]
missed_2nd = two[two.pred_d2 == BLANK]
only_d1 = two[(two.pred_d1 != two.true_d1) & (two.pred_d2 == two.true_d2)]
only_d2 = two[(two.pred_d1 == two.true_d1) & (two.pred_d2 != two.true_d2)]

print("2-digit numbers wrong:", len(two))
print("  order swapped (e.g. 23 read as 32):", len(swapped))
print("  2nd digit predicted as blank      :", len(missed_2nd))
print("  only digit1 wrong                 :", len(only_d1))
print("  only digit2 wrong                 :", len(only_d2))
print()

extra_2nd = one[one.pred_d2 != BLANK]
print("1-digit numbers wrong:", len(one))
print("  model added a 2nd digit           :", len(extra_2nd))
print("  digit1 wrong                      :", len(one[one.pred_d1 != one.true_d1]))
print()

print("Top 10 digit1 confusions (true -> pred):")
w1 = d[d.pred_d1 != d.true_d1]
print(w1.groupby(["true_d1", "pred_d1"]).size().sort_values(ascending=False).head(10).to_string(), "\n")

print("Top 10 digit2 confusions (true -> pred):")
w2 = d[d.pred_d2 != d.true_d2]
print(w2.groupby(["true_d2", "pred_d2"]).size().sort_values(ascending=False).head(10).to_string())
