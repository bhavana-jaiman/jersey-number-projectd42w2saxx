"""
add_dump.py -- makes a COPY of the testing script that also saves the raw model outputs
(state / digit-1 / digit-2 logits) of every test body to a .pkl file.
The original testing script is NOT changed.

Run (in workspace_16_10_2026, where the testing script is):
  python add_dump.py
  -> creates jerseyNumberRecognitiorWithState_Testing_dump.py
"""
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "jerseyNumberRecognitiorWithState_Testing.py"
DST = SRC.replace(".py", "_dump.py")
lines = open(SRC, encoding="utf8").read().split("\n")


def insert(anchor, new, after=True):
    idx = [i for i, l in enumerate(lines) if anchor in l and not l.lstrip().startswith("#")]
    if len(idx) != 1:
        sys.exit(f"ERROR: line '{anchor}' found {len(idx)} times (need exactly 1) -> send me this message")
    i = idx[0]
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    pos = i + 1 if after else i
    lines[pos:pos] = [ind + x for x in new]


insert("opt = parser.parse_args()", [
    'parser.add_argument("--dump", type=str, default="test_outputs.pkl", help="where to save raw outputs")',
], after=False)
insert("opt = parser.parse_args()", ["dump = []"], after=True)

insert("_, pred_d2 = torch.max(logits_d2, 1)", [
    "dump.append(dict(img=os.path.basename(img), n=len(all_jersey_numbers),",
    "                 d1=int(jersey_number[0]), d2=int(jersey_number[1]),",
    "                 w=float(body_rect[2]), h=float(body_rect[3]),",
    "                 ls=logits_state[0].float().cpu().numpy(),",
    "                 l1=logits_d1[0].float().cpu().numpy(),",
    "                 l2=logits_d2[0].float().cpu().numpy()))",
], after=True)

insert('print("----")', [
    "import pickle",
    "pickle.dump(dump, open(opt.dump, 'wb'))",
    "print('saved raw outputs of', len(dump), 'bodies ->', opt.dump)",
], after=False)

open(DST, "w", encoding="utf8").write("\n".join(lines))
print("created", DST)
print("run it exactly like the normal testing script, plus:  --dump <name>.pkl")
