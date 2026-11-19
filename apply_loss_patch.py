"""
Programmatically patches utils/loss.py to add a 1.3x loss weight for the
confusable digit classes {1,2,3,5,6,7,8,9} to both the digit_1 (criterion1)
and digit_2 (criterion2) CrossEntropyLoss heads.

Uses a REGEX match (tolerant of whitespace/tabs) on the block that builds
self.criterion1 / self.criterion2, right after self.state_criterion is
built. Makes a timestamped .bak copy before writing.

Usage:
    python apply_loss_patch.py utils/loss.py

If the match fails, the script prints what it searched for and makes NO
changes -- safe to re-run after the match pattern is corrected.
"""
import re
import sys
import shutil
import time

# Matches the block, in order, whitespace-insensitive:
#   self.criterion1 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
#   d2_weights = torch.ones(11)
#   d2_weights[10] = blank_weight    # 10 = blank
#   self.criterion2 = nn.CrossEntropyLoss(weight = d2_weights, ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)
# Any commented-out lines directly above (e.g. "# self.criterion1 = ...")
# are tolerated and left in place (not part of the match).
PATTERN = re.compile(
    r"self\.criterion1\s*=\s*nn\.CrossEntropyLoss\(\s*ignore_index\s*=\s*IGNORE_INDEX\s*,"
    r"\s*label_smoothing\s*=\s*label_smoothing\s*\)\s*\n"
    r"\s*d2_weights\s*=\s*torch\.ones\(\s*11\s*\)\s*\n"
    r"\s*d2_weights\[\s*10\s*\]\s*=\s*blank_weight[^\n]*\n"
    r"\s*self\.criterion2\s*=\s*nn\.CrossEntropyLoss\(\s*weight\s*=\s*d2_weights\s*,"
    r"\s*ignore_index\s*=\s*IGNORE_INDEX\s*,\s*\n?"
    r"\s*label_smoothing\s*=\s*label_smoothing\s*\)",
    re.DOTALL,
)


def build_new_block(indent):
    # indent = the leading whitespace captured from the original
    # self.criterion1 line, reused so the patched block matches the
    # surrounding code's indentation style exactly (tabs or spaces).
    lines = [
        "# ---------------HCL_Changes----------------",
        "# Confusable-digit reweighting: validation-set failure analysis",
        "# (inspect_small_scale_failures_v3.py, run on",
        "# JerseyNumber_Validation_Dataset) found the dominant error pattern",
        "# is genuine visual-shape confusion between specific digit pairs:",
        "#   7<->1 (71 total), 6->8 (20), 3->2 (19), 9->8 (18),",
        "#   2->1 (18), 5->6 (18)",
        "# Per-class error rates were also highest for 7 (23.5%), 4 (18.7%),",
        "# 3 (18.4%), 5 (18.0%). Class BALANCE in training data was checked",
        "# and found even (3700-4200 instances per digit), ruling out a",
        "# frequency-imbalance explanation -- this is a confusability issue,",
        "# not a volume issue, so we use a flat multiplier on the implicated",
        "# classes rather than inverse-frequency weights.",
        "# digit 4 had an elevated error rate but no clear confusable",
        "# partner in the pair analysis, so it is NOT included in this first",
        "# pass -- keeping the change isolated to the confirmed confusion",
        "# pairs makes it easier to attribute any accuracy change correctly.",
        "CONFUSABLE_DIGIT_WEIGHT = 1.3",
        "CONFUSABLE_DIGITS = [1, 2, 3, 5, 6, 7, 8, 9]",
        "",
        "d1_weights = torch.ones(10)",
        "for d in CONFUSABLE_DIGITS:",
        indent + "d1_weights[d] = CONFUSABLE_DIGIT_WEIGHT",
        "",
        "d2_weights = torch.ones(11)",
        "d2_weights[10] = blank_weight    # 10 = blank (unchanged, existing behavior)",
        "for d in CONFUSABLE_DIGITS:",
        indent + "d2_weights[d] = CONFUSABLE_DIGIT_WEIGHT",
        "# ---------------------------------------------",
        "",
        "self.criterion1 = nn.CrossEntropyLoss(weight=d1_weights, ignore_index=IGNORE_INDEX,",
        indent + "                           label_smoothing=label_smoothing)",
        "self.criterion2 = nn.CrossEntropyLoss(weight=d2_weights, ignore_index=IGNORE_INDEX,",
        indent + "                           label_smoothing=label_smoothing)",
    ]
    return "\n".join(indent + ln if ln else "" for ln in lines)


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_loss_patch.py <path_to_loss.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    matches = list(PATTERN.finditer(content))

    if not matches:
        print("ERROR: no match for the criterion1/criterion2 block was found.")
        print("No changes made.")
        print()
        print("This regex looks for, in order (whitespace-insensitive):")
        print("  self.criterion1 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)")
        print("  d2_weights = torch.ones(11)")
        print("  d2_weights[10] = blank_weight    # 10 = blank")
        print("  self.criterion2 = nn.CrossEntropyLoss(weight = d2_weights, ignore_index=IGNORE_INDEX,")
        print("                                         label_smoothing=label_smoothing)")
        print()
        print("Run this and paste the output so the pattern can be corrected:")
        print("  grep -n -B2 -A6 'self.criterion1' <your_loss.py>")
        sys.exit(1)

    if len(matches) > 1:
        print(f"ERROR: found {len(matches)} matches, expected exactly 1. "
              "Refusing to patch to avoid changing the wrong spot. "
              "Tell me and I'll make the match more specific.")
        sys.exit(1)

    m = matches[0]
    matched_text = m.group(0)

    # figure out the indentation used on the self.criterion1 line, so the
    # new block lines up with the surrounding code (this file appears to
    # use tabs elsewhere, but we detect it directly rather than assume)
    line_start = content.rfind("\n", 0, m.start()) + 1
    indent_match = re.match(r"[ \t]*", content[line_start:m.start()])
    indent = indent_match.group(0) if indent_match else ""
    # m.start() should already be at the start of "self.criterion1" on its
    # own line in the common case; if there's leading whitespace captured
    # by \s* at the front of the pattern it's harmless since DOTALL \s*
    # before "self" isn't in this particular pattern -- indent comes from
    # the raw text before the match on that line.
    if not indent:
        # self.criterion1 might not be at column 0; re-derive indent from
        # the first line of the match itself if it starts mid-line context
        first_line = matched_text.splitlines()[0]
        indent_match2 = re.match(r"[ \t]*", first_line)
        indent = indent_match2.group(0) if indent_match2 else "        "

    print(f"Matched criterion1/criterion2 block. Detected indent: {repr(indent)}")

    new_block = build_new_block(indent)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_content = content[: line_start] + new_block + content[m.end():]
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: criterion1 now weighted (digits 1,2,3,5,6,7,8,9 get 1.3x), "
          "criterion2 extended beyond blank-only weighting to also weight those digits 1.3x.")


if __name__ == "__main__":
    main()
