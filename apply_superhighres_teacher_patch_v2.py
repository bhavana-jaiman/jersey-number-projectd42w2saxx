"""
Programmatically patches subModules/backbone_ying.py to add a THIRD
feature-map-size mode to featureExtractor_Ying: a 24x24 "super_high_res"
teacher, alongside the existing 6x6 student (high_res=False) and 12x12
teacher (high_res=True) modes.

Why: three data-reweighting experiments this session (confusable-digit
loss/oversampling x2, then scale-based oversampling) all made strict
2-digit accuracy WORSE than the 79.74% baseline. The baseline was already
produced via knowledge distillation from a 12x12 teacher. Since the
student's own params/MACs/FLOPs must NOT increase (hard constraint), but
the TEACHER is discarded after training, the next lever is: make an even
BIGGER teacher (24x24 feature map) for richer soft labels, distilled into
the SAME unchanged student.

Mechanism: featureExtractor_Ying already does this trick ONCE, to go from
6x6 (student) to 12x12 (high_res teacher) -- block4_1's stride drops
2->1, and block4_1/block5/block5_1's dilation doubles 1->2 to compensate:

    s4, dil = (1, 2) if high_res else (2, 1)
    self.block4_1 = block(64, 96, s4, 4, 0, 1)
    self.block5   = block(96, 96, 1, 2, 0, 0, dilation=dil)
    self.block5_1 = block(96, 96, 1, 2, 0, 0, dilation=dil)

This applies the SAME trick one stage earlier (block3_1, always stride=2
in the original, taking 24px->12px), gated on a NEW `super_high_res`
flag, with the existing high_res dilation COMPOUNDING the new one:

    s3, dil3 = (1, 2) if super_high_res else (2, 1)
    self.block3_1 = block(32, 64, s3, 3, 0, 1, dilation=dil3)
    self.block4   = block(64, 64, 1, 2, 0, 0, dilation=dil3)

    s4, dil4_base = (1, 2) if high_res else (2, 1)
    dil4 = dil4_base * dil3
    self.block4_1 = block(64, 96, s4, 4, 0, 1, dilation=dil4)
    self.block5   = block(96, 96, 1, 2, 0, 0, dilation=dil4)
    self.block5_1 = block(96, 96, 1, 2, 0, 0, dilation=dil4)

Truth table:
    high_res=False, super_high_res=False -> original student,      6x6  (UNCHANGED)
    high_res=True,  super_high_res=False -> existing 12x12 teacher,12x12 (UNCHANGED)
    high_res=True,  super_high_res=True  -> NEW teacher,          24x24

Design note: this version uses TWO small, separately-anchored regex
patches instead of one large multi-line block match, specifically to
avoid swallowing/duplicating surrounding indentation or lines (a bug
found and fixed during testing of an earlier, more ambitious single-block
version of this script). Patch A only touches the self.block3_1 = ...
line (adds dilation=dil3 via a one-line replace, after inserting the s3/
dil3 computation immediately before it). Patch B only touches the
existing 3-line s4/dil/.../block4_1/block5/block5_1 HCL_Changes block
(replaces "dil" with the new compounded "dil4" throughout, and inserts
the dil4_base/dil4 computation). Patch C adds dilation=dil3 to the
self.block4 = block(64, 64, 1, 2, 0, 0) line specifically (the ONE
occurrence between block3_1 and the HCL_Changes marker).

Usage:
    python apply_superhighres_teacher_patch.py subModules/backbone_ying.py
"""
import re
import sys
import shutil
import time

SIG_PATTERN = re.compile(
    r"def __init__\(self, out_channels=256, high_res=False\) -> None:"
)
SIG_NEW = "def __init__(self, out_channels=256, high_res=False, super_high_res=False) -> None:"

# Patch A: the block3_1 construction line, anchored tightly (must be the
# ONLY line matching this exact original text in the file).
BLOCK3_1_OLD = "self.block3_1 = block(32, 64, 2, 3, 0, 1)"
BLOCK3_1_NEW_TEMPLATE = (
    "s3, dil3 = (1, 2) if super_high_res else (2, 1)\n"
    "{indent}self.block3_1 = block(32, 64, s3, 3, 0, 1, dilation=dil3)"
)

# Patch B: the self.block4 construction line IMMEDIATELY following
# block3_1 (original, no dilation kwarg yet). Anchored by exact original
# text; must be the only such occurrence.
BLOCK4_OLD = "self.block4 = block(64, 64, 1, 2, 0, 0)"
BLOCK4_NEW = "self.block4 = block(64, 64, 1, 2, 0, 0, dilation=dil3)"

# Patch C: the existing HCL_Changes 4-line block (s4/dil + block4_1/block5/
# block5_1), matched tightly and whitespace-tolerantly, replaced with the
# dil4-compounding version.
HCL_OLD_PATTERN = re.compile(
    r"s4,\s*dil\s*=\s*\(1,\s*2\)\s*if\s*high_res\s*else\s*\(2,\s*1\)\s*\n"
    r"(?P<indent>[ \t]*)self\.block4_1\s*=\s*block\(64,\s*96,\s*s4,\s*4,\s*0,\s*1\)\s*\n"
    r"[ \t]*self\.block5\s*=\s*block\(96,\s*96,\s*1,\s*2,\s*0,\s*0,\s*dilation\s*=\s*dil\)\s*\n"
    r"[ \t]*self\.block5_1\s*=\s*block\(96,\s*96,\s*1,\s*2,\s*0,\s*0,\s*dilation\s*=\s*dil\)"
)


def build_hcl_new(indent):
    return (
        "s4, dil4_base = (1, 2) if high_res else (2, 1)\n"
        f"{indent}dil4 = dil4_base * dil3\n"
        f"{indent}self.block4_1 = block(64, 96, s4, 4, 0, 1, dilation=dil4)\n"
        f"{indent}self.block5 = block(96, 96, 1, 2, 0, 0, dilation=dil4)\n"
        f"{indent}self.block5_1 = block(96, 96, 1, 2, 0, 0, dilation=dil4)"
    )


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_superhighres_teacher_patch.py <path_to_backbone_ying.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "super_high_res" in content and "dil4 = dil4_base * dil3" in content:
        print("File already fully patched (super_high_res + dil4 compounding present). No changes made.")
        return

    # --- Scope to ONLY the featureExtractor_Ying class, not featureExtractor_Ying_Small ---
    # (confirmed via live grep: MultiTaskLearnerWithState -- the model actually
    # used for training -- instantiates featureExtractor_Ying, not _Small; both
    # classes contain textually-identical lines like "self.block3_1 = block(32,
    # 64, 2, 3, 0, 1)", so every anchor below is restricted to the slice of the
    # file between "class featureExtractor_Ying(" and the next top-level class.)
    class_start_m = re.search(r"^class featureExtractor_Ying\(nn\.Module\):", content, re.MULTILINE)
    if not class_start_m:
        print("ERROR: could not find 'class featureExtractor_Ying(nn.Module):'. No changes made.")
        print("Run: grep -n 'class featureExtractor_Ying' subModules/backbone_ying.py")
        sys.exit(1)
    # next top-level class after it (e.g. featureExtractor_Ying_Small) bounds the slice
    next_class_m = re.search(r"^class \w+", content[class_start_m.end():], re.MULTILINE)
    class_start = class_start_m.start()
    class_end = class_start_m.end() + (next_class_m.start() if next_class_m else len(content) - class_start_m.end())
    scope = content[class_start:class_end]
    print(f"Scoped to featureExtractor_Ying class only: chars [{class_start}:{class_end}] "
          f"({class_end - class_start} chars, excludes featureExtractor_Ying_Small).")

    # --- Checks: each anchor must appear EXACTLY ONCE within that scope ---
    sig_count = len(SIG_PATTERN.findall(scope))
    if sig_count != 1:
        print(f"ERROR: __init__ signature pattern found {sig_count} times within featureExtractor_Ying, expected 1. No changes made.")
        sys.exit(1)

    block3_1_count = scope.count(BLOCK3_1_OLD)
    if block3_1_count != 1:
        print(f"ERROR: '{BLOCK3_1_OLD}' found {block3_1_count} times within featureExtractor_Ying, expected exactly 1. No changes made.")
        sys.exit(1)

    block4_count = scope.count(BLOCK4_OLD)
    if block4_count != 1:
        print(f"ERROR: '{BLOCK4_OLD}' found {block4_count} times within featureExtractor_Ying, expected exactly 1. No changes made.")
        sys.exit(1)

    hcl_matches = list(HCL_OLD_PATTERN.finditer(scope))
    if len(hcl_matches) != 1:
        print(f"ERROR: HCL_Changes s4/dil block found {len(hcl_matches)} times within featureExtractor_Ying, expected exactly 1. No changes made.")
        sys.exit(1)

    # Detect real indent from the block3_1 line's own leading whitespace (within scope).
    idx = scope.index(BLOCK3_1_OLD)
    line_start = scope.rfind("\n", 0, idx) + 1
    indent_match = re.match(r"[ \t]*", scope[line_start:idx])
    indent = indent_match.group(0) if indent_match else "        "

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_scope = scope

    # Patch 1: signature
    new_scope = SIG_PATTERN.sub(SIG_NEW, new_scope, count=1)

    # Patch 2 (Patch A): block3_1 -- insert s3/dil3 line before it, add dilation kwarg
    block3_1_new = BLOCK3_1_NEW_TEMPLATE.format(indent=indent)
    new_scope = new_scope.replace(BLOCK3_1_OLD, block3_1_new, 1)

    # Patch 3 (Patch B): block4 -- add dilation=dil3
    new_scope = new_scope.replace(BLOCK4_OLD, BLOCK4_NEW, 1)

    # Patch 4 (Patch C): HCL_Changes block -- compound dilation
    hcl_m = HCL_OLD_PATTERN.search(new_scope)
    hcl_indent = hcl_m.group("indent")
    hcl_new = build_hcl_new(hcl_indent)
    new_scope = HCL_OLD_PATTERN.sub(lambda _m: hcl_new.replace("\\", "\\\\"), new_scope, count=1)

    # Reassemble: only the featureExtractor_Ying slice changes; everything
    # before class_start and after class_end (including featureExtractor_Ying_Small
    # in full, untouched) is preserved exactly.
    new_content = content[:class_start] + new_scope + content[class_end:]

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print()
    print("Truth table:")
    print("  high_res=False, super_high_res=False -> original student,        6x6  (UNCHANGED)")
    print("  high_res=True,  super_high_res=False -> existing teacher,       12x12 (UNCHANGED)")
    print("  high_res=True,  super_high_res=True  -> NEW teacher,            24x24")


if __name__ == "__main__":
    main()
