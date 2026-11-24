"""
Adds a --super_high_res CLI flag to jerseyNumberRecognitior_Testing.py (the
"no WithState" script, which -- despite its name -- already imports and uses
MultiTaskLearnerWithState, per the HCL_Changes comment block at the top of
the file). This lets you test the 24x24 teacher checkpoint
(trained with --high_res --super_high_res) with THIS script too, the same
way we already did for jerseyNumberRecognitiorWithState_Testing.py.

Confirmed via live grep (this session) the EXACT current code:

    line 110:
        "--high_res", action="store_true", help="12x12 feature map model"

    line 115:
        model = MultiTaskLearnerWithState(high_res=opt.high_res).cuda()

(Note: this script uses .cuda() directly, not .to(device) -- the patch
preserves that style exactly, it does not change it.)

Changes:
  1. New --super_high_res argparse flag (mirrors --high_res, same style).
  2. model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res).cuda()

Anchors are found by regex (tolerant of the actual quote char/whitespace in
the file, not a hardcoded copy-pasted literal) and the real indentation of
each insertion point is detected from the file itself (not assumed), same
approach used successfully for the two previous CLI patches in this project.

Usage:
    python apply_nostate_superhighres_patch.py jerseyNumberRecognitior_Testing.py
"""
import re
import sys
import shutil
import time


def backup(path):
    bpath = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, bpath)
    print(f"Backup saved to: {bpath}")
    return bpath


def leading_indent_of_match(full_content, match_obj):
    start = match_obj.start()
    line_start = full_content.rfind("\n", 0, start) + 1
    prefix = full_content[line_start:start]
    return prefix if prefix.strip() == "" else ""


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_nostate_superhighres_patch.py <path_to_jerseyNumberRecognitior_Testing.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "--super_high_res" in content and "super_high_res=opt.super_high_res" in content:
        print(f"{path}: already patched. No changes made.")
        return

    # --- Anchor 1: the --high_res argparse line ---
    # Matches: "--high_res", action="store_true", help="12x12 feature map model"
    # (quote-tolerant via backreference, and tolerant of single- vs double-quotes)
    HIGH_RES_RE = re.compile(
        r'(.)--high_res\1,\s*action=(.)store_true\2,\s*help=(.)12x12 feature map model\3'
    )
    hr_matches = list(HIGH_RES_RE.finditer(content))
    if len(hr_matches) != 1:
        print(f"ERROR: --high_res argparse line found {len(hr_matches)} times via regex, expected 1. No changes made.")
        sys.exit(1)
    hr_m = hr_matches[0]
    hr_old_text = hr_m.group(0)
    quote = hr_m.group(1)
    hr_line_indent = leading_indent_of_match(content, hr_m)

    # --- Anchor 2: the model construction line ---
    MODEL_CTOR_RE = re.compile(
        r'model = MultiTaskLearnerWithState\(high_res=opt\.high_res\)\.cuda\(\)'
    )
    model_matches = list(MODEL_CTOR_RE.finditer(content))
    if len(model_matches) != 1:
        print(f"ERROR: model construction line found {len(model_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    model_old_text = model_matches[0].group(0)

    print("Anchor found. Detected quote char:", repr(quote))
    print("Detected --high_res line's own indent:", repr(hr_line_indent))

    q = quote
    super_high_res_line = (
        f'{hr_line_indent}{q}--super_high_res{q}, action={q}store_true{q}, '
        f'help={q}24x24 feature map model (stack with --high_res){q}'
    )

    new_model_ctor = (
        'model = MultiTaskLearnerWithState(high_res=opt.high_res, '
        'super_high_res=opt.super_high_res).cuda()'
    )

    # Find the start of the ENCLOSING parser.add_argument(...) statement,
    # not just the line the matched fragment happens to sit on -- the real
    # --high_res call may span multiple lines (it does, confirmed by a test
    # fixture mirroring the real file: the call opens on one line and the
    # "--high_res", action=..., help=... fragment is indented on the next).
    # We scan backward from the match to the nearest "parser.add_argument("
    # that precedes it with only whitespace/newlines in between.
    ADD_ARG_OPEN_RE = re.compile(r'parser\.add_argument\(\s*$')
    start = hr_m.start()
    search_region = content[:start]
    open_matches = list(ADD_ARG_OPEN_RE.finditer(search_region))
    if open_matches:
        stmt_start = open_matches[-1].start()
    else:
        # Fallback: the --high_res fragment is on the same line as
        # "parser.add_argument(" itself (single-line call). Use the start
        # of that line.
        stmt_start = content.rfind("\n", 0, start) + 1

    # Indent of the enclosing statement itself (not the fragment's own
    # possibly-deeper indent if it's on a continuation line).
    stmt_line_start = content.rfind("\n", 0, stmt_start) + 1
    stmt_indent_prefix = content[stmt_line_start:stmt_start]
    stmt_indent = stmt_indent_prefix if stmt_indent_prefix.strip() == "" else hr_line_indent

    backup(path)

    new_content = content
    insertion = (
        f'{stmt_indent}parser.add_argument({super_high_res_line.strip()})\n'
    )
    new_content = new_content[:stmt_line_start] + insertion + new_content[stmt_line_start:]

    new_content = new_content.replace(model_old_text, new_model_ctor, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("  New flag: --super_high_res")
    print("  model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res).cuda()")
    print()
    print("Next: python -c \"import ast; ast.parse(open('" + path + "').read()); print('SYNTAX_OK')\"")


if __name__ == "__main__":
    main()
