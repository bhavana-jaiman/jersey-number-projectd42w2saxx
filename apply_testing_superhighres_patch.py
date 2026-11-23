"""
Adds a --super_high_res CLI flag to jerseyNumberRecognitiorWithState_Testing.py
(the companion to apply_cli_superhighres_patch_v3.py, which already did this
for the TRAINING script). Needed so you can later test against a checkpoint
trained with --high_res --super_high_res (the new 24x24 teacher) -- without
this, MultiTaskLearnerWithState is built as a 6x6/12x12 model and
load_state_dict(..., strict=True) will fail to match the 24x24 checkpoint's
weight shapes.

Confirmed via live grep (this session) the EXACT current code:

    parser.add_argument("--high_res", action="store_true",
                help="12x12 feature map model")

    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = MultiTaskLearnerWithState().to(device)
    model = MultiTaskLearnerWithState(high_res=opt.high_res).to(device)
    #model = MultiTaskLearner().to(device)

Changes:
  1. New --super_high_res argparse flag (mirrors --high_res, same style).
  2. model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res).to(device)

Anchors are found by regex (tolerant of the actual quote char/whitespace in
the file, not a hardcoded copy-pasted literal), same approach used for the
training script patch after the first hardcoded-literal attempt failed.

Usage:
    python apply_testing_superhighres_patch.py jerseyNumberRecognitiorWithState_Testing.py
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
        print("Usage: python apply_testing_superhighres_patch.py <path_to_testing_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "--super_high_res" in content and "super_high_res=opt.super_high_res" in content:
        print(f"{path}: already patched. No changes made.")
        return

    # --- Anchor 1: the --high_res argparse block ---
    HIGH_RES_RE = re.compile(
        r'parser\.add_argument\((.)--high_res\1,\s*action=(.)store_true\2,\n'
        r'([ \t]*)help=(.)12x12 feature map model\4\)'
    )
    hr_matches = list(HIGH_RES_RE.finditer(content))
    if len(hr_matches) != 1:
        print(f"ERROR: --high_res argparse block found {len(hr_matches)} times via regex, expected 1. No changes made.")
        sys.exit(1)
    hr_m = hr_matches[0]
    hr_old_text = hr_m.group(0)
    quote = hr_m.group(1)
    help_indent = hr_m.group(3)
    hr_line_indent = leading_indent_of_match(content, hr_m)

    # --- Anchor 2: the model construction line ---
    MODEL_CTOR_RE = re.compile(
        r'model = MultiTaskLearnerWithState\(high_res=opt\.high_res\)\.to\(device\)'
    )
    model_matches = list(MODEL_CTOR_RE.finditer(content))
    if len(model_matches) != 1:
        print(f"ERROR: model construction line found {len(model_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    model_old_text = model_matches[0].group(0)

    print("Anchor found. Detected quote char:", repr(quote))
    print("Detected --high_res line's own indent:", repr(hr_line_indent))
    print("Detected help= continuation indent:", repr(help_indent))

    q = quote
    super_high_res_block = (
        f'{hr_line_indent}parser.add_argument({q}--super_high_res{q}, action={q}store_true{q},\n'
        f'{help_indent}help={q}24x24 feature map model (stack with --high_res){q})'
    )

    new_model_ctor = (
        'model = MultiTaskLearnerWithState(high_res=opt.high_res, '
        'super_high_res=opt.super_high_res).to(device)'
    )

    backup(path)

    new_content = content
    new_content = new_content.replace(hr_old_text, hr_old_text + "\n" + super_high_res_block, 1)
    new_content = new_content.replace(model_old_text, new_model_ctor, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("  New flag: --super_high_res")
    print("  model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res)")
    print()
    print("Next: python -c \"import ast; ast.parse(open('" + path + "').read()); print('SYNTAX_OK')\"")


if __name__ == "__main__":
    main()
