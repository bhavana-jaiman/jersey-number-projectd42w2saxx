"""
v2 of the final wiring patch for jerseyNumberRecognitiorWithState_Training.py
(utils/kd.py is ALREADY patched successfully -- this script does NOT touch it).

Why v2: the first attempt hardcoded straight ASCII quotes '"--high_res"' as
the match anchor and failed with "substring not found". The real file (per
dump_argparse_block.py's repr() output) actually uses the same block, byte
for byte, matching a search on the raw bytes -- but to be safe against any
further transcription risk, this version finds every anchor by SEARCHING
the real file's content with regex (so it works regardless of exact quote
glyphs/whitespace) instead of hardcoding a literal string copied from a
photo, then uses the ACTUAL matched text (not a retyped copy) as the base
for building the replacement. This removes the whole class of "my
transcription doesn't match the file" bugs.

Confirmed via dump_argparse_block.py's real repr() output:
    343: parser.add_argument("--high_res",
    344:         action="store_true",
    345:         help="12x12 feature map model (block4_1 stride 1)"
    346:         )
    347: parser.add_argument("--teacher",
    348:         type=str,
    349:         default="",
    350:         help="12x12 checkpoint(trained with --high_res) model (block4_1 stride 1)"
    351:         )
    352: parser.add_argument("--kd_alpha",
    ...
    461: model = MultiTaskLearnerWithState(high_res=opt.high_res).to(device)
    462: teacher = load_teacher(opt.teacher, high_res=True, device=device) if opt.teacher else None
    463: if teacher is not None:
    464:     print(f"[KD] teacher loaded: {opt.teacher}  alpha={opt.kd_alpha} T={opt.kd_T}")

Usage:
    python apply_cli_superhighres_patch_v2.py jerseyNumberRecognitiorWithState_Training.py
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_cli_superhighres_patch_v2.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "--super_high_res" in content and "--teacher_super_high_res" in content:
        print(f"{path}: already patched. No changes made.")
        return

    # --- Anchor 1: the --high_res parser.add_argument(...) block, found by
    # regex (tolerant of whatever quote character is actually in the file),
    # captured as a single string so the ORIGINAL bytes are reused verbatim. ---
    HIGH_RES_RE = re.compile(
        r'parser\.add_argument\((.)--high_res\1,\n'
        r'([ \t]*)action=(.)store_true\3,\n'
        r'[ \t]*help=(.)12x12 feature map model \(block4_1 stride 1\)\4\n'
        r'[ \t]*\)'
    )
    hr_matches = list(HIGH_RES_RE.finditer(content))
    if len(hr_matches) != 1:
        print(f"ERROR: --high_res argparse block found {len(hr_matches)} times via regex, expected 1. No changes made.")
        if not hr_matches:
            print("Run dump_argparse_block.py again and share the exact repr() output around --high_res.")
        sys.exit(1)
    hr_m = hr_matches[0]
    hr_old_text = hr_m.group(0)
    quote = hr_m.group(1)          # the actual quote char used for arg names, e.g. " or “
    indent = hr_m.group(2)          # the actual indent used for action=/help=/)

    def leading_indent_of_match(full_content, match_obj):
        """Indent of the line the match STARTS on (e.g. 'parser.add_argument(' itself),
        which may differ from the indent used for its continuation lines (action=/help=/)).
        Found by looking backwards from match start to the previous newline -- works no
        matter what enclosing block (function/if/etc.) this argparse call lives inside,
        without needing to know or assume what that enclosing block is."""
        start = match_obj.start()
        line_start = full_content.rfind("\n", 0, start) + 1
        prefix = full_content[line_start:start]
        return prefix if prefix.strip() == "" else ""

    hr_line_indent = leading_indent_of_match(content, hr_m)

    # --- Anchor 2: the --teacher parser.add_argument(...) block ---
    TEACHER_RE = re.compile(
        r'parser\.add_argument\((.)--teacher\1,\n'
        r'([ \t]*)type=str,\n'
        r'[ \t]*default=(.)\3,\n'
        r'[ \t]*help=(.)12x12 checkpoint\(trained with --high_res\) model \(block4_1 stride 1\)\4\n'
        r'[ \t]*\)'
    )
    teacher_matches = list(TEACHER_RE.finditer(content))
    if len(teacher_matches) != 1:
        print(f"ERROR: --teacher argparse block found {len(teacher_matches)} times via regex, expected 1. No changes made.")
        sys.exit(1)
    teacher_m = teacher_matches[0]
    teacher_old_text = teacher_m.group(0)
    teacher_indent = teacher_m.group(2)
    teacher_line_indent = leading_indent_of_match(content, teacher_m)

    # --- Anchor 3: the model construction line ---
    MODEL_CTOR_RE = re.compile(
        r'model = MultiTaskLearnerWithState\(high_res=opt\.high_res\)\.to\(device\)'
    )
    model_matches = list(MODEL_CTOR_RE.finditer(content))
    if len(model_matches) != 1:
        print(f"ERROR: model construction line found {len(model_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    model_old_text = model_matches[0].group(0)

    # --- Anchor 4: the teacher-loading call ---
    TEACHER_CALL_RE = re.compile(
        r'teacher = load_teacher\(opt\.teacher, high_res=True, device=device\) if opt\.teacher else None'
    )
    tc_matches = list(TEACHER_CALL_RE.finditer(content))
    if len(tc_matches) != 1:
        print(f"ERROR: load_teacher(...) call found {len(tc_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    teacher_call_old_text = tc_matches[0].group(0)

    # --- Anchor 5: the print line ---
    PRINT_RE = re.compile(
        r'print\(f(.)\[KD\] teacher loaded: \{opt\.teacher\}  alpha=\{opt\.kd_alpha\} T=\{opt\.kd_T\}\1\)'
    )
    print_matches = list(PRINT_RE.finditer(content))
    if len(print_matches) != 1:
        print(f"ERROR: '[KD] teacher loaded' print line found {len(print_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    print_m = print_matches[0]
    print_old_text = print_m.group(0)
    print_quote = print_m.group(1)

    print("All 5 anchors found exactly once. Detected quote char:", repr(quote))
    print("Detected argparse continuation indent:", repr(indent))
    print("Detected --high_res line's own indent:", repr(hr_line_indent))
    print("Detected --teacher line's own indent:", repr(teacher_line_indent))

    # --- Build replacements, reusing the REAL matched text + REAL quote/indent.
    # Every new line -- including the "parser.add_argument(" line itself -- is
    # prefixed with the SAME indent its neighboring original line actually uses,
    # so the new block sits at the identical indentation level as --high_res /
    # --teacher, whatever enclosing function/if-block they're inside. ---
    q = quote
    closing_line_match = re.search(r'\n([ \t]*)\)$', hr_old_text)
    close_indent = closing_line_match.group(1) if closing_line_match else indent
    super_high_res_block = (
        f'{hr_line_indent}parser.add_argument({q}--super_high_res{q},\n'
        f'{indent}action={q}store_true{q},\n'
        f'{indent}help={q}24x24 feature map model (block3_1 + block4_1 both stride 1; '
        f'stack with --high_res to compound; student-side only, mirrors --high_res){q}\n'
        f'{close_indent})'
    )

    teacher_closing_match = re.search(r'\n([ \t]*)\)$', teacher_old_text)
    teacher_close_indent = teacher_closing_match.group(1) if teacher_closing_match else teacher_indent
    teacher_super_high_res_block = (
        f'{teacher_line_indent}parser.add_argument({q}--teacher_super_high_res{q},\n'
        f'{teacher_indent}action={q}store_true{q},\n'
        f'{teacher_indent}help={q}load --teacher as a 24x24 model (block3_1+block4_1 stride 1); '
        f'set this when --teacher was itself trained with --high_res --super_high_res{q}\n'
        f'{teacher_close_indent})'
    )

    new_model_ctor = (
        'model = MultiTaskLearnerWithState(high_res=opt.high_res, '
        'super_high_res=opt.super_high_res).to(device)'
    )
    new_teacher_call = (
        'teacher = load_teacher(opt.teacher, high_res=True, '
        'super_high_res=opt.teacher_super_high_res, device=device) if opt.teacher else None'
    )

    pq = print_quote
    new_print = (
        f'print(f{pq}[KD] teacher loaded: {{opt.teacher}}  alpha={{opt.kd_alpha}} T={{opt.kd_T}} {pq}\n'
        f'          f{pq}teacher_super_high_res={{opt.teacher_super_high_res}} {pq}\n'
        f'          f{pq}student_super_high_res={{opt.super_high_res}}{pq})'
    )

    backup(path)

    new_content = content
    new_content = new_content.replace(hr_old_text, hr_old_text + "\n" + super_high_res_block, 1)
    new_content = new_content.replace(teacher_old_text, teacher_old_text + "\n" + teacher_super_high_res_block, 1)
    new_content = new_content.replace(model_old_text, new_model_ctor, 1)
    new_content = new_content.replace(teacher_call_old_text, new_teacher_call, 1)
    new_content = new_content.replace(print_old_text, new_print, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("  New flags: --super_high_res (student), --teacher_super_high_res (teacher)")
    print("  model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res)")
    print("  teacher = load_teacher(opt.teacher, high_res=True, super_high_res=opt.teacher_super_high_res, ...)")
    print()
    print("Next: python -c \"import ast; ast.parse(open('" + path + "').read()); print('SYNTAX_OK')\"")


if __name__ == "__main__":
    main()
