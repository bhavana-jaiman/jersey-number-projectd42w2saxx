"""
Final wiring patch: adds a --super_high_res CLI flag to
jerseyNumberRecognitiorWithState_Training.py and updates utils/kd.py's
load_teacher(...) so it can construct/load a teacher built with
super_high_res=True (the new 24x24 feature map mode).

Confirmed via live grep (this session) the EXACT current code:

  utils/kd.py:
    def load_teacher(ckpt, high_res=True, device="cuda"):
        t = MultiTaskLearnerWithState(high_res=high_res).to(device)
        sd = torch.load(ckpt, map_location=device)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        t.load_state_dict(sd, strict=True)
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        return t

  jerseyNumberRecognitiorWithState_Training.py:
    parser.add_argument("--high_res",
        action="store_true",
        help="12x12 feature map model (block4_1 stride 1)"
        )
    parser.add_argument("--teacher",
        type=str,
        default="",
        help="12x12 checkpoint(trained with --high_res) model (block4_1 stride 1)"
        )
    parser.add_argument("--kd_alpha", ...)
    parser.add_argument("--kd_T", ...)
    ...
    # ================HCL_Changes================
    # model = MultiTaskLearner().to(device)
    #model = MultiTaskLearnerWithState().to(device)
    model = MultiTaskLearnerWithState(high_res=opt.high_res).to(device)
    teacher = load_teacher(opt.teacher, high_res=True, device=device) if opt.teacher else None
    if teacher is not None:
        print(f"[KD] teacher loaded: {opt.teacher}  alpha={opt.kd_alpha} T={opt.kd_T}")
    # ================================================

IMPORTANT: load_teacher's teacher-construction call is hardcoded to
high_res=True (NOT opt.high_res) -- i.e. today, the teacher is ALWAYS built
as the 12x12 model regardless of what the student (--high_res) is set to.
This patch keeps that same "teacher flags are independent of student flags"
design: it adds a NEW --super_high_res CLI flag (for the STUDENT's own
construction, mirroring --high_res exactly) and a SEPARATE --teacher_super_high_res
flag specifically for the teacher-loading call, so you can e.g. train/load
a 24x24 teacher (--teacher_super_high_res) while the student itself stays a
plain 6x6 model (no --high_res/--super_high_res passed to it) -- this is
the actual intended use case (bigger teacher, unchanged student).

Changes:
  utils/kd.py:
    load_teacher(ckpt, high_res=True, super_high_res=False, device="cuda")
    -> MultiTaskLearnerWithState(high_res=high_res, super_high_res=super_high_res)

  jerseyNumberRecognitiorWithState_Training.py:
    1. New --super_high_res flag (mirrors --high_res, for the STUDENT)
    2. New --teacher_super_high_res flag (for the TEACHER being loaded)
    3. model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res).to(device)
    4. teacher = load_teacher(opt.teacher, high_res=True, super_high_res=opt.teacher_super_high_res, device=device) if opt.teacher else None
    5. print line also reports super_high_res flags

All replacements are exact-string, each checked to occur exactly once
before any write. Backup taken before any change.

Usage:
    python apply_cli_superhighres_patch.py utils/kd.py jerseyNumberRecognitiorWithState_Training.py
"""
import sys
import shutil
import time


def backup(path):
    bpath = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, bpath)
    print(f"Backup saved to: {bpath}")
    return bpath


def require_exactly_one(content, needle, label, path):
    n = content.count(needle)
    if n != 1:
        print(f"ERROR: in {path}, '{label}' found {n} times, expected exactly 1. No changes made to {path}.")
        return False
    return True


def patch_kd_py(path):
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "super_high_res=super_high_res" in content and "def load_teacher(ckpt, high_res=True, super_high_res=False" in content:
        print(f"{path}: already patched. No changes made.")
        return True

    SIG_OLD = 'def load_teacher(ckpt, high_res=True, device="cuda"):'
    SIG_NEW = 'def load_teacher(ckpt, high_res=True, super_high_res=False, device="cuda"):'

    CALL_OLD = "t = MultiTaskLearnerWithState(high_res=high_res).to(device)"
    CALL_NEW = "t = MultiTaskLearnerWithState(high_res=high_res, super_high_res=super_high_res).to(device)"

    checks = [
        (SIG_OLD, "load_teacher signature"),
        (CALL_OLD, "MultiTaskLearnerWithState(high_res=high_res) call in load_teacher"),
    ]
    for needle, label in checks:
        if not require_exactly_one(content, needle, label, path):
            return False

    backup(path)
    new_content = content.replace(SIG_OLD, SIG_NEW, 1)
    new_content = new_content.replace(CALL_OLD, CALL_NEW, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("  load_teacher(...) now accepts super_high_res=False and passes it through.")
    return True


def patch_training_script(path):
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "--super_high_res" in content and "--teacher_super_high_res" in content:
        print(f"{path}: already patched. No changes made.")
        return True

    # --- 1. Insert new --super_high_res flag right after the existing --high_res block ---
    HIGH_RES_ARG_OLD = (
        'parser.add_argument("--high_res",\n'
        '        action="store_true",\n'
        '        help="12x12 feature map model (block4_1 stride 1)"\n'
        '        )'
    )

    # --- 2. Insert new --teacher_super_high_res flag right after the existing --teacher block ---
    TEACHER_ARG_OLD = (
        'parser.add_argument("--teacher",\n'
        '        type=str,\n'
        '        default="",\n'
        '        help="12x12 checkpoint(trained with --high_res) model (block4_1 stride 1)"\n'
        '        )'
    )

    def detect_indent(full_content, needle):
        idx = full_content.index(needle)
        line_start = full_content.rfind("\n", 0, idx) + 1
        prefix = full_content[line_start:idx]
        return prefix if prefix.strip() == "" else ""

    indent_hr = detect_indent(content, HIGH_RES_ARG_OLD)
    indent_teacher = detect_indent(content, TEACHER_ARG_OLD)

    SUPER_HIGH_RES_ARG_NEW = (
        HIGH_RES_ARG_OLD + "\n"
        f'{indent_hr}parser.add_argument("--super_high_res",\n'
        f'{indent_hr}        action="store_true",\n'
        f'{indent_hr}        help="24x24 feature map model (block3_1 + block4_1 both stride 1; "\n'
        f'{indent_hr}             "stack with --high_res to compound; student-side only, mirrors --high_res)"\n'
        f'{indent_hr}        )'
    )

    TEACHER_ARG_NEW = (
        TEACHER_ARG_OLD + "\n"
        f'{indent_teacher}parser.add_argument("--teacher_super_high_res",\n'
        f'{indent_teacher}        action="store_true",\n'
        f'{indent_teacher}        help="load --teacher as a 24x24 model (block3_1+block4_1 stride 1); "\n'
        f'{indent_teacher}             "set this when --teacher was itself trained with --high_res --super_high_res"\n'
        f'{indent_teacher}        )'
    )

    # --- 3. Student construction: thread opt.super_high_res through ---
    MODEL_CTOR_OLD = "model = MultiTaskLearnerWithState(high_res=opt.high_res).to(device)"
    MODEL_CTOR_NEW = "model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res).to(device)"

    # --- 4. Teacher loading call: thread opt.teacher_super_high_res through ---
    TEACHER_CALL_OLD = 'teacher = load_teacher(opt.teacher, high_res=True, device=device) if opt.teacher else None'
    TEACHER_CALL_NEW = (
        "teacher = load_teacher(opt.teacher, high_res=True, "
        "super_high_res=opt.teacher_super_high_res, device=device) if opt.teacher else None"
    )

    # --- 5. Print line: report the new flags too ---
    PRINT_OLD = 'print(f"[KD] teacher loaded: {opt.teacher}  alpha={opt.kd_alpha} T={opt.kd_T}")'

    checks = [
        (HIGH_RES_ARG_OLD, "--high_res argparse block"),
        (TEACHER_ARG_OLD, "--teacher argparse block"),
        (MODEL_CTOR_OLD, "MultiTaskLearnerWithState(high_res=opt.high_res) construction"),
        (TEACHER_CALL_OLD, "load_teacher(opt.teacher, high_res=True, ...) call"),
        (PRINT_OLD, "[KD] teacher loaded print line"),
    ]
    for needle, label in checks:
        if not require_exactly_one(content, needle, label, path):
            return False

    print_idx = content.index(PRINT_OLD)
    print_line_start = content.rfind("\n", 0, print_idx) + 1
    print_prefix = content[print_line_start:print_idx]
    print_indent = print_prefix if print_prefix.strip() == "" else ""
    # Align continuation lines under the opening "print(" paren.
    cont_indent = print_indent + " " * len("print(")
    PRINT_NEW = (
        'print(f"[KD] teacher loaded: {opt.teacher}  alpha={opt.kd_alpha} T={opt.kd_T} "\n'
        f'{cont_indent}f"teacher_super_high_res={{opt.teacher_super_high_res}} "\n'
        f'{cont_indent}f"student_super_high_res={{opt.super_high_res}}")'
    )

    backup(path)
    new_content = content
    new_content = new_content.replace(HIGH_RES_ARG_OLD, SUPER_HIGH_RES_ARG_NEW, 1)
    new_content = new_content.replace(TEACHER_ARG_OLD, TEACHER_ARG_NEW, 1)
    new_content = new_content.replace(MODEL_CTOR_OLD, MODEL_CTOR_NEW, 1)
    new_content = new_content.replace(TEACHER_CALL_OLD, TEACHER_CALL_NEW, 1)
    new_content = new_content.replace(PRINT_OLD, PRINT_NEW, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("  New flags: --super_high_res (student), --teacher_super_high_res (teacher)")
    print("  model = MultiTaskLearnerWithState(high_res=opt.high_res, super_high_res=opt.super_high_res)")
    print("  teacher = load_teacher(opt.teacher, high_res=True, super_high_res=opt.teacher_super_high_res, ...)")
    return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python apply_cli_superhighres_patch.py <path_to_kd.py> <path_to_training_script.py>")
        sys.exit(1)

    kd_path, train_path = sys.argv[1], sys.argv[2]

    ok1 = patch_kd_py(kd_path)
    ok2 = patch_training_script(train_path)

    if not (ok1 and ok2):
        print()
        print("One or both files were NOT patched due to errors above. Review and re-run.")
        sys.exit(1)

    print()
    print("All done. New usage example (bigger teacher, student UNCHANGED):")
    print("  1) Train the 24x24 teacher itself (standalone, no --teacher flag, needs BOTH flags):")
    print("     python jerseyNumberRecognitiorWithState_Training.py --high_res --super_high_res ...")
    print("  2) Distill into the plain student using that teacher checkpoint:")
    print("     python jerseyNumberRecognitiorWithState_Training.py --teacher <24x24_ckpt> --teacher_super_high_res --kd_alpha 0.5 --kd_T 4.0 ...")
    print("     (no --high_res / --super_high_res here -- student stays the original 6x6 model)")


if __name__ == "__main__":
    main()
