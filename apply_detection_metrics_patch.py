"""
Adds Jersey-Detection-style TP / FP / FN / Precision / Recall counters to
jerseyNumberRecognitiorWithState_Testing.py, computed from the existing
per-sample `state` (ground truth) and `predicted_state` (model output)
variables that already exist in the test loop -- plus the existing
jerseyNumber_correct-style digit match check.

IMPORTANT -- what this does and does NOT give you:
  - TP / FP / FN / Precision / Recall are computed and are meaningful,
    derived honestly from your classifier's real state + digit outputs.
  - AP@0.5 is NOT computed and is NOT added by this patch. Your pipeline
    does not produce confidence-ranked bounding-box predictions scored
    against IoU thresholds, so a true AP@0.5 cannot be computed from it.
    Do not fabricate this number -- if your table requires it, that must
    come from an actual object detector, or be explicitly dropped/footnoted
    as "not applicable to this classification-based pipeline" when you
    present results.

Definition used (confirmed with you before writing this):
  - GT (ground truth positive)  = a sample where state != 0 (a real jersey
    number is present, state 1 or state 2).
  - TP (true positive)  = state != 0 AND predicted_state != 0 AND the full
    digit match (pred_d1==jersey_number[0] and pred_d2==jersey_number[1])
    is correct.
  - FN (false negative) = state != 0 AND (predicted_state == 0 OR the
    digit match is wrong). I.e. a real jersey number that was either
    missed entirely (predicted empty) or read incorrectly.
  - FP (false positive) = state == 0 AND predicted_state != 0. I.e. the
    model hallucinated a number where none exists.
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)   (this equals TP / GT_count)

Confirmed exact anchor lines (from live grep/screenshots, this session):

    state_correct += int(predicted_state == state)
    ...
    jerseyNumber_correct += pred_d1.eq(jersey_number[0]).logical_and(
        pred_d2.eq(jersey_number[1]))

This patch:
  1. Adds counter initialization (det_tp = det_fp = det_fn = 0) near the
     other counter initializations (wherever state_correct/jerseyNumber_correct
     are first initialized to 0 -- found by regex, not assumed line number).
  2. Adds the TP/FP/FN accumulation logic immediately after the existing
     jerseyNumber_correct += ... statement, using the SAME predicted_state,
     state, pred_d1, pred_d2, jersey_number variables already in scope there
     -- no new model calls, no new forward passes, purely derived from
     values already computed by the existing test loop.
  3. Adds a final print of GT / TP / FP / FN / Precision / Recall, placed
     right after the existing "the accuracy of the ... is ..." print block
     (found by anchoring on the existing print of "the accuracy of the state1 is").

Usage:
    python apply_detection_metrics_patch.py jerseyNumberRecognitiorWithState_Testing.py
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
        print("Usage: python apply_detection_metrics_patch.py <path_to_jerseyNumberRecognitiorWithState_Testing.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "det_tp" in content and "det_fp" in content and "det_fn" in content:
        print(f"{path}: already patched (det_tp/det_fp/det_fn found). No changes made.")
        return

    # --- Anchor 1: initialization of state_correct = 0 (to place our new
    # counters' initialization right alongside it, same indent) ---
    INIT_RE = re.compile(r'state_correct\s*=\s*0\b')
    init_matches = list(INIT_RE.finditer(content))
    if len(init_matches) < 1:
        print("ERROR: could not find 'state_correct = 0' initialization. No changes made.")
        sys.exit(1)
    init_m = init_matches[0]
    init_indent = leading_indent_of_match(content, init_m)
    init_line_start = content.rfind("\n", 0, init_m.start()) + 1
    init_line_end = content.find("\n", init_m.start())

    # --- Anchor 2: the jerseyNumber_correct += ... statement (may span
    # multiple lines, ending in a closing paren on its own line per the
    # screenshot: "pred_d2.eq(jersey_number[1]))" ) ---
    JN_CORRECT_RE = re.compile(
        r'jerseyNumber_correct\s*\+=\s*pred_d1\.eq\(jersey_number\[0\]\)\.logical_and\(\s*'
        r'pred_d2\.eq\(jersey_number\[1\]\)\)'
    )
    jn_matches = list(JN_CORRECT_RE.finditer(content))
    if len(jn_matches) != 1:
        print(f"ERROR: jerseyNumber_correct += ... statement found {len(jn_matches)} times via regex, expected 1. No changes made.")
        print("The statement may be formatted slightly differently than expected -- paste the exact lines around it if this fails.")
        sys.exit(1)
    jn_m = jn_matches[0]
    jn_indent = leading_indent_of_match(content, jn_m)
    jn_stmt_end = jn_m.end()

    # --- Anchor 3: the final print of "the accuracy of the state1 is" (to
    # place our summary print right after the existing accuracy prints) ---
    FINAL_PRINT_RE = re.compile(
        r'print\([^\n]*the accuracy of the state1 is[^\n]*\)'
    )
    final_print_matches = list(FINAL_PRINT_RE.finditer(content))
    if len(final_print_matches) != 1:
        print(f"ERROR: final 'the accuracy of the state1 is' print found {len(final_print_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    fp_m = final_print_matches[0]
    fp_indent = leading_indent_of_match(content, fp_m)
    fp_stmt_end = fp_m.end()

    print("All anchors found.")
    print("  init indent:", repr(init_indent))
    print("  jerseyNumber_correct += ... indent:", repr(jn_indent))
    print("  final print indent:", repr(fp_indent))

    backup(path)

    new_content = content

    # 1) Insert counter initialization right after state_correct = 0 line.
    counters_init = (
        f"\n{init_indent}det_tp = 0\n"
        f"{init_indent}det_fp = 0\n"
        f"{init_indent}det_fn = 0"
    )
    new_content = new_content[:init_line_end] + counters_init + new_content[init_line_end:]

    # Recompute jn_m position-dependent offsets are now invalid since we
    # inserted text before them; re-find the jerseyNumber_correct anchor
    # in the UPDATED content to get the correct insertion point.
    jn_matches2 = list(JN_CORRECT_RE.finditer(new_content))
    if len(jn_matches2) != 1:
        print("ERROR: lost track of jerseyNumber_correct anchor after first insertion. Aborting without writing.")
        sys.exit(1)
    jn_m2 = jn_matches2[0]
    jn_stmt_end2 = jn_m2.end()

    # 2) Insert TP/FP/FN accumulation right after the jerseyNumber_correct
    # statement, using the SAME variables already in scope at this point
    # in the loop (state, predicted_state, pred_d1, pred_d2, jersey_number).
    detection_logic = (
        f"\n{jn_indent}_digit_match = bool(pred_d1.eq(jersey_number[0]).logical_and(\n"
        f"{jn_indent}    pred_d2.eq(jersey_number[1])))\n"
        f"{jn_indent}if state != 0:\n"
        f"{jn_indent}    if predicted_state != 0 and _digit_match:\n"
        f"{jn_indent}        det_tp += 1\n"
        f"{jn_indent}    else:\n"
        f"{jn_indent}        det_fn += 1\n"
        f"{jn_indent}else:\n"
        f"{jn_indent}    if predicted_state != 0:\n"
        f"{jn_indent}        det_fp += 1"
    )
    new_content = new_content[:jn_stmt_end2] + detection_logic + new_content[jn_stmt_end2:]

    # 3) Insert final summary print after the existing state1-accuracy print.
    # Re-find in updated content.
    final_print_matches2 = list(FINAL_PRINT_RE.finditer(new_content))
    if len(final_print_matches2) != 1:
        print("ERROR: lost track of final print anchor after earlier insertions. Aborting without writing.")
        sys.exit(1)
    fp_m2 = final_print_matches2[0]
    fp_stmt_end2 = fp_m2.end()

    summary_print = (
        f"\n{fp_indent}_det_gt = det_tp + det_fn\n"
        f"{fp_indent}_det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0\n"
        f"{fp_indent}_det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0\n"
        f"{fp_indent}print(f\"[Jersey Detection] Total GT={{_det_gt}} TP={{det_tp}} FP={{det_fp}} FN={{det_fn}} \"\n"
        f"{fp_indent}      f\"Precision={{_det_precision*100:.2f}}% Recall={{_det_recall*100:.2f}}%\")\n"
        f"{fp_indent}print(\"[Jersey Detection] NOTE: AP@0.5 is not computed -- this pipeline does not \"\n"
        f"{fp_indent}      \"produce confidence-ranked bounding-box predictions scored against IoU, \"\n"
        f"{fp_indent}      \"which true AP@0.5 requires.\")"
    )
    new_content = new_content[:fp_stmt_end2] + summary_print + new_content[fp_stmt_end2:]

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("New output added at the end of the test run:")
    print("  [Jersey Detection] Total GT=... TP=... FP=... FN=... Precision=...% Recall=...%")
    print()
    print("Next: python -c \"import ast; ast.parse(open('" + path + "').read()); print('SYNTAX_OK')\"")


if __name__ == "__main__":
    main()
