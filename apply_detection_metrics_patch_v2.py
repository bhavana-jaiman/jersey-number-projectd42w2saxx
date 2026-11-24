"""
v2 -- rewritten against the REAL confirmed code (via sed -n '440,445p;548,583p'
on your actual jerseyNumberRecognitiorWithState_Testing.py), after v1's blind
anchors failed (state_correct is declared in a multi-variable tuple
assignment, not "state_correct = 0" alone; jerseyNumber_correct is only
accumulated when 0 < len(jersey_number) <= 2).

Confirmed exact real code (this session, live sed output):

    jerseyNumber_correct, state_correct, total_body, count_2digit_bodies, cont_3digit_bodies = 0, 0, 0, 0, 0
    state_correct_3digit, state_correct_less3digit = 0, 0
    cont_0digit_bodies, state_correct_0digit = 0, 0

    ...
    _, pred_d1 = torch.max(logits_d1, 1)
    _, pred_d2 = torch.max(logits_d2, 1)
    prob_state = torch.softmax(logits_state, dim=1)
    predicted_state = prob_state.argmax(1).item()
    if predicted_state == 2 and prob_state[0, 2].item() < 0.4:
        predicted_state = prob_state[0, :2].argmax().item()
    state_correct += int(predicted_state == state)
    if state == 2 and predicted_state == 2:
        state_correct_3digit += 1
    elif state == 1 and predicted_state == 1:
        state_correct_less3digit += 1
    elif state == 0 and predicted_state == 0:
        state_correct_0digit += 1
    ...
    image_name = f"{pred_d1.item()}_{pred_d2.item()}"
    plug_save_image(body_img, image_name, "Testing3")
    ...
    if 0 < len(jersey_number) <= 2:
        jerseyNumber_correct += pred_d1.eq(jersey_number[0]).logical_and(
            pred_d2.eq(jersey_number[1]))

    print("----")
    print("the accuracy of the 2 digit is %2f" % ((jerseyNumber_correct.item()/count_2digit_bodies)*100))
    print(f'the accuracy of the state is {100 * (state_correct/total_body):.2f}')
    print(f'the accuracy of the state2 is {100 * (state_correct_3digit/cont_3digit_bodies):.2f}')
    print(f'the accuracy of the state1 is {100 * (state_correct_less3digit/count_2digit_bodies):.2f}')
    if cont_0digit_bodies:
        print(f'the accuracy of the state0 is {100 * (state_correct_0digit/count_0digit_bodies):.2f}')

Definition used (same as discussed and confirmed before writing this):
  - GT (ground truth positive)  = a sample where state != 0 (a real jersey
    number is present -- state 1 or state 2).
  - TP (true positive)  = state != 0 AND predicted_state != 0 AND the full
    digit match (pred_d1==jersey_number[0] and pred_d2==jersey_number[1])
    is correct. For state == 2 (3+ digit) samples, jersey_number may have
    more than 2 entries and the existing script does not digit-match those
    at all (the `if 0 < len(jersey_number) <= 2:` guard skips them) -- so
    for state==2 samples, TP here is based on predicted_state alone (did
    the model correctly say "a number is present and it's 3+ digits"),
    since no ground-truth digit target exists to compare against for
    state2 in this script's own logic.
  - FN (false negative) = state != 0 AND the sample was not counted as TP
    above (either predicted_state == 0, or -- for state==1 samples only --
    the 2-digit match was wrong).
  - FP (false positive) = state == 0 AND predicted_state != 0.
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)

This patch:
  1. Adds det_tp/det_fp/det_fn = 0, 0, 0 right after the existing
     cont_0digit_bodies, state_correct_0digit = 0, 0 initialization line.
  2. Adds the TP/FP/FN accumulation logic right after the EXISTING
     `if 0 < len(jersey_number) <= 2: jerseyNumber_correct += ...` block,
     reusing predicted_state, state, pred_d1, pred_d2, jersey_number
     already computed above it in the loop -- no new model calls.
  3. Adds a final summary print right after the existing
     "the accuracy of the state1 is ..." print statement.

Usage:
    python apply_detection_metrics_patch_v2.py jerseyNumberRecognitiorWithState_Testing.py
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
        print("Usage: python apply_detection_metrics_patch_v2.py <path_to_jerseyNumberRecognitiorWithState_Testing.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "det_tp" in content and "det_fp" in content and "det_fn" in content:
        print(f"{path}: already patched (det_tp/det_fp/det_fn found). No changes made.")
        return

    # --- Anchor 1: cont_0digit_bodies, state_correct_0digit = 0, 0 ---
    INIT_RE = re.compile(
        r'cont_0digit_bodies,\s*state_correct_0digit\s*=\s*0,\s*0'
    )
    init_matches = list(INIT_RE.finditer(content))
    if len(init_matches) != 1:
        print(f"ERROR: init line 'cont_0digit_bodies, state_correct_0digit = 0, 0' found {len(init_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    init_m = init_matches[0]
    init_indent = leading_indent_of_match(content, init_m)
    init_stmt_end = init_m.end()

    # --- Anchor 2: the `if 0 < len(jersey_number) <= 2:` block containing
    # jerseyNumber_correct += pred_d1.eq(...).logical_and(pred_d2.eq(...)) ---
    JN_BLOCK_RE = re.compile(
        r'if 0 < len\(jersey_number\) <= 2:\s*\n'
        r'([ \t]*)jerseyNumber_correct \+= pred_d1\.eq\(jersey_number\[0\]\)\.logical_and\(\s*\n'
        r'[ \t]*pred_d2\.eq\(jersey_number\[1\]\)\)'
    )
    jn_matches = list(JN_BLOCK_RE.finditer(content))
    if len(jn_matches) != 1:
        print(f"ERROR: 'if 0 < len(jersey_number) <= 2: jerseyNumber_correct += ...' block found {len(jn_matches)} times via regex, expected 1. No changes made.")
        sys.exit(1)
    jn_m = jn_matches[0]
    if_indent = leading_indent_of_match(content, jn_m)  # indent of "if 0 < len..."
    inner_indent = jn_m.group(1)  # indent of the jerseyNumber_correct line inside the if
    jn_stmt_end = jn_m.end()

    # --- Anchor 3: the final "the accuracy of the state1 is ..." print ---
    FINAL_PRINT_RE = re.compile(
        r"print\(f'the accuracy of the state1 is[^\n]*\)"
    )
    final_print_matches = list(FINAL_PRINT_RE.finditer(content))
    if len(final_print_matches) != 1:
        print(f"ERROR: final state1-accuracy print found {len(final_print_matches)} times, expected 1. No changes made.")
        sys.exit(1)
    fp_m = final_print_matches[0]
    fp_indent = leading_indent_of_match(content, fp_m)
    fp_stmt_end = fp_m.end()

    print("All anchors found.")
    print("  init indent:", repr(init_indent))
    print("  if-block indent:", repr(if_indent), " inner indent:", repr(inner_indent))
    print("  final print indent:", repr(fp_indent))

    backup(path)

    new_content = content

    # 1) counters init, right after cont_0digit_bodies, state_correct_0digit = 0, 0
    counters_init = (
        f"\n{init_indent}det_tp, det_fp, det_fn = 0, 0, 0"
    )
    new_content = new_content[:init_stmt_end] + counters_init + new_content[init_stmt_end:]

    # Re-find the jerseyNumber_correct if-block in the UPDATED content
    jn_matches2 = list(JN_BLOCK_RE.finditer(new_content))
    if len(jn_matches2) != 1:
        print("ERROR: lost track of jerseyNumber_correct if-block after first insertion. Aborting without writing.")
        sys.exit(1)
    jn_m2 = jn_matches2[0]
    jn_stmt_end2 = jn_m2.end()
    if_indent2 = leading_indent_of_match(new_content, jn_m2)

    # 2) TP/FP/FN logic, placed at the SAME indent level as the
    # "if 0 < len(jersey_number) <= 2:" statement itself (i.e. as a
    # sibling block after it, not nested inside it), since state==0 and
    # state==2 samples must also be handled and they don't enter that if.
    detection_logic = (
        f"\n{if_indent2}if state != 0:\n"
        f"{if_indent2}    _digit_ok = True\n"
        f"{if_indent2}    if state == 1:\n"
        f"{if_indent2}        _digit_ok = bool(pred_d1.eq(jersey_number[0]).logical_and(\n"
        f"{if_indent2}            pred_d2.eq(jersey_number[1])))\n"
        f"{if_indent2}    if predicted_state != 0 and _digit_ok:\n"
        f"{if_indent2}        det_tp += 1\n"
        f"{if_indent2}    else:\n"
        f"{if_indent2}        det_fn += 1\n"
        f"{if_indent2}else:\n"
        f"{if_indent2}    if predicted_state != 0:\n"
        f"{if_indent2}        det_fp += 1"
    )
    new_content = new_content[:jn_stmt_end2] + detection_logic + new_content[jn_stmt_end2:]

    # 3) final summary print, after the existing state1-accuracy print
    final_print_matches2 = list(FINAL_PRINT_RE.finditer(new_content))
    if len(final_print_matches2) != 1:
        print("ERROR: lost track of final print anchor after earlier insertions. Aborting without writing.")
        sys.exit(1)
    fp_m2 = final_print_matches2[0]
    fp_stmt_end2 = fp_m2.end()
    fp_indent2 = leading_indent_of_match(new_content, fp_m2)

    summary_print = (
        f"\n{fp_indent2}_det_gt = det_tp + det_fn\n"
        f"{fp_indent2}_det_precision = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0.0\n"
        f"{fp_indent2}_det_recall = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0.0\n"
        f"{fp_indent2}print(f\"[Jersey Detection] Total GT={{_det_gt}} TP={{det_tp}} FP={{det_fp}} FN={{det_fn}} \"\n"
        f"{fp_indent2}      f\"Precision={{_det_precision*100:.2f}}% Recall={{_det_recall*100:.2f}}%\")\n"
        f"{fp_indent2}print(\"[Jersey Detection] NOTE: AP@0.5 is not computed -- this pipeline does not \"\n"
        f"{fp_indent2}      \"produce confidence-ranked bounding-box predictions scored against IoU, \"\n"
        f"{fp_indent2}      \"which true AP@0.5 requires.\")"
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
