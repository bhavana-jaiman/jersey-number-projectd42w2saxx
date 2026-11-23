"""
Dumps the EXACT repr() of the --high_res argparse block and the
MultiTaskLearnerWithState(...) construction site in
jerseyNumberRecognitiorWithState_Testing.py, so a --super_high_res flag can
be added to the testing script the same way it was added to the training
script -- matched exactly against the real file instead of guessed from a
photo.

Usage:
    python dump_testing_script.py jerseyNumberRecognitiorWithState_Testing.py
"""
import sys

if len(sys.argv) != 2:
    print("Usage: python dump_testing_script.py <path_to_testing_script.py>")
    sys.exit(1)

path = sys.argv[1]
with open(path, "r", encoding="utf8") as f:
    lines = f.readlines()


def find_and_print(marker, context_before=1, context_after=8):
    found_any = False
    for i, line in enumerate(lines):
        if marker in line:
            found_any = True
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after)
            print(f"--- match for {marker!r} at line {i+1} ---")
            for j in range(start, end):
                print(f"{j+1:5d}: {lines[j]!r}")
            print()
    if not found_any:
        print(f"--- NO MATCH for {marker!r} ---\n")


find_and_print("high_res", context_before=1, context_after=6)
find_and_print("MultiTaskLearnerWithState(", context_before=2, context_after=2)
find_and_print("load_state_dict", context_before=2, context_after=2)
