"""
Dumps the EXACT bytes (with repr() so whitespace/quotes are unambiguous) of
the --high_res and --teacher argparse blocks in
jerseyNumberRecognitiorWithState_Training.py, plus the two call-site lines
and the print line -- everything apply_cli_superhighres_patch.py needs to
match exactly. The previous patch attempt failed with "substring not found"
for HIGH_RES_ARG_OLD, meaning my transcription of that block (from a photo
of a terminal) doesn't byte-match the real file. This prints the real
lines with repr() so there's no ambiguity about spaces/quotes/wrapping.

Usage:
    python dump_argparse_block.py jerseyNumberRecognitiorWithState_Training.py
"""
import sys

if len(sys.argv) != 2:
    print("Usage: python dump_argparse_block.py <path_to_training_script.py>")
    sys.exit(1)

path = sys.argv[1]
with open(path, "r", encoding="utf8") as f:
    lines = f.readlines()

def find_and_print(marker, context_before=1, context_after=8):
    for i, line in enumerate(lines):
        if marker in line:
            start = max(0, i - context_before)
            end = min(len(lines), i + context_after)
            print(f"--- match for {marker!r} at line {i+1} ---")
            for j in range(start, end):
                print(f"{j+1:5d}: {lines[j]!r}")
            print()
            return
    print(f"--- NO MATCH for {marker!r} ---\n")

find_and_print('"--high_res"', context_after=6)
find_and_print('"--teacher"', context_after=8)
find_and_print("MultiTaskLearnerWithState(high_res=opt.high_res)", context_before=3, context_after=1)
find_and_print("load_teacher(opt.teacher", context_before=0, context_after=1)
find_and_print("[KD] teacher loaded", context_before=0, context_after=1)
