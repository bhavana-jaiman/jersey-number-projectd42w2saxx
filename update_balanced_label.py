from pathlib import Path


# ============================================================
# INPUT PATH
# ============================================================

# Give your validation_new folder here
INPUT_DIR = Path("/path/to/validation_new")


# ============================================================
# PROCESS LABEL FILES
# ============================================================

label_files = list(INPUT_DIR.rglob("*.txt"))

print("=" * 60)
print("UPDATING VALIDATION LABELS")
print("=" * 60)

print(f"Input folder : {INPUT_DIR}")
print(f"Label files  : {len(label_files)}")

processed = 0
empty_files = 0
invalid_lines = 0


for label_file in label_files:

    # Read original label
    with open(label_file, "r") as f:
        lines = f.readlines()

    class_numbers = []

    for line in lines:

        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        parts = line.split()

        # ----------------------------------------------------
        # First value is assumed to be the class number
        # ----------------------------------------------------

        try:
            class_number = int(float(parts[0]))
            class_numbers.append(str(class_number))

        except (ValueError, IndexError):

            print(
                f"WARNING: Invalid line in {label_file}:\n"
                f"    {line}"
            )

            invalid_lines += 1

    # --------------------------------------------------------
    # Write only class numbers
    # --------------------------------------------------------

    new_content = " ".join(class_numbers)

    with open(label_file, "w") as f:
        f.write(new_content)

    if len(class_numbers) == 0:
        empty_files += 1

    processed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("COMPLETED")
print("=" * 60)

print(f"Processed labels : {processed}")
print(f"Empty labels     : {empty_files}")
print(f"Invalid lines    : {invalid_lines}")

print("\nExample conversion:")

print("""
BEFORE:

3 0.421 0.532 0.123 0.245
7 0.612 0.431 0.151 0.289

AFTER:

3 7
""")
