"""
Fixes the _sample_weights_from_labels_list function that apply_sampler_patch_v3.py
inserted into jerseyNumberRecognitiorWithState_Training.py.

Root cause: the function assumed `dataset.labels_list` is a pre-built
attribute (a list of per-sample digit-class arrays). That attribute IS
defined on some dataset classes in utils/jersey_dataset.py (jersey_Dataset,
jerseyNumber_ValidationDataset), but the actual class used for training --
jerseyWithState_Dataset -- has that line COMMENTED OUT (line 100) and only
has `self.labels_files` (an array of label FILE PATHS). This caused:
    AttributeError: 'jerseyWithState_Dataset' object has no attribute
    'labels_list'. Did you mean: 'labels_files'?

Fix: read dataset.labels_files instead, and call np.loadtxt(path) on each
path directly inside the weight-building function, matching exactly what
jersey_Dataset's own (working) self.labels_list construction does:
    np.loadtxt(path).reshape(-1, 5)[:, 0] for path in self.labels_files

Usage:
    python fix_labels_list_attr.py jerseyNumberRecognitiorWithState_Training.py
"""
import re
import sys
import shutil
import time

OLD = '''def _sample_weights_from_labels_list(dataset):
\tweights = []
\tfor labels in dataset.labels_list:
\t\thas_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))
\t\tweights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)
\treturn weights'''

NEW = '''def _sample_weights_from_labels_list(dataset):
\tweights = []
\tfor path in dataset.labels_files:
\t\tlabels = np.loadtxt(path).reshape(-1, 5)[:, 0]
\t\thas_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))
\t\tweights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)
\treturn weights'''


def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_labels_list_attr.py <path_to_training_script.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    # Try exact tab-based match first, then fall back to a whitespace-
    # tolerant regex in case indentation differs slightly in the live file.
    if OLD in content:
        matched_old = OLD
        new_block = NEW
    else:
        pattern = re.compile(
            r"def _sample_weights_from_labels_list\(dataset\):\s*\n"
            r"[ \t]*weights = \[\]\s*\n"
            r"[ \t]*for labels in dataset\.labels_list:\s*\n"
            r"[ \t]*has_confusable = any\(int\(d\) in CONFUSABLE_DIGITS_SET for d in np\.atleast_1d\(labels\)\)\s*\n"
            r"[ \t]*weights\.append\(OVERSAMPLE_WEIGHT if has_confusable else 1\.0\)\s*\n"
            r"[ \t]*return weights"
        )
        m = pattern.search(content)
        if not m:
            print("ERROR: could not find the _sample_weights_from_labels_list function to fix.")
            print("No changes made. Run this and paste the output:")
            print("  grep -n -A6 '_sample_weights_from_labels_list' <your_file.py>")
            sys.exit(1)
        matched_old = m.group(0)
        # rebuild NEW using the same leading indent found before "weights = []"
        indent_match = re.search(r"\n([ \t]*)weights = \[\]", matched_old)
        indent = indent_match.group(1) if indent_match else "\t"
        new_block = (
            "def _sample_weights_from_labels_list(dataset):\n"
            f"{indent}weights = []\n"
            f"{indent}for path in dataset.labels_files:\n"
            f"{indent}{indent}labels = np.loadtxt(path).reshape(-1, 5)[:, 0]\n"
            f"{indent}{indent}has_confusable = any(int(d) in CONFUSABLE_DIGITS_SET for d in np.atleast_1d(labels))\n"
            f"{indent}{indent}weights.append(OVERSAMPLE_WEIGHT if has_confusable else 1.0)\n"
            f"{indent}return weights"
        )

    occurrences = content.count(matched_old)
    if occurrences > 1:
        print(f"ERROR: found {occurrences} matches, expected exactly 1. Refusing to patch.")
        sys.exit(1)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    new_content = content.replace(matched_old, new_block)
    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: _sample_weights_from_labels_list now reads dataset.labels_files "
          "and calls np.loadtxt(path) per file, instead of assuming a pre-built "
          "dataset.labels_list attribute (which jerseyWithState_Dataset doesn't have).")


if __name__ == "__main__":
    main()
