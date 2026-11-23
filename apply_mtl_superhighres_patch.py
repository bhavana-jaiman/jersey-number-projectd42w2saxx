"""
Patches MultiTaskLearnerWithState in subModules/backbone_ying.py so it
threads a new `super_high_res` parameter through to its
featureExtractor_Ying(...) call, matching the backbone patch already
applied (apply_superhighres_teacher_patch_v2.py), which added
super_high_res=False as a new parameter to featureExtractor_Ying.

Confirmed via live grep (this session) that MultiTaskLearnerWithState is
the ACTUAL model class used for training (not MultiTaskLearner, the other
class in this file), and that it currently has:

    class MultiTaskLearnerWithState(nn.Module):
        def __init__(self, out_channels=256, high_res=False):
            ...
            #self.feature_extractor = featureExtractor_Ying(out_channels)
            self.feature_extractor = featureExtractor_Ying(out_channels, high_res=high_res)

This patch changes ONLY two things, both tightly anchored to exact text:
1. The __init__ signature: adds super_high_res=False as a new parameter.
2. The featureExtractor_Ying(...) call: adds super_high_res=super_high_res.

With super_high_res left at its default (False) everywhere it's not
explicitly passed, MultiTaskLearnerWithState's behavior is byte-identical
to before this patch -- this only adds the ABILITY to request the new
24x24 teacher mode, it doesn't change any existing behavior by itself.

Usage:
    python apply_mtl_superhighres_patch.py subModules/backbone_ying.py
"""
import re
import sys
import shutil
import time

SIG_OLD_PATTERN = re.compile(
    r"class MultiTaskLearnerWithState\(nn\.Module\):\s*\n"
    r"([ \t]*)def __init__\(self, out_channels=256, high_res=False\):"
)

CALL_OLD = "self.feature_extractor = featureExtractor_Ying(out_channels, high_res=high_res)"
CALL_NEW = "self.feature_extractor = featureExtractor_Ying(out_channels, high_res=high_res, super_high_res=super_high_res)"


def main():
    if len(sys.argv) != 2:
        print("Usage: python apply_mtl_superhighres_patch.py <path_to_backbone_ying.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf8") as f:
        content = f.read()

    if "super_high_res=super_high_res" in content and "def __init__(self, out_channels=256, high_res=False, super_high_res=False):" in content:
        print("MultiTaskLearnerWithState already patched -- no changes made.")
        return

    sig_matches = list(SIG_OLD_PATTERN.finditer(content))
    if len(sig_matches) != 1:
        print(f"ERROR: MultiTaskLearnerWithState __init__ signature found {len(sig_matches)} times, expected 1. No changes made.")
        if not sig_matches:
            print("Run: grep -n -A2 'class MultiTaskLearnerWithState' subModules/backbone_ying.py")
        sys.exit(1)

    # The call site text must appear exactly once IN THE WHOLE FILE for a plain
    # string replace to be safe (featureExtractor_Ying_Small and
    # MultiTaskLearner, the other classes in this file, don't call
    # featureExtractor_Ying with exactly this high_res=high_res kwarg form --
    # confirmed via the live grep that showed only ONE such call, at the
    # MultiTaskLearnerWithState location).
    call_count = content.count(CALL_OLD)
    if call_count != 1:
        print(f"ERROR: '{CALL_OLD}' found {call_count} times, expected exactly 1. No changes made.")
        if call_count == 0:
            print("Run: grep -n 'featureExtractor_Ying(out_channels, high_res=high_res)' subModules/backbone_ying.py")
        sys.exit(1)

    backup_path = f"{path}.bak.{int(time.time())}"
    shutil.copy2(path, backup_path)
    print(f"Backup saved to: {backup_path}")

    m = sig_matches[0]
    sig_old_text = m.group(0)
    sig_new_text = sig_old_text.replace(
        "def __init__(self, out_channels=256, high_res=False):",
        "def __init__(self, out_channels=256, high_res=False, super_high_res=False):",
    )
    new_content = content.replace(sig_old_text, sig_new_text, 1)
    new_content = new_content.replace(CALL_OLD, CALL_NEW, 1)

    with open(path, "w", encoding="utf8") as f:
        f.write(new_content)

    print(f"Patched successfully: {path}")
    print("Change: MultiTaskLearnerWithState.__init__ now accepts super_high_res=False")
    print("and passes it through to featureExtractor_Ying. Default behavior unchanged;")
    print("pass super_high_res=True (with high_res=True) to build the new 24x24 teacher.")


if __name__ == "__main__":
    main()
