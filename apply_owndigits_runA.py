"""
apply_owndigits_runA.py
-----------------------
Applies "own-digit copy" + "Run A" to YOUR CURRENT files, in place.

  * Old code is COMMENTED OUT, never deleted.
  * Every change is wrapped in  # ---------------HCL_Changes---------------  markers.
  * Nothing outside the listed blocks is touched.
  * A backup of each file is written first (*.bak_owndigits_runA).
  * Safe to run twice: blocks that are already changed are skipped.
  * If an expected block is not found, the script stops and changes NOTHING.

Run from New_arcitecture_2:
    python apply_owndigits_runA.py
"""
import re
import shutil
import sys

TRAIN_FILE = "train_New_changed.py"
DATASET_FILE = "utils/jersey_dataset_New_changed.py"

MARK = "# ---------------HCL_Changes---------------"
END = "# -------------------------------------------"


def fail(msg):
    print("ERROR:", msg)
    print("No file was changed.")
    sys.exit(1)


def comment_block(text):
    """Comment out every line of `text`, keeping its indentation."""
    out = []
    for line in text.rstrip("\n").split("\n"):
        if not line.strip():
            out.append("")
            continue
        ind = len(line) - len(line.lstrip())
        out.append(line[:ind] + "# " + line[ind:])
    return "\n".join(out) + "\n"


def indent(text, n):
    pad = " " * n
    return "\n".join(pad + l if l.strip() else "" for l in text.strip("\n").split("\n")) + "\n"


def class_region(src, class_name):
    """(start, end) of a class body: from 'class X' to the next top-level class / __main__."""
    m = re.search(r"^class " + re.escape(class_name) + r"\b", src, re.M)
    if not m:
        fail(f"class {class_name} not found")
    nxt = re.search(r"^(class |if __name__)", src[m.end():], re.M)
    return m.start(), (m.end() + nxt.start()) if nxt else len(src)


# ======================================================================
# New code blocks (indentation is added automatically)
# ======================================================================
HELPERS = '''
# ---------------HCL_Changes---------------
# Own-digit copy: the extra digits come from the SAME image (same font,
# colour, size, lighting) instead of a sticker from another image.
# _get_tight_digit_sticker above is kept but no longer used.
def _digit_boxes_px(self, label, w_factor, h_factor, pad):
    """YOLO rows -> digit boxes [x1, y1, x2, y2] in padded-image pixels, sorted left->right."""
    boxes = []
    for l in label:
        x1 = w_factor * (l[1] - l[3] / 2) + pad[0]
        y1 = h_factor * (l[2] - l[4] / 2) + pad[2]
        x2 = w_factor * (l[1] + l[3] / 2) + pad[0]
        y2 = h_factor * (l[2] + l[4] / 2) + pad[2]
        boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
    boxes.sort(key=lambda b: b[0])
    return boxes

def _copy_own_digits(self, image, boxes, need):
    """Paste `need` copies of the image's own digits to the right of the number.
    image: tensor C x H x W (padded). Returns (new_image, new_right_edge) or (image, None)."""
    _, H, W = image.shape
    dh = max(b[3] for b in boxes) - min(b[1] for b in boxes)
    if len(boxes) >= 2:
        gap = max(1, boxes[1][0] - boxes[0][2])           # same gap as the real number
    else:
        gap = max(1, int(dh * random.uniform(0.10, 0.20)))
    out = image.clone()
    x = max(b[2] for b in boxes)                           # right edge of the number
    for _ in range(need):
        x1, y1, x2, y2 = random.choice(boxes)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        pw, ph = x2 - x1, y2 - y1
        px = x + gap
        if pw <= 0 or ph <= 0 or px + pw > W:
            return image, None                             # doesn't fit -> no change
        out[:, y1:y2, px:px + pw] = image[:, y1:y2, x1:x2]
        x = px + pw
    return out, x
# -------------------------------------------
'''

OWN_DIGIT_BLOCK = '''
sticker = None                     # old cross-image sticker no longer used
x_original = x2
p_sticker = 0.15 if jerseyNumber_len == 1 else 0.05
if state == 1 and random.random() < p_sticker:
    boxes = self._digit_boxes_px(label, w_factor, h_factor, pad)
    # only when every box belongs to the number (skips the "far-apart" 2-box case)
    if len(boxes) == jerseyNumber_len:
        need = 3 - jerseyNumber_len
        new_image, new_x2 = self._copy_own_digits(image, boxes, need)
        if new_x2 is not None:
            image = new_image
            x2 = new_x2                    # crop box now includes the copied digits
            state = 2
            jerseyNumber_len = jerseyNumber_len + need
            digit_number = [self.IGNORE_INDEX, self.IGNORE_INDEX]
'''

RUNA_CROP = '''
# The real image area inside the padded square (padding is black):
ox1, oy1 = pad[0], pad[2]
ox2, oy2 = pad[0] + w_factor, pad[2] + h_factor

if state == 0 or random.random() < 0.5:
    # 50%: the whole player image (closest to the client's crop)
    x_min, y_min, x_max, y_max = ox1, oy1, ox2, oy2
else:
    # 50%: anything between "number + small margin" and "whole player".
    # x1, y1, x2, y2 = number box (includes copied digits from the own-digit step).
    H = max(1.0, y2 - y1)
    tx1, ty1 = max(ox1, x1 - 0.37 * H), max(oy1, y1 - 0.48 * H)   # tightest allowed box
    tx2, ty2 = min(ox2, x2 + 0.48 * H), min(oy2, y2 + 0.53 * H)
    x_min = ox1 + (tx1 - ox1) * random.random()   # each edge randomly between
    y_min = oy1 + (ty1 - oy1) * random.random()   # tight and full image
    x_max = tx2 + (ox2 - tx2) * random.random()
    y_max = ty2 + (oy2 - ty2) * random.random()

image = image[:, int(y_min):int(y_max), int(x_min):int(x_max)]
image, _ = pad_to_square(image, 0)                # same rule as validation and test
'''

VAL_BLOCK = '''
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Unable to read image: {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)            # Run A: RGB, like training
h, w = image.shape[:2]
s = max(h, w)                                             # Run A: pad to square (black)
canvas = np.zeros((s, s, 3), dtype=image.dtype)
canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = image
image = canvas.astype("float32") / 255.0
resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)   # no 114 / centre crop
'''


def wrap(title, old, new, ind):
    """HCL marker + title + old code commented + new code + end marker."""
    pad = " " * ind
    return (pad + MARK + "\n"
            + "".join(pad + "# " + t + "\n" for t in title)
            + comment_block(old)
            + indent(new, ind)
            + pad + END + "\n")


# ======================================================================
# Dataset file
# ======================================================================
def patch_dataset(src):
    done = []

    # ---- 1. helper methods ------------------------------------------
    if "def _copy_own_digits" in src:
        print("[dataset] helpers already present -> skipped")
    else:
        a, b = class_region(src, "JerseyWithState_Dataset")
        anchor = re.search(r"^( *)return image\[:, min_y:max_y, min_x:max_x\], sticker_digits\n",
                           src[a:b], re.M)
        if not anchor:
            fail("end of _get_tight_digit_sticker not found in JerseyWithState_Dataset")
        method_ind = len(anchor.group(1)) - 4
        pos = a + anchor.end()
        src = src[:pos] + indent(HELPERS, method_ind) + src[pos:]
        done.append("helpers _digit_boxes_px / _copy_own_digits added")

    # ---- 2. sticker block -> own-digit copy --------------------------
    if "self._copy_own_digits(image, boxes, need)" in src and "Own-digit copy replaces" in src:
        print("[dataset] own-digit block already present -> skipped")
    else:
        a, b = class_region(src, "JerseyWithState_Dataset")
        body = src[a:b]
        m1 = re.search(r"^( *)sticker = None\s*\n\s*x_original = x2\s*\n", body, re.M)
        m2 = re.search(r"^ *label\[0, 1\] = \(\(x1 \+ x2\) / 2\) / padded_w", body, re.M)
        if not m1 or not m2 or m2.start() < m1.start():
            fail("sticker block (sticker = None / x_original = x2 ... label[0, 1] = ...) not found")
        ind = len(m1.group(1))
        old = body[m1.start():m2.start()]
        new = wrap(["Own-digit copy replaces the cross-image sticker.",
                    "Old sticker block (commented out, not deleted):"], old, OWN_DIGIT_BLOCK, ind) + "\n"
        src = src[:a + m1.start()] + new + src[a + m2.start():]
        done.append("sticker block commented out, own-digit copy inserted")

    # ---- 3. tight crop -> Run A context crop -------------------------
    if "Run A: context crop" in src:
        print("[dataset] Run A crop already present -> skipped")
    else:
        a, b = class_region(src, "JerseyWithState_Dataset")
        body = src[a:b]
        m1 = re.search(r"^( *)x_min = label\[0, 1\] \* padded_w - label\[0, 3\] \* padded_w / 2\.0", body, re.M)
        if not m1:
            fail("tight crop start (x_min = label[0, 1] * padded_w ...) not found")
        m2 = re.compile(r"^ *image = image\[:, int\(y_min\):int\(y_max\), int\(x_min\):int\(x_max\)\]\n",
                        re.M).search(body, m1.start())
        if not m2:
            fail("tight crop end (image = image[:, int(y_min):...]) not found")
        ind = len(m1.group(1))
        old = body[m1.start():m2.end()]
        new = wrap(["Run A: context crop (number-level ... whole player), then pad to square.",
                    "Old tight crop (commented out, not deleted):"], old, RUNA_CROP, ind)
        src = src[:a + m1.start()] + new + src[a + m2.end():]
        done.append("tight crop commented out, Run A context crop inserted")

    # ---- 4. validation preprocessing ---------------------------------
    if "Run A: RGB, like training" in src:
        print("[dataset] validation change already present -> skipped")
    else:
        a, b = class_region(src, "JerseyNumber_ValidationDataset_V2")
        body = src[a:b]
        m1 = re.search(r"^( *)image = cv2\.imread\(image_path\)", body, re.M)
        m2 = re.search(r"^ *resized_image = resized_image\.transpose\(2, 0, 1\)", body, re.M)
        if not m1 or not m2 or m2.start() < m1.start():
            fail("validation image block (cv2.imread ... transpose) not found in V2")
        ind = len(m1.group(1))
        old = body[m1.start():m2.start()]
        new = wrap(["Run A: RGB + pad to square + resize 96 (same rule as training).",
                    "Old BGR + 114 -> centre-crop 96 (commented out, not deleted):"], old, VAL_BLOCK, ind)
        src = src[:a + m1.start()] + new + src[a + m2.start():]
        done.append("validation: BGR/114-centre-crop commented out, RGB + pad-to-square inserted")

    return src, done


# ======================================================================
# Training file
# ======================================================================
def patch_train(src):
    if "Run A: image is already square" in src:
        print("[train] transform change already present -> skipped")
        return src, []
    m = re.search(r"^( *)(Transforms\.RandomResizedCrop\(\(96, 96\).*\),)[ \t]*\n", src, re.M)
    if not m:
        fail("Transforms.RandomResizedCrop((96, 96), ...) line not found in training file")
    ind = len(m.group(1))
    pad = " " * ind
    new = (pad + MARK + "\n"
           + pad + "# Run A: the dataset returns a square (padded) context crop, so only resize.\n"
           + pad + "# RandomResizedCrop could cut the number off while the label still says \"27\".\n"
           + pad + "# " + m.group(2) + "\n"
           + pad + "Transforms.Resize((96, 96), antialias=True),          # Run A: image is already square (padded)\n"
           + pad + END + "\n")
    return src[:m.start()] + new + src[m.end():], ["RandomResizedCrop commented out, Resize((96, 96)) inserted"]


if __name__ == "__main__":
    ds_src = open(DATASET_FILE).read()
    tr_src = open(TRAIN_FILE).read()

    # build both results first; write only if everything was found
    new_ds, ds_done = patch_dataset(ds_src)
    new_tr, tr_done = patch_train(tr_src)

    for path, old, new in [(DATASET_FILE, ds_src, new_ds), (TRAIN_FILE, tr_src, new_tr)]:
        compile(new, path, "exec")                      # syntax check before writing
        if new != old:
            shutil.copy(path, path + ".bak_owndigits_runA")
            open(path, "w").write(new)

    print("\nChanges made:")
    for d in ds_done:
        print("  ", DATASET_FILE, "->", d)
    for d in tr_done:
        print("  ", TRAIN_FILE, "->", d)
    if not ds_done and not tr_done:
        print("   nothing (already applied)")
    else:
        print("\nBackups: *.bak_owndigits_runA   (restore with: cp X.bak_owndigits_runA X)")
