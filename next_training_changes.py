# =====================================================================
# NEXT TRAINING (exp6): all changes, in order
# Lines starting with "# OLD:" are the lines to replace; keep them commented out.
# Files:
#   A) utils/jersey_dataset.py
#   B) jerseyNumberRecognitiorWithState_Training.py
#   C) dataset folders (bash)
#   D) check + run command
# Status: [DONE] = you already did it, only verify   [NEW] = do it now   [OPTIONAL]
# =====================================================================


# #####################################################################
# A) utils/jersey_dataset.py
# #####################################################################

# ---- A1 [DONE] body_like_box: more small digits (centre 0.24 -> 0.20) ----
#      inside def body_like_box(...)   (BODY_R stays (0.14, 0.60))
    # r = float(np.clip(np.exp(random.gauss(np.log(0.24), 0.4)), BODY_R[0], BODY_R[1]))
    r = float(np.clip(np.exp(random.gauss(np.log(0.20), 0.4)), BODY_R[0], BODY_R[1]))


# ---- A2 [NEW] jerseyNumber_ValidationDataset_V2.__init__ : keep_3plus switch ----
#      REPLACE the def line, and put your existing 3+ skip block under "if not keep_3plus:"
#      (the skip block moves 4 spaces to the right; nothing else in __init__ changes)
    # OLD:
    # def __init__(self, root):
    def __init__(self, root, keep_3plus=False):
        super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
        # ... image_dir / images_path / labels_path lines unchanged ...

        # ---------------HCL_Changes---------------
        # validation_body -> keep_3plus=False (skip fake multi-player 3+)
        # syn3_dataset    -> keep_3plus=True  (synthetic 3-digit, state 2)
        if not keep_3plus:
            keep = [i for i, lp in enumerate(self.labels_path)
                    if np.loadtxt(lp, ndmin=1).size <= 2]
            self.images_path = [self.images_path[i] for i in keep]
            self.labels_path = [self.labels_path[i] for i in keep]
        # -------------------------------------------


# ---- A3 [VERIFY] new_collate_fn : 3+ label -> [-1, -1]  (must be BEFORE the raise) ----
        for label in individual_digits_labels:
            t_label = torch.from_numpy(label).view(-1)
            if t_label.size(0) == 1:
                t_label = F.pad(t_label, (0, 1), value=10)
            elif t_label.size(0) > 2:                          # 3+ digits -> state 2
                t_label = torch.tensor([-1.0, -1.0])
            elif t_label.size(0) != 2:
                raise ValueError(f"Unexpected label shape: {label.shape}")
            collated_labels.append(t_label)


# #####################################################################
# B) jerseyNumberRecognitiorWithState_Training.py
# #####################################################################

# ---- B1 [DONE] imports + RandomLowRes class (above def train_one_epoch) ----
import random
from PIL import Image

class RandomLowRes:
    def __init__(self, min_size=24, max_size=64, p=1.0):
        self.min_size, self.max_size, self.p = min_size, max_size, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.randint(self.min_size, self.max_size)
        small = img.resize((s, s), random.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST]))
        return small.resize((w, h), random.choice([Image.BILINEAR, Image.BICUBIC]))


# ---- B2 [NEW] import ConcatDataset (top of file) ----
from torch.utils.data import ConcatDataset


# ---- B3 [DONE] train_transform_list : low-res after Resize, rotation +-30 ----
        Transforms.Resize((96, 96)),                                          # (already there)
        Transforms.RandomApply([RandomLowRes(24, 64, p=1.0)], p=0.5),         # p=1.0 inside!
        # ...
        Transforms.RandomApply([Transforms.RandomRotation(30, fill=128)], p=0.4),   # was 20


# ---- B4 [VERIFY] eval_one_epoch : -1 -> state 2, digits ignored, 0.9 threshold ----
#      (these lines must exist; add them if grep in D1 does not find them)
            state_labels = (digit_number_labels[:, 0] != 10).long()
            state_labels[digit_number_labels[:, 0] == -1] = 2              # 3+ -> state 2
            digit_number_labels[state_labels != 1] = -100                  # digits not scored

            # ... model + loss ...

            prob = torch.softmax(logits_state, 1)
            _, pre_state = torch.max(logits_state, 1)
            weak = (pre_state == 2) & (prob[:, 2] < 0.9)                   # 3+ only if >= 0.9
            pre_state[weak] = torch.argmax(prob[weak, :2], 1)

            digits_ok = (correct_d1 & correct_d2) | (state_labels != 1)    # != 1  (not != 0)


# ---- B5 [NEW] learning rate for fine-tuning from exp5 ----
    # OLD:
    # initial_learning_rate = 1e-3
    initial_learning_rate = 3e-4


# ---- B6 [NEW] validation loader : validation_body + syn3_dataset ----
    val_dataset = jerseyNumber_ValidationDataset_V2(validation_path)          # (unchanged)

    # OLD:
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
    #                         num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)
    # ---------------HCL_Changes---------------
    syn3_path = "../Datasets/70_30_dataset/syn3_dataset"     # <- your syn3 folder (images/ + labels/)
    syn3_dataset = jerseyNumber_ValidationDataset_V2(syn3_path, keep_3plus=True)
    print("validation: body", len(val_dataset), "+ synthetic 3-digit", len(syn3_dataset))
    val_loader = DataLoader(ConcatDataset([val_dataset, syn3_dataset]), batch_size=64, shuffle=False,
                            num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)
    # -------------------------------------------


# #####################################################################
# C) dataset folders (bash, from New_arcitecture_2) - change paths to yours
# #####################################################################

# ---- C1 [DONE] 369 fake 3+ removed from validation_body ----

# ---- C2 [OPTIONAL, recommended] take the 800 source photos of syn3 out of training ----
"""
mkdir -p ./datasets/training_used_for_syn3
for f in ../Datasets/70_30_dataset/syn3_dataset/labels/*.txt; do
  n=$(basename "$f" .txt)
  mv ./datasets/training_dataset_Ying/"$n".jpg ./datasets/training_dataset_Ying/"$n".txt \
     ./datasets/training_used_for_syn3/ 2>/dev/null
done
ls ./datasets/training_used_for_syn3 | wc -l        # about 1600 (800 jpg + 800 txt)
"""


# #####################################################################
# D) check + run
# #####################################################################
"""
D1  check the important lines are there:
    grep -n "keep_3plus" utils/jersey_dataset.py
    grep -n "size(0) > 2" utils/jersey_dataset.py
    grep -n "== -1\|< 0.9\|state_labels != 1\|initial_learning_rate\|RandomLowRes\|syn3_path" \
         jerseyNumberRecognitiorWithState_Training.py

D2  syntax check:
    python -m py_compile utils/jersey_dataset.py jerseyNumberRecognitiorWithState_Training.py

D3  run (fine-tune from exp5, checkpoints at the end of each LR cycle):
    python jerseyNumberRecognitiorWithState_Training.py \
        --pre_trained_model <exp5 _best.pth> --epochs 480 --checkpoint_interval 40

D4  at start of log:   validation: body N + synthetic 3-digit 800
    first validation:  state confusion matrix has a "true 2" row with 800 images
    look at OP3_Training images: some blurry / small digits, digits still readable

D5  after training: test _best.pth, _ckp_120.pth, _ckp_280.pth on the client test,
    compare with exp5 (76.32 %, scale bins 9-18 px, state2 / state1)
"""
