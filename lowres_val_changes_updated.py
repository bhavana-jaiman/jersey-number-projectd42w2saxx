# =====================================================================
# exp6 changes: all in one place
#   Change 1: low-res augmentation      (training script)
#   Change 2: more small digits         (utils/jersey_dataset.py)
#   Change 3: low-res validation        (both files)
# =====================================================================


# #####################################################################
# CHANGE 1: Low-resolution augmentation
# FILE: jerseyNumberRecognitiorWithState_Training.py
# #####################################################################

# ---- 1a) ADD at the top with the imports (skip a line if already imported) ----
import random
from PIL import Image


# ---- 1b) ADD this class ABOVE  def train_one_epoch(...) ----
# ---------------HCL_Changes (low-res aug)---------------
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
# --------------------------------------------------------


# ---- 1c) train_transform_list : ADD right AFTER  Transforms.Resize((96, 96)), ----
        Transforms.Resize((96, 96)),                                   # (already there)
        # ---------------HCL_Changes (low-res aug)---------------
        Transforms.RandomApply([RandomLowRes(24, 64, p=1.0)], p=0.5),
        # --------------------------------------------------------


# #####################################################################
# CHANGE 2: Show the model more small digits
# FILE: utils/jersey_dataset.py  ->  def body_like_box(...)
# #####################################################################

# ---- 2a) REPLACE the  r = ...  line (BODY_R stays (0.14, 0.60)) ----
    # r = float(np.clip(np.exp(random.gauss(np.log(0.24), 0.4)), BODY_R[0], BODY_R[1]))
    r = float(np.clip(np.exp(random.gauss(np.log(0.20), 0.4)), BODY_R[0], BODY_R[1]))


# #####################################################################
# CHANGE 3: Low-res validation during training
# #####################################################################

# =====================================================================
# FILE: utils/jersey_dataset.py  ->  class jerseyNumber_ValidationDataset_V2
# =====================================================================

# ---- 3a) __init__ : REPLACE the first two lines ----
    # def __init__(self, root):
    #     super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
    def __init__(self, root, low_res=False):
        super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
        self.low_res = low_res
        # ... rest of __init__ unchanged (image_dir, images_path, labels_path, 3+ skip)


# ---- 3b) __getitem__ : ADD after imread / cvtColor, BEFORE the resize to 96 ----
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # (already there)

        # ---------------HCL_Changes (low-res val)---------------
        if self.low_res:
            rng = random.Random(index)            # same shrink every epoch
            s = rng.randint(24, 48)               # harsher: (20, 40)
            image = cv2.resize(image, (s, s), interpolation=cv2.INTER_AREA)
        # --------------------------------------------------------

        image = image.astype("float32") / 255.0               # (already there)
        resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)  # (already there)


# =====================================================================
# FILE: jerseyNumberRecognitiorWithState_Training.py
# =====================================================================

# ---- 3c) imports : ADD at the top ----
from torch.utils.data import Subset


# ---- 3d) __main__ : ADD right after the existing  val_loader = DataLoader(...) ----
    # ---------------HCL_Changes (low-res val)---------------
    val_small_full = jerseyNumber_ValidationDataset_V2(validation_path, low_res=True)
    val_small = Subset(val_small_full, list(range(0, len(val_small_full), 3)))  # every 3rd image
    val_small_loader = DataLoader(val_small, batch_size=64, shuffle=False,
                                  num_workers=opt.n_cpu, pin_memory=True,
                                  collate_fn=val_small_full.new_collate_fn)
    # --------------------------------------------------------


# ---- 3e) epoch loop : ADD inside  if epoch % 5 == 0 and epoch != 0:
#          right AFTER your existing  val_loss, accuracy, ... = eval_one_epoch(model, val_loader, ...)
            # ---------------HCL_Changes (low-res val)---------------
            print("------ low-res val ------")
            s_loss, s_acc, s_state = eval_one_epoch(model, val_small_loader, loss_fn, device, epoch)
            logwriter.add_scalar("Evaluation/LowRes_Accuracy", s_acc, epoch)
            score = (accuracy + s_acc) / 2
            print(f"[Score] normal {accuracy:.4f}  low-res {s_acc:.4f}  avg {score:.4f}")
            # --------------------------------------------------------


# ---- 3f) best-model check : REPLACE (pick _best.pth on the average score) ----
        # if pre_accuracy < accuracy:
        if epoch % 5 == 0 and epoch != 0 and pre_accuracy < score:

            torch.save(model.state_dict(), (output + "_best.pth"))   # (unchanged)

            # print(f"the best accuracy changed from {pre_accuracy:.4f} to {accuracy:.4f}")
            print(f"the best score changed from {pre_accuracy:.4f} to {score:.4f}")

            # pre_accuracy = accuracy
            pre_accuracy = score
