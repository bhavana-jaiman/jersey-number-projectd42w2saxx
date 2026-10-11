# =====================================================================
# Low-res validation during training: all changes
# Files: utils/jersey_dataset.py  and  jerseyNumberRecognitiorWithState_Training.py
# =====================================================================


# =====================================================================
# FILE 1: utils/jersey_dataset.py  ->  class jerseyNumber_ValidationDataset_V2
# =====================================================================

# ---- 1a) __init__ : REPLACE the first two lines ----
    # def __init__(self, root):
    #     super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
    def __init__(self, root, low_res=False):
        super(jerseyNumber_ValidationDataset_V2, self).__init__(root)
        self.low_res = low_res
        # ... rest of __init__ unchanged (image_dir, images_path, labels_path, 3+ skip)


# ---- 1b) __getitem__ : ADD after imread / cvtColor, BEFORE the resize to 96 ----
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
# FILE 2: jerseyNumberRecognitiorWithState_Training.py
# =====================================================================

# ---- 2a) imports : ADD at the top ----
from torch.utils.data import Subset


# ---- 2b) __main__ : ADD right after the existing  val_loader = DataLoader(...) ----
    # ---------------HCL_Changes (low-res val)---------------
    val_small_full = jerseyNumber_ValidationDataset_V2(validation_path, low_res=True)
    val_small = Subset(val_small_full, list(range(0, len(val_small_full), 3)))  # every 3rd image
    val_small_loader = DataLoader(val_small, batch_size=64, shuffle=False,
                                  num_workers=opt.n_cpu, pin_memory=True,
                                  collate_fn=val_small_full.new_collate_fn)
    # --------------------------------------------------------


# ---- 2c) epoch loop : ADD inside  if epoch % 5 == 0 and epoch != 0:
#          right AFTER your existing  val_loss, accuracy, ... = eval_one_epoch(model, val_loader, ...)
            # ---------------HCL_Changes (low-res val)---------------
            print("------ low-res val ------")
            s_loss, s_acc, s_state = eval_one_epoch(model, val_small_loader, loss_fn, device, epoch)
            logwriter.add_scalar("Evaluation/LowRes_Accuracy", s_acc, epoch)
            score = (accuracy + s_acc) / 2
            print(f"[Score] normal {accuracy:.4f}  low-res {s_acc:.4f}  avg {score:.4f}")
            # --------------------------------------------------------


# ---- 2d) best-model check : REPLACE (pick _best.pth on the average score) ----
        # if pre_accuracy < accuracy:
        if epoch % 5 == 0 and epoch != 0 and pre_accuracy < score:

            torch.save(model.state_dict(), (output + "_best.pth"))   # (unchanged)

            # print(f"the best accuracy changed from {pre_accuracy:.4f} to {accuracy:.4f}")
            print(f"the best score changed from {pre_accuracy:.4f} to {score:.4f}")

            # pre_accuracy = accuracy
            pre_accuracy = score
