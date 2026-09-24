import time
import datetime
import logging
from tqdm import tqdm

import argparse

import numpy as np
import pandas as pd
import math

from utils.util import *
# ---------------HCL_Changes---------------
# Training now uses the state-aware dataset (returns images, digit_number,
# jerseyNumber_len, state). Use the exact class names from your jersey_dataset.py.
# from utils.jersey_dataset import jersey_Dataset, jerseyNumber_ValidationDataset_V2
from utils.jersey_dataset import JerseyWithState_Dataset, jerseyNumber_ValidationDataset_V2
# -------------------------------------------
from utils.loss import make_loss_fn
from utils.autoAugment import AutoAugment

from subModules.backbone_ying import MultiTaskLearner

import csv
import random
import matplotlib
matplotlib.use("Agg")  # no display needed - safe for terminal/SSH use
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as Transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset
from torch.amp.autocast_mode import autocast

from torch.utils.tensorboard import SummaryWriter

# If your dataset pads a "missing" second digit with a placeholder value
# (for single-digit jersey numbers), set that value here so it gets excluded
# from per-digit accuracy stats. Leave as None if every label is a real 0-9 digit.
# ---------------HCL_Changes---------------
# Blank (10) and ignore (-100) targets are already skipped by the
# "true_d not in per_digit_total" check in eval_one_epoch, so None is fine.
# -------------------------------------------
IGNORE_DIGIT_LABEL = None

# ---------------HCL_Changes---------------
# Must match IGNORE_INDEX in utils/loss.py and the dataset.
IGNORE_INDEX = -100
BLANK_DIGIT = 10


def load_matching_weights(model, state_dict):
    """
    Load only tensors whose name AND shape match the current model.
    Needed for old-architecture checkpoints: strict=False skips missing /
    unexpected keys (e.g. the removed digital head) but still raises on a
    shape mismatch (e.g. digit1 head 11 -> 10 classes).
    """
    model_state = model.state_dict()
    filtered = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = [k for k in state_dict if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} tensors; skipped {len(skipped)} "
          f"(removed or shape changed): {skipped}")


def compute_sample_correct(pred_d1, pred_d2, pred_state, digit_labels, state_labels):
    """
    A sample is correct when:
      - predicted state == true state, AND
      - if true state is 1 (readable 1-2 digit number): digit1 and digit2 also match.
    For state 0 / 2 there are no digits to check.
    """
    state_ok = pred_state == state_labels
    digits_ok = (pred_d1 == digit_labels[:, 0]) & (pred_d2 == digit_labels[:, 1])
    needs_digits = state_labels == 1
    return state_ok & (digits_ok | ~needs_digits), state_ok
# -------------------------------------------


def is_milestone_epoch(epoch):
    """
    Returns True for the epochs we want a clean train/val accuracy + loss
    snapshot for: epoch 1, then every 10th epoch (10, 20, 30, 40, ...).
    """
    return epoch == 1 or (epoch % 10 == 0 and epoch != 0)


def plot_digit_accuracy(per_digit_stats, save_path, title="Per-Digit Accuracy"):
    digits = list(range(10))
    totals = [per_digit_stats["total"][d] for d in digits]
    corrects = [per_digit_stats["correct"][d] for d in digits]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(digits, totals, color="#e8e8e8", label="Total")
    ax.bar(digits, corrects, color="#1f6f9c", label="Corrected Jersey Number")

    for i, d in enumerate(digits):
        ax.text(d, totals[i] + max(totals) * 0.015, str(totals[i]),
                ha="center", va="bottom", fontsize=9)
        ax.text(d, corrects[i] * 0.5, str(corrects[i]),
                ha="center", va="center", fontsize=9, color="white")

    ax.set_xlabel("Jersey Number")
    ax.set_ylabel("Number of Jersey Numbers")
    ax.set_title(title)
    ax.set_xticks(digits)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved per-digit accuracy plot to {save_path}")


def save_mismatches_csv(mismatches, save_path):
    if not mismatches:
        print("No mismatches to save.")
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mismatches[0].keys())
        writer.writeheader()
        writer.writerows(mismatches)
    print(f"Saved {len(mismatches)} misclassified samples to {save_path}")


def save_milestone_metrics_csv(milestone_metrics, save_path):
    if not milestone_metrics:
        print("No milestone metrics to save.")
        return
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=milestone_metrics[0].keys())
        writer.writeheader()
        writer.writerows(milestone_metrics)
    print(f"Saved milestone metrics for " f"{len(milestone_metrics)} epochs to {save_path}")


def plot_training_validation_loss(milestone_metrics, save_path):
    """
    plot and save training loss vs validation loss by using the milestone epochs"""
    if not milestone_metrics:
        print("No milestone metrics available for loss plot.")
    epochs = [m["epoch"] for m in milestone_metrics]
    train_losses = [m["train_loss"] for m in milestone_metrics]
    val_losses = [m["val_loss"] for m in milestone_metrics]
    plt.plot(
        epochs,
        train_losses,
        marker='o',
        linewidth=2,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        marker='o',
        linewidth=2,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved training/validation loss plot to {save_path}")


def build_checkpoint(epoch, model, optimizer, lr_scheduler, scaler,
                     pre_accuracy, epochs_since_improvement, milestone_metrics):
    """
    Build a full training-state checkpoint that can be used to resume training
    exactly where it left off. Keeps the "state_dict" key so older weight-only
    loaders (e.g. --pre_trained_model) stay backward compatible.
    """
    return {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "pre_accuracy": pre_accuracy,
        "epochs_since_improvement": epochs_since_improvement,
        "milestone_metrics": milestone_metrics,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Training Epoch{epoch}")

    avg_loss = 0
    correct = 0
    total = 0
    # ---------------HCL_Changes---------------
    state_correct = 0
    # -------------------------------------------

    # ---------------HCL_Changes---------------
    # JerseyWithState_Dataset.collate_fn returns (images, digit_number, jerseyNumber_len, state)
    # for batch_i, (images, whole_number_labels, digit_number_labels, _) in enumerate(pbar):
    #     images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)
    for batch_i, (images, digit_number_labels, _, state_labels) in enumerate(pbar):
        images, digit_number_labels, state_labels = images.to(device), digit_number_labels.to(device), state_labels.to(device)
        # -------------------------------------------

        optimizer.zero_grad()

        with autocast("cuda", enabled=True):
            # ---------------HCL_Changes---------------
            # digital head removed, state head added.
            # ASSUMED model output order: feat, digit1, digit2, state -- match your MultiTaskLearner.forward
            # _, logits_whole, logits_digit1, logits_digit2 = model(images)
            _, logits_digit1, logits_digit2, logits_state = model(images)
            # -------------------------------------------
            #print("IMAGE DTYPE:", images.dtype)
            #print("LOGITS DTYPE:", logits_whole.dtype)
            # ---------------HCL_Changes---------------
            # loss = criterion(
            #         logits_whole,
            #         logits_digit1,
            #         logits_digit2,
            #         whole_number_labels,
            #         digit_number_labels,
            #         )
            loss = criterion(
                    logits_digit1,
                    logits_digit2,
                    logits_state,
                    digit_number_labels,
                    state_labels,
                    )
            # -------------------------------------------

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        avg_loss += loss.item()

        # --- training accuracy tracking (digit-level match, same rule as eval) ---
        with torch.no_grad():
            _, pred_d1 = torch.max(logits_digit1, 1)
            _, pred_d2 = torch.max(logits_digit2, 1)
            # ---------------HCL_Changes---------------
            _, pred_state = torch.max(logits_state, 1)

            # correct_d1 = (pred_d1 == digit_number_labels[:, 0])
            # correct_d2 = (pred_d2 == digit_number_labels[:, 1])
            #
            # correct += (correct_d1 & correct_d2).sum().item()
            sample_ok, state_ok = compute_sample_correct(
                pred_d1, pred_d2, pred_state, digit_number_labels, state_labels
            )
            correct += sample_ok.sum().item()
            state_correct += state_ok.sum().item()
            # -------------------------------------------
            total += images.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.1f}",
            acc=f"{(correct / total) * 100:.1f}%",
            # ---------------HCL_Changes---------------
            state_acc=f"{(state_correct / total) * 100:.1f}%",
            # -------------------------------------------
        )

    train_accuracy = correct / total
    return avg_loss / len(loader), train_accuracy


def eval_one_epoch(model, val_loader, criterion, device, epoch, collect_mismatches=False):
    model.eval()
    pbar = tqdm(val_loader, desc=f"Evaluating Epoch{epoch}")

    val_loss = 0
    correct = 0
    # ---------------HCL_Changes---------------
    state_correct = 0
    # -------------------------------------------

    per_digit_correct = {d: 0 for d in range(10)}
    per_digit_total = {d: 0 for d in range(10)}
    mismatches = []

    print(pbar)
    with torch.no_grad():
        for batch_i, (images, digit_number_labels, whole_number_labels ) in enumerate(pbar):
            images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)
                #for batch_i, (images, digit_number_labels, whole_number_labels, filenames) in enumerate(pbar):

            # ---------------HCL_Changes---------------
            # jerseyNumber_ValidationDataset_V2 has no state label, so derive it:
            #   digit1 == 10 (blank) -> no number -> state 0
            #   otherwise            -> 1-2 digit number -> state 1
            # (V2's collate only accepts 1 or 2 digits, so state 2 never occurs here.)
            # For state 0 the digit targets become -100 so the loss ignores them.
            state_labels = torch.where(
                digit_number_labels[:, 0] == BLANK_DIGIT,
                torch.zeros_like(digit_number_labels[:, 0]),
                torch.ones_like(digit_number_labels[:, 0]),
            )
            digit_targets = digit_number_labels.clone()
            digit_targets[state_labels != 1] = IGNORE_INDEX

            # feat, logits_whole, logits_digit1, logits_digit2 = model(images)
            feat, logits_digit1, logits_digit2, logits_state = model(images)

            # loss = criterion(logits_whole, logits_digit1, logits_digit2,
            #                  whole_number_labels, digit_number_labels)
            loss = criterion(logits_digit1, logits_digit2, logits_state,
                             digit_targets, state_labels)
            # -------------------------------------------

            val_loss += loss.item()

            _, pre_d1 = torch.max(logits_digit1, 1)
            _, pre_d2 = torch.max(logits_digit2, 1)
            # ---------------HCL_Changes---------------
            _, pre_state = torch.max(logits_state, 1)
            # -------------------------------------------

            # ---------------HCL_Changes---------------
            # correct_d1 = (pre_d1 == digit_number_labels[:, 0])
            # correct_d2 = (pre_d2 == digit_number_labels[:, 1])
            # correct += (correct_d1 & correct_d2).sum().item()
            correct_d1 = (pre_d1 == digit_targets[:, 0])
            correct_d2 = (pre_d2 == digit_targets[:, 1])
            sample_ok, state_ok = compute_sample_correct(
                pre_d1, pre_d2, pre_state, digit_targets, state_labels
            )
            correct += sample_ok.sum().item()
            state_correct += state_ok.sum().item()
            # -------------------------------------------

            for pos, (preds, labels, is_correct) in enumerate([
                # ---------------HCL_Changes---------------
                # (pre_d1, digit_number_labels[:, 0], correct_d1),
                # (pre_d2, digit_number_labels[:, 1], correct_d2),
                (pre_d1, digit_targets[:, 0], correct_d1),
                (pre_d2, digit_targets[:, 1], correct_d2),
                # -------------------------------------------
            ]):
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                correct_np = is_correct.cpu().numpy()

                for sample_i in range(len(labels_np)):
                    true_d = int(labels_np[sample_i])

                    if IGNORE_DIGIT_LABEL is not None and true_d == IGNORE_DIGIT_LABEL:
                        continue
                    if true_d not in per_digit_total:
                        continue

                    per_digit_total[true_d] += 1
                    if correct_np[sample_i]:
                        per_digit_correct[true_d] += 1
                    elif collect_mismatches:
                        # ---------------HCL_Changes---------------
                        # 'filenames' was never defined (V2 doesn't return it), so this
                        # crashed on the final eval. Look the path up from the dataset
                        # instead (valid because val_loader uses shuffle=False).
                        global_i = batch_i * val_loader.batch_size + sample_i
                        # -------------------------------------------
                        mismatches.append({
                            "batch_index": batch_i,
                            "sample_in_batch": sample_i,
                            "digit_position": pos + 1,
                            "true_digit": true_d,
                            "predicted_digit": int(preds_np[sample_i]),
                            # ---------------HCL_Changes---------------
                            # "filename": filenames[sample_i],
                            "filename": val_loader.dataset.images_path[global_i],
                            "predicted_state": int(pre_state[sample_i].item()),
                            # -------------------------------------------
                        })

            pbar.set_postfix(
                val_loss=f"{loss.item():.1f}",
            )

    accuracy = correct / len(val_loader.dataset)
    # ---------------HCL_Changes---------------
    state_accuracy = state_correct / len(val_loader.dataset)
    # print(f"Val Loss: {val_loss/len(val_loader):.4f}, State Accuracy: {100 * accuracy:.2f}%")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Accuracy: {100 * accuracy:.2f}%, State Accuracy: {100 * state_accuracy:.2f}%")
    # -------------------------------------------

    per_digit_stats = {"correct": per_digit_correct, "total": per_digit_total}

    if collect_mismatches:
        return val_loss / len(val_loader), accuracy, per_digit_stats, mismatches
    return val_loss / len(val_loader), accuracy, per_digit_stats


if __name__ == "__main__":
    SEED_VALUE = 42
    set_seed(SEED_VALUE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch"
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="./datasets/training_dataset_Ying",
        help="path to data config file",
    )

    parser.add_argument(
        "--n_cpu",
        type=int,
        default=8,
        help="number of cpu threads to use during batch generation",
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="interval between saving model weights",
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="interval (in epochs) between validation runs",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="epochs with no val-accuracy improvement before early stopping",
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="if set, randomly subsets training data to this many samples",
    )

    parser.add_argument(
        "--output", type=str, default="./training_runs", help="root folder for run outputs"
    )
    parser.add_argument(
        "--model_name", type=str, default="jerseyNumberRecognizer", help="model name"
    )
    parser.add_argument(
        "--pre_trained_model",
        type=str,
        default=False,
        help="path of the pre trained model",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint to resume from (restores model, optimizer, "
             "scheduler, scaler, epoch, best-accuracy and RNG state)",
    )

    # Autoaugment
    parser.add_argument(
        "--autoAugment",
        action="store_false",
        default=True,
        help="gradient accumulation step",
    )

    #opt = parser.parse_args()
    #now = time.localtime()
    #time_now = str(time.strftime("%Y%m%d%H%M%S", now))
    #run_dir = os.path.join(opt.output, time_now)
    #os.makedirs(run_dir, exist_ok=True)
    #output = os.path.join(run_dir, opt.model_name)

    opt = parser.parse_args()
    os.makedirs("output", exist_ok=True)

    now = time.localtime()
    date_str = str(time.strftime("%d%m%Y", now))

    os.makedirs(opt.output, exist_ok=True)
    existing_versions = [
        d for d in os.listdir(opt.output)
        if d.startswith(f"train_{date_str}_v") and os.path.isdir(opt.output + "/" + d)
    ]
    if existing_versions:
        version_numbers = [
            int(d.split("_v")[-1]) for d in existing_versions if d.split("_v")[-1].isdigit()
        ]
        next_version = max(version_numbers) + 1 if version_numbers else 1
    else:
        next_version = 1

    time_now = f"train_{date_str}_v{next_version}"
    run_dir = opt.output + "/" + time_now
    os.makedirs(run_dir, exist_ok=True)
    output = run_dir + "/" + opt.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("cpu_count():", os.cpu_count())
    print("device:", device)
    print("run output folder:", output)
    torch.backends.cudnn.benchmark = True

    train_path = opt.data_config

    validation_path = "../../Datasets/validation_dataset_Ying"#"../../Datasets/validation_new" #"./simple2D_validation" #"./datasets/validation_dataset_Ying"

    train_transform_list = [
        Transforms.RandomResizedCrop((96, 96), scale=(0.4, 1.0)),
        Transforms.RandomApply(
            [Transforms.ElasticTransform(alpha=35.0, sigma=5.0)], p=0.6
        ),
    ]

    '''if opt.autoAugment:
        train_transform_list.extend([
            Transforms.Lambda(
                lambda x: Transforms.ToPILImage()(x) if torch.is_tensor(x) else x
            ),
            AutoAugment()
        ])

    train_transform_list.extend(
        [
            Transforms.RandomApply(
                [Transforms.RandomPerspective(distortion_scale=0.4, p=1)],
                p=0.4
            ),
            Transforms.RandomApply(
                [Transforms.ColorJitter(brightness=0.15, contrast=0.05)],
                p=0.5,
            ),
        ]
    )'''
    #---------------------------------HCL_Changes---------------------------------
    if opt.autoAugment:
        train_transform_list.append(AutoAugment())

    train_transform_list.extend(
        [
            Transforms.RandomApply(
                [Transforms.RandomRotation(degrees=(-12, 12),fill=0)], p=0.5
            ),
            Transforms.RandomApply(
                [Transforms.RandomPerspective(distortion_scale=0.2, p=1)], p=0.25
            ),
            Transforms.RandomGrayscale(p=0.2),
            Transforms.RandomApply(
                [Transforms.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.2,hue=0.03)],
                p=0.7,
            ),
            Transforms.RandomApply([Transforms.GaussianBlur(kernel_size=3,sigma=(0.1,1.2))], p=0.3),
            Transforms.ToTensor(),
            #Transforms.Lambda(
                #lambda x: x if torch.is_tensor(x) else Transforms.ToTensor()(x)
            #),
        ]
    )

    # ---------------HCL_Changes---------------
    # train_dataset = jersey_Dataset(
    #     train_path, transform=Transforms.Compose(train_transform_list)
    # )
    train_dataset = JerseyWithState_Dataset(
        train_path, transform=Transforms.Compose(train_transform_list)
    )
    # -------------------------------------------

    if opt.max_train_samples is not None and opt.max_train_samples < len(train_dataset):
        rng = random.Random(SEED_VALUE)
        indices = list(range(len(train_dataset)))
        rng.shuffle(indices)
        indices = indices[: opt.max_train_samples]
        train_dataset = Subset(train_dataset, indices)
        print(f"Limited training set to {opt.max_train_samples} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, "collate_fn") else train_dataset.dataset.collate_fn,
    )

    model = MultiTaskLearner().to(device)

    if opt.pre_trained_model:
        # ---------------HCL_Changes---------------
        # Old checkpoints have the digital head and an 11-class digit1 head;
        # strict=False still errors on the digit1 shape mismatch, so load only
        # tensors that match. The digit1 and state heads start fresh in that case.
        # model.load_state_dict(torch.load(opt.pre_trained_model), strict=False)
        load_matching_weights(model, torch.load(opt.pre_trained_model, map_location=device))
        # -------------------------------------------
        print("Loaded pretrained model!")

    val_dataset = jerseyNumber_ValidationDataset_V2(
        validation_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)

    #whole_class_counts = np.load("whole_number_class_counts.npy")
    #whole_class_counts = np.clip(whole_class_counts, a_min=1, a_max=None)

    #loss_fn = make_loss_fn(whole_class_counts=whole_class_counts)
    # ---------------HCL_Changes---------------
    # whole-number head removed. Optional: weight the state loss by class frequency, e.g.
    # state_class_counts = [n_state0, n_state1, n_state2]
    # loss_fn = make_loss_fn(state_class_counts=state_class_counts).to(device)
    # loss_fn = make_loss_fn()
    loss_fn = make_loss_fn().to(device)

    # print(loss_fn.whole_criterion.weight)
    print(loss_fn.state_criterion.weight)
    # -------------------------------------------
    print(loss_fn.criterion1.weight)

    initial_learning_rate = 1e-3

    #optimizer = torch.optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-2)
    #---------------------------------HCL_Changes---------------------------------
    #optimizer = torch.optim.SGD(model, lr=initial_learning_rate,, momentum = 0.00001, weight_decay=1e-2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay = 0.00001, nesterov=True)
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=40,
                                                  cycle_mult=2.0,
                                                  max_lr=initial_learning_rate,
                                                  min_lr=1e-6,
                                                  warmup_steps=5,
                                                  gamma=0.85)

    logwriter = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    scaler = torch.amp.GradScaler(enabled=False)    #True

    pre_accuracy, accuracy = float("-inf"), float("-inf")
    best_ckpt_path = output + "_best.pth"
    epochs_since_improvement = 0
    stopped_early = False

    # Holds the train/val loss + accuracy snapshot at milestone epochs
    # (epoch 1, then every 10th epoch), so you can build a clean
    # accuracy/loss-vs-epoch table for reporting.
    milestone_metrics = []

    # ----------------------------------------------------------------------
    # Resume logic: restores full training state so training continues exactly
    # where it stopped. Must run AFTER model/optimizer/scheduler/scaler and the
    # best-accuracy bookkeeping vars exist, since it overwrites them.
    # ----------------------------------------------------------------------
    start_epoch = 0

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {opt.resume}")

        print(f"Resuming from checkpoint: {opt.resume}")
        ckpt = torch.load(opt.resume, map_location="cpu")  # cpu load keeps RNG restore clean

        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            # NOTE: full resume only works with a checkpoint saved by the NEW
            # architecture; old-architecture checkpoints will fail here (use
            # --pre_trained_model for those instead).
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            scaler.load_state_dict(ckpt["scaler"])

            start_epoch = ckpt["epoch"] + 1
            pre_accuracy = ckpt.get("pre_accuracy", pre_accuracy)
            epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
            milestone_metrics = ckpt.get("milestone_metrics", [])

            rng = ckpt.get("rng")
            if rng is not None:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])

            print(f"Resumed at epoch {start_epoch} "
                  f"(best val acc so far: {pre_accuracy:.4f})")
        else:
            # bare state_dict / weights-only file -> can't fully resume, just load weights
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            # ---------------HCL_Changes---------------
            # model.load_state_dict(state, strict=False)
            load_matching_weights(model, state)
            # -------------------------------------------
            print("Weights-only checkpoint: loaded weights, starting from epoch 0.")

    for epoch in range(start_epoch, opt.epochs + 1):
        loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch)
        lr_scheduler.step()

        logwriter.add_scalar("Train/Loss", loss, epoch)
        logwriter.add_scalar("Train/Accuracy", train_accuracy, epoch)
        logwriter.add_scalar("Train/Lr", lr_scheduler.get_lr()[0], epoch)

        milestone = is_milestone_epoch(epoch)
        regular_eval = (epoch % opt.eval_interval == 0 and epoch != 0)

        if regular_eval or milestone:
            val_loss, accuracy, _ = eval_one_epoch(model, val_loader, loss_fn, device, epoch)

            logwriter.add_scalar("Evaluation/Accuracy", accuracy, epoch)
            logwriter.add_scalar("Evaluation/Loss", val_loss, epoch)

            if milestone:
                milestone_metrics.append({
                    "epoch": epoch,
                    "train_loss": loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": accuracy,
                })
                print(
                    f"[Milestone] Epoch {epoch}: "
                    f"Train Loss={loss:.4f}, Train Acc={100*train_accuracy:.2f}% | "
                    f"Val Loss={val_loss:.4f}, Val Acc={100*accuracy:.2f}%"
                )

            if regular_eval:
                if pre_accuracy < accuracy:
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"the best accuracy changed from {pre_accuracy:.4f} to {accuracy:.4f}")
                    pre_accuracy = accuracy
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += opt.eval_interval

                if opt.patience is not None and epochs_since_improvement >= opt.patience:
                    print(f"No improvement for {epochs_since_improvement} epochs. Stopping early at epoch {epoch}.")
                    stopped_early = True
                    break

        if epoch % opt.checkpoint_interval == 0:
            # Full resumable checkpoint (model + optimizer + scheduler + scaler
            # + epoch + best-acc bookkeeping + RNG). Feed this path to --resume.
            torch.save(
                build_checkpoint(epoch, model, optimizer, lr_scheduler, scaler,
                                 pre_accuracy, epochs_since_improvement, milestone_metrics),
                output + "_ckp_%d.pth" % epoch,
            )

        if epoch == (opt.epochs - 1) and not stopped_early:
            torch.save(model.state_dict(), output + "_%d_final.pth" % epoch)

    print(f"The best accuracy is {pre_accuracy:.4f}")

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path))
        print(f"Reloaded best checkpoint from {best_ckpt_path} for final reporting.")

    final_val_loss, final_accuracy, per_digit_stats, mismatches = eval_one_epoch(
        model, val_loader, loss_fn, device, epoch="final", collect_mismatches=True
    )
    print(f"Final (best-model) validation accuracy: {100 * final_accuracy:.2f}%")

    save_mismatches_csv(mismatches, os.path.join(run_dir, "wrong_predictions.csv"))
    save_milestone_metrics_csv(milestone_metrics, os.path.join(run_dir, "milestone_metrics.csv"))
    plot_training_validation_loss(
        milestone_metrics,
        os.path.join(run_dir, "training_validation_loss.png"))
    plot_digit_accuracy(
        per_digit_stats,
        os.path.join(run_dir, "per_digit_accuracy.png"),
        title=f"Per-Digit Accuracy (best model, val acc {100*final_accuracy:.2f}%)",
    )
    logwriter.close()
