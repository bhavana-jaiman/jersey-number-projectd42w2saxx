import os
import time
import datetime
import logging
from tqdm import tqdm

import argparse

from Analysis_plugin_2 import plug_save_image_batch
import numpy as np
import pandas as pd
import math

from utils.util import *
# ---------------HCL_Changes---------------
# with-state training: jerseyWithState_Dataset (state 0 / 2 samples carry
# -100 digit labels), loss with state term
# from utils.jersey_dataset import jersey_Dataset, jerseyNumber_ValidationDataset_V2
from utils.jersey_dataset import jerseyWithState_Dataset, jerseyNumber_ValidationDataset_V2
from utils.loss import make_loss_fn
# -------------------------------------------
from utils.autoAugment import AutoAugment

# ---------------HCL_Changes---------------
# from subModules.backbone_ying import MultiTaskLearner
from subModules.backbone_ying import MultiTaskLearnerWithState
# -------------------------------------------

import torch
import torchvision.transforms as Transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast

from torch.utils.tensorboard import SummaryWriter

# ---------------HCL_Changes---------------
# confusion matrices + loss / accuracy curves
from collections import Counter
import matplotlib
matplotlib.use("Agg")  # save plots to file, no display needed on the server
import matplotlib.pyplot as plt


def jersey_number(d1, d2):
    # [d, 10] -> d ; [d1, d2] -> d1d2
    return d1 if d2 == 10 else d1 * 10 + d2


def print_confusion(title, cm, names):
    # prints a confusion matrix, rows = true, cols = predicted
    print(title)
    print("true\\pred " + "".join(f"{n:>7}" for n in names))
    for name, row in zip(names, cm):
        print(f"{name:>9} " + "".join(f"{v:>7}" for v in row))


def plot_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax1.plot(history["val_epoch"], history["val_loss"], "o-", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(history["epoch"], [100 * a for a in history["train_acc"]], label="Train acc")
    ax2.plot(history["val_epoch"], [100 * a for a in history["val_acc"]], "o-", label="Val acc")
    ax2.plot(history["val_epoch"], [100 * a for a in history["val_state_acc"]], "s--", label="Val state acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)"); ax2.set_title("Accuracy")
    ax2.grid(True, alpha=0.3); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
# -------------------------------------------


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Training Epoch{epoch}")

    avg_loss = 0
    # ---------------HCL_Changes---------------
    # training accuracy (on augmented images)
    correct, state_correct, total = 0, 0, 0
    # -------------------------------------------

    # ---------------HCL_Changes---------------
    # jerseyWithState_Dataset returns: images, digit_number, jerseyNumber_len, state
    # for batch_i, (images, whole_number_labels, digit_number_labels, _) in enumerate(pbar):
    #     plug_save_image_batch(images, "OP3_Training")
    #     images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)
    for batch_i, (images, digit_number_labels, _, state_labels) in enumerate(pbar):
        plug_save_image_batch(images, "OP3_Training")
        images, digit_number_labels, state_labels = images.to(device), digit_number_labels.to(device), state_labels.to(device)
    # -------------------------------------------

        optimizer.zero_grad()

        with autocast("cuda", enabled=True):
            # ---------------HCL_Changes---------------
            # _, logits_whole, logits_digit1, logits_digit2 = model(images)
            _, logits_digit1, logits_digit2, logits_state = model(images)
            # -------------------------------------------

        # ---------------HCL_Changes---------------
        # no whole-number head; state loss added
        # loss = criterion(
        #         logits_whole,
        #         logits_digit1,
        #         logits_digit2,
        #         whole_number_labels,
        #         digit_number_labels,
        #     )
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

        # ---------------HCL_Changes---------------
        # sample correct = state right and, for state 1, both digits right
        with torch.no_grad():
            pre_d1 = logits_digit1.argmax(1)
            pre_d2 = logits_digit2.argmax(1)
            pre_state = logits_state.argmax(1)
            state_ok = pre_state == state_labels
            digits_ok = (pre_d1 == digit_number_labels[:, 0]) & (pre_d2 == digit_number_labels[:, 1])
            sample_ok = state_ok & ((state_labels != 1) | digits_ok)
            correct += sample_ok.sum().item()
            state_correct += state_ok.sum().item()
            total += images.size(0)

        # pbar.set_postfix(
        #     loss=f"{loss.item():.1f}",
        # )
        pbar.set_postfix(
            loss=f"{loss.item():.1f}",
            acc=f"{100 * correct / total:.1f}%",
            state_acc=f"{100 * state_correct / total:.1f}%",
        )

    # return avg_loss / len(loader)
    return avg_loss / len(loader), correct / total, state_correct / total
    # -------------------------------------------


def eval_one_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    pbar = tqdm(val_loader, desc=f"Evaluating Epoch{epoch}")

    val_loss = 0

    correct = 0
    # ---------------HCL_Changes---------------
    state_correct = 0
    state_cm = [[0] * 3 for _ in range(3)]      # rows = true state, cols = predicted
    digit1_cm = [[0] * 10 for _ in range(10)]   # state-1 samples only
    digit2_cm = [[0] * 11 for _ in range(11)]
    jersey_total, jersey_wrong = 0, 0
    wrong_pairs = Counter()                     # (true number, predicted number) -> count
    # -------------------------------------------

    with torch.no_grad():
        for batch_i, (images, digit_number_labels, whole_number_labels) in enumerate(pbar):
            plug_save_image_batch(images, "OP3_validation")
            images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)

            # ---------------HCL_Changes---------------
            # jerseyNumber_ValidationDataset_V2 has no state label, derive it:
            #   "no number" label [10, 10] -> state 0, digits [-100, -100] (not scored)
            #   otherwise (1-2 digits)      -> state 1
            # (V2 only accepts 1-2 digit labels, so state 2 does not occur here)
            state_labels = (digit_number_labels[:, 0] != 10).long()
            digit_number_labels[state_labels == 0] = -100

            # feat, logits_whole, logits_digit1, logits_digit2 = model(images)
            feat, logits_digit1, logits_digit2, logits_state = model(images)

            # loss = criterion(logits_whole, logits_digit1, logits_digit2,
            #                  whole_number_labels, digit_number_labels)
            loss = criterion(logits_digit1, logits_digit2, logits_state,
                             digit_number_labels, state_labels)
            # -------------------------------------------

            val_loss += loss.item()

            _, pre_d1 = torch.max(logits_digit1, 1)
            _, pre_d2 = torch.max(logits_digit2, 1)

            # ---------------HCL_Changes---------------
            # a sample is correct when the state is right and, for state 1,
            # both digits are right ("no number" correct = predicted state 0)
            _, pre_state = torch.max(logits_state, 1)
            state_ok = (pre_state == state_labels)
            state_correct += state_ok.sum().item()

            # correct_d1 = (pre_d1 == digit_number_labels[:, 0])
            # correct_d2 = (pre_d2 == digit_number_labels[:, 1])
            # correct += (correct_d1 & correct_d2).sum().item()
            correct_d1 = (pre_d1 == digit_number_labels[:, 0])
            correct_d2 = (pre_d2 == digit_number_labels[:, 1])
            digits_ok = (correct_d1 & correct_d2) | (state_labels == 0)
            correct += (state_ok & digits_ok).sum().item()

            # confusion matrices
            for i in range(images.size(0)):
                t_state, p_state = state_labels[i].item(), pre_state[i].item()
                state_cm[t_state][p_state] += 1
                if t_state == 1:
                    t1, t2 = digit_number_labels[i, 0].item(), digit_number_labels[i, 1].item()
                    p1, p2 = pre_d1[i].item(), pre_d2[i].item()
                    digit1_cm[t1][p1] += 1
                    digit2_cm[t2][p2] += 1
                    jersey_total += 1
                    if (t1, t2) != (p1, p2):
                        jersey_wrong += 1
                        wrong_pairs[(jersey_number(t1, t2), jersey_number(p1, p2))] += 1
            # -------------------------------------------

            pbar.set_postfix(
                Val_loss=f"{loss.item():.1f}",
            )

    # ---------------HCL_Changes---------------
    # print(f"Val Loss: {val_loss/len(val_loader):.4f}, State Accuracy: {100 * correct / len(val_loader.dataset):.2f}%")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / len(val_loader.dataset):.2f}%, "
          f"State Accuracy: {100 * state_correct / len(val_loader.dataset):.2f}%")

    # 1) state confusion (0 = no number, 1 = 1-2 digits, 2 = 3+ digits)
    print_confusion("State confusion (rows = true, cols = predicted):",
                    state_cm, ["0", "1", "2"])
    # 2) jersey numbers (state-1 samples): how many wrong, which mistakes
    if jersey_total > 0:
        print(f"Jersey numbers: {jersey_total} total, {jersey_wrong} wrong "
              f"({100 * jersey_wrong / jersey_total:.2f}%), "
              f"jersey accuracy {100 * (jersey_total - jersey_wrong) / jersey_total:.2f}%")
        print("Top 15 mistakes (true -> predicted : count):")
        for (t, p), n in wrong_pairs.most_common(15):
            print(f"   {t:>3} -> {p:<3} : {n}")
        print_confusion("Digit 1 confusion (rows = true, cols = predicted):",
                        digit1_cm, [str(d) for d in range(10)])
        print_confusion("Digit 2 confusion (rows = true, cols = predicted, 10 = blank):",
                        digit2_cm, [str(d) for d in range(11)])

    # return val_loss/len(val_loader), correct / len(val_loader.dataset)
    return (val_loss / len(val_loader), correct / len(val_loader.dataset),
            state_correct / len(val_loader.dataset))
    # -------------------------------------------


if __name__ == "__main__":
    SEED_VALUE = 42
    set_seed(SEED_VALUE)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=280, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="size of each image batch"
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
        "--output", type=str, default="./checkpoints", help="model name"
    )
    parser.add_argument(
        "--model_name", type=str, default="jerseyNumberRecognitior", help="model name"
    )
    parser.add_argument(
        "--pre_trained_model",
        type=str,
        default=False,
        help="path of the pre trained model",
    )

    # Autoaugment
    parser.add_argument(
        "--autoAugment",
        action="store_false",
        default=True,
        help="gradient accumulation step",
    )

    opt = parser.parse_args()
    os.makedirs("output", exist_ok=True)
    now = time.localtime()
    time_now = str(time.strftime("%Y%m%d%H%M%S", now))
    os.makedirs(opt.output + "/" + time_now, exist_ok=True)
    output = opt.output + "/" + time_now + "/" + opt.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("cpu_count():", os.cpu_count())
    print("device:", device)
    torch.backends.cudnn.benchmark = True

    train_path = opt.data_config

    # validation_path = "./datasets/validation_dataset_Ying"
    validation_path = "/home/eng_bhavana/workspace_bhavana/Datasets/70_30_dataset/validation_final"
    train_transform_list = [
        Transforms.RandomResizedCrop((96, 96), scale=(0.4, 1.0)),
        # Transforms.RandomApply(
        #     [Transforms.ElasticTransform(alpha=35.0, sigma=5.0)], p=0.6
        # ),
    ]

    if opt.autoAugment:
        train_transform_list.append(AutoAugment())

    train_transform_list.extend(
        [
            Transforms.RandomApply(
                [Transforms.RandomPerspective(distortion_scale=0.4, p=1)], p=0.4
            ),
            Transforms.RandomApply(
                [Transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)],
                p=0.5,
            ),
            Transforms.RandomApply(
                [Transforms.RandomRotation(degrees=(-30, 30), fill=128)], p=0.4
            ),
            Transforms.RandomApply(
                [Transforms.ElasticTransform(alpha=50.0, sigma=5.0, fill=128)], p=0.6
            ),
            Transforms.RandomApply([Transforms.GaussianBlur(kernel_size=3)], p=0.3),
            Transforms.ToTensor(),
            Transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=0.5),
        ]
    )

    # ---------------HCL_Changes---------------
    # train_dataset = jersey_Dataset(
    train_dataset = jerseyWithState_Dataset(
        train_path, transform=Transforms.Compose(train_transform_list)
    )
    # -------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        # ---------------HCL_Changes---------------
        # state head has BatchNorm1d: a last batch of 1 sample crashes in training
        drop_last=True,
        # -------------------------------------------
        collate_fn=train_dataset.collate_fn,
    )  #

    # ---------------HCL_Changes---------------
    # model = MultiTaskLearner().to(device)
    model = MultiTaskLearnerWithState().to(device)
    # -------------------------------------------

    if opt.pre_trained_model:
        model.load_state_dict(torch.load(opt.pre_trained_model), strict=False)
        print("Loaded pretrained model!")

    # ---------------HCL_Changes---------------
    # val_dataset = jerseyNumber_ValidationDataset_V2(
    val_dataset = jerseyNumber_ValidationDataset_V2(
            validation_path)
    # -------------------------------------------
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)

    # ---------------HCL_Changes---------------
    # loss_fn = make_loss_fn()
    loss_fn = make_loss_fn(lambda_state=0.3)
    # -------------------------------------------

    initial_learning_rate = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_learning_rate, weight_decay=1e-2)

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                 first_cycle_steps=40,
                                                 cycle_mult=2.0,
                                                 max_lr=initial_learning_rate,
                                                 min_lr=1e-6,
                                                 warmup_steps=5,
                                                 gamma=0.85)

    logwriter = SummaryWriter(log_dir="./logs/%s/" % time_now)

    scaler = torch.amp.GradScaler(enabled=True)

    pre_accuracy, accuracy = float("-inf"), float("-inf")

    # ---------------HCL_Changes---------------
    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_epoch": [], "val_loss": [], "val_acc": [], "val_state_acc": []}
    # -------------------------------------------

    for epoch in range(0, opt.epochs + 1):

        # ---------------HCL_Changes---------------
        # loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch)
        loss, train_acc, train_state_acc = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch)
        # -------------------------------------------
        lr_scheduler.step()

        logwriter.add_scalar("Train/Loss", loss, epoch)
        logwriter.add_scalar("Train/Lr", lr_scheduler.get_lr()[0], epoch)
        # ---------------HCL_Changes---------------
        logwriter.add_scalar("Train/Accuracy", train_acc, epoch)
        logwriter.add_scalar("Train/StateAccuracy", train_state_acc, epoch)
        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        # -------------------------------------------

        if epoch % 5 == 0 and epoch != 0:
            # ---------------HCL_Changes---------------
            # val_loss, accuracy = eval_one_epoch(model, val_loader, loss_fn, device, epoch)
            val_loss, accuracy, val_state_acc = eval_one_epoch(model, val_loader, loss_fn, device, epoch)
            # -------------------------------------------

            logwriter.add_scalar("Evaluation/Accuracy", accuracy, epoch)
            logwriter.add_scalar("Evaluation/Loss", val_loss, epoch)

            # ---------------HCL_Changes---------------
            logwriter.add_scalar("Evaluation/StateAccuracy", val_state_acc, epoch)
            history["val_epoch"].append(epoch)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(accuracy)
            history["val_state_acc"].append(val_state_acc)
            print(f"[Milestone] Epoch {epoch}: Train Loss={loss:.4f}, Train Acc={100 * train_acc:.2f}%, "
                  f"Train State Acc={100 * train_state_acc:.2f}% | Val Loss={val_loss:.4f}, "
                  f"Val Acc={100 * accuracy:.2f}%, Val State Acc={100 * val_state_acc:.2f}%")
            plot_curves(history, output + "_curves.png")   # overwritten at every validation
            # -------------------------------------------

        if pre_accuracy < accuracy:

            torch.save(
                model.state_dict(),
                (
                    output + "_best.pth"
                ),
            )

            print(
                f"the best accuracy changed from {pre_accuracy:.4f} to {accuracy:.4f}"
            )

            pre_accuracy = accuracy

        if epoch % opt.checkpoint_interval == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
            }

            torch.save(checkpoint, output + "_ckp_%d.pth" % epoch)

        if epoch == (opt.epochs - 1):

            torch.save(model.state_dict(), output + "_%d_final.pth" % epoch)

    print(f"The best accuracy is {pre_accuracy:.4f}")
