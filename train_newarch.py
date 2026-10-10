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
# new-architecture dataset / loss (digit1 0-9, no whole-number label;
# no-number and 3+ digit samples carry -100 digit labels)
# from utils.jersey_dataset import jersey_Dataset, jerseyNumber_ValidationDataset_V2
# from utils.loss import make_loss_fn
from utils.jersey_dataset_New_3 import JerseyDataset, JerseyNumber_ValidationDataset_V2
from utils.loss_New import make_loss_fn
# -------------------------------------------
from utils.autoAugment import AutoAugment

# ---------------HCL_Changes---------------
# from subModules.backbone_ying import MultiTaskLearner
from subModules.backbone_New import MultiTaskLearner
# -------------------------------------------

import torch
import torchvision.transforms as Transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast

from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch):
    model.train()
    pbar = tqdm(loader, desc=f"Training Epoch{epoch}")

    avg_loss = 0

    # ---------------HCL_Changes---------------
    # new dataset returns: images, digit_number, jerseyNumber_len, state (no whole number)
    # for batch_i, (images, whole_number_labels, digit_number_labels, _) in enumerate(pbar):
    #     plug_save_image_batch(images, "OP3_Training")
    #     images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)
    for batch_i, (images, digit_number_labels, _, _) in enumerate(pbar):
        plug_save_image_batch(images, "OP3_Training")
        images, digit_number_labels = images.to(device), digit_number_labels.to(device)
    # -------------------------------------------

        optimizer.zero_grad()

        with autocast("cuda", enabled=True):
            # ---------------HCL_Changes---------------
            # _, logits_whole, logits_digit1, logits_digit2 = model(images)
            _, logits_digit1, logits_digit2 = model(images)
            # -------------------------------------------

        # ---------------HCL_Changes---------------
        # no whole-number head; no state in this model -> state arguments None
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
                None,
                digit_number_labels,
                None,
            )
        # -------------------------------------------

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        avg_loss += loss.item()

        pbar.set_postfix(
            loss=f"{loss.item():.1f}",
        )

    return avg_loss / len(loader)


def eval_one_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    pbar = tqdm(val_loader, desc=f"Evaluating Epoch{epoch}")

    val_loss = 0

    correct = 0

    with torch.no_grad():
        for batch_i, (images, digit_number_labels, whole_number_labels) in enumerate(pbar):
            plug_save_image_batch(images, "OP3_validation")
            images, digit_number_labels, whole_number_labels = images.to(device), digit_number_labels.to(device), whole_number_labels.to(device)

            # ---------------HCL_Changes---------------
            # digit1 head has no class 10 any more: a "no number" label [10, 10]
            # becomes [-100, -100] (not scored by the loss, counted as not correct)
            digit_number_labels[digit_number_labels[:, 0] == 10] = -100

            # feat, logits_whole, logits_digit1, logits_digit2 = model(images)
            feat, logits_digit1, logits_digit2 = model(images)

            # loss = criterion(logits_whole, logits_digit1, logits_digit2,
            #                  whole_number_labels, digit_number_labels)
            loss = criterion(logits_digit1, logits_digit2, None,
                             digit_number_labels, None)
            # -------------------------------------------

            val_loss += loss.item()

            _, pre_d1 = torch.max(logits_digit1, 1)
            _, pre_d2 = torch.max(logits_digit2, 1)

            correct_d1 = (pre_d1 == digit_number_labels[:, 0])
            correct_d2 = (pre_d2 == digit_number_labels[:, 1])
            correct += (correct_d1 & correct_d2).sum().item()

            pbar.set_postfix(
                Val_loss=f"{loss.item():.1f}",
            )

    print(f"Val Loss: {val_loss/len(val_loader):.4f}, State Accuracy: {100 * correct / len(val_loader.dataset):.2f}%")

    return val_loss/len(val_loader), correct / len(val_loader.dataset)


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

    validation_path = "./datasets/validation_dataset_Ying"
    train_transform_list = [
        Transforms.RandomResizedCrop((96, 96), scale=(0.4, 1.0)),
        Transforms.RandomApply(
            [Transforms.ElasticTransform(alpha=35.0, sigma=5.0)], p=0.6
        ),
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
            # Transforms.RandomApply([Transforms.GaussianBlur(kernel_size=3)], p=0.3),
            Transforms.ToTensor(),
        ]
    )

    # ---------------HCL_Changes---------------
    # train_dataset = jersey_Dataset(
    train_dataset = JerseyDataset(
        train_path, transform=Transforms.Compose(train_transform_list)
    )
    # -------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )  #

    model = MultiTaskLearner().to(device)

    if opt.pre_trained_model:
        model.load_state_dict(torch.load(opt.pre_trained_model), strict=False)
        print("Loaded pretrained model!")

    # ---------------HCL_Changes---------------
    # val_dataset = jerseyNumber_ValidationDataset_V2(
    val_dataset = JerseyNumber_ValidationDataset_V2(
            validation_path)
    # -------------------------------------------
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)

    loss_fn = make_loss_fn()

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

    for epoch in range(0, opt.epochs + 1):

        loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, epoch)
        lr_scheduler.step()

        logwriter.add_scalar("Train/Loss", loss, epoch)
        logwriter.add_scalar("Train/Lr", lr_scheduler.get_lr()[0], epoch)

        if epoch % 5 == 0 and epoch != 0:
            val_loss, accuracy = eval_one_epoch(model, val_loader, loss_fn, device, epoch)

            logwriter.add_scalar("Evaluation/Accuracy", accuracy, epoch)
            logwriter.add_scalar("Evaluation/Loss", val_loss, epoch)

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
