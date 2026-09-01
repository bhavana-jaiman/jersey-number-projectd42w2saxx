import os
import time
import datetime
import logging
from tqdm import tqdm

import argparse

import numpy as np
import pandas as pd
import math

from utils.util import *
from utils.jersey_dataset import jersey_Dataset, jerseyNumber_ValidationDataset_V2
from utils.loss import make_loss_fn
from utils.autoAugment import AutoAugment

from subModules.backbone_ying import build_model
from utils.multitask_loss import JerseyMultiTaskLoss, compute_kd_loss

import torch
import torchvision.transforms as Transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast

from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, epoch,
                     opt, teacher_model=None):
    model.train()
    pbar = tqdm(loader, desc=f"Training Epoch{epoch}")

    avg_loss = 0

    for batch_i, (images, whole_number_labels, digit_number_labels, _) in enumerate(pbar):
        images = images.to(device)
        digit_number_labels = digit_number_labels.to(device)
        whole_number_labels = whole_number_labels.to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=True):
            student_out = model(images)

            if opt.no_state:
                _, logits_whole, logits_digit1, logits_digit2 = student_out
                state_logits, state_targets = None, None
            else:
                _, logits_whole, logits_digit1, logits_digit2, state_logits = student_out
                # NOTE: your current dataset/collate_fn does not yet return
                # state labels in this loop's unpacking (whole_number_labels,
                # digit_number_labels, _). If/when your jersey_Dataset starts
                # yielding a state label as that 4th (currently discarded)
                # element, wire it in here as state_targets.
                state_targets = None

            losses = loss_fn(
                logits_whole,
                logits_digit1,
                logits_digit2,
                whole_number_labels,
                digit_number_labels[:, 0],
                digit_number_labels[:, 1],
                state_logits=state_logits,
                state_targets=state_targets,
            )

            loss = losses["total"]

            if opt.use_kd and teacher_model is not None:
                with torch.no_grad():
                    teacher_out = teacher_model(images)

                kd = compute_kd_loss(
                    student_out,
                    teacher_out,
                    temperature=opt.kd_temperature,
                    use_state=not opt.no_state,
                )

                loss = loss + opt.kd_weight * kd

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


def eval_one_epoch(model, val_loader, loss_fn, device, epoch, opt):
    model.eval()
    pbar = tqdm(val_loader, desc=f"Evaluating Epoch{epoch}")

    val_loss = 0

    correct = 0

    with torch.no_grad():
        for batch_i, (images, digit_number_labels, whole_number_labels) in enumerate(pbar):
            images = images.to(device)
            digit_number_labels = digit_number_labels.to(device)
            whole_number_labels = whole_number_labels.to(device)

            student_out = model(images)

            if opt.no_state:
                feat, logits_whole, logits_digit1, logits_digit2 = student_out
                state_logits, state_targets = None, None
            else:
                feat, logits_whole, logits_digit1, logits_digit2, state_logits = student_out
                state_targets = None

            losses = loss_fn(
                logits_whole,
                logits_digit1,
                logits_digit2,
                whole_number_labels,
                digit_number_labels[:, 0],
                digit_number_labels[:, 1],
                state_logits=state_logits,
                state_targets=state_targets,
            )

            loss = losses["total"]

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

    # ------------------------------------------------------------
    # Architecture / loss experiment flags
    # ------------------------------------------------------------
    parser.add_argument("--use-stn", action="store_true",
        help="Enable lightweight Spatial Transformer Network")
    parser.add_argument("--attention", type=str, default="none",
        choices=["none", "se", "eca", "both"], help="Attention mechanism")
    parser.add_argument("--attention-stages", nargs="+", default=[],
        choices=["block3", "block4", "final"], help="Where to apply attention")
    parser.add_argument("--no-state", action="store_true",
        help="Disable state classification head")
    parser.add_argument("--lambda-digit1", type=float, default=1.0)
    parser.add_argument("--lambda-digit2", type=float, default=1.0)
    parser.add_argument("--lambda-whole", type=float, default=0.3)
    parser.add_argument("--lambda-state", type=float, default=0.1)
    parser.add_argument("--use-consistency", action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--whole-loss", type=str, default="ce",
        choices=["ce", "weighted_ce"])
    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--kd-weight", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.0)

    opt = parser.parse_args()

    if opt.use_kd and not opt.teacher_checkpoint:
        parser.error("--teacher-checkpoint is required when --use-kd is set")

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
                [Transforms.ColorJitter(brightness=0.15, contrast=0.05)],
                p=0.5,
            ),
            # Transforms.RandomApply([Transforms.GaussianBlur(kernel_size=3)], p=0.3),
            Transforms.ToTensor(),
        ]
    )

    train_dataset = jersey_Dataset(
        train_path, transform=Transforms.Compose(train_transform_list)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )  #

    # ------------------------------------------------------------
    # Model creation (now fully driven by the CLI flags above)
    # ------------------------------------------------------------
    model = build_model(
        use_state=not opt.no_state,
        use_stn=opt.use_stn,
        attention=opt.attention,
        attention_stages=opt.attention_stages,
        out_channels=128,
        dropout=opt.dropout,
    ).to(device)

    if opt.pre_trained_model:
        model.load_state_dict(torch.load(opt.pre_trained_model), strict=False)
        print("Loaded pretrained model!")

    teacher_model = None
    if opt.use_kd:
        teacher_model = build_model(
            use_state=not opt.no_state,
            out_channels=128,
        ).to(device)
        teacher_model.load_state_dict(
            torch.load(opt.teacher_checkpoint), strict=False
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print(f"Loaded teacher model from {opt.teacher_checkpoint}")

    val_dataset = jerseyNumber_ValidationDataset_V2(
        validation_path
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             num_workers=opt.n_cpu, pin_memory=True, collate_fn=val_dataset.new_collate_fn)

    # ------------------------------------------------------------
    # Loss (now the configurable JerseyMultiTaskLoss)
    # ------------------------------------------------------------
    loss_fn = JerseyMultiTaskLoss(
        lambda_digit1=opt.lambda_digit1,
        lambda_digit2=opt.lambda_digit2,
        lambda_whole=opt.lambda_whole,
        lambda_state=opt.lambda_state,
        use_consistency=opt.use_consistency,
        consistency_weight=opt.consistency_weight,
    ).to(device)

    initial_learning_rate = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_learning_rate, weight_decay=1e-2
    )

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

        loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn,
                                device, epoch, opt, teacher_model=teacher_model)
        lr_scheduler.step()

        logwriter.add_scalar("Train/Loss", loss, epoch)
        logwriter.add_scalar("Train/Lr", lr_scheduler.get_lr()[0], epoch)

        if epoch % 5 == 0 and epoch != 0:
            val_loss, accuracy = eval_one_epoch(model, val_loader, loss_fn, device, epoch, opt)

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
