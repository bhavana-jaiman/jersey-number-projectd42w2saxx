import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

# ---------------HCL_Changes---------------
# New architecture:
#   digit1 head : 10 classes (0-9), no blank class
#   digit2 head : 11 classes (0-9, 10 = blank)
#   state head  : 3 classes (0 = no digit, 1 = one/two digits, 2 = three+ digits)
#   digital (whole-number) head : REMOVED
# The dataset sets digit1/digit2 targets to IGNORE_INDEX (-100) for state 0/2
# samples, so the digit losses must ignore -100.
IGNORE_INDEX = -100
# -------------------------------------------


def calculate_weights(counts):
    counts = np.array(counts, dtype=np.float32)
    total_samples = counts.sum()
    num_classes = len(counts)
    class_weights = total_samples / (num_classes * counts)
    return torch.tensor(class_weights, dtype=torch.float32)


# ---------------HCL_Changes---------------
def _masked_ce(criterion, logits, target):
    # If every target in the batch is IGNORE_INDEX (e.g. a batch of only
    # state 0/2 samples), CrossEntropyLoss(reduction='mean') returns NaN.
    # Return a zero that stays connected to the graph instead.
    if (target != IGNORE_INDEX).any():
        return criterion(logits, target)
    return logits.sum() * 0.0
# -------------------------------------------


class make_loss_fn(nn.Module):
    # ---------------HCL_Changes---------------
    # whole_class_counts -> state_class_counts (whole-number head removed).
    # Pass per-state sample counts [n_state0, n_state1, n_state2] to weight the
    # state loss against class imbalance; None = unweighted.
    # def __init__(self, whole_class_counts=None) -> None:
    def __init__(self, state_class_counts=None, lambda_state: float = 1.0) -> None:
        # -------------------------------------------
        super(make_loss_fn, self).__init__()

        weights_d2 = torch.ones(11).to(torch.device("cuda"))
        weights_d2[10] = 0.2

        # ---------------HCL_Changes---------------
        # if whole_class_counts is not None:
        #     whole_weights = calculate_weights(whole_class_counts).to(torch.device("cuda"))
        # else:
        #     whole_weights = None
        # #self.whole_criterion = nn.CrossEntropyLoss()
        # self.whole_criterion = nn.CrossEntropyLoss(weight=whole_weights)
        if state_class_counts is not None:
            state_weights = calculate_weights(state_class_counts)
        else:
            state_weights = None
        # register_buffer so the weights follow the module's .to(device)
        self.register_buffer("state_weights", state_weights)
        self.state_criterion = nn.CrossEntropyLoss(weight=state_weights)

        # digit1: 10 classes, digit2: 11 classes; both ignore state 0/2 samples
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # label_smoothing=0.05
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # label_smoothing=0.05
        # NOTE: weights_d2 above was never passed to criterion2 in the original
        # file either. Left unused; to down-weight the blank class use:
        # self.criterion2 = nn.CrossEntropyLoss(weight=weights_d2, ignore_index=IGNORE_INDEX)

        # self.lambda_whole = 0.3 #0.5 done
        self.lambda_state = lambda_state
        # -------------------------------------------

    # ---------------HCL_Changes---------------
    # def forward(self, whole_logits, digit1_logits: torch.tensor, digit2_logits: torch.tensor, whole_number_labels, digitNumerLables):
    def forward(self, digit1_logits: torch.Tensor, digit2_logits: torch.Tensor,
                state_logits: torch.Tensor, digitNumerLables, state_labels):
        # whole_number_loss = self.whole_criterion(whole_logits.float(), whole_number_labels.long())
        digitNumerLables = digitNumerLables.long()
        digit1_loss = _masked_ce(self.criterion1, digit1_logits.float(), digitNumerLables[:, 0])
        digit2_loss = _masked_ce(self.criterion2, digit2_logits.float(), digitNumerLables[:, 1])
        state_loss = self.state_criterion(state_logits.float(), state_labels.long())

        loss_digits = digit1_loss + digit2_loss
        # total_loss = loss_digits + self.lambda_whole * whole_number_loss
        total_loss = loss_digits + self.lambda_state * state_loss
        return total_loss
    # -------------------------------------------


class FocalLoss(nn.Module):
    # ---------------HCL_Changes---------------
    # ignore_index added so FocalLoss can also be used for digit1/digit2
    # (gather() crashes on -100 targets otherwise).
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None,
                 reduction: str = 'mean', ignore_index: int = IGNORE_INDEX) -> None:
        # -------------------------------------------
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

        self.reduction = reduction

    def forward(self, input, target):
        # ---------------HCL_Changes---------------
        valid = target != self.ignore_index
        input = input[valid]
        target = target[valid]
        if target.numel() == 0:
            return input.sum() * 0.0
        # -------------------------------------------

        logpt_prob = F.log_softmax(input, dim=-1)
        logpt = logpt_prob.gather(1, target.view(-1, 1)).squeeze(1)

        pt = logpt.exp()

        # ---------------HCL_Changes---------------
        # Original applied alpha to logpt AFTER the loss was computed, so alpha
        # had no effect; and target.device() is not callable (.device is a
        # property). Both fixed: alpha is applied before computing the loss.
        if self.alpha is not None:
            if self.alpha.device != target.device:
                self.alpha = self.alpha.to(target.device)
            at = self.alpha.gather(0, target)
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        # -------------------------------------------

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError


class DistillationalLoss(nn.Module):
    def __init__(self, base_ciriterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float) -> None:
        super().__init__()
        self.base_ciriterion = base_ciriterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    # ---------------HCL_Changes---------------
    # digital_logits / lenLables removed; state_logits / state_labels added.
    # epoch kept (unused) so existing call sites only need the logit/label change.
    # def forward(self, inputs, feat_student, digital_logits: torch.tensor, digit1_logits: torch.tensor, digit2_logits: torch.tensor, lenLables, digitNumerLables, epoch):
    def forward(self, inputs, feat_student, digit1_logits: torch.Tensor,
                digit2_logits: torch.Tensor, state_logits: torch.Tensor,
                digitNumerLables, state_labels, epoch):
        # base_ciriterion = self.base_ciriterion(
        #     inputs, feat_student, digital_logits, digit1_logits, digit2_logits, lenLables, digitNumerLables, epoch)
        base_ciriterion = self.base_ciriterion(
            digit1_logits, digit2_logits, state_logits, digitNumerLables, state_labels)
        # -------------------------------------------

        if self.distillation_type == 'none':
            return base_ciriterion

        # ---------------HCL_Changes---------------
        # Teacher must use the NEW architecture (feat, digit1, digit2, state).
        # An old teacher with a digital head and 11-class digit1 will not match.
        with torch.no_grad():
            # feat, digital_t, digit1_t, digit2_t = self.teacher_model(inputs)
            feat, digit1_t, digit2_t, state_t = self.teacher_model(inputs)
        # -------------------------------------------

        if self.distillation_type == 'soft':
            T = self.tau
            # ---------------HCL_Changes---------------
            # Original passed F.softmax(teacher) with log_target=True, which is
            # wrong (log_target=True expects log-probabilities). Fixed by using
            # F.log_softmax for the teacher. digital term removed, state term added.
            # distillationLoss = F.kl_div(F.log_softmax(digital_logits / T, dim=1), F.softmax(digital_t / T, dim=1), reduction='batchmean', log_target=True) * (T*T) + ...
            distillationLoss = (
                F.kl_div(F.log_softmax(digit1_logits / T, dim=1),
                         F.log_softmax(digit1_t / T, dim=1),
                         reduction='batchmean', log_target=True) * (T * T)
                + F.kl_div(F.log_softmax(digit2_logits / T, dim=1),
                           F.log_softmax(digit2_t / T, dim=1),
                           reduction='batchmean', log_target=True) * (T * T)
                + F.kl_div(F.log_softmax(state_logits / T, dim=1),
                           F.log_softmax(state_t / T, dim=1),
                           reduction='batchmean', log_target=True) * (T * T)
            )
            # -------------------------------------------

        elif self.distillation_type == 'hard':
            # distillationLoss = F.cross_entropy(digital_logits, digital_t.argmax(dim=1)) + F.cross_entropy(
            #     digit1_logits, digit1_t.argmax(dim=1)) + F.cross_entropy(digit2_logits, digit2_t.argmax(dim=1))
            distillationLoss = F.mse_loss(feat_student, feat)

        loss = base_ciriterion * (1 - self.alpha) + \
            distillationLoss * self.alpha

        return loss
