import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def calculate_weights(counts):
    """Inverse-frequency class weights: w_i = N / (C * n_i).
    Feed this the REAL per-class counts from class_counts.py / dataset_audit.py
    — do not guess these numbers."""
    counts = np.array(counts, dtype=np.float32)
    total_samples = counts.sum()
    num_classes = len(counts)
    class_weights = total_samples / (num_classes * counts)
    return torch.tensor(class_weights, dtype=torch.float32)


class make_loss_fn(nn.Module):
    """
    CHANGES vs. the original:

    1. weights_d1 / weights_d2 are now actually PASSED into the criteria
       (previously computed and silently ignored). Pass real per-digit
       counts in via `digit1_counts` / `digit2_counts` at construction —
       get these from class_counts.py or dataset_audit.py, not guessed.

    2. Optional `use_focal=True` swaps CrossEntropyLoss for the (bug-fixed)
       FocalLoss below on the digit heads, to better handle the thin
       "absent digit" (class 10) coverage found during the dataset audit.

    3. Optional `use_uncertainty_weighting=True` replaces the fixed
       lambda_whole=0.3 with learnable per-task log-variances — this
       finishes the feature that backbone_ying.py's get_loss_params()/
       get_uncertainties() were clearly built for but never completed
       (those methods reference self.cls_log_var / cls1_log_var /
       cls2_log_var, which are never defined anywhere).

       IMPORTANT: if you enable this, the log-variance parameters must be
       added to the optimizer alongside the model's own parameters, e.g.:
           optimizer = torch.optim.AdamW(
               list(model.parameters()) + list(loss_fn.parameters()),
               lr=1e-3, weight_decay=1e-2,
           )
       Forgetting this means the log-variances never update and the loss
       behaves as if uncertainty weighting were silently disabled.
    """

    def __init__(
        self,
        digit1_counts=None,
        digit2_counts=None,
        absent_class_weight: float = 0.2,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        use_uncertainty_weighting: bool = False,
        lambda_whole: float = 0.3,
        device: str = "cuda",
    ) -> None:
        super(make_loss_fn, self).__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.device = device

        # ------------------------------------------------------------
        # Build real class weights from real counts, instead of the
        # original's `weights_d2 = torch.ones(11); weights_d2[10] = 0.2`
        # placeholder that was never even applied.
        # ------------------------------------------------------------
        weights_d1 = self._build_weights(digit1_counts, absent_class_weight, device)
        weights_d2 = self._build_weights(digit2_counts, absent_class_weight, device)

        if use_focal:
            self.criterion1 = FocalLoss(gamma=focal_gamma, alpha=weights_d1)
            self.criterion2 = FocalLoss(gamma=focal_gamma, alpha=weights_d2)
        else:
            self.criterion1 = nn.CrossEntropyLoss(weight=weights_d1)
            self.criterion2 = nn.CrossEntropyLoss(weight=weights_d2)

        self.whole_criterion = nn.CrossEntropyLoss()

        if use_uncertainty_weighting:
            # learnable log-variances, one per task -- see class docstring
            # for the required optimizer change.
            self.log_var_whole = nn.Parameter(torch.zeros(1, device=device))
            self.log_var_d1 = nn.Parameter(torch.zeros(1, device=device))
            self.log_var_d2 = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.lambda_whole = lambda_whole

    @staticmethod
    def _build_weights(counts, absent_class_weight, device):
        if counts is None:
            # no real counts supplied -> fall back to uniform weights
            # (equivalent to the ORIGINAL unweighted behavior), but still
            # down-weight the "absent digit" class 10 by default, since
            # the dataset audit found real coverage of that class is thin.
            w = torch.ones(11, dtype=torch.float32)
            w[10] = absent_class_weight
            return w.to(device)
        w = calculate_weights(counts)
        return w.to(device)

    def get_loss_params(self):
        """Mirrors the get_loss_params()/get_uncertainties() interface
        backbone_ying.py already expects, so this can be swapped in as a
        drop-in completion of that half-built feature."""
        if not self.use_uncertainty_weighting:
            return None
        return {
            "log_var_whole": self.log_var_whole,
            "log_var_d1": self.log_var_d1,
            "log_var_d2": self.log_var_d2,
        }

    def forward(self, whole_logits, digit1_logits: torch.Tensor, digit2_logits: torch.Tensor,
                whole_number_labels, digitNumerLables):

        whole_number_loss = self.whole_criterion(whole_logits, whole_number_labels)
        digit1_loss = self.criterion1(digit1_logits.float(), digitNumerLables[:, 0])
        digit2_loss = self.criterion2(digit2_logits.float(), digitNumerLables[:, 1])

        if self.use_uncertainty_weighting:
            precision_whole = torch.exp(-self.log_var_whole)
            precision_d1 = torch.exp(-self.log_var_d1)
            precision_d2 = torch.exp(-self.log_var_d2)

            total_loss = (
                precision_d1 * digit1_loss + self.log_var_d1
                + precision_d2 * digit2_loss + self.log_var_d2
                + precision_whole * whole_number_loss + self.log_var_whole
            )
            return total_loss.squeeze()

        loss_digits = digit1_loss + digit2_loss
        total_loss = loss_digits + self.lambda_whole * whole_number_loss
        return total_loss


class FocalLoss(nn.Module):
    """
    CHANGES vs. the original:
      - alpha is now applied to `loss` (after pt is computed), not to
        `logpt` — the original multiplied it into the wrong term, which
        diluted its intended effect.
      - `target.device` is now used as a property, not called as a method
        (`target.device()` would have raised a TypeError the moment alpha
        was ever actually passed in — which it never was, in the original).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        logpt_prob = F.log_softmax(input, dim=-1)
        logpt = logpt_prob.gather(1, target.view(-1, 1)).squeeze(1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.alpha is not None:
            if self.alpha.device != target.device:  # property, not target.device()
                self.alpha = self.alpha.to(target.device)
            at = self.alpha.gather(0, target)
            loss = loss * at  # scale the final loss, not logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class DistillationLoss(nn.Module):
    """Unchanged from the original — still requires a pretrained,
    frozen teacher_model to be wired in before this does anything.
    Recommended to only enable this after Phases 1-4 are stable and
    you have a trained teacher checkpoint to distill from."""

    def __init__(self, base_ciriterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float) -> None:
        super().__init__()
        self.base_ciriterion = base_ciriterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, feat_student, digital_logits: torch.Tensor,
                digit1_logits: torch.Tensor, digit2_logits: torch.Tensor,
                lenLables, digitNumerLables, epoch):
        base_ciriterion = self.base_ciriterion(
            inputs, feat_student, digital_logits, digit1_logits, digit2_logits, lenLables, digitNumerLables, epoch
        )

        if self.distillation_type == 'none':
            return base_ciriterion

        with torch.no_grad():
            feat, digital_t, digit1_t, digit2_t = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillationLoss = (
                F.kl_div(F.log_softmax(digital_logits / T, dim=1), F.softmax(digital_t / T, dim=1),
                          reduction='batchmean', log_target=False) * (T * T)
                + F.kl_div(F.log_softmax(digit1_logits / T, dim=1), F.softmax(digit1_t / T, dim=1),
                            reduction='batchmean', log_target=False) * (T * T)
                + F.kl_div(F.log_softmax(digit2_logits / T, dim=1), F.softmax(digit2_t / T, dim=1),
                            reduction='batchmean', log_target=False) * (T * T)
            )
        elif self.distillation_type == 'hard':
            distillationLoss = (
                F.cross_entropy(digital_logits, digital_t.argmax(dim=1))
                + F.cross_entropy(digit1_logits, digit1_t.argmax(dim=1))
                + F.cross_entropy(digit2_logits, digit2_t.argmax(dim=1))
            )
        else:
            raise ValueError(f"Unknown distillation_type: {self.distillation_type}")

        loss = base_ciriterion * (1 - self.alpha) + distillationLoss * self.alpha
        return loss
