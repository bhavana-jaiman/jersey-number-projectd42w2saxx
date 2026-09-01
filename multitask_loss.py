import torch
import torch.nn as nn
import torch.nn.functional as F


class JerseyMultiTaskLoss(nn.Module):
    """
    Supervised multitask loss for digit1 / digit2 / whole-number /
    (optional) state.

    Digit heads use 11 classes:
        0-9 = actual digit
        10  = padding / no second digit

    Whole-number head uses 100 classes (00-99). The consistency
    term compares this against the joint distribution implied by
    digit1 * digit2, restricted to classes 0-9 on each digit head
    (i.e. the two-digit case). It intentionally does NOT try to
    force single-digit whole-number labels into that joint space -
    verify your label-generation code's exact single-digit mapping
    before enabling --use-consistency.
    """

    def __init__(
        self,
        lambda_digit1=1.0,
        lambda_digit2=1.0,
        lambda_whole=0.3,
        lambda_state=0.1,
        use_consistency=False,
        consistency_weight=0.1,
        whole_class_weights=None
    ):
        super().__init__()

        self.lambda_digit1 = lambda_digit1
        self.lambda_digit2 = lambda_digit2
        self.lambda_whole = lambda_whole
        self.lambda_state = lambda_state

        self.use_consistency = use_consistency
        self.consistency_weight = consistency_weight

        self.ce_digit1 = nn.CrossEntropyLoss()

        self.ce_digit2 = nn.CrossEntropyLoss()

        self.ce_whole = nn.CrossEntropyLoss(
            weight=whole_class_weights
        )

        self.ce_state = nn.CrossEntropyLoss()

    def forward(
        self,
        whole_logits,
        digit1_logits,
        digit2_logits,
        whole_targets,
        digit1_targets,
        digit2_targets,
        state_logits=None,
        state_targets=None
    ):

        loss_digit1 = self.ce_digit1(
            digit1_logits,
            digit1_targets
        )

        loss_digit2 = self.ce_digit2(
            digit2_logits,
            digit2_targets
        )

        loss_whole = self.ce_whole(
            whole_logits,
            whole_targets
        )

        total_loss = (
            self.lambda_digit1 * loss_digit1
            +
            self.lambda_digit2 * loss_digit2
            +
            self.lambda_whole * loss_whole
        )

        loss_state = torch.tensor(
            0.0,
            device=whole_logits.device
        )

        if (
            state_logits is not None
            and state_targets is not None
        ):

            loss_state = self.ce_state(
                state_logits,
                state_targets
            )

            total_loss += (
                self.lambda_state *
                loss_state
            )

        # ----------------------------------------------------
        # Digit / whole consistency
        # ----------------------------------------------------

        consistency_loss = torch.tensor(
            0.0,
            device=whole_logits.device
        )

        if self.use_consistency:

            consistency_loss = self.consistency_loss(
                whole_logits,
                digit1_logits,
                digit2_logits
            )

            total_loss += (
                self.consistency_weight *
                consistency_loss
            )

        return {
            "total": total_loss,
            "digit1": loss_digit1,
            "digit2": loss_digit2,
            "whole": loss_whole,
            "state": loss_state,
            "consistency": consistency_loss
        }

    @staticmethod
    def consistency_loss(
        whole_logits,
        digit1_logits,
        digit2_logits
    ):

        # ----------------------------------------------------
        # Convert digit probabilities into joint probabilities
        #
        # P(d1,d2) = P(d1) * P(d2)
        # ----------------------------------------------------

        p1 = F.softmax(
            digit1_logits,
            dim=1
        )

        p2 = F.softmax(
            digit2_logits,
            dim=1
        )

        # Ignore class 10 for whole-number construction.
        #
        # Classes 0-9 correspond to digits.
        #
        # Whole numbers 10-99:
        #
        # 10 -> digit1=1, digit2=0
        # 11 -> digit1=1, digit2=1
        # ...
        # 99 -> digit1=9, digit2=9
        #
        # Single digit classes:
        #
        # 0-9 -> digit1=digit, digit2=10
        # ----------------------------------------------------

        joint = (
            p1[:, :10].unsqueeze(2)
            *
            p2[:, :10].unsqueeze(1)
        )

        joint = joint.reshape(
            joint.size(0),
            100
        )

        # Whole-number probability
        whole_probability = F.softmax(
            whole_logits,
            dim=1
        )

        # Consistency only compares the 00-99 space.
        #
        # KL is more stable than directly forcing the logits
        # to be identical.
        #

        eps = 1e-8

        joint = joint + eps
        whole_probability = whole_probability[:, :100] + eps

        joint = joint / joint.sum(
            dim=1,
            keepdim=True
        )

        whole_probability = (
            whole_probability /
            whole_probability.sum(
                dim=1,
                keepdim=True
            )
        )

        return F.kl_div(
            torch.log(whole_probability),
            joint,
            reduction="batchmean"
        )


# ============================================================
# Knowledge Distillation
# ============================================================

def kd_loss(
    student_logits,
    teacher_logits,
    temperature=4.0
):
    """
    Standard soft-target KD loss (Hinton et al.), scaled by T^2
    so gradient magnitude stays comparable to the hard-label CE
    terms as temperature changes.
    """

    student_log_prob = F.log_softmax(
        student_logits / temperature,
        dim=1
    )

    teacher_prob = F.softmax(
        teacher_logits / temperature,
        dim=1
    )

    loss = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="batchmean"
    )

    return loss * (temperature ** 2)


def compute_kd_loss(
    student_outputs,
    teacher_outputs,
    temperature=4.0,
    use_state=True
):
    """
    student_outputs / teacher_outputs are expected to be the raw
    model forward() tuples:

        (feat, whole_logits, digit1_logits, digit2_logits[, state_logits])

    Only includes the state term if use_state is True AND both
    tuples actually contain a state_logits element (length 5).
    """

    _, s_whole, s_d1, s_d2 = student_outputs[:4]
    _, t_whole, t_d1, t_d2 = teacher_outputs[:4]

    loss = (
        kd_loss(s_whole, t_whole, temperature)
        +
        kd_loss(s_d1, t_d1, temperature)
        +
        kd_loss(s_d2, t_d2, temperature)
    )

    has_student_state = len(student_outputs) == 5
    has_teacher_state = len(teacher_outputs) == 5

    if use_state and has_student_state and has_teacher_state:
        s_state = student_outputs[4]
        t_state = teacher_outputs[4]

        loss = loss + kd_loss(s_state, t_state, temperature)

    return loss
