"""
utils/kd.py  -- knowledge distillation from a (bigger) teacher model.

The teacher (e.g. the 12x12 model) is used ONLY during training; the deployed
student keeps its own FLOPs.

    teacher = load_teacher(ckpt, high_res=True, device=device)
    ...
    kd = kd_loss((l1, l2, ls), teacher_logits(teacher, images), T=4.0)
    loss = (1 - alpha) * loss + alpha * kd
"""
import torch
import torch.nn.functional as F

from subModules.backbone_ying import MultiTaskLearnerWithState


def load_teacher(ckpt, high_res=True, device="cuda"):
    t = MultiTaskLearnerWithState(high_res=high_res).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    t.load_state_dict(sd, strict=True)          # must match exactly
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def teacher_logits(teacher, images):
    _, t1, t2, ts = teacher(images)
    return t1.float(), t2.float(), ts.float()


def kd_loss(student, teacher, T=4.0):
    """Sum of KL(teacher || student) over digit1, digit2 and state heads, scaled by T^2."""
    loss = 0.0
    for s, t in zip(student, teacher):
        loss = loss + F.kl_div(F.log_softmax(s.float() / T, 1),
                               F.softmax(t / T, 1), reduction="batchmean")
    return loss * T * T
