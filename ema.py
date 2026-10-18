"""
utils/ema.py  -- exponential moving average of the model weights (EMA).

ema = ModelEMA(model, decay=0.999)      # after the model is built
ema.update(model)                        # after every optimizer step
ema.module                               # use for evaluation / saving

The EMA copy is a smoothed average of the last ~1/(1-decay) steps (0.999 -> ~1000
steps), so it is less noisy than the last step's weights; no extra cost at inference.
"""
import copy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:                                   # e.g. BatchNorm num_batches_tracked
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
