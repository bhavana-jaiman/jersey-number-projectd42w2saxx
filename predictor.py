"""Plug YOUR jersey-number recognizer in here. Only these two functions are used by safe_crop.py."""


def load():
    # TODO: load your model/checkpoint once, e.g.  global MODEL; MODEL = ...; MODEL.eval().cuda()
    raise NotImplementedError("edit predictor.py: load()")


def predict(pil_images):
    """list[PIL.Image (RGB)] -> list[str] of digits, e.g. ['7', '77', '16'].
    Use YOUR usual preprocessing and decoding, and run the list as one batch for speed."""
    # TODO
    raise NotImplementedError("edit predictor.py: predict()")
