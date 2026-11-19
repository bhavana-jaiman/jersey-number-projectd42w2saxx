# Patch 1 of 2: utils/loss.py — per-class weight boost for confusable digits

## What this does
Adds a flat 1.3x loss weight for digits {1,2,3,5,6,7,8,9} (the classes
involved in the confusion pairs found in validation: 7<->1, 6->8, 3->2,
9->8, 2->1, 5->6) to BOTH digit_1 and digit_2 CrossEntropyLoss criteria.
Digits 0 and 4 keep weight 1.0 (not implicated in the confusion pairs;
digit 4 had elevated error rate but no clear confusable partner, so left
alone for this first pass to isolate the effect of the confusable-pair fix).

criterion1 currently has NO weight argument at all -- this adds one.
criterion2 already has a weight vector but only sets index 10 (blank) --
this extends it to also weight the confusable digit classes 0-9.

## Where to make the change

Find this block (around line 60-75 in your utils/loss.py, right after
`super(make_loss_fn, self).__init__()`):

```python
        if state_class_counts is not None:
            state_weights = calculate_weights(state_class_counts)
        else:
            state_weights = None
        self.state_criterion = nn.CrossEntropyLoss(weight=state_weights)

        # self.criterion1 = nn.CrossEntropyLoss()  # label_smoothing=0.05
        # self.criterion2 = nn.CrossEntropyLoss()  # label_smoothing=0.05
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX,
                                               label_smoothing=label_smoothing)
        d2_weights = torch.ones(11)
        d2_weights[10] = blank_weight    # 10 = blank
        self.criterion2 = nn.CrossEntropyLoss(weight = d2_weights, ignore_index=IGNORE_INDEX,
                                               label_smoothing=label_smoothing)
```

## Replace it with:

```python
        if state_class_counts is not None:
            state_weights = calculate_weights(state_class_counts)
        else:
            state_weights = None
        self.state_criterion = nn.CrossEntropyLoss(weight=state_weights)

        # ---------------HCL_Changes----------------
        # Confusable-digit reweighting: validation-set failure analysis
        # (inspect_small_scale_failures_v3.py, run on
        # JerseyNumber_Validation_Dataset) found the dominant error pattern
        # is genuine visual-shape confusion between specific digit pairs:
        #   7<->1 (71 total), 6->8 (20), 3->2 (19), 9->8 (18),
        #   2->1 (18), 5->6 (18)
        # Per-class error rates were also highest for 7 (23.5%), 4 (18.7%),
        # 3 (18.4%), 5 (18.0%). Class BALANCE in training data was checked
        # and found even (3700-4200 instances per digit), ruling out a
        # frequency-imbalance explanation -- this is a confusability issue,
        # not a volume issue, so we use a flat multiplier on the implicated
        # classes rather than inverse-frequency weights.
        # digit 4 had an elevated error rate but no clear confusable
        # partner in the pair analysis, so it is NOT included in this first
        # pass -- keeping the change isolated to the confirmed confusion
        # pairs makes it easier to attribute any accuracy change correctly.
        CONFUSABLE_DIGIT_WEIGHT = 1.3
        CONFUSABLE_DIGITS = [1, 2, 3, 5, 6, 7, 8, 9]

        d1_weights = torch.ones(10)
        for d in CONFUSABLE_DIGITS:
            d1_weights[d] = CONFUSABLE_DIGIT_WEIGHT

        d2_weights = torch.ones(11)
        d2_weights[10] = blank_weight    # 10 = blank (unchanged, existing behavior)
        for d in CONFUSABLE_DIGITS:
            d2_weights[d] = CONFUSABLE_DIGIT_WEIGHT
        # ---------------------------------------------

        self.criterion1 = nn.CrossEntropyLoss(weight=d1_weights, ignore_index=IGNORE_INDEX,
                                               label_smoothing=label_smoothing)
        self.criterion2 = nn.CrossEntropyLoss(weight=d2_weights, ignore_index=IGNORE_INDEX,
                                               label_smoothing=label_smoothing)
```

## Important: device placement

`state_weights` (from `calculate_weights`) does NOT appear to be
explicitly moved to the GPU device in this constructor either (no
`.to(device)` call visible in what we've seen) -- `nn.CrossEntropyLoss`
accepts a CPU weight tensor and PyTorch will usually handle the device
transfer automatically when `.to(device)` is called on the whole
`make_loss_fn` module (since `weight` is registered as a buffer inside
the nn.CrossEntropyLoss submodule). This should work the same way
automatically -- but if you hit a "tensors on different devices" error
when running the first training step, that confirms it needs an explicit
`.to(device)`; in that case tell me and I'll adjust.
