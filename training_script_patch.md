# Patch notes for jerseyNumberRecognitior_Training.py

I don't have your full training script in front of me (only the argparse
block and model-creation line from our chat), so rather than guess at the
rest of the file and risk breaking your training loop, here are the exact
edits to drop into your existing file.

## 1. Import

Replace:

    from subModules.backbone_ying import MultiTaskLearner

with:

    from backbone import build_model
    from multitask_loss import JerseyMultiTaskLoss, compute_kd_loss

## 2. Add to your argparse section (alongside --epochs, --batch_size, etc.)

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

## 3. Replace model creation

Old:

    model = MultiTaskLearner().to(device)

New:

    model = build_model(
        use_state=not opt.no_state,
        use_stn=opt.use_stn,
        attention=opt.attention,
        attention_stages=opt.attention_stages,
        out_channels=128,
        dropout=opt.dropout,
    ).to(device)

    loss_fn = JerseyMultiTaskLoss(
        lambda_digit1=opt.lambda_digit1,
        lambda_digit2=opt.lambda_digit2,
        lambda_whole=opt.lambda_whole,
        lambda_state=opt.lambda_state,
        use_consistency=opt.use_consistency,
        consistency_weight=opt.consistency_weight,
    ).to(device)

    teacher_model = None
    if opt.use_kd:
        assert opt.teacher_checkpoint, "--teacher-checkpoint required with --use-kd"
        teacher_model = build_model(use_state=not opt.no_state).to(device)
        teacher_model.load_state_dict(torch.load(opt.teacher_checkpoint), strict=False)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

## 4. Inside train_one_epoch, replace the forward/loss block

Your current code does roughly:

    _, logits_whole, logits_digit1, logits_digit2 = model(images)
    loss = criterion(logits_whole, logits_digit1, ...)

Replace with (state-aware + optional KD):

    student_out = model(images)  # (feat, whole, d1, d2[, state])

    if opt.no_state:
        _, whole_logits, d1_logits, d2_logits = student_out
        state_logits, state_targets = None, None
    else:
        _, whole_logits, d1_logits, d2_logits, state_logits = student_out
        state_targets = state_number_labels  # however your loader names this

    losses = loss_fn(
        whole_logits, d1_logits, d2_logits,
        whole_number_labels, digit1_labels, digit2_labels,
        state_logits=state_logits, state_targets=state_targets,
    )
    loss = losses["total"]

    if opt.use_kd and teacher_model is not None:
        with torch.no_grad():
            teacher_out = teacher_model(images)
        kd = compute_kd_loss(
            student_out, teacher_out,
            temperature=opt.kd_temperature,
            use_state=not opt.no_state,
        )
        loss = loss + opt.kd_weight * kd

Everything after this (scaler.scale(loss).backward(), clip_grad_norm_,
scaler.step(optimizer), etc.) stays exactly as you already have it — only
the forward pass and loss computation change.

## 5. eval_one_epoch

Same substitution: swap the direct `model(images)` unpacking + manual
criterion call for the `student_out = model(images)` + `loss_fn(...)`
pattern above (no KD needed at eval time).

---

If you paste your actual training script, I can apply this directly and
hand back the complete file instead of a patch.
