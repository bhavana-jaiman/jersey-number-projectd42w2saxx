def adaptive_number_scale(
    current_number_height,
    image_height,
    base_scale=1.10,
    min_height_ratio=0.12,
    max_height_ratio=0.20,
    min_scale=1.05,
    max_scale=1.30,
):
    """
    Adaptively determine the scale of the generated jersey number.

    current_number_height:
        Height of the current/reference number bbox in pixels.

    image_height:
        Height of the jersey crop in pixels.

    base_scale:
        Normal enlargement when the current number size is reasonable.

    min_height_ratio:
        Minimum acceptable number height as a fraction of image height.

    max_height_ratio:
        Maximum acceptable number height as a fraction of image height.

    min_scale / max_scale:
        Safety limits on the final scale.
    """

    if image_height <= 0 or current_number_height <= 0:
        return base_scale

    current_ratio = current_number_height / float(image_height)

    # Number is too small → enlarge it more
    if current_ratio < min_height_ratio:
        target_height = image_height * min_height_ratio
        scale = target_height / float(current_number_height)

    # Number is too large → reduce it
    elif current_ratio > max_height_ratio:
        target_height = image_height * max_height_ratio
        scale = target_height / float(current_number_height)

    # Number is already in a reasonable range
    else:
        scale = base_scale

    # Safety limits
    scale = max(min_scale, min(scale, max_scale))

    return scale














# ---------------------------------------------------------
# ADAPTIVE NUMBER SIZE
# ---------------------------------------------------------

current_number_height = max(1, y2 - y1)

scale = adaptive_number_scale(
    current_number_height=current_number_height,
    image_height=H,
    base_scale=args.scale_factor,
    min_height_ratio=args.min_height_ratio,
    max_height_ratio=args.max_height_ratio,
    min_scale=args.min_scale,
    max_scale=args.max_scale,
)


















parser.add_argument(
    "--min-height-ratio",
    type=float,
    default=0.12,
    help="Minimum desired jersey-number height as a fraction of crop height.",
)

parser.add_argument(
    "--max-height-ratio",
    type=float,
    default=0.20,
    help="Maximum desired jersey-number height as a fraction of crop height.",
)

parser.add_argument(
    "--min-scale",
    type=float,
    default=1.05,
    help="Minimum adaptive scale factor.",
)

parser.add_argument(
    "--max-scale",
    type=float,
    default=1.30,
    help="Maximum adaptive scale factor.",
)
