216
217
218
219
220
221
222
223
224
225
226
227
228
229
230
231
232
233
234
235
236
237
238
239
240
241
242
243
244
245
246
247
248
249
250
251
252
253
254
255
256
257
258
259
260
261
262
263
264
265
266
267
268
269
270
271
#!/usr/bin/env python3
        type=int,
        default=256,
        help=(
            "Model feature/output channels. Default: 256. "
            "Must match the checkpoint architecture."
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("JERSEY NUMBER RECOGNITION - ONNX EXPORT")
    print("=" * 70)
    print(f"Checkpoint : {Path(args.checkpoint).resolve()}")
    print(f"Backbone   : {Path(args.backbone).resolve()}")
    print(f"Channels   : {args.channels}")
    print(f"Input      : 3 x {args.height} x {args.width}")
    print(f"Output     : {Path(args.output).resolve()}")
    print("=" * 70)

    # Load the requested backbone file.
    ModelClass = load_backbone_class(args.backbone)

    print()
    print(f"Loaded model class: {ModelClass.__name__}")

    # Your shown backbone defaults to 256, but we pass it explicitly.
    try:
        model = ModelClass(out_channels=args.channels)
    except TypeError as e:
        raise TypeError(
            "Could not construct MultiTaskLearnerWithState with "
            f"out_channels={args.channels}. "
            "Check the constructor in the supplied backbone file."
        ) from e

    print(f"Created model with out_channels={args.channels}")

    # Load trained weights.
    model = load_checkpoint(model, args.checkpoint)

    print("Checkpoint loaded successfully.")

    # Export.
    export_onnx(
        model=model,
        output_path=args.output,
        height=args.height,
        width=args.width,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
