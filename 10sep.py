def visualize_dataset(dataset_path, output_path):

    # Check whether dataset has images/labels subdirectories
    image_subdir = os.path.join(dataset_path, "images")
    label_subdir = os.path.join(dataset_path, "labels")

    if os.path.isdir(image_subdir) and os.path.isdir(label_subdir):
        # Structure:
        # dataset/
        #   images/
        #   labels/
        image_dir = image_subdir
        label_dir = label_subdir

    else:
        # Structure:
        # dataset/
        #   image1.jpg
        #   image1.txt
        #   image2.jpg
        #   image2.txt
        image_dir = dataset_path
        label_dir = dataset_path

    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"ERROR: Image directory not found: {image_dir}")
        return

    # Supported image extensions
    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]

    images.sort()

    print(f"Dataset: {dataset_path}")
    print(f"Images found: {len(images)}")
    print("-" * 50)

    processed = 0
    missing_labels = 0

    for image_name in images:

        image_path = os.path.join(image_dir, image_name)

        # Corresponding label
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        image = cv2.imread(image_path)

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        # Missing label
        if not os.path.exists(label_path):
            missing_labels += 1
            print(f"WARNING: No label: {image_name}")
            continue

        # Draw YOLO boxes
        image = draw_yolo_boxes(
            image,
            label_path
        )

        # Save
        output_image_path = os.path.join(
            output_path,
            image_name
        )

        cv2.imwrite(
            output_image_path,
            image
        )

        processed += 1

    print("\nFinished")
    print(f"Processed images : {processed}")
    print(f"Missing labels   : {missing_labels}")
    print(f"Output directory : {output_path}")
