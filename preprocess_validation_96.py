import os
import cv2
import shutil
import argparse


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Crop newly added validation images to 96x96 "
                "and convert YOLO labels to 1D labels."
)

parser.add_argument(
    "--validation_dir",
    required=True,
    help="Path to validation directory containing existing, real, synthetic"
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output processed validation directory"
)

parser.add_argument(
    "--size",
    type=int,
    default=96,
    help="Output image size"
)

parser.add_argument(
    "--margin",
    type=int,
    default=20,
    help="Extra margin around annotation before making 96x96 crop"
)

args = parser.parse_args()


VALIDATION_DIR = args.validation_dir
OUTPUT_DIR = args.output_dir
SIZE = args.size
MARGIN = args.margin


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

for category in [
    "existing",
    "real",
    "synthetic"
]:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            category,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            category,
            "labels"
        ),
        exist_ok=True
    )


# ============================================================
# READ YOLO LABEL
# ============================================================

def read_yolo_label(label_path):

    labels = []

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:

                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                labels.append(
                    (
                        class_id,
                        xc,
                        yc,
                        w,
                        h
                    )
                )

            except ValueError:

                continue

    return labels


# ============================================================
# GET 96x96 CROP
# ============================================================

def crop_to_96x96(
    image,
    labels,
    margin=20
):

    """
    Finds the union of all annotated boxes,
    adds margin, makes the crop square,
    then resizes to 96x96.
    """

    if not labels:
        return None


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Convert YOLO boxes to pixel coordinates
    # --------------------------------------------------------

    boxes = []

    for (
        class_id,
        xc,
        yc,
        box_w,
        box_h
    ) in labels:

        x_center = xc * width
        y_center = yc * height

        w = box_w * width
        h = box_h * height

        x1 = x_center - w / 2
        y1 = y_center - h / 2

        x2 = x_center + w / 2
        y2 = y_center + h / 2

        boxes.append(
            (x1, y1, x2, y2)
        )


    # --------------------------------------------------------
    # Union of all digit boxes
    # --------------------------------------------------------

    x1 = min(
        box[0]
        for box in boxes
    )

    y1 = min(
        box[1]
        for box in boxes
    )

    x2 = max(
        box[2]
        for box in boxes
    )

    y2 = max(
        box[3]
        for box in boxes
    )


    # --------------------------------------------------------
    # Add margin
    # --------------------------------------------------------

    x1 -= margin
    y1 -= margin

    x2 += margin
    y2 += margin


    # --------------------------------------------------------
    # Make square crop
    # --------------------------------------------------------

    crop_width = x2 - x1
    crop_height = y2 - y1

    crop_size = max(
        crop_width,
        crop_height
    )


    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2


    x1 = center_x - crop_size / 2
    x2 = center_x + crop_size / 2

    y1 = center_y - crop_size / 2
    y2 = center_y + crop_size / 2


    # --------------------------------------------------------
    # Clip to image
    # --------------------------------------------------------

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))

    x2 = int(min(width, x2))
    y2 = int(min(height, y2))


    if x2 <= x1 or y2 <= y1:

        return None


    crop = image[
        y1:y2,
        x1:x2
    ]


    # --------------------------------------------------------
    # Resize to exactly 96x96
    # --------------------------------------------------------

    crop = cv2.resize(
        crop,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )


    return crop


# ============================================================
# CONVERT YOLO LABEL TO 1D
# ============================================================

def convert_to_1d_label(labels):

    """
    YOLO:

        3 x y w h
        7 x y w h

    becomes:

        3 7
    """

    if not labels:

        return ""


    # --------------------------------------------------------
    # Extract classes
    # --------------------------------------------------------

    classes = [
        label[0]
        for label in labels
    ]


    # --------------------------------------------------------
    # Sort digits from left to right
    # --------------------------------------------------------

    if len(labels) > 1:

        labels_sorted = sorted(
            labels,
            key=lambda x: x[1]
        )

        classes = [
            label[0]
            for label in labels_sorted
        ]


    return " ".join(
        str(int(class_id))
        for class_id in classes
    )


# ============================================================
# PROCESS NEW VALIDATION DATA
# ============================================================

def process_new_category(
    category
):

    input_images = os.path.join(
        VALIDATION_DIR,
        category,
        "images"
    )

    input_labels = os.path.join(
        VALIDATION_DIR,
        category,
        "labels"
    )


    output_images = os.path.join(
        OUTPUT_DIR,
        category,
        "images"
    )

    output_labels = os.path.join(
        OUTPUT_DIR,
        category,
        "labels"
    )


    if not os.path.isdir(input_images):

        print(
            f"WARNING: {input_images} not found"
        )

        return


    if not os.path.isdir(input_labels):

        print(
            f"WARNING: {input_labels} not found"
        )

        return


    image_files = [

        f
        for f in os.listdir(input_images)

        if f.endswith(IMAGE_EXTENSIONS)

    ]


    print("\n" + "=" * 60)

    print(
        f"PROCESSING {category.upper()}"
    )

    print("=" * 60)

    print(
        f"Images found: {len(image_files)}"
    )


    processed = 0
    missing_labels = 0
    failed = 0


    for image_name in image_files:

        image_path = os.path.join(
            input_images,
            image_name
        )

        label_name = (
            os.path.splitext(image_name)[0]
            + ".txt"
        )

        label_path = os.path.join(
            input_labels,
            label_name
        )


        if not os.path.exists(label_path):

            print(
                f"Missing label: {image_name}"
            )

            missing_labels += 1

            continue


        image = cv2.imread(
            image_path
        )


        if image is None:

            print(
                f"Cannot read: {image_name}"
            )

            failed += 1

            continue


        labels = read_yolo_label(
            label_path
        )


        if not labels:

            print(
                f"Invalid/empty YOLO label: "
                f"{label_name}"
            )

            failed += 1

            continue


        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        cropped = crop_to_96x96(
            image,
            labels,
            margin=MARGIN
        )


        if cropped is None:

            print(
                f"Crop failed: {image_name}"
            )

            failed += 1

            continue


        # ----------------------------------------------------
        # Create 1D label
        # ----------------------------------------------------

        new_label = convert_to_1d_label(
            labels
        )


        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_image_path = os.path.join(
            output_images,
            image_name
        )

        cv2.imwrite(
            output_image_path,
            cropped
        )


        # ----------------------------------------------------
        # Save label
        # ----------------------------------------------------

        output_label_path = os.path.join(
            output_labels,
            label_name
        )

        with open(
            output_label_path,
            "w"
        ) as f:

            f.write(new_label)


        processed += 1


    print(
        f"\nProcessed       : {processed}"
    )

    print(
        f"Missing labels  : {missing_labels}"
    )

    print(
        f"Failed          : {failed}"
    )


# ============================================================
# COPY EXISTING VALIDATION UNCHANGED
# ============================================================

def copy_existing_validation():

    input_images = os.path.join(
        VALIDATION_DIR,
        "existing",
        "images"
    )

    input_labels = os.path.join(
        VALIDATION_DIR,
        "existing",
        "labels"
    )

    output_images = os.path.join(
        OUTPUT_DIR,
        "existing",
        "images"
    )

    output_labels = os.path.join(
        OUTPUT_DIR,
        "existing",
        "labels"
    )


    if not os.path.isdir(input_images):

        print(
            "\nWARNING: Existing validation "
            "images directory not found."
        )

        return


    print("\n" + "=" * 60)

    print(
        "COPYING EXISTING VALIDATION"
    )

    print("=" * 60)


    copied = 0


    for filename in os.listdir(
        input_images
    ):

        if not filename.endswith(
            IMAGE_EXTENSIONS
        ):
            continue


        image_source = os.path.join(
            input_images,
            filename
        )

        image_destination = os.path.join(
            output_images,
            filename
        )


        shutil.copy2(
            image_source,
            image_destination
        )


        label_name = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_source = os.path.join(
            input_labels,
            label_name
        )


        if os.path.exists(
            label_source
        ):

            shutil.copy2(
                label_source,
                os.path.join(
                    output_labels,
                    label_name
                )
            )


        copied += 1


    print(
        f"Existing validation copied: "
        f"{copied}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "VALIDATION DATASET PREPROCESSING"
    )

    print("=" * 60)

    print(
        f"Output size: {SIZE} x {SIZE}"
    )

    print(
        f"Crop margin: {MARGIN}px"
    )


    # Existing validation stays unchanged
    copy_existing_validation()


    # Newly added real validation
    process_new_category(
        "real"
    )


    # Newly added synthetic validation
    process_new_category(
        "synthetic"
    )


    print("\n" + "=" * 60)

    print(
        "PROCESSING COMPLETED"
    )

    print("=" * 60)

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )
