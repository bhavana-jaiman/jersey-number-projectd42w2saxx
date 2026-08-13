import os
import cv2


# =========================
# Dataset paths
# =========================

base_path = "/home/ying/Desktop/Dataset/jersey_number_detection"

datasets = {
    "training": os.path.join(base_path, "training_dataset_Ying"),
    "validation": os.path.join(base_path, "validation_dataset_Ying")
}

output_base = os.path.join(base_path, "annotation")


# =========================
# Process each dataset
# =========================

for dataset_type, dataset_path in datasets.items():

    image_dir = os.path.join(dataset_path, "images")
    label_dir = os.path.join(dataset_path, "labels")

    output_dir = os.path.join(output_base, dataset_type)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {dataset_type} dataset...")

    # Get all images
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    for image_file in image_files:

        # -------------------------
        # Image path
        # -------------------------
        image_path = os.path.join(image_dir, image_file)

        # -------------------------
        # Corresponding label path
        # -------------------------
        image_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(label_dir, image_name + ".txt")

        # Read image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {image_file}")
            continue

        # Image dimensions
        image_height, image_width = image.shape[:2]

        # -------------------------
        # Read YOLO annotation
        # -------------------------
        if os.path.exists(label_path):

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # YOLO format:
                # class_id center_x center_y width height
                parts = line.split()

                if len(parts) != 5:
                    print(f"Invalid label in: {label_path}")
                    continue

                class_id = int(parts[0])

                center_x = float(parts[1])
                center_y = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # -------------------------
                # Convert YOLO coordinates
                # to pixel coordinates
                # -------------------------

                center_x = center_x * image_width
                center_y = center_y * image_height

                box_width = box_width * image_width
                box_height = box_height * image_height

                # x1, y1 = top-left
                # x2, y2 = bottom-right

                x1 = int(center_x - box_width / 2)
                y1 = int(center_y - box_height / 2)

                x2 = int(center_x + box_width / 2)
                y2 = int(center_y + box_height / 2)

                # Keep coordinates inside image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_width - 1, x2)
                y2 = min(image_height - 1, y2)

                # -------------------------
                # Draw bounding box
                # -------------------------

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # -------------------------
                # Draw class ID
                # -------------------------

                label = str(class_id)

                cv2.putText(
                    image,
                    label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:
            print(f"Label not found for: {image_file}")

        # -------------------------
        # Save annotated image
        # -------------------------

        output_path = os.path.join(output_dir, image_file)

        cv2.imwrite(output_path, image)

    print(f"Saved annotations to: {output_dir}")

print("\nDone!")
