predicted_number = f"{digit1_prediction.item()}{digit2_prediction.item()}"

os.makedirs("results/predicted_images", exist_ok=True)

display_img = body_img.copy()

cv2.putText(
    display_img,
    f"Predicted: {predicted_number}",
    (5, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imwrite(
    f"results/predicted_images/{os.path.basename(img)}",
    display_img
)
