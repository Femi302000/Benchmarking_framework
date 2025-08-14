# draw_bboxes.py

import json
from PIL import Image, ImageDraw, ImageFont

def draw_bboxes(
    image_path="/home/femi/Benchmarking_framework/scripts/pre_processing/overlaysss/scene_000_synthetic_rgb_range.png",
    detections_json_path="/home/femi/Downloads/SAM-6D/SAM-6D/Data/Example/outputs/sam6d_results/detection_ism.json",
    output_path="/home/femi/Downloads/SAM-6D/SAM-6D/Data/Example/outputs/output_with_bboxes.jpg"
):
    """
    Draws bounding boxes on an image based on JSON detections,
    placing each confidence score in the bottom-right corner of its box.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Load detections JSON
    with open(detections_json_path, "r") as f:
        detections = json.load(f)

    # Draw each bounding box and score
    for det in detections:
        x, y, w, h = det["bbox"]
        score = det.get("score", 0.0)
        color = "green" if score >= 0.25 else "red"

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # Prepare score label
        text = f"{score:.2f}"
        # Compute text size via textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Compute bottom-right corner position of label
        tx1 = x + w - text_width
        ty1 = y + h
        tx2 = x + w
        ty2 = y + h + text_height

        # Draw label background
        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        # Draw score text
        draw.text((tx1, ty1), text, fill="white", font=font)

    # Save output image
    image.save(output_path)
    print(f"Annotated image saved to: {output_path}")


if __name__ == "__main__":
    draw_bboxes()
