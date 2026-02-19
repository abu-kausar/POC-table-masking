import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
DATASET_DIR = "train/label-studio-annotation-2026-02-13"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")
CLASSES_FILE = os.path.join(DATASET_DIR, "classes.txt")

NUM_IMAGES = 9
GRID_COLS = 3
SEED = 42
# ----------------------------------------

random.seed(SEED)

# Load class names (if available)
class_names = []
if os.path.exists(CLASSES_FILE):
    with open(CLASSES_FILE, "r") as f:
        class_names = [c.strip() for c in f.readlines()]

# Get images
images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)
images = images[:NUM_IMAGES]


def draw_polygons(image, label_path, class_names):
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.strip().split()))

            # Case B: class + polygon
            if len(values) == 9:
                cls = int(values[0])
                coords = values[1:]
                label = class_names[cls] if cls < len(class_names) else str(cls)

            # Case A: polygon only
            elif len(values) == 8:
                coords = values
                label = None

            else:
                print(f"⚠️ Skipping invalid label: {label_path}")
                continue

            # Convert normalized → pixel coords
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                points.append([x, y])

            pts = np.array(points, np.int32).reshape((-1, 1, 2))

            color = (0, 255, 0)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

            # Put label text
            if label:
                x_text, y_text = points[0]
                cv2.putText(
                    image,
                    label,
                    (x_text, max(y_text - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

    return image


# Plot images
rows = (NUM_IMAGES + GRID_COLS - 1) // GRID_COLS
plt.figure(figsize=(15, 5 * rows))

for idx, img_name in enumerate(images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = draw_polygons(img, label_path, class_names)

    plt.subplot(rows, GRID_COLS, idx + 1)
    plt.imshow(img)
    plt.title(img_name)
    plt.axis("off")

# Display the image
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis('off')
plt.title('Visualization of Annotated Images')
plt.savefig(os.path.join('train/visualizations', 'visualization_of_annotated_images.png'), dpi=300, bbox_inches='tight')
# Save the figure to a file
plt.close()