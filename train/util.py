import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil

def visualize_annotations(dataset_dir, num_images=9, grid_cols=3, seed=42):
    # ---------------- CONFIG ----------------
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    classes_file = os.path.join(dataset_dir, "classes.txt")
    # ----------------------------------------

    random.seed(seed)

    # Load class names (if available)
    class_names = []
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            class_names = [c.strip() for c in f.readlines()]

    # Get images
    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)
    images = images[:num_images]


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
    rows = (num_images + grid_cols - 1) // grid_cols
    plt.figure(figsize=(15, 5 * rows))

    for idx, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = draw_polygons(img, label_path, class_names)

        plt.subplot(rows, grid_cols, idx + 1)
        plt.imshow(img)
        plt.title(img_name)
        plt.axis("off")
    
    # make output directory if not exists
    output_dir = os.path.join("train", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Display the image
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.title('Visualization of Annotated Images')
    plt.imshow(img)
    plt.savefig(os.path.join(output_dir, 'visualization_of_annotated_images.png'), dpi=300, bbox_inches='tight')
    # Save the figure to a file
    plt.close()
    
    
    
def split_dataset(dataset_dir, output_dir, train_ratio=0.8, seed=42):
    """
    This function splits the dataset into training and validation sets.
        It copies the images and their corresponding labels into separate directories for train and val.
    """
    # ---------------- CONFIG ----------------
    IMAGE_DIR = os.path.join(dataset_dir, "images")
    LABEL_DIR = os.path.join(dataset_dir, "labels")
    CLASSES_FILE = os.path.join(dataset_dir, "classes.txt")

    # ----------------------------------------

    random.seed(seed)

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Copy classes.txt
    shutil.copy(CLASSES_FILE, os.path.join(output_dir, "classes.txt"))

    # Get all images
    images = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]


    def copy_files(image_list, split):
        for img_name in image_list:
            img_src = os.path.join(IMAGE_DIR, img_name)
            lbl_src = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")

            img_dst = os.path.join(output_dir, "images", split, img_name)
            lbl_dst = os.path.join(output_dir, "labels", split, os.path.basename(lbl_src))

            shutil.copy(img_src, img_dst)

            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, lbl_dst)
            else:
                print(f"⚠️ Warning: label missing for {img_name}")


    copy_files(train_images, "train")
    copy_files(val_images, "val")

    print("✅ YOLO dataset split completed!")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")



if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    DATASET_DIR = "train/label-studio-annotation-2026-02-13"
    IMAGE_DIR = os.path.join(DATASET_DIR, "images")
    LABEL_DIR = os.path.join(DATASET_DIR, "labels")
    CLASSES_FILE = os.path.join(DATASET_DIR, "classes.txt")

    NUM_IMAGES = 9
    GRID_COLS = 3
    SEED = 42
    visualize_annotations(dataset_dir=DATASET_DIR, num_images=NUM_IMAGES, grid_cols=GRID_COLS)