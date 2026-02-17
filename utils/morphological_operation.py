import os
import json
import cv2
import numpy as np

def visualize_connected_components(mask_path, output_path=None):
    """
    Visualize connected components in the mask for debugging.
    
    Args:
        mask_path (str): Path to the binary mask image.
        output_path (str, optional): Path to save the visualization. If None, it won't be saved.
    Returns:
        np.ndarray: Image with connected components visualized.
    """
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create colored visualization
    colored_labels = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Assign random colors to each component
    np.random.seed(42)  # For reproducibility
    for i in range(1, num_labels):  # Skip background
        color = np.random.randint(0, 255, size=3).tolist()
        colored_labels[labels == i] = color
        
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, colored_labels)
    
    return colored_labels, num_labels


def refine_mask_with_morphology(mask, output_path=None):
    """
    Refine mask using morphological operation to better separate words.
    
    Args:
        mask_path (str): Path to the binary mask image.
        output_path (str, optional): Path to save the refined mask. If None, it won't be saved.
    Returns:
        np.ndarray: Refined binary mask.
    """
    
    # Invert if mas has white bacground (text is white on black is better for connected components)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
        # print("Mask inverted: background was white, now black")
    # Apply morphological opening to remove noise
    kernel_open = np.ones((2,2), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # save refined mask
    if output_path:
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, mask_opened)

    return mask_opened

def get_word_bounding_boxes_from_mask(image_path, min_area=20, max_area=50000):
    """
    Get bounding boxes of words from the binary mask.
    
    Args:
        mask_path (str): Path to the binary mask image.
        min_area (int): Minimum area of connected component to be considered a word.
        max_area (int): Maximum area of connected component to be considered a word.
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h) for each detected word.
    """
    # Read original image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step-1: Apply erosion to separate connected words
    kernel = np.ones((4, 4), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=3)
    
    # Step-2: Refine mask with morphological operations
    refined_mask = refine_mask_with_morphology(img_erosion)
    
    # Step-3: Make Binary Image and Find Connected Components
    # make binary image on refined_mask
    _, refined_mask_binary = cv2.threshold(refined_mask, 90, 255, cv2.THRESH_BINARY)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_mask_binary, connectivity=8)

    FILL_THRESHOLD = 0.25   # tune this (0.9 = very strict)

    boxes = []

    for i in range(1, num_labels):  # skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bbox_area = w * h
        fill_ratio = area / bbox_area

        if fill_ratio >= FILL_THRESHOLD:
            boxes.append((x, y, w, h))

    return img_erosion, refined_mask, refined_mask_binary, boxes


def main():
    IMAGE_FOLDER = "/home/kawsar/Downloads/label-studio-annotation-2026-02-13/images"
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    morphological_output_dir = "morphological_outputs"
    os.makedirs(morphological_output_dir, exist_ok=True)

    for i, image_name in enumerate(os.listdir(IMAGE_FOLDER)):
        image_path = os.path.join(IMAGE_FOLDER, image_name)
        img_erosion, refined_mask, refined_mask_binary, word_boxes = get_word_bounding_boxes_from_mask(image_path)
        
        image = cv2.imread(image_path)
        # draw boxes on original image for visualization
        img_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in word_boxes:
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        output_dir = os.path.join(OUTPUT_DIR, os.path.splitext(image_name)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        # save all intermediate results
        cv2.imwrite(os.path.join(output_dir, f'original_{image_name}.png'), image)
        cv2.imwrite(os.path.join(output_dir, f'eroded_{image_name}.png'), img_erosion)
        cv2.imwrite(os.path.join(output_dir, f'refined_mask_{image_name}.png'), refined_mask)
        cv2.imwrite(os.path.join(output_dir, f'refined_mask_binary_{image_name}.png'), refined_mask_binary)
        cv2.imwrite(os.path.join(output_dir, f'detected_words_{image_name}.png'), img_color)
        cv2.imwrite(os.path.join(morphological_output_dir, f'{os.path.splitext(image_name)[0]}_morphology.png'), img_color)
        print(f"{i+1}. Processed {image_name}, found {len(word_boxes)} word boxes.")
        
        
if __name__ == "__main__":
    main()