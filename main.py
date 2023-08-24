import cv2
import numpy as np


def extract_regions_between_lines(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use the Hough transform to detect horizontal lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=2)

    # If lines is None or less than 2 lines are detected, return an empty list
    if lines is None or len(lines) < 2:
        return []

    # Sort lines by their y-coordinate
    sorted_lines = sorted(lines, key=lambda x: x[0][1])

    cropped_images = []
    offset = 20  # Adjust based on the thickness of your lines

    for i in range(len(sorted_lines) - 1):
        y1 = sorted_lines[i][0][1]
        y2 = sorted_lines[i + 1][0][1]

        # Crop the region between two consecutive lines from the original colored image, excluding the lines
        cropped = image[y1 + offset:y2 - offset, :]
        cropped_images.append(cropped)

    return cropped_images


def save_cropped_images(images, base_filename):
    for idx, img in enumerate(images):
        if img is not None and not img.size == 0:
            cv2.imwrite(f"{base_filename}_{idx}.png", img)
        else:
            print(f"Image {idx} is empty. Skipping...")


def process_image(image_path, output_base_filename):
    cropped_regions = extract_regions_between_lines(image_path)
    save_cropped_images(cropped_regions, output_base_filename)

if __name__ == "__main__":
    # Example usage:
    process_image("./collectionpdf/design.png", 'output_image')
