"""
Performs adaptive binarization on images in 'train', 'valid', and 'test' folders within a specified root directory.
"""
import os
import cv2

# Define the image file extensions to be processed
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def binarize_images(root_dir, block_size=11, c=2):
    """
    Performs adaptive binarization on images in 'train', 'valid', and 'test' folders
    within the specified root directory using OpenCV.

    Args:
        root_dir (str): The path to the root directory containing 'train', 'valid', and 'test' folders.
        block_size (int, optional): Size of the pixel neighborhood used for adaptive thresholding.
                                     Must be a positive odd integer. Defaults to 11.
        c (int, optional): Constant subtracted from the mean or weighted mean. Typically positive.
                             Defaults to 2.
    """
    sub_dirs = ['train', 'valid', 'test']
    for sub_dir in sub_dirs:
        input_dir = os.path.join(root_dir, sub_dir)
        output_dir = os.path.join(root_dir, f"{sub_dir}_bin_adaptive")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {sub_dir} folder (adaptive binarization, blockSize={block_size}, C={c})...")
        image_count = 0
        for filename in os.listdir(input_dir):
            if filename.endswith(IMAGE_EXTENSIONS):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Apply adaptive thresholding using mean and a constant offset
                        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY_INV, block_size, c)
                        cv2.imwrite(output_path, thresh)
                        # print(f"Adaptively binarized: {input_path} -> {output_path}")
                        image_count += 1
                    else:
                        print(f"Warning: Could not read image {input_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
        print(f"{sub_dir} folder processing completed, {image_count} images processed (adaptive binarization).\n")

if __name__ == "__main__":
    data_root = 'nycu-hw2-data'
    binarize_images(data_root, block_size=15, c=5) # You can adjust the parameters as needed
    print("Adaptive binarization processing completed for all folders!")