"""This script converts labeled training images into COCO annotation format."""
import os
import json
import cv2
import numpy as np
import skimage.io as sio
from utils import encode_mask


def generate_coco_annotations(train_dir, output_path):
    """
    Generate COCO-format annotations from class-labeled masks in subfolders.

    Args:
        train_dir (str): Path to the root training directory, with subfolders for each image set.
        output_path (str): Path where the output COCO JSON file will be saved.

    Returns:
        None
    """
    folder_names = sorted([f for f in os.listdir(
        train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    images = []
    annotations = []
    categories = [{"id": 1, "name": "1"}, {"id": 2, "name": "2"}, {
        "id": 3, "name": "3"}, {"id": 4, "name": "4"}]
    ann_id = 0
    img_id = 0
    for folder in folder_names:
        folder_path = os.path.join(train_dir, folder)
        print(folder_path)
        img_id += 1
        for file in os.listdir(folder_path):
            if file.startswith("image"):
                img_path = os.path.join(folder_path, file)
                print(img_path)
                img = cv2.imread(str(img_path))  # pylint: disable=no-member
                img_h = int(img.shape[0])
                img_w = int(img.shape[1])

                image = {
                    "id": img_id,
                    "file_name": folder + '/' + file,
                    "height": img_h,
                    "width": img_w
                }
                images.append(image)
            elif file.startswith("class"):
                # 讀取 classi.tif (class1, class2, class3, class4)
                class_path = os.path.join(folder_path, file)
                class_mask = sio.imread(class_path)
                print(class_path)

                # class_mask 的物件編號
                unique_values = np.unique(class_mask)

                # 根據每個連通區域處理 bbox 和 segmentation
                for i in unique_values:
                    if i == 0:
                        continue
                    ann_id += 1
                    binary_mask = class_mask == i
                    ys, xs = np.where(binary_mask)
                    xmin, xmax = xs.min(), xs.max()
                    ymin, ymax = ys.min(), ys.max()
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    # 計算 bbox
                    bbox = [float(xmin), float(ymin), float(w), float(h)]

                    area = int(np.sum(binary_mask))

                    rle_mask = encode_mask(binary_mask=binary_mask)

                    # 設定每個物件的 category_id，這裡假設每個物件的類別都屬於同一個類別
                    category_id = int(file[5])  # class1 -> 1, class2 -> 2, ...

                    # 整理出一個 annotation
                    annotation = {
                        "id": ann_id,
                        "image_id": img_id,
                        "bbox": bbox,
                        "category_id": category_id,
                        "segmentation": rle_mask,
                        "area": area
                    }
                    annotations.append(annotation)
    coco_format_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    output_dir = os.path.dirname(output_path)
    if output_dir != '' and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 indent 讓 JSON 更易讀
        json.dump(coco_format_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved COCO annotations to {output_path}")


TRAIN_PATH = './data/train/'
OUTPUT_PATH = './data/train.json'
generate_coco_annotations(TRAIN_PATH, OUTPUT_PATH)
