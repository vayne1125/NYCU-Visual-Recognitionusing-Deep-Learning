import os
from PIL import Image, ImageOps
import cv2

def binarize_images(root_dir):
    """
    對指定根目錄下的 train, valid, test 資料夾中的圖像進行自適應二值化處理 (使用 OpenCV)。
    """
    sub_dirs = ['train', 'valid', 'test']
    for sub_dir in sub_dirs:
        input_dir = os.path.join(root_dir, sub_dir)
        output_dir = os.path.join(root_dir, f"{sub_dir}_bin_adaptive")
        os.makedirs(output_dir, exist_ok=True)

        print(f"正在處理 {sub_dir} 資料夾 (自適應二值化)...")
        for filename in os.listdir(input_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # 使用自適應閾值 (這裡使用均值和一個小的常數偏移)
                        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 2) # 調整 block size 和 C
                        cv2.imwrite(output_path, thresh)
                        # print(f"已自適應二值化: {input_path} -> {output_path}")
                    else:
                        print(f"警告: 無法讀取圖像 {input_path}")
                except Exception as e:
                    print(f"處理 {input_path} 時發生錯誤: {e}")
        print(f"{sub_dir} 資料夾處理完成 (自適應二值化)。\n")

if __name__ == "__main__":
    data_root = 'nycu-hw2-data'
    binarize_images(data_root)
    print("所有資料夾的自適應二值化處理完成！")