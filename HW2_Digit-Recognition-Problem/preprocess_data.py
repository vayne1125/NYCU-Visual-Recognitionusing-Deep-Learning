import os
from PIL import Image, ImageOps

def binarize_images(root_dir, threshold=200):
    """
    對指定根目錄下的 train, valid, test 資料夾中的圖像進行二值化處理。

    Args:
        root_dir (str): 包含 train, valid, test 資料夾的根目錄路徑。
        threshold (int): 二值化的閾值 (0-255)。你可以根據需要調整這個值。
    """
    sub_dirs = ['train', 'valid', 'test']
    for sub_dir in sub_dirs:
        input_dir = os.path.join(root_dir, sub_dir)
        output_dir = os.path.join(root_dir, f"{sub_dir}_bin")
        os.makedirs(output_dir, exist_ok=True)  # 創建輸出資料夾，如果不存在

        print(f"正在處理 {sub_dir} 資料夾...")
        for filename in os.listdir(input_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # 根據你的圖片格式調整
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                try:
                    image = Image.open(input_path).convert("L")  # 轉換為灰度圖像
                    image = ImageOps.invert(image)  # 反轉顏色，假設數字是深色背景是淺色
                    binarized_image = image.point(lambda p: p > threshold and 255)  # 使用閾值進行二值化 (白色為前景)
                    binarized_image = ImageOps.invert(binarized_image) # 再次反轉顏色，讓數字是深色 (如果你的模型預期數字是深色)
                    binarized_image.save(output_path)
                    print(f"已二值化: {input_path} -> {output_path}")
                except Exception as e:
                    print(f"處理 {input_path} 時發生錯誤: {e}")
        print(f"{sub_dir} 資料夾處理完成。\n")

if __name__ == "__main__":
    data_root = 'nycu-hw2-data'  # 你的資料根目錄
    binarize_images(data_root)
    print("所有資料夾的二值化處理完成！")