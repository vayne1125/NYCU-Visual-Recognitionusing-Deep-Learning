import numpy as np
import os
import re # 用於自然排序檔案名

# --- 輔助函數：自然排序鍵 ---
def natural_sort_key(s):
    """ 用於自然排序檔案名的鍵函數。 """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def ensemble_npz_results(input_npz_paths, output_npz_path):
    """
    讀取多個 .npz 檔案，對其中的影像進行像素級平均 (ensemble)，
    並將集成後的結果儲存到一個新的 .npz 檔案中。

    Args:
        input_npz_paths (list): 包含要集成之 .npz 檔案路徑的列表。
                                例如: ['../local/results_run1/pred.npz', '../local/results_run2/pred.npz']
        output_npz_path (str): 儲存集成結果的新 .npz 檔案的路徑。
    """
    if not input_npz_paths:
        print("錯誤: 未提供任何輸入 .npz 檔案路徑。")
        return

    print(f"正在載入 {len(input_npz_paths)} 個 .npz 檔案進行集成...")

    all_loaded_data = []
    # 逐一載入所有的 .npz 檔案
    for i, npz_path in enumerate(input_npz_paths):
        if not os.path.exists(npz_path):
            print(f"警告: 檔案 {npz_path} 不存在，已跳過。")
            continue
        try:
            data = np.load(npz_path)
            all_loaded_data.append(data)
            print(f"成功載入: {npz_path}")
        except Exception as e:
            print(f"錯誤: 載入 {npz_path} 失敗: {e}，已跳過。")
            continue

    if not all_loaded_data:
        print("沒有成功載入的 .npz 檔案，無法進行集成。")
        return

    # 獲取所有 .npz 檔案中包含的所有影像檔案名稱
    # 假設所有 npz 檔案都包含相同的影像集
    # 如果它們包含的影像集可能不同，你需要更複雜的邏輯來處理缺失的影像
    image_filenames = sorted(all_loaded_data[0].files, key=natural_sort_key)
    print(f"在載入的 .npz 檔案中找到 {len(image_filenames)} 張影像。")

    ensembled_results = {}

    print("正在進行影像集成...")
    for filename in image_filenames:
        images_for_averaging = []
        
        for data in all_loaded_data:
            if filename in data:
                # 載入 (C, H, W) 格式的 uint8 影像
                img_np_uint8 = data[filename]
                # 轉換為 float32 並歸一化到 [0, 1] 範圍，以便進行精確平均
                img_float = img_np_uint8.astype(np.float32) / 255.0
                images_for_averaging.append(img_float)
            else:
                print(f"警告: 影像 {filename} 未在某個 .npz 檔案中找到，該檔案將不參與此影像的平均。")

        if images_for_averaging:
            # 將所有浮點數影像堆疊起來，然後沿著新的批次維度進行平均
            stacked_images = np.stack(images_for_averaging, axis=0) # 形狀為 (num_sources, C, H, W)
            ensembled_image_float = np.mean(stacked_images, axis=0) # 平均後形狀為 (C, H, W)

            # 將結果截斷到 [0, 1] 範圍，然後轉換回 [0, 255] 的 uint8
            ensembled_image_uint8 = (np.clip(ensembled_image_float, 0.0, 1.0) * 255.0).astype(np.uint8)
            ensembled_results[filename] = ensembled_image_uint8
        else:
            print(f"警告: 影像 {filename} 在所有輸入 .npz 檔案中均未找到，已跳過。")

    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_npz_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 儲存集成後的結果
    try:
        np.savez(output_npz_path, **ensembled_results)
        print(f"\n集成結果已成功儲存到 {output_npz_path}")
    except Exception as e:
        print(f"錯誤: 儲存集成 .npz 檔案 {output_npz_path} 失敗: {e}")

    # 關閉所有載入的 npz 檔案
    for data in all_loaded_data:
        data.close()
    print("所有載入的 .npz 檔案已關閉。")


if __name__ == "__main__":
    # --- 配置你的輸入和輸出路徑 ---
    # 假設你的原始推斷結果儲存在這些目錄下的 pred.npz 中
    # 你可以放多個你覺得表現不錯的 run 的 pred.npz 路徑
    INPUT_NPZ_PATHS = [
        '../local/results/finetune_28.60.npz',  # 第一次 Fine-tune 的結果
        '../local/results/tta_aaa.npz',  # 第二次 Fine-tune (調整 LR/WD/Loss) 的結果
        '../local/results/28.54.npz',  # 第三次 Fine-tune (不同 LR/WD/Loss) 的結果
        # ... 你可以根據實際情況添加更多路徑 ...
    ]

    # 集成後的結果將儲存到這個路徑
    OUTPUT_ENSEMBLED_NPZ_PATH = '../local/results_ensembled/pred_ensembled.npz'

    # 執行集成函數
    ensemble_npz_results(INPUT_NPZ_PATHS, OUTPUT_ENSEMBLED_NPZ_PATH)

    print("\n集成腳本運行結束。")