import os
import shutil
import random
import glob
from tqdm import tqdm

def organize_dataset(source_folder, test_ratio=0.1):
    """
    将source_folder中的图片文件按比例分割为训练集和测试集。
    保持原文件名的连续性，只是将部分文件移动到测试集目录。
    
    Args:
        source_folder (str): 原始图片文件夹
        test_ratio (float): 测试集比例，默认为0.1
    """
    print(f"处理文件夹: {source_folder}")
    
    # 确定文件夹名称
    folder_name = os.path.basename(source_folder)
    train_folder = source_folder
    test_folder = os.path.join("test", folder_name)
    
    # 创建测试集文件夹
    os.makedirs(test_folder, exist_ok=True)
    
    # 获取所有jpg文件
    jpg_files = sorted(glob.glob(os.path.join(source_folder, "*.jpg")))
    total_files = len(jpg_files)
    
    if total_files == 0:
        print(f"警告: {source_folder} 中未找到jpg文件")
        return
    
    # 计算测试集大小
    test_size = int(total_files * test_ratio)
    print(f"总文件数: {total_files}, 将移动 {test_size} 文件到测试集")
    
    # 随机选择测试集文件
    random.seed(42)  # 设置随机种子以确保结果可复现
    test_indices = random.sample(range(total_files), test_size)
    
    # 移动文件到测试集
    for idx in tqdm(sorted(test_indices), desc=f"移动 {folder_name} 文件到测试集"):
        src_file = jpg_files[idx]
        dst_file = os.path.join(test_folder, os.path.basename(src_file))
        shutil.move(src_file, dst_file)
    
    print(f"{folder_name} 处理完成: {total_files - test_size} 文件在训练集, {test_size} 文件在测试集")

def main():
    # 创建测试集根目录
    os.makedirs("test", exist_ok=True)
    
    # 处理good_food文件夹
    organize_dataset("good_food")
    
    # 处理bad_food文件夹
    organize_dataset("bad_food")
    
    print("\n数据集整理完成！")
    print("- 训练集: good_food/ 和 bad_food/")
    print("- 测试集: test/good_food/ 和 test/bad_food/")

if __name__ == "__main__":
    main() 