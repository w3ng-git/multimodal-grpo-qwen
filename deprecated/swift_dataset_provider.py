"""
Swift数据集提供器 - 允许Swift框架直接从内存加载数据集
不再需要保存到中间文件
"""

from train_food_swift import create_food_classifier_dataset

# 使用惰性加载避免模块导入时就加载数据集
_food_dataset = None

def get_food_dataset():
    """返回准备好的食物分类数据集"""
    global _food_dataset
    if _food_dataset is None:
        print("首次调用，开始创建数据集...")
        _food_dataset = create_food_classifier_dataset()
    return _food_dataset
