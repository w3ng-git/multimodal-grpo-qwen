"""
测试Swift数据集提供器
"""

import swift_dataset_provider

# 测试获取数据集
dataset = swift_dataset_provider.get_food_dataset()
print(f"成功获取数据集，包含 {len(dataset)} 个样本")
print(f"数据集列: {dataset.column_names}")

# 显示样本示例
if len(dataset) > 0:
    example = dataset[0]
    print("\n样本示例:")
    print(f"- 图像路径: {example['images'][0]}")
    print(f"- 解决方案: {example['solution']}")
    
    # 打印提示文本的前100个字符
    prompt_text = example["messages"][0]["content"][1]["text"]
    print(f"- 提示前缀: {prompt_text[:100]}..." if len(prompt_text) > 100 else prompt_text)

print("\n测试完成，数据集提供器正常工作！")
