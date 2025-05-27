#!/bin/bash
# 单GPU训练脚本 - 避免分布式训练问题

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 只使用一个GPU

# 执行训练
python train_food_classifier.py

echo "训练完成！" 