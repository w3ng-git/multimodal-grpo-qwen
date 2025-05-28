#!/bin/bash

# 设置环境变量，限制GPU内存使用
export CUDA_VISIBLE_DEVICES=0

# 确保依赖已安装
pip install -r requirements.txt

# 运行评测
python eval_vllm.py --model output/GRPO_FoodClassifier/v0-20250528-003934/checkpoint-1150 --output results.json

# 显示结果摘要
echo "评测完成，详细结果保存在 results.json" 