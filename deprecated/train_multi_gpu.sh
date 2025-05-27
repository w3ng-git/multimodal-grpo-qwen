#!/bin/bash
# 多GPU训练脚本 - 使用3张GPU进行训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2  # 使用前3个GPU进行训练
# GPU 3 将用于vLLM推理（在train_food_classifier.py中已配置）

# 添加Torch分布式张量控制变量
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # 启用详细调试信息
export TORCH_DISTRIBUTED_TENSOR_FALLBACK=1  # 允许张量类型混合

# 注意：此脚本假设vLLM服务器已由用户手动启动
# 如果需要运行vLLM服务器，请先执行: bash start_vllm_server.sh

# 执行多GPU训练 - 使用单节点方式启动
echo "开始多GPU训练..."
# 方法1：使用指定配置启动
accelerate launch --config_file=accelerate_config.yaml train_food_classifier.py

# 如果上面的方法失败，可以尝试使用下面的备选方法（取消注释使用）
# 方法2：使用更简单的启动方式
#PYTHONPATH="." accelerate launch --multi_gpu --num_processes=3 train_food_classifier.py

echo "训练完成！" 