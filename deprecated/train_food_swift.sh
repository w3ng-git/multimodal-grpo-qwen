#!/bin/bash
# filepath: /home/cpns1107/Qwen/train_food_swift.sh
# Swift框架多GPU GRPO训练启动脚本
# 使用0,1,2三张显卡用于训练，3号显卡用于vLLM快速rollout

# 检查是否安装了Swift
if ! command -v swift &> /dev/null; then
    echo "错误: Swift框架未安装。请先安装Swift:"
    echo "pip install ms-swift"
    exit 1
fi

# 设置显存大小限制，避免OOM
export MAX_PIXELS=401408  # 控制图像处理的最大像素数

# 检查vLLM服务是否在运行
if ! curl -s 127.0.0.1:8000 &> /dev/null; then
    echo "提醒: vLLM服务器似乎未运行。是否要启动vLLM服务?"
    read -p "启动vLLM服务? (y/n): " start_vllm
    if [[ "$start_vllm" == "y" ]]; then
        echo "启动vLLM服务器..."
        # 后台启动vLLM服务
        bash start_swift_vllm.sh &
        
        # 等待vLLM服务启动
        echo "等待vLLM服务器启动..."
        sleep 15
    else
        echo "警告: 继续训练但没有vLLM服务器，这可能会导致错误。"
        echo "如需启动vLLM服务器，请运行: bash start_swift_vllm.sh"
    fi
fi

# 训练主函数
function run_training() {
    echo "========================================"
    echo "    开始Swift GRPO多模态训练 - 食物分类器    "
    echo "========================================"
    echo "训练设备: CUDA 0,1,2"
    echo "vLLM服务: CUDA 3"
    echo "========================================"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=0,1,2  # 训练使用前3张卡
    export NPROC_PER_NODE=3            # 进程数量与显卡数量一致
    export MAX_PIXELS=262144           # 最大图像像素数限制
    
    # 准备数据集
    echo "准备数据集和插件..."
    
    # 运行数据准备脚本以创建插件
    python train_food_swift.py
    
    # 检查插件是否成功创建
    if [ ! -f "plugins/food_classifier_plugin.py" ]; then
        echo "错误: 未能创建奖励函数插件文件"
        exit 1
    fi
    
    # 启动Swift训练
    swift rlhf \
        --rlhf_type grpo \
        --model vl-3b-inst \
        --external_plugins plugins/food_classifier_plugin.py \
        --reward_funcs food_classifier format \
        --use_vllm true \
        --vllm_mode server \
        --vllm_server_host 127.0.0.1 \
        --vllm_server_port 8000 \
        --train_type full \
        --torch_dtype bfloat16 \
        --max_completion_length 512 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --learning_rate 1e-6 \
        --gradient_accumulation_steps 4 \
        --save_strategy 'steps' \
        --eval_strategy 'steps' \
        --eval_steps 50 \
        --save_steps 50 \
        --save_total_limit 5 \
        --logging_steps 1 \
        --output_dir output/swift_grpo_food \
        --warmup_ratio 0.01 \
        --dataloader_num_workers 2 \
        --num_generations 4 \
        --temperature 0.7 \
        --top_p 0.8 \
        --top_k 20 \
        --repetition_penalty 1.05 \
        --deepspeed zero2 \
        --log_completions true \
        --num_iterations 1 \
        --beta 0.04 \
        --max_grad_norm 0.5 \
        --dataset_provider "swift_dataset_provider:get_food_dataset"
}

# 安装Swift框架（如果需要）
function install_swift() {
    echo "安装Swift框架..."
    pip install -U ms-swift
    
    # 检查安装结果
    if ! command -v swift &> /dev/null; then
        echo "错误: Swift安装失败，请检查错误信息并手动安装"
        exit 1
    fi
    echo "Swift安装成功!"
}

# 检查并创建必要的数据和插件
function prepare_environment() {
    echo "准备训练环境..."
    
    # 运行数据准备脚本
    python train_food_swift.py
    
    # 检查插件是否成功创建
    if [ ! -f "plugins/food_classifier_plugin.py" ]; then
        echo "错误: 未能创建奖励函数插件文件"
        exit 1
    fi
    
    echo "环境准备完成!"
}

# 主执行流程
echo "开始Swift GRPO训练准备工作..."

# 检查Swift是否安装，如果未安装则询问是否安装
if ! command -v swift &> /dev/null; then
    echo "未检测到Swift框架"
    read -p "是否安装Swift? (y/n): " install_swift_prompt
    if [[ "$install_swift_prompt" == "y" ]]; then
        install_swift
    else
        echo "错误: 无法继续，Swift框架未安装"
        exit 1
    fi
fi

# 准备环境
prepare_environment

# 启动训练
run_training

echo "训练完成!"
echo "模型保存在: output/swift_grpo_food"
