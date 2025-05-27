#!/bin/bash

# 设置默认参数
MODEL_NAME="vl-3b-inst"
HOST="0.0.0.0"
PORT="8000"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=1024

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tp)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

echo "正在启动Qwen2.5-VL模型服务..."
echo "模型: $MODEL_NAME"
echo "主机: $HOST"
echo "端口: $PORT"
echo "张量并行大小: $TENSOR_PARALLEL_SIZE"
echo "最大模型长度: $MAX_MODEL_LEN"

# 确保使用最新版本的vLLM (0.7.2+)
VLLM_VERSION=$(pip show vllm | grep Version | awk '{print $2}')
echo "vLLM版本: $VLLM_VERSION"

export CUDA_VISIBLE_DEVICES=3
# 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --device cuda \
    --host $HOST \
    --port $PORT \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --block-size 16 \
    --trust-remote-code 