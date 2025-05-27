#!/bin/bash
# filepath: /home/cpns1107/Qwen/check_vllm.sh
# 检查和管理vLLM服务脚本

# 设置
PORT=8000
HOST="127.0.0.1"
VLLM_GPU=3
MODEL_NAME="vl-3b-inst"

# 功能: 检查vLLM服务状态
check_vllm() {
  if curl -s $HOST:$PORT &> /dev/null; then
    echo "✅ vLLM服务正在运行 ($HOST:$PORT)"
    return 0
  else
    echo "❌ vLLM服务未在运行 ($HOST:$PORT)"
    return 1
  fi
}

# 功能: 启动vLLM服务
start_vllm() {
  echo "🚀 启动vLLM服务..."
  
  # 设置环境变量
  export CUDA_VISIBLE_DEVICES=$VLLM_GPU
  export MAX_PIXELS=153600  # 降低像素数限制
  
  # 以下三种方式尝试启动，取消注释你需要的方式
  
  # 1. 使用Swift自带的vLLM启动方式
  echo "尝试使用Swift rollout启动vLLM服务..."
  MAX_PIXELS=153600 swift rollout \
    --model $MODEL_NAME \
    --gpu_memory_utilization 0.4 \
    --host $HOST \
    --port $PORT &
  
  # 2. 如果上面的方式不行，可以尝试直接使用vLLM的方式
  # echo "尝试使用原生vLLM启动服务..."
  # python -m vllm.entrypoints.openai.api_server \
  #  --model $MODEL_NAME \
  #  --host $HOST \
  #  --port $PORT \
  #  --tensor-parallel-size 1 \
  #  --gpu-memory-utilization 0.4 \
  #  --max-model-len 4096 &
  
  # 3. 使用start_swift_vllm.sh脚本
  # echo "尝试使用start_swift_vllm.sh启动服务..."
  # bash start_swift_vllm.sh &
  
  # 等待服务启动
  echo "等待vLLM服务启动..."
  for i in {1..30}; do
    sleep 2
    if check_vllm; then
      echo "✅ vLLM服务已成功启动!"
      return 0
    fi
    echo "等待中... ($i/30)"
  done
  
  echo "❌ vLLM服务启动超时，请检查日志排查问题"
  return 1
}

# 功能: 停止vLLM服务
stop_vllm() {
  echo "🛑 正在停止vLLM服务..."
  
  # 查找运行在指定端口的进程
  PID=$(lsof -t -i:$PORT)
  
  if [ -z "$PID" ]; then
    echo "没有找到运行在端口 $PORT 的进程"
    return 0
  fi
  
  echo "发现运行在端口 $PORT 的进程 (PID: $PID)，正在停止..."
  kill $PID
  
  # 等待进程终止
  for i in {1..10}; do
    if ! ps -p $PID > /dev/null; then
      echo "✅ vLLM服务已成功停止"
      return 0
    fi
    sleep 1
  done
  
  # 如果进程仍在运行，强制终止
  echo "进程仍在运行，尝试强制终止..."
  kill -9 $PID
  
  if ! ps -p $PID > /dev/null; then
    echo "✅ vLLM服务已强制停止"
    return 0
  else
    echo "❌ 无法停止vLLM服务，请手动终止进程 PID: $PID"
    return 1
  fi
}

# 主命令处理
case "$1" in
  start)
    check_vllm || start_vllm
    ;;
  stop)
    stop_vllm
    ;;
  restart)
    stop_vllm
    sleep 2
    start_vllm
    ;;
  status)
    check_vllm
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac

exit 0
