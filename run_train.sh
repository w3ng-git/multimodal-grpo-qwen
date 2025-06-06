WANDB_API_KEY=834078d14052291df0cdf5561e4018947444cfa1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
WANDB_PROJECT=swift_grpo_food_classifier2 \
MAX_PIXELS=401408 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model vl-3b-inst \
    --external_plugins food_plugin.py \
    --reward_funcs food_classification format \
    --use_vllm true \
    --vllm_device 6 \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 8192 \
    --vllm_limit_mm_per_prompt '{"image": 1}' \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'food_dataset/dataset.json' \
    --max_completion_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 1 \
    --output_dir output/GRPO_FoodClassifier \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 2 \
    --num_generations 6 \
    --temperature 1.0 \
    --repetition_penalty 1.1 \
    --system 'system_prompt.txt' \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --log_completions true \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --report_to wandb 
