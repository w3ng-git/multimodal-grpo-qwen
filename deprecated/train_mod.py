from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoTokenizer, 
    AutoProcessor,
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
import json
import os
from qwen_vl_utils import process_vision_info

# 2. 模型和处理器加载
def load_model_and_processors():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "qwen2.5vl",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()
    
    tokenizer = AutoTokenizer.from_pretrained("qwen2.5vl")
    processor = AutoProcessor.from_pretrained("qwen2.5vl")
    
    return model, tokenizer, processor

# 3. 数据集准备
def prepare_dataset():
    data_path = "./data.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
        train_data = data[:-4]
        test_data = data[-4:]
    
    with open("train_data.json", "w", encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open("test_data.json", "w", encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    return Dataset.from_json("train_data.json")

# 4. 数据预处理函数
def process_func(example, processor, tokenizer):
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"{file_path}", "resized_height": 256, "resized_width": 256},
                {"type": "text", "text": "请描述这张图片的内容。"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}
    
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_values": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }

# 5. 配置LoRA
def setup_lora(model):
    config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, config)

# 6. 主函数
def main():
    # 加载模型和处理器
    model, tokenizer, processor = load_model_and_processors()
    
    # 准备数据集
    train_ds = prepare_dataset()
    
    # 处理数据集
    train_dataset = train_ds.map(lambda x: process_func(x, processor, tokenizer))
    
    # 打印数据集信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(train_dataset[0])
    
    # 配置LoRA
    peft_model = setup_lora(model)
    
    # 配置训练参数
    args = TrainingArguments(
        output_dir="output/Qwen2.5-VL-LoRA",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=5,
        save_steps=74,
        learning_rate=1e-4,
        gradient_checkpointing=True,
    )
    
    # 初始化训练器
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()