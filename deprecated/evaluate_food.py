import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "vl-3b-inst",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("vl-3b-inst")

# 加载图片
image_paths = ["./bad_food/0001.jpg", "./bad_food/0002.jpg", "./bad_food/0003.jpg", "./bad_food/0004.jpg", "./bad_food/0005.jpg"]
images = [Image.open(path) for path in image_paths]

# 为每张图片生成评价
for i, image in enumerate(images):
    # 构造消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # PIL Image 会被自动处理
                {"type": "text", 
                 "text": "请给这道菜写幽默锐评，并用顺序的完整闭合格式 <analyse></analyse><comment></comment><answer></answer> 回答，"
                         "analyse部分是详细分析这道菜的成分和特点，详细信息越多越好，comment部分是幽默评论，answer 可以选择“黑暗料理”或“正常食物”，作为你判断的这是属于哪一类（二分类问题）答案，例如<answer>正常食物</answer>"}
            ]
        }
    ]
    
    # 使用处理器将消息转换为模型输入
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    inputs = processor(
        text=[text], 
        images=[image], 
        return_tensors="pt"
    ).to(model.device)
    
    # 生成评价
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # 最大生成长度
            do_sample=True,       # 启用采样
            temperature=0.7,      # 控制随机性
            top_p=0.8,            # 核采样
            top_k=20,             # 限制采样
            repetition_penalty=1.05,  # 重复惩罚
            min_p=0.0,            # this parameter is optional
        )
    
    # 解码输出
    response = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    print(f"图片 {image_paths[i]} 的评价:")
    print(response)
    print("-" * 50) 