"""
Qwen VL 工具函数 - 提供处理图像和聊天模板的工具函数
"""

import os
from PIL import Image
import numpy as np
import base64
import io
from typing import Any, Dict, List, Union

def process_vision_info(image_data: Any) -> Image.Image:
    """
    处理各种格式的图像数据，返回PIL.Image对象
    
    支持以下格式:
    - PIL.Image对象
    - 图像文件路径(字符串)
    - base64编码的图像字符串
    - numpy数组
    - 字节数据
    
    Args:
        image_data: 各种格式的图像数据
        
    Returns:
        PIL.Image: 处理后的图像对象
    """
    if image_data is None:
        raise ValueError("图像数据不能为None")
        
    # 如果已经是PIL.Image对象，直接返回
    if isinstance(image_data, Image.Image):
        return image_data
        
    # 如果是字符串，可能是文件路径或base64
    if isinstance(image_data, str):
        # 检查是否是文件路径
        if os.path.isfile(image_data):
            return Image.open(image_data)
            
        # 检查是否是base64编码
        if image_data.startswith(('data:image', 'base64:')):
            # 提取base64部分
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            elif image_data.startswith('base64:'):
                image_data = image_data[7:]
                
            # 解码base64
            image_bytes = base64.b64decode(image_data)
            return Image.open(io.BytesIO(image_bytes))
            
        raise ValueError(f"无法识别的图像字符串格式: {image_data[:30]}...")
        
    # 如果是numpy数组
    if isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data)
        
    # 如果是字节或字节流
    if isinstance(image_data, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_data))
        
    # 如果是文件对象
    if hasattr(image_data, 'read'):
        return Image.open(image_data)
        
    raise TypeError(f"不支持的图像数据类型: {type(image_data)}")

def apply_chat_template(messages: Dict[str, List[Dict]], tokenizer: Any) -> Dict[str, str]:
    """
    将会话消息应用到聊天模板中
    
    Args:
        messages: 包含会话消息的字典，格式为{"messages": [...]}
        tokenizer: 使用的分词器，需要有apply_chat_template方法
        
    Returns:
        Dict: 包含处理后文本的字典，格式为{"text": processed_text}
    """
    if not messages or "messages" not in messages:
        return {"text": ""}
        
    # 获取消息列表
    message_list = messages["messages"]
    
    # 如果tokenizer支持apply_chat_template方法，使用它
    if hasattr(tokenizer, "apply_chat_template"):
        # 处理消息中可能包含的图像
        processed_messages = []
        for msg in message_list:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # 用户消息可能包含图像
                content_list = msg["content"]
                # 过滤掉图像，仅保留文本内容用于聊天模板
                text_content = ""
                for item in content_list:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                processed_messages.append({"role": "user", "content": text_content})
            else:
                processed_messages.append(msg)
                
        # 应用聊天模板
        formatted_text = tokenizer.apply_chat_template(
            processed_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return {"text": formatted_text}
    
    # 如果没有apply_chat_template方法，使用简单拼接
    formatted_text = ""
    for msg in message_list:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # 处理内容可能是列表的情况(多模态)
        if isinstance(content, list):
            text_content = ""
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_content += item.get("text", "")
                # 图像内容在这里被忽略
            content = text_content
            
        formatted_text += f"{role.capitalize()}: {content}\n"
        
    return {"text": formatted_text}