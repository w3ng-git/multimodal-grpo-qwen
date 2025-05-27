"""
Qwen VL GRPO Trainer - 支持多模态(图像+文本)输入的GRPO训练器
此文件继承自grpo_trainer.py，添加了对图像处理的支持
"""

import torch
from typing import Any, Callable, Optional, Union
from collections.abc import Sized
from contextlib import nullcontext
import warnings

from datasets import Dataset, IterableDataset
from transformers import (
    AutoTokenizer, 
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl import GRPOTrainer, GRPOConfig

# 从grpo_trainer.py导入RewardFunc类型
from trl.trainer.grpo_trainer import RewardFunc


class QwenVLGRPOTrainer(GRPOTrainer):
    """
    GRPO训练器的多模态扩展，支持图像和文本的混合输入。
    专为Qwen2.5 VL模型设计，可以处理包含图像的对话格式数据。
    
    Example:
    ```python
    trainer = QwenVLGRPOTrainer(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
        processor=processor,
    )
    
    trainer.train()
    ```
    
    Args:
        processor (`Any`, *optional*):
            用于处理多模态输入的处理器，通常是AutoProcessor。如果为None，
            则会尝试从模型名称加载。
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        processor: Optional[Any] = None,  # 新增的处理器参数
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional[Any] = None,
    ):
        # 调用父类初始化
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        
        # 保存processor
        self.processor = processor
        if self.processor is None:
            # 如果没有提供processor，尝试从模型加载
            from transformers import AutoProcessor
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model if isinstance(model, str) else model.config._name_or_path
                )
            except Exception as e:
                warnings.warn(
                    f"无法加载AutoProcessor: {e}。如果您使用的是多模态模型，请确保明确提供processor。"
                )
    
    def _prepare_inputs(self, generation_batch):
        """
        准备模型训练/评估的输入。处理生成完成和批次管理。
        针对多模态输入进行了扩展。
        """
        return super()._prepare_inputs(generation_batch)
    
    def _generate_and_score_completions(self, inputs):
        """
        为输入生成完成并计算奖励。处理包含图像的多模态输入。
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        
        # 处理多模态输入，依赖于process_vision_info实用函数
        from qwen_vl_utils import process_vision_info
        
        # 使用process_vision_info处理prompts中的图像信息
        processed_prompts = []
        for p in prompts:
            if isinstance(p, list) and len(p) > 0:
                # 处理会话格式，可能包含图像
                processed_p = []
                for msg in p:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        # 处理用户消息，可能包含图像
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # 内容是一个列表，可能包含图像
                            processed_content = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "image":
                                    # 处理图像项
                                    image_data = item.get("image")
                                    # 确保图像已正确加载
                                    processed_image = process_vision_info(image_data)
                                    processed_content.append({"type": "image", "image": processed_image})
                                else:
                                    # 保持文本项不变
                                    processed_content.append(item)
                            msg = {"role": msg["role"], "content": processed_content}
                        processed_p.append(msg)
                    else:
                        # 保持非用户消息不变
                        processed_p.append(msg)
                processed_prompts.append(processed_p)
            else:
                # 非对话格式，按原样处理
                processed_prompts.append(p)
        
        # 使用processor处理多模态输入，转化为模型输入格式
        if self.processor is not None:
            # 对于多模态输入使用processor
            try:
                from qwen_vl_utils import apply_chat_template
                # 创建会话格式输入
                messages = [{"messages": p} for p in processed_prompts]
                # 应用聊天模板
                prompt_inputs = [apply_chat_template(msg, self.processing_class) for msg in messages]
                # 使用processor处理
                prompt_inputs = self.processor(
                    prompt_inputs, return_tensors="pt", padding=True, padding_side="left"
                )
                prompt_inputs = super()._prepare_inputs(prompt_inputs)
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            except Exception as e:
                warnings.warn(f"使用processor处理输入失败: {e}。回退到标准文本处理。")
                # 回退到标准处理
                prompt_inputs = super()._generate_and_score_completions(inputs)
                return prompt_inputs
        else:
            # 回退到标准处理
            prompt_inputs = super()._generate_and_score_completions(inputs)
            return prompt_inputs
        
        # 限制prompt长度
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            
        # 其余代码与原始_generate_and_score_completions方法相同
        # 调用父类方法的剩余部分来处理生成和评分
        # 这部分需要重写完整的生成和评分逻辑，但为简化示例，我们只处理图像特定部分
        
        # 为此示例，我们只处理了多模态输入处理部分
        # 实际实现需要完整重写生成和评分部分
        
        # 现在使用原始方法完成其余部分
        inputs_with_processed_prompts = [{"prompt": p} for p in processed_prompts]
        for i, inp in enumerate(inputs):
            for k, v in inp.items():
                if k != "prompt":
                    inputs_with_processed_prompts[i][k] = v
                    
        return super()._generate_and_score_completions(inputs_with_processed_prompts)