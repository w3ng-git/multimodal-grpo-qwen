import re
from typing import Dict, List, Union
# 导入swift.plugin.orm以获取orms字典
from swift.plugin.orm import orms, ORM



# 直接定义基类，不再继承orms.ORM
class FoodClassificationORM(ORM):
    """奖励函数：检查<answer>标签中的回答是否与标准答案匹配"""

    def __call__(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        *args,
        **kwargs,
    ) -> Union[float, List[float]]:
        if isinstance(predictions, str):
            predictions = [predictions]
        if isinstance(references, str):
            references = [references]

        scores = []
        for p, r in zip(predictions, references):
            # 提取预测答案中的<answer>内容
            answer_match = re.search(r'<answer>(.*?)</answer>', p, re.DOTALL)
            pred_answer = answer_match.group(1).strip() if answer_match else ""
            
            # 提取参考答案中的<answer>内容
            ref_match = re.search(r'<answer>(.*?)</answer>', r, re.DOTALL)
            ref_answer = ref_match.group(1).strip() if ref_match else ""

            # 计算分数
            if pred_answer.lower() == ref_answer.lower():
                score = 1.0  # 完全匹配
            elif "黑暗料理" in pred_answer and "黑暗料理" in ref_answer:
                score = 0.8  # 部分匹配
            elif "正常食物" in pred_answer and "正常食物" in ref_answer:
                score = 0.8  # 部分匹配
            else:
                score = 0.0  # 不匹配
                
            scores.append(score)

        if len(scores) == 1:
            return scores[0]
        return scores


class FormatORM(ORM):
    """奖励函数：检查是否包含<analyse>、<comment>、<answer>三个完整标签"""

    def __call__(
        self,
        predictions: Union[List[str], str],
        references: Union[List[str], str],
        *args,
        **kwargs,
    ) -> Union[float, List[float]]:
        if isinstance(predictions, str):
            predictions = [predictions]

        scores = []
        for p in predictions:
            score = 0.0
            
            # 检查analyse标签
            if re.search(r'<analyse>.*?</analyse>', p, re.DOTALL):
                score += 0.3
                
            # 检查comment标签
            if re.search(r'<comment>.*?</comment>', p, re.DOTALL):
                score += 0.3
                
            # 检查answer标签
            if re.search(r'<answer>.*?</answer>', p, re.DOTALL):
                score += 0.4
                
            scores.append(score)

        if len(scores) == 1:
            return scores[0]
        return scores

# 直接注册奖励函数
orms["food_classification"] = FoodClassificationORM
orms["format"] = FormatORM 