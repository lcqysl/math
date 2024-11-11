import json
import torch
import re
from llama_factory import Llama
from transformers import AutoTokenizer

# 初始化 Qwen2.5-Math-7b 模型和 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-Math-7b")  # 请确保路径正确
model = Llama.from_pretrained("Qwen2.5-Math-7b")  # 假设 llama-factory 支持直接加载模型

# 如果有 GPU，使用 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 读取文件中的数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 提取输出中的最后一个数字
def extract_answer(output_text):
    # 使用正则表达式匹配输出中的所有数字
    matches = re.findall(r"\$?(\d+(\.\d+)?)\$", output_text)
    if matches:
        # 返回最后一个匹配的数字
        return float(matches[-1][0])
    else:
        return None

# 对每个问题使用 Qwen2.5-Math-7b 模型进行推理
def validate_data(data):
    valid_count = 0
    for entry in data:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        expected_output = entry.get("output", "")

        # 构造模型输入
        prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer:"

        # Tokenize 输入
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 生成模型输出
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs['input_ids'])

        # 解码模型输出
        model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取模型输出中的最后一个答案
        model_answer = extract_answer(model_output)

        # 提取期望的输出中的最后一个答案
        expected_answer = extract_answer(expected_output)

        # 比较模型输出与期望输出的答案
        if model_answer is not None and expected_answer is not None and abs(model_answer - expected_answer) < 1e-3:
            valid_count += 1
            print(f"Correct output" )
        else:
            print(f"Incorrect output for input: {input_text}")
            print(f"Expected: {expected_answer}")
            print(f"Got: {model_answer}")

    return valid_count

# 主函数
if __name__ == "__main__":
    # 读取包含问题数据的文件
    file_path = "/data/hzf/LLaMA-Factory/data/math_all.json"  
    data = load_data(file_path)

    # 验证数据的正确性
    valid_count = validate_data(data)
    print(f"Correct answers: {valid_count}/{len(data)}")
