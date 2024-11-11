import re
import json
from vllm import LLM, SamplingParams

# 文件路径
path = "/data/hzf/LLaMA-Factory/data/math_test.json"
wrong_output_path = "/data/hzf/all_result_1"  # 错误输出文件路径

# 加载JSON数据
with open(path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 从JSON数据中提取`input`字段和正确答案
prompts = [item["instruction"] + " " + item["input"] for item in data]
correct_answers = [item["output"] for item in data]

# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10000)

# 加载模型
llm = LLM(model="/data/qmy/models/Qwen2.5-Math-1.5B")

# 执行推理
outputs = llm.generate(prompts, sampling_params)

# 提取文本中的最后一个数字
def extract_last_number(text: str) -> str:
    # 使用正则表达式匹配所有数字（包括整数和小数）
    matches = re.findall(r"(\d+(\.\d+)?)", text)
    if matches:
        # 返回最后一个匹配的数字字符串
        return matches[-1][0]  # 返回第一个匹配组，即数字部分
    return None

correct_count = 0
total_count = 0

# 打开文件用于写入错误索引
with open(wrong_output_path, "w", encoding="utf-8") as wrong_file:
    # 输出推理结果并提取最后一个数字
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        # 提取最后一个数字
        last_number = extract_last_number(generated_text)
        correct_answer = extract_last_number(correct_answers[i])
        
        # 打印与正确答案的对比
        print(f"Extracted last number: {last_number}")
        print(f"Correct answer: {correct_answer}")
        
        # if last_number == correct_answer:
        #     print("The model's output is correct!\n")
        #     correct_count += 1
        # else:
        #     print("The model's output is incorrect.\n")
        #     # 错误时写入文件
        #     wrong_file.write(f"Error at index {i}\n")
        #     wrong_file.write(f"last_number: {last_number}\n")
        #     wrong_file.write(f"correct_answer: {correct_answer}\n")
        #     wrong_file.write(f"generated_text: {generated_text}\n\n\n\n\n\n")
        if last_number == correct_answer:
            wrong_file.write(f"正确\n")
            wrong_file.write(f"{correct_answer}\n")
        else:
            wrong_file.write(f"错误\n")
            wrong_file.write(f"{correct_answer}\n")
            wrong_file.write(f"{last_number}\n")
        
        total_count += 1

# 输出总结果
if total_count > 0:
    print(f"Total count: {total_count}")
    print(f"Correct count: {correct_count}")
    accuracy = correct_count / total_count * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No data processed.")
