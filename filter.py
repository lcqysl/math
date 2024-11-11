import json

# 文件路径
math_all_path = '/data/hzf/LLaMA-Factory/data/math_all.json'
wrong_path = '/data/hzf/wrong'

# 读取math_all.json文件
with open(math_all_path, 'r', encoding='utf-8') as f:
    math_data = json.load(f)

# 读取wrong文件，获取需要舍去的索引
with open(wrong_path, 'r', encoding='utf-8') as f:
    wrong_indexes = []
    for line in f:
        if line.startswith("Error at index"):
            # 提取索引数字
            index = int(line.strip().split()[-1])
            wrong_indexes.append(index)

# 移除对应索引的数据
filtered_data = [item for idx, item in enumerate(math_data) if idx not in wrong_indexes]

# 保存处理后的数据到新的JSON文件
output_path = '/data/hzf/LLaMA-Factory/data/math_all_filtered.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print(f"新文件已保存为 {output_path}")
