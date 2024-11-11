# api_call_example.py
from openai import OpenAI
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
instruction = "Please answer the math question."
input_text = "147 passengers fit into 7 buses. How many passengers fit in 4 buses?"
content = '{"instruction": "' + instruction + '", "input": "' + input_text + '"}'
print(content)
#messages = [{"role": "user", "content": "' + content + '"}]
messages=[  {
    "instruction": "Please answer the math question.",
    "input": "147 passengers fit into 7 buses. How many passengers fit in 4 buses?",
    "output": ""
 }]
#messages=[  {"147 passengers fit into 7 buses. How many passengers fit in 4 buses?"}]
result = client.chat.completions.create(messages=messages, model="/data/qmy/models/Qwen2.5-Math-1.5B")
print(result.choices[0].message)
#python llm.py