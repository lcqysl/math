

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model directly
import torch
from prompt_utils import get_prompt
import json
import argparse
import utils
from prompt_utils import *
from data_loader import BatchDatasetLoader
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

import random
import re
import matplotlib.pyplot as plt
from collections import Counter

# model_path = "/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/Qwen2-7B-Instruct"
# llm = LLM(model=model_path, tensor_parallel_size=1)
# tokenizer = llm.get_tokenizer()

# back_prompt =  """
# Please act as a professional math teacher.
# Your goal is to verify if the answer to a math word problem is correct by checking if the answer satisfies the conditions of the original question.
# To achieve the goal, you have two jobs.
# # Substituting the given answer back into the problem to check if it aligns with the problem's conditions.
# # Determine whether the answer is correct or incorrect based on this verification.

# You have one principles to do this.
# # Clearly state whether the answer is correct or incorrect.

# Given Question: {question}
# Given Answer: {answer}
# Your output should be in the following format:
# FINAL JUDGEMENT: The answer is <correct/incorrect> based on the verification
# """

back_prompt = """
Please act as a professional math teacher.
Your goal is to accurately solve a math word problem by first clarifying the question and then verifying if the answer satisfies the problem's conditions.
To achieve the goal, you have two jobs.
# Clarify and restate the Given Question to avoid any ambiguity.
# Substitute the given answer back into the restated problem to check if it aligns with the problem's conditions.

You have two principles to do this.
# Ensure the problem is clearly and unambiguously stated.
# Ensure the verification process checks if the answer is consistent with the restated problem.

Given Question: {question}
Given Answer: {answer}
Your output should be in the following format:
FINAL JUDGEMENT: The answer is <correct/incorrect> based on the verification
"""

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        id = -1
        for line in f1:
            id += 1
            print(id)
            data = json.loads(line)
            con.append(data)
    print(con[0])        
    return con


def view_datalen(random_list, save_path, name):

    score = 0
    for i in range(0, len(random_list)):
        score += random_list[i]
    print('avg: ', score/len(random_list))

    count = Counter(random_list)
    numbers = list(count.keys())
    counts = list(count.values())

    print('** : ', numbers)
    print('** : ', counts)

    plt.figure(figsize=(5, 4))
    bars=plt.bar(numbers, counts, color='#AFEEEE', edgecolor='blue')

    # plt.title('Frequency of Numbers', fontsize=16, fontweight='bold')
    plt.xlabel(name+' Candidate Set Size', fontsize=18)
    plt.ylabel('Number', fontsize=18)
    # plt.xticks(numbers)  # 设置x轴刻度

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)


def inference(input_prompt):

    prompt = []
    prompt_i = input_prompt
    tmp = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_i}],
        tokenize=False,
    )
    prompt.append(tmp)
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=2048, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
    outputs = llm.generate(prompt, sampling_params)
    res_data = []
    for j in range(0, len(outputs)):
        output = outputs[j]
        prompt = output.prompt
        response = output.outputs[0].text
        res_data.append(response)

    return res_data[0].replace('response: ', '').replace('assistant\n', '').replace('assistant: ', '').replace('user: ', '').replace('solution: ', '').replace('Assistant: ', '').replace('answer\n', '').strip()
    

def show_img_len():
    data = load_file_2('/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MCTS/math_tree_bfs4_depth6_width4_actionPormpt3_llama_math500.json')
    number = []
    for i in range(0, len(data)):
        number.append(len(data[i]['tree_solution']))
    for i in range(1, 13):
        number.append(i)
    
    view_datalen(number, '/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MCTS/image/length-math_tree_bfs4_depth8_width3_SVAMP_yi.png', 'SVAMP')


def eval_gsm8k_verify():
    data = load_file_2('/mnt/petrelfs/zhaozhengyang/sunlinzhuang/MCTS/math_tree_bfs3_depth6_width4_gsm8k.json')
    base_gsm8k = load_file_2('/mnt/petrelfs/zhaozhengyang/sunlinzhuang/eval/MAmmoTH-main/math_eval/dataset/gsm8k/gsm8k.jsonl')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # print('** back verify **')
            input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            print('** input_prompt: ', input_prompt)
            output = inference(input_prompt)
            print('** output: ', output)
            if(output.lower().find('incorrect') == -1):
                print('** answer: ', answer)
                ans_list.append(answer)
            print('='*15)
            
            ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(base_gsm8k[i]['answer'].split('\n#### ')[-1])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Accuracy: ', correct / (correct + wrong))




def eval_math_svamp_backVerify():
    data = load_file_2('/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MCTS/yi/math_tree_bfs4_depth8_width3_simuleq_yi.json')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # print('** back verify **')
            input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            # print('** input_prompt: ', input_prompt)
            output = inference(input_prompt)
            # print('** output: ', output)
            if(output.lower().find('incorrect') == -1):
                # print('** answer: ', answer)
                ans_list.append(answer)
            # print('='*15)
            
            # ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Accuracy: ', correct / (correct + wrong))



def eval_math_svamp_backVerify_2(data):
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # print('** back verify **')
            input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            # print('** input_prompt: ', input_prompt)
            output = inference(input_prompt)
            # print('** output: ', output)
            if(output.lower().find('incorrect') == -1):
                # print('** answer: ', answer)
                ans_list.append(answer)
            # print('='*15)
            
            # ans_list.append(answer)
        if(data[i]['question'].find("If $a+b=7$ and $a^3+b^3=42$, what is the value of the sum") != -1):
            exit(0)
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Second Accuracy: ', correct / (correct + wrong))
    return correct, wrong



def eval_math_svamp_backVerify_1():
    data = load_file_2('/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MCTS/math_tree_bfs3_depth6_width4_actionPormpt3_llama_math500.json')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # # print('** back verify **')
            # input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            # # print('** input_prompt: ', input_prompt)
            # output = inference(input_prompt)
            # # print('** output: ', output)
            # if(output.lower().find('incorrect') == -1):
            #     # print('** answer: ', answer)
            #     ans_list.append(answer)
            # # print('='*15)
            
            ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    reuse_data = []
    id = -1
    for pred, gold in zip(pred_ans, gold_ans):
        id += 1
        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            reuse_data.append(data[id])
            print('** wrong')
            wrong += 1
            
        print('** ', correct, wrong)
    
    if(len(reuse_data) > 0):
        correct_2, wrong_2 = eval_math_svamp_backVerify_2(reuse_data)
        correct += correct_2
        wrong = wrong_2
    print('Accuracy: ', correct / (correct + wrong))



def eval_math_svamp():
    data = load_file_2('/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/math_mm/math_cot_Llama-3.2-11B-Vision-Instruct.json')
    print(data[0].keys())
    
    
    number = []
    # print(data[0]['tree_solution'])
    pred_ans = []
    gold_ans = []
    for i in range(0, len(data)):

        if(i % 50 ==0):
            print(f'BackVerify: {i}')

        ans_list = []
        for j in range(0, len(data[i]['tree_solution'])):
            pred = ' '.join(data[i]['tree_solution'][j][1:])
            answer = utils.answer_clean('math', ['The answer is:', 'The answer is', 'the answer is'], pred)
            

            # # print('** back verify **')
            # input_prompt = back_prompt.format(question=data[i]['question'], answer=answer)
            # # print('** input_prompt: ', input_prompt)
            # output = inference(input_prompt)
            # # print('** output: ', output)
            # if(output.lower().find('incorrect') == -1):
            #     # print('** answer: ', answer)
            #     ans_list.append(answer)
            # # print('='*15)
            
            ans_list.append(answer)
            
        
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {gold_ans[-1]}")


    correct = 0
    wrong = 0
    ii = -1
    for pred, gold in zip(pred_ans, gold_ans):
        ii += 1
        print('='*10)
        print('query: ', data[ii]['query'])
        print('answer: ', data[ii]['answer'])


        if isinstance(gold, (float, int)):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            print('** correct')
            correct += 1
        else:
            print('** wrong')
            wrong += 1
        print('** ', correct, wrong)
    print('Accuracy: ', correct / (correct + wrong))





if __name__ == "__main__":
    eval_math_svamp()
    # eval_math_svamp_backVerify_1()
    # show_img_len()
    # data = load_file_2('/mnt/petrelfs/zhaozhengyang/sunlinzhuang/MCTS/math_tree_depth8_width2.json')
    # print(len(data))