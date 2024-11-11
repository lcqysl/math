
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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


back_prompt =  """
Please act as a professional math teacher.
Your goal is to verify if the answer to a math word problem is correct by checking if the answer satisfies the conditions of the original question.
To achieve the goal, you have two jobs.
# Substituting the given answer back into the problem to check if it aligns with the problem's conditions.
# Determine whether the answer is correct or incorrect based on this verification.

You have two principles to do this.
# Ensure the solution is step-by-step and logically checks the validity of the answer.
# Clearly state whether the answer is correct or incorrect after verification.

Given Question: {question}
Given Answer: {answer}
Your output should be in the following format:
SOLUTION: <your detailed verification by substituting the answer back into the question>
FINAL JUDGEMENT: The answer is <correct/incorrect> based on the verification
"""

def load_file_2(load_path):
    with open(load_path, 'r', encoding='utf-8') as f1:
        con = []
        for line in f1:
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

    plt.figure(figsize=(8, 6))
    bars=plt.bar(numbers, counts, color='#AFEEEE', edgecolor='blue')

    # plt.title('Frequency of Numbers', fontsize=16, fontweight='bold')
    plt.xlabel(name+' Length', fontsize=18)
    plt.ylabel('Data Number', fontsize=18)
    # plt.xticks(numbers)  # 设置x轴刻度

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)

def show_img_len():
    data = load_file_2('/mnt/petrelfs/zhaozhengyang/sunlinzhuang/MCTS/math_tree_bfs3_depth6_width4.json')
    number = []
    for i in range(0, len(data)):
        number.append(len(data[i]['tree_solution']))
    view_datalen(number, '/mnt/petrelfs/zhaozhengyang/sunlinzhuang/MCTS/length-math_tree_bfs3_depth6_width4.png', 'test')


def eval():
    data = load_file_2('/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MCTS/math_tree_bfs4_depth8_width3_gsm8k_llama.json')
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
            ans_list.append(answer)
            print(answer)
        counter = Counter(ans_list)
        if(len(ans_list) != 0):
            most_common = counter.most_common(1)[0]
        else:
            most_common = ['none']
        
        
        data[i]['tree_pred_answer'] = most_common
        
        pred_ans.append(most_common[0])
        gold_ans.append(data[i]['answer'])
        print(f"** most_common: {most_common}, gold_ans: {data[i]['answer']}")


    correct = 0
    wrong = 0
    for pred, gold in zip(pred_ans, gold_ans):
        if isinstance(gold, float):
            gold = [str(gold), gold]
        if isinstance(gold, str):
            gold = [gold]
        # print('** gold: ', gold)
        if utils.compare_answer_with_groundtruth(str(pred), *gold):
            correct += 1
        else:
            wrong += 1
        print('** ', correct, wrong)
    print('Accuracy: ', correct / (correct + wrong))

        



if __name__ == "__main__":
    eval()
    # show_img_len()
    # data = load_file_2('/mnt/petrelfs/zhaozhengyang/sunlinzhuang/MCTS/math_tree_depth8_width2.json')
    # print(len(data))