### For open-eneded questions, the dataset should be one of 
### ['nu m g le', 'numglue', 'math', 'numglue', 'deepmind', 'simuleq'] 
### We first try PoT and if the generated program is not executable, we shift to CoT

# python run_choice.py \
#   --model "/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/MAmmoTH2-8B" \
#   --shots 0 \
#   --stem_flan_type "pot_prompt" \
#   --dataset "aqua" \
#   --model_max_length 1500 \
#   --print \
#   --cot_backup \

export NCCL NVLS_ENABLE=0

python run_open.py \
  --model '/cpfs/29f69eb5e2e60f26/user/sft_intern/slz/Meta-Llama-3-8B-Instruct' \
  --shots 0 \
  --stem_flan_type "cot" \
  --dataset "gsm8k" \
  --model_max_length 2048 \
  --print \

