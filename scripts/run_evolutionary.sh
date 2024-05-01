#!/usr/bin/bash

current_dir=$(pwd)
python -u ./evolutionary_merge.py \
  --GPU_idx "0" \
  --tasks "mmlu_tiny,winogrande,arc_challenge,gsm8k,humanevalsynthesize-js,mbpp" \
  --models "migtissera/Synthia-7B-v1.2,teknium/OpenHermes-7B,meta-llama/Llama-2-7b-chat-hf,lmsys/vicuna-7b-v1.5" \
  --merge_method "linear" \
  --seed 42 \
  --n_trials 200 \
  --num_group 1 \
  --output_path "./output.txt" 


