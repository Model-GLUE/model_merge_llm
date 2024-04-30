## Environment
1. **lm-evaluation-harness**

   Install a new version of lm_eval

   ```shell
   git clone https://github.com/s1ghhh/lm-evaluation-harness.git
   cd lm-evaluation-harness
   git checkout offset_by_id
   
   git log
   #commit cc499b52c265853fe92305af1897eb137f3036e3 (HEAD -> offset_by_id, #origin/offset_by_id)
    #Author: s1ghhh <sghinscu@163.com>
    #Date:   Sun Apr 21 02:31:59 2024 +0000
   
   ```
   
   Ensure the version of the following Python libraries
   
   ```shell
   optuna=3.6.1
   accelerate=0.27.2
   transformers=4.36.2
   lm_eval=0.4.0
   torch=2.1.2
   ```
   
   Then run the following scripts to check if the result is `0.7372`. If the result is `0.7372`, then the version is correct; otherwise, please refer to `./requirements_lm_eval.txt`

   ```shell
    CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
     --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
     --tasks winogrande \
     --num_fewshot 5 \
     --batch_size 1 
   
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m lm_eval --model hf \
     --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
     --tasks winogrande \
     --num_fewshot 5 \
     --batch_size 1 
   ```

   Now please add the dataset of tinybenchmarks to the lm_eval

   ```shell
      cp -r <path of this repo, not the lm_eval>/lm_eval/tasks/tinyMMLU <path of lm_eval>/lm_eval/tasks/
      cp <path of this repo, not the lm_eval>/lm_eval/tasks/arc/arc_easy.yaml <path of lm_eval>/lm_eval/tasks/arc/
      cp <path of this repo, not the lm_eval>/lm_eval/tasks/gsm8k/gsm8k.yaml <path of lm_eval>/lm_eval/tasks/gsm8k/
      cp <path of this repo, not the lm_eval>/lm_eval/tasks/winogrande/default.yaml <path of lm_eval>/lm_eval/tasks/winogrande/
   ```
   For tasks such as `mmlu_tiny, gsm8k, arc_challenge, winogrande` in the lm-eval, we will use the dataset of tinybenchmarks.


2. **bigcode-evaluation-harness**

   Install bigcode-evaluation-harness

   ```shell
      git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
      cd bigcode-evaluation-harness
      git checkout 00967d12093ef614de7bdad0772aed8e4118f1fd
   ```
   
   Now please replace the `mbpp.py` to use a smaller validation set for evaluation
   
   ```shell
      cp <path of this repo, not the bigcode>/bigcode_eval/tasks/mbpp.py <path of bigcode>/bigcode_eval/tasks/
   ```

## Run and eval

1. **Running evolutionary strategies**

   Currently, only single cards are supported, and disk reads are frequent. The code is not efficient and will be improved in the future.

   For scripts `ties_which8.py`, `ties_which4.py`, `linear_which8.py`, `linear_which4.py`, the following parameters need to be adjusted.
   ```shell
      # The root directory for executing this script, and each .py script should have a separate root directory
      root_path='/workspace/0423_tinybenchmarks/SEARCH/evo/0428_new_zoo/which8/tinybench_linear'
      # Which GPU is the script running on
      GPU_idx='5'
      # the path of bigcode-evaluation-harness
      path_of_bigcode='/workspace/0423_tinybenchmarks/bigcode-evaluation-harness'
   ```
   
   Then please run the script and collect the results of each iteration.

   ```shell
      nohup python -u ties_which8.py >> ties_which8.out &
   ```
   
   The results of each round will be recorded in `ties_which8.out`

2. **Running evolutionary strategies**

   After obtaining the best combination, please evaluate it using normal lm-eval (w.o tinybenchmarks) and bigcode (use testset of mbpp instead of valset).