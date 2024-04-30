import optuna
import subprocess
import os
import json
env = os.environ.copy()
env['MKL_THREADING_LAYER'] = 'GNU'
env['MKL_SERVICE_FORCE_INTEL'] = '1'

root_path='/workspace/0423_tinybenchmarks/SEARCH/evo/0428_new_zoo/which8/tinybench_ties'
GPU_idx='4'
path_of_bigcode='/workspace/0423_tinybenchmarks/bigcode-evaluation-harness'

def objective(trial):
    weights_list = [
        trial.suggest_float(f"weight_{i}", 0.0, 1.0)
        for i in range(16)
    ]
    try:
        shell_script_content = f"""#!/bin/bash
root_path={root_path}
models=("lmsys/vicuna-7b-v1.5" "meta-llama/Llama-2-7b-chat-hf" "teknium/OpenHermes-7B" "garage-bAInd/Platypus2-7B" "neuralmagic/Llama-2-7b-evolcodealpaca" "meta-math/MetaMath-7B-V1.0" "migtissera/Synthia-7B-v1.2" "PygmalionAI/pygmalion-2-7b")

weights=("{weights_list[0]}" "{weights_list[1]}" "{weights_list[2]}" "{weights_list[3]}" "{weights_list[4]}" "{weights_list[5]}" "{weights_list[6]}" "{weights_list[7]}")
densities=("{weights_list[8]}" "{weights_list[9]}" "{weights_list[10]}" "{weights_list[11]}" "{weights_list[12]}" "{weights_list[13]}" "{weights_list[14]}" "{weights_list[15]}")



echo -e "models:\\n  - model: meta-llama/Llama-2-7b-hf\\n" > tmpyml.yml
w_count=0
for model in "${{models[@]}}"; do
    echo -e "  - model: $model\\n    parameters:\\n      weight: ${{weights[$((w_count))]}}\\n      density: ${{densities[$((w_count))]}}\\n" >> tmpyml.yml
    ((w_count++))
done
    echo -e "merge_method: ties\\nbase_model: meta-llama/Llama-2-7b-hf\\ndtype: bfloat16" >> tmpyml.yml

# unset models[${{#models[@]}}-1]
CUDA_VISIBLE_DEVICES={GPU_idx} mergekit-yaml tmpyml.yml ./tmp_model --cuda --random-seed 42
# cp ./config.json ./tmp_model

result_float=0
tasks=("mmlu_tiny" "gsm8k" "arc_challenge" "winogrande")
key1=("mmlu_tiny" "gsm8k" "arc_challenge" "winogrande")
key2=("acc,none" "exact_match,get-answer" "acc_norm,none" "acc,none")
k=0
for task in "${{tasks[@]}}"; do
    CUDA_VISIBLE_DEVICES={GPU_idx} lm_eval --model hf \
    --model_args pretrained=./tmp_model,trust_remote_code=True,dtype="bfloat16" \
    --tasks $task \
    --batch_size 1 \
    --output_path ./${{task}}_tmp.json
    result=$(jq -r ".results.${{key1[$k]}}.\\"${{key2[$k]}}\\"" ./${{task}}_tmp.json)
    tmp_result=$(echo "$result" | bc)
    result_float=$(echo "$result_float + $tmp_result" | bc)
    rm ./${{task}}_tmp.json
    ((k++))
done


# cd {path_of_bigcode}
export CUDA_VISIBLE_DEVICES={GPU_idx}
python {path_of_bigcode}/main.py \
    --model $root_path/tmp_model \
    --tasks mbpp \
    --max_length_generation 1024 \
    --do_sample True \
    --n_samples 1 \
    --top_p 0.95 \
    --batch_size 1 \
    --temperature 0.2 \
    --precision bf16 \
    --trust_remote_code \
    --allow_code_execution \
    --use_auth_token \
    --metric_output_path $root_path/mbpp_tmp.json


# cd $root_path
result=$(jq -r '.mbpp."pass@1"' ./mbpp_tmp.json)
tmp_result=$(echo "$result" | bc)
result_float=$(echo "$result_float + $tmp_result" | bc)
rm ./mbpp_tmp.json


echo $result_float
echo -e $result_float > ./tmp_results.txt

"""
        script_filename = 'tmp_script.sh'
        with open(script_filename, 'w') as script_file:
            script_file.write(shell_script_content)

        os.chmod(script_filename, 0o755)

        result = subprocess.run(['./tmp_script.sh'], capture_output=True, text=True, shell=True, env=env)

        if result.returncode == 0:
            with open('./tmp_results.txt', 'r') as f:
                acc = float(f.readline())

            print('------------------------------')
            print('weights_list: ', weights_list)
            print('acc: ', acc)
            os.remove('./tmp_results.txt')
            return -float(acc)
        else:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            print("Return Code:", result.returncode)
            return 0.0
    except:
        return 0.0

sampler = optuna.samplers.CmaEsSampler(popsize=12, sigma0=1/6, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=500, show_progress_bar=True)
