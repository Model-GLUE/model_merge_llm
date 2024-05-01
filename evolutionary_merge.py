import optuna
import subprocess
import os
import json
import yaml
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
import torch
import math
from merge_tools import build_model_config, run_merge
from eval_tools import evaluate_model
from mergekit.options import MergeOptions

env = os.environ.copy()
env['MKL_THREADING_LAYER'] = 'GNU'
env['MKL_SERVICE_FORCE_INTEL'] = '1'



def get_popsize():
    if 'dare' in args.merge_method or 'ties' in args.merge_method:
        num_param = len(args.model_list) * 2
    else:
        num_param = len(args.model_list)
    popsize = 4 + 3 * math.log(num_param)
    return round(popsize)

def execute_merge(merged_model_state_dict, model_state_dict_list, coe_list, range_list, n_layers=32):
    if merged_model_state_dict == None:
        merged_model_state_dict = {}

    with open(model_idx_path, 'r') as f:
        index = json.load(f)
    model_structure = index["weight_map"]
    for tensor_name in model_structure.keys():
        flag = 0
        for selected_layer in range_list:
            # print(selected_layer, range_list)
            if selected_layer == n_layers:
                match_str = 'lm_head.weight'
            elif selected_layer == n_layers + 1:
                match_str = 'model.embed_tokens.weight'
            elif selected_layer == n_layers + 2:
                match_str = 'model.norm.weight'
            else:
                match_str = f"layers.{selected_layer}."
            if match_str in tensor_name:
                flag = 1
                break
        if flag == 1 and 'rotary_emb.inv_freq' not in tensor_name:
            # do merge
            for model_state_dict, weight in zip(model_state_dict_list, coe_list):
                if match_str == 'lm_head.weight' or match_str == 'model.embed_tokens.weight':

                    if tensor_name in merged_model_state_dict:
                        merged_model_state_dict[tensor_name] += torch.tensor(model_state_dict[tensor_name][:32000, :], dtype=torch.bfloat16) * torch.tensor(weight, dtype=torch.bfloat16)
                    else:
                        merged_model_state_dict[tensor_name] = torch.tensor(model_state_dict[tensor_name][:32000, :], dtype=torch.bfloat16) * torch.tensor(weight,
                                                                                              dtype=torch.float)
                else:
                    if tensor_name in merged_model_state_dict:
                        merged_model_state_dict[tensor_name] += torch.tensor(model_state_dict[tensor_name], dtype=torch.bfloat16) * torch.tensor(weight, dtype=torch.bfloat16)
                    else:
                        merged_model_state_dict[tensor_name] = torch.tensor(model_state_dict[tensor_name], dtype=torch.bfloat16) * torch.tensor(weight,
                                                                                                                                                dtype=torch.float)

    return merged_model_state_dict

def normalize_list(coe_list):
    return [x / sum(coe_list) for x in coe_list]

def merge_models_group(coefficient_list, model_list, n_of_group, n_layers=32, batch_merging_group_size=4):
    if len(model_list) < batch_merging_group_size:
        batch_merging_group_size = len(model_list)
    coe_by_range_list = []
    for j in range(n_of_group):
        tmp_coe_by_range_list = []
        for k in range(len(model_list)):
            i = j * len(model_list) + k
            tmp_coe_by_range_list.append(coefficient_list[i])
        coe_by_range_list.append(normalize_list(tmp_coe_by_range_list))

    layer_idxs = list(range(n_layers))
    layer_idxs_by_group = [layer_idxs[i:i + n_layers // n_of_group] for i in range(0, n_layers, n_layers // n_of_group)]

    # lm_head and embed
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    coe_by_range_list.append(normalize_list(coefficient_list[-len(model_list):]))
    layer_idxs_by_group.extend([[n_layers], [n_layers + 1], [n_layers + 2]])

    coe_by_group_list = []
    for coe_by_range in coe_by_range_list:
        coe_group_tmp = [coe_by_range[i:i + batch_merging_group_size] for i in range(0, len(coe_by_range), batch_merging_group_size)]
        if coe_by_group_list == []:
            for i in range(0, len(coe_by_range), batch_merging_group_size):
                coe_by_group_list.append([])

        for i in range(len(coe_by_range) // batch_merging_group_size):
            coe_by_group_list[i].append(coe_group_tmp[i])

    merged_model_state_dict = None
    model_group_list = [model_list[i:i + batch_merging_group_size] for i in range(0, len(model_list), batch_merging_group_size)]


    for model_group, coe_group in zip(model_group_list, coe_by_group_list):
        model2del = []
        model_state_dict_list = []
        for model_name in model_group:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
            model_state_dict_list.append(model.state_dict())
            model2del.append(model)

        for ranges, coe_group_by_range in zip(layer_idxs_by_group, coe_group):
            # print(ranges, coe_group_by_range)
            merged_model_state_dict = execute_merge(merged_model_state_dict, model_state_dict_list, coe_group_by_range, ranges, n_layers)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    model2eval = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', torch_dtype=torch.bfloat16)
    model2eval.load_state_dict(merged_model_state_dict)

    return model2eval, tokenizer



def objective(trial):
    if args.num_group == 1:
        if 'dare' in args.merge_method or 'ties' in args.merge_method:
            n = len(args.model_list) * 2
        else:
            n = len(args.model_list)

        param_list = [
            trial.suggest_float(f"weight_{i}", args.range_min, args.range_max)
            for i in range(n)
        ]
        
        weight_list = param_list[:len(args.model_list)]
        density_list = param_list[len(args.model_list):]
        cfg = build_model_config(model_list=args.model_list, weight_list=weight_list, 
                                 density_list=density_list, merge_method=args.merge_method, 
                                 base_model='meta-llama/Llama-2-7b-hf' if 'dare' in args.merge_method or 'ties' in args.merge_method else None)
        model, tokenizer = run_merge(cfg, options=MergeOptions())
    else:
        n = len(args.model_list) * (args.num_group + 1)
        param_list = [
            trial.suggest_float(f"weight_{i}", args.range_min, args.range_max)
            for i in range(n)
        ]

        model, tokenizer = merge_models_group(param_list, args.model_list, args.num_group)
    model.to("cuda")
    acc = evaluate_model(model, tokenizer, bigcode_cfg_path='./files/generation_config.json', task_list=args.task_list)
        
    print('------------------------------')
    print('Param_list: ', param_list)
    print('Acc: ', acc)
    return -float(acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--GPU_idx", type=str, required=True)
    parser.add_argument("--merge_method", type=str, default='linear')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma0", type=float, default=1/6)
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--num_group", type=int, default=1, required=True)
    parser.add_argument("--range_max", type=float, default=1.0)
    parser.add_argument("--range_min", type=float, default=0.0)
    parser.add_argument("--output_path", type=str, default='./output.txt')
    
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.GPU_idx
    args.model_list = args.models.split(',')
    args.task_list = args.tasks.split(',')
    # popsize = 4+3*ln(num_param)
    popsize = get_popsize()
    sys.stdout = open(args.output_path, 'a')
    sampler = optuna.samplers.CmaEsSampler(popsize=popsize, sigma0=args.sigma0, seed=args.seed)
    study = optuna.create_study(sampler=sampler)

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    sys.stdout.close()
    sys.stdout = sys.__stdout__

