
import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from heuristic_merge import get_sim



def clustering_by_architecture(wild_models):
    model_families = []
    model_familie_arch = []
    for model_path in wild_models:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = config.architectures
        find_the_fml = False
        for i, a in enumerate(model_familie_arch):
            if a == arch:
                model_families[i].append(model_path)
                find_the_fml = True
                break
        if not find_the_fml:
            model_families.append([model_path])
            model_familie_arch.append(arch)
            
    return model_families

def clustering_by_similarity(wild_models, sim_of_delta_param=False, threshold=0.95):
    model_families = []
    for model_path in wild_models:
        if model_families == []:
            model_families.append([model_path])
        else:
            find_the_fml = False
            for i, fml in enumerate(model_families):
                representative_model_path = fml[0]
                sim = get_sim(representative_model_path, model_path, sim_of_delta_param, True)
                if sim['sim_all'] > threshold:
                    model_families[i].append(model_path)
                    find_the_fml = True
                    break
            if not find_the_fml:
                model_families.append([model_path])

    return model_families


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    model_list = args.models.split(',')
    
    print("Clustering based on architecture...")
    families_by_architecture = clustering_by_architecture(model_list)
    print("Done...")
    print(families_by_architecture)
    print("Clustering based on similarity...")
    mergeable_families = []
    for fml_by_arch in families_by_architecture:
        families_by_similarity = clustering_by_similarity(fml_by_arch, False, args.threshold)
        mergeable_families.extend(families_by_similarity)
    print("Done...")
    
    for i, m_f in enumerate(mergeable_families):
        print(f"Mergeable Family {i+1}: {m_f}")