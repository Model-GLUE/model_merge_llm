# Model-GLUE : Democratized LLM Scaling for A Large Model Zoo in the Wild

This is the official code for the merging benchmark in the ``Model-GLUE : Democratized LLM Scaling for A Large Model Zoo in the Wild``.

## Quick start

### Requirements
* Python 3.10.13
* Other dependencies: 
   ```shell
   optuna=3.6.1
   cmaes=0.10.0
   accelerate=0.27.2
   transformers=4.36.2
   lm_eval=0.4.0
   torch=2.1.2
   mergekit=0.0.4
   ```
* lm-evaluation-harness

   ```shell
   git clone https://github.com/s1ghhh/lm-evaluation-harness.git
   cd lm-evaluation-harness
   pip install -e .
   
   ```
   

### Running the code

1. **Clustering**
   ```shell
      bash ./scripts/run_clustering.sh
   ```
2. **Heuristic (Average)**
   ```shell
      bash ./scripts/run_heuristic_average.sh
   ```
3. **Heuristic (Coefficient)**
   ```shell
      bash ./scripts/run_heuristic_coefficient.sh
   ```
   
4. **Heuristic (Similarity)**
   ```shell
      bash ./scripts/run_heuristic_similarity_descending.sh
      bash ./scripts/run_heuristic_similarity_ascending.sh
   ```

5. **Evolutionary Strategy**
   ```shell
      bash ./scripts/run_evolutionary.sh
   ```

## Acknowledgement

Part of the codes are adapted from [mergekit](https://github.com/arcee-ai/mergekit), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness).