#!/bin/bash 


#SBATCH -J trans					  # name of job
#SBATCH -p gpu-ms					  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/student_models/
#SBATCH --gres=gpu:1 
#SBATCH  --constraint="gpuram24G|gpuram40G|gpuram48G|gpuram95G"
#SBATCH  --mem=25G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)

conda activate /lnet/work/people/jon/wmt25/env
export HOME=/lnet/work/people/jon/llm_agents
export OPENAI_API_KEY=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
nohup vllm serve --max-num-seqs 64 --max-model-len 1024 --port 5005 CohereForAI/aya-expanse-8b  > vllm.log &
sleep 5m; 
python3 eval_arabench.py     --dataset-dir AraBench_dataset     --output-dir arabench_translations_aya   --model CohereForAI/aya-expanse-8b  --base-url http://localhost:5005/v1 > ara.log 
