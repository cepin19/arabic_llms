#!/bin/bash 


#SBATCH -J trans					  # name of job
#SBATCH -p gpu-troja					  # name of partition or queue (if not specified default partition is used
#SBATCH -D /lnet/work/people/jon/student_models/
#SBATCH --gres=gpu:2 
#SBATCH  --constraint="gpuram48G"
#SBATCH  --mem=35G
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

eval $(conda shell.bash hook)

conda activate /lnet/work/people/jon/wmt25/env
export HOME=/lnet/work/people/jon/llm_agents
export OPENAI_API_KEY=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
for model in tiiuae/Falcon-H1-34B-Instruct
do
	m=$(echo $model | sed 's/.*\///g')

	nohup vllm serve --max-num-seqs 8 --max-model-len 1024 --port 5005 --tensor_parallel_size=2 $model  > vllm_$m.log 2>&1 &
sleep 35m; 
m=$(echo $model | sed 's/.*\///g')
python3 eval_arabench.py   --dataset-dir AraBench_dataset     --output-dir arabench_translations_$m   --model $model --base-url http://localhost:5005/v1 > $m.log 
pkill -f vllm
done
