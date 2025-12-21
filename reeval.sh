for model in inceptionai/Jais-2-8B-Chat CohereLabs/command-a-translate-08-2025 inceptionai/Jais-2-70B-Chat meta-llama/Llama-3.3-70B-Instruct   CohereLabs/aya-expanse-32b   google/gemma-3-27b-it CohereLabs/c4ai-command-r7b-arabic-02-2025 CohereLabs/aya-expanse-32b google/gemma-3-27b-it  utter-project/EuroLLM-9B-Instruct google/gemma-3-4b-it  Qwen/Qwen3-4B-Instruct-2507 gpt-4.1-nano gpt-4.1-mini mistralai/Mistral-Small-3.2-24B-Instruct-2506  CohereLabs/c4ai-command-r-v01  CohereLabs/c4ai-command-r-08-2024 CohereLabs/aya-expanse-8b  
do
m=$(echo $model | sed 's/.*\///g')
python3 eval_arabench.py   --dataset-dir AraBench_dataset     --output-dir translations/arabench_translations_$m   --model $model --base-url http://localhost:5005/v1

python3 eval_arabench.py   --dataset-dir AraBench_dataset     --output-dir fixed_translations/arabench_translations_$m   --model $model --base-url http://localhost:5005/v1 

done


for model in inceptionai/Jais-2-8B-Chat CohereLabs/command-a-translate-08-2025  inceptionai/Jais-2-70B-Chat CohereLabs/aya-expanse-32b   google/gemma-3-27b-it CohereLabs/c4ai-command-r7b-arabic-02-2025 CohereLabs/aya-expanse-32b google/gemma-3-27b-it  utter-project/EuroLLM-9B-Instruct google/gemma-3-4b-it  Qwen/Qwen3-4B-Instruct-2507 gpt-4.1-nano gpt-4.1-mini mistralai/Mistral-Small-3.2-24B-Instruct-2506  CohereLabs/c4ai-command-r-v01  CohereLabs/c4ai-command-r-08-2024 CohereLabs/aya-expanse-8b
do
m=$(echo $model | sed 's/.*\///g')
python3 eval_arabench.py   --dataset-dir AraBench_dataset     --output-dir translations/arabench_aren_translations_$m --reverse   --model $model --base-url http://localhost:5005/v1

python3 eval_arabench.py   --dataset-dir AraBench_dataset     --output-dir fixed_translations/arabench_aren_translations_$m --reverse   --model $model --base-url http://localhost:5005/v1

done

