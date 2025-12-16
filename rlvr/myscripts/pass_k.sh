# Base
python pass_k.py --task activity --model Qwen/Qwen2.5-7B-Instruct --k 256
python pass_k.py --task lis --model Qwen/Qwen2.5-7B-Instruct --k 256

# RLVR with answer
python pass_k.py --task activity --model Tanvirul/activity_answer_qwen7b --k 256
python pass_k.py --task lis --model Tanvirul/lis_answer_qwen7b --k 256

# RLVR with id (prefix)
python pass_k.py --task activity --model Tanvirul/activity_id_qwen7b --k 256
python pass_k.py --task lis --model Tanvirul/lis_id_qwen7b --k 256

# RLVR with id (exact)
python pass_k.py --task activity --model Tanvirul/activity_id_exact --k 256
python pass_k.py --task lis --model Tanvirul/lis_id_exact_qwen7b --k 256

# run prefix id-reward model from one task on another
python pass_k.py --task activity --model Tanvirul/lis_id_qwen7b --k 256
python pass_k.py --task lis --model Tanvirul/activity_id_qwen7b --k 256

# run answer-reward model from one task on another
python pass_k.py --task activity --model Tanvirul/lis_answer_qwen7b --k 256
python pass_k.py --task lis --model Tanvirul/activity_answer_qwen7b --k 256

# run exact id-reward model from one task on another
python pass_k.py --task activity --model Tanvirul/lis_id_exact_qwen7b --k 256
python pass_k.py --task lis --model Tanvirul/activity_id_exact --k 256

# reasoning prompt with LIS
python pass_k.py --task lis --model Tanvirul/lis_answer_reason --k 256

# RLVR with LLAMA r_ans
python pass_k.py --task activity --model Tanvirul/activity_answer_llama --k 256
python pass_k.py --task lis --model Tanvirul/lis_answer_llama --k 256

# Base LLAMA
python pass_k.py --task activity --model meta-llama/Llama-3.1-8B-Instruct --k 256
python pass_k.py --task lis --model meta-llama/Llama-3.1-8B-Instruct --k 256