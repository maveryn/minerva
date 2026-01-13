python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/minerva/minerva_base_grpo_llama8b/best/actor/ \
    --target_dir checkpoints/minerva/minerva_base_grpo_llama8b/best/hf \
    --hf_upload_path athena-security/minerva_grpo_llama8b_500 \
    --private




# LIS with RLVR + answer
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/lis_answer_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/lis_answer_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_answer_qwen7b \
    --private

# Activity with RLVR + answer
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/activity_answer_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/activity_answer_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/activity_answer_qwen7b \
    --private

# LIS with RLVR + id
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/lis_id_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/lis_id_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_id_qwen7b \
    --private


# LIS with RLVR + id exact
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/lis_id_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/lis_id_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_id_qwen7b \
    --private


# Activity with RLVR + id
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/activity_id_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/activity_id_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/activity_id_qwen7b \
    --private


# Activity with RLVR + id exact
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/activity_id_exact_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/activity_id_exact_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/activity_id_exact \
    --private


# LIS with reasoning prompt 
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/lis_answer_reason/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/lis_answer_reason/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_answer_reason \
    --private


# LLAMA Activity with RLVR + answer
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/activity_answer_llama_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/activity_answer_llama_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/activity_answer_llama \
    --private

# LLAMA LIS with RLVR + answer
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/lis_answer_llama_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/lis_answer_llama_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_answer_llama \
    --private

# Qwen r_all
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/rlvr/activity_rall_qwen2_7b/global_step_120/actor/ \
    --target_dir checkpoints/rlvr/activity_rall_qwen2_7b/global_step_120/hf \
    --hf_upload_path Tanvirul/lis_rall_qwen2_7b \
    --private
