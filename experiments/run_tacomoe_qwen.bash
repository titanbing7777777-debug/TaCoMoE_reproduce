lora_rank=16
#lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
lora_trainable='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=16870
SAVE_STEPS=1687
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
USER_DIR="/share/home/tm891982051140000/a945582010/"
model_name_or_path="resources/qwen2-7b"
your_data_path="data/en/task"
TIME=$(date +%Y%m%d_%H%M%S)
your_checkpopint_path="$USER_DIR/saved/saved_$TIME/qwen2-7b/en/task"
MAX_SOURCE_LENGTH=2048

export DS_SKIP_CUDA_CHECK=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# Prefer the conda env C++ runtime to satisfy CXXABI_1.3.15 for _sqlite3/ICU
if [ -z "$CONDA_PREFIX" ]; then
    export CONDA_PREFIX="/root/miniconda3/envs/MOE"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# # Training Command

deepspeed --num_gpus=2 --master_port $MASTER_PORT run_tacomoe.py \
    --deepspeed src/ds.config \
    --do_train \
    --train_file $your_data_path/train.json \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 100 \
    --save_steps ${SAVE_STEPS} \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --bf16 \
    --lora_name moelora \
    --expert_num 8

# deepspeed --num_gpus=1 --master_port $MASTER_PORT run_tacomoe.py \
#     --do_predict \
#     --test_file $your_data_path/test.json \
#     --cache_dir $your_data_path \
#     --overwrite_cache \
#     --prompt_column input \
#     --response_column target \
#     --model_name_or_path $model_name_or_path \
#     --peft_path $your_checkpopint_path/checkpoint-45000 \
#     --output_dir results/pred/moelora/qwen_en/45000 \
#     --overwrite_output_dir \
#     --max_source_length $MAX_SOURCE_LENGTH \
#     --max_target_length 512 \
#     --per_device_eval_batch_size 4 \
#     --predict_with_generate \
#     --lora_name moelora \
#     --expert_num 8
