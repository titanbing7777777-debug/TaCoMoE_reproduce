lora_rank=16
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
#lora_trainable='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=17220
SAVE_STEPS=1722
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
USER_DIR="/share/home/tm891982051140000/a945582010/"
model_name_or_path="$USER_DIR/chatglm3-6b"   
your_data_path="data/en/task"
your_checkpopint_path="$USER_DIR/saved/saved_20260110_154653/chatglm3/en/task"
MAX_SOURCE_LENGTH=2048

# 允许通过第一个位置参数传入 checkpoint 步数，默认 5166
CKPT_STEP="${1:-5166}"
OUTPUT_DIR_BASE="results/pred/moelora/chatglm3-6b_en"

DS_SKIP_CUDA_CHECK=1

export DEEPSPEED_LOG_LEVEL=debug
export NCCL_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
ulimit -c unlimited

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

deepspeed --num_gpus=2 --master_port $MASTER_PORT run_tacomoe.py \
    --do_predict \
    --test_file $your_data_path/test.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path "$your_checkpopint_path/checkpoint-$CKPT_STEP" \
    --output_dir "$OUTPUT_DIR_BASE/$CKPT_STEP" \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 8
