lora_rank=16
#lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
lora_trainable='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj'
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
MAX_STEPS=16870
SAVE_STEPS=1687
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
USER_DIR="/share/home/tm891982051140000/a945582010"
model_name_or_path="resources/qwen2-7b"
your_data_path="data/en/task"
TIME=$(date +%Y%m%d_%H%M%S)
your_checkpopint_path="$USER_DIR/saved/saved_20260120_101654/qwen2-7b/en/task"
MAX_SOURCE_LENGTH=2048
# 默认 checkpoint，可用 -c 覆盖；未来可继续拓展其它选项
checkpoint_step=844
while getopts "c:" opt; do
    case $opt in
        c) checkpoint_step="$OPTARG" ;;
        *) echo "用法: $0 [-c checkpoint_step]"; exit 1 ;;
    esac
done
shift $((OPTIND-1))

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

deepspeed --num_gpus=2 --master_port $MASTER_PORT run_tacomoe.py \
    --do_predict \
    --test_file $your_data_path/test.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $your_checkpopint_path/checkpoint-$checkpoint_step \
    --output_dir results/pred/moelora/qwen_en/$checkpoint_step \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 512 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --lora_name moelora \
    --expert_num 8
