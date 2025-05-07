N=8
MODEL_NAME="Qwen3-30B-A3B-FP8"
SAVE_DIR="/home/$USER/LongBench/LongBenchv1/results"
RUN_FILE="/home/$USER/LongBench/LongBenchv1/pred_v1.py"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$N $RUN_FILE \
    --model $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --datasets all \
    --desc 32k \