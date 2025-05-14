N=4
MODEL_NAME="Llama-3-8B-Instruct-262k"
SAVE_DIR="/home/$USER/LongBench/LongBenchv1/results"
RUN_FILE="/home/$USER/LongBench/LongBenchv1/pred_v1.py"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$N $RUN_FILE \
    --save_dir $SAVE_DIR \
    --model $MODEL_NAME \
    --datasets all \
    --desc "triton" \