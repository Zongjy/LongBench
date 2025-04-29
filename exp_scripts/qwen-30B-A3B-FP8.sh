N=8
MODEL_NAME="Qwen3-30B-A3B-FP8"
SAVE_DIR="/home/liyi/LongBench/LongBenchv1/results"
RUN_FILE="/home/liyi/LongBench/LongBenchv1/pred_v1.py"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$N $RUN_FILE \
    --model $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --desc "40k" \
    --datasets qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,musique,qmsum,multi_news,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,lcc,repobench-p,trec \