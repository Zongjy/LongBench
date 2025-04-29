N=8
MODEL_NAME="Llama-3-8B-Instruct-262k"
SAVE_DIR="/home/liyi/LongBench/LongBenchv1/results"
RUN_FILE="/home/liyi/LongBench/LongBenchv1/pred_v1.py"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$N $RUN_FILE \
    --save_dir $SAVE_DIR \
    --model $MODEL_NAME \
    --desc "cuboid-4-0.6-8" \
    --datasets qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,musique,qmsum,multi_news,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,lcc,repobench-p,trec