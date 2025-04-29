N=2
MODEL_NAME="Llama-3.1-8B-Instruct"
SAVE_DIR="/home/liyi/LongBench/LongBenchv1/results"
RUN_FILE="/home/liyi/LongBench/LongBenchv1/pred_v1.py"

export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=$N $RUN_FILE \
    --model $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --desc "cuboid-4-0.6-8" \
    --datasets qasper,multifieldqa_en,hotpotqa,2wikimqa,gov_report,musique,qmsum,multi_news,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,lcc,repobench-p,trec \