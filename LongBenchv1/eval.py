# eval.py
import os
import json
import argparse
import numpy as np
import glob

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="/home/liyi/LongBench/LongBenchv1/results")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--desc', type=str, default=None)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            if prediction is not None:
                prediction = prediction.lstrip('\n').split('\n')[0]
            else:
                prediction = ""
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    model_name = args.model + ("_" + args.desc if args.desc else "")
    if args.e:
        base_pred_path = os.path.join(args.save_dir, "pred_e", model_name)
    else:
        base_pred_path = os.path.join(args.save_dir, "pred", model_name)
    if not os.path.exists(base_pred_path):
        print(f"Error: Prediction directory not found at {base_pred_path}")
        exit()

    all_files = glob.glob(os.path.join(base_pred_path, "*.jsonl"))
    dataset_data = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)

        # Determine the base dataset name (handling _rank.jsonl files)
        base_name_without_ext = filename[:-len(".jsonl")] # Remove .jsonl
        parts = base_name_without_ext.rsplit('_', 1)
        # Check if the part after the last '_' is purely digits
        if len(parts) == 2 and parts[1].isdigit():
            dataset_name = parts[0]
        else:
            dataset_name = base_name_without_ext

        if dataset_name not in dataset_data:
            dataset_data[dataset_name] = {'predictions': [], 'answers': [], 'lengths': [], 'all_classes': None}
    
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {line}, file_path: {file_path}, expection from: {e}")
                    exit(1)
                dataset_data[dataset_name]['predictions'].append(data.get("pred", ""))
                dataset_data[dataset_name]['answers'].append(data.get("answers", []))
                if args.e and "length" in data:
                    dataset_data[dataset_name]['lengths'].append(data["length"])
                if dataset_data[dataset_name]['all_classes'] is None and "all_classes" in data:
                        dataset_data[dataset_name]['all_classes'] = data["all_classes"]

    print("Calculating scores for datasets:")
    # Now iterate through the collected data per dataset and calculate scores
    for dataset_name, data_lists in dataset_data.items():
        predictions = data_lists['predictions']
        answers = data_lists['answers']
        lengths = data_lists['lengths']
        all_classes = data_lists['all_classes'] # Use the collected all_classes

        if not predictions: # Skip if no data was collected for this dataset
            print(f"  No valid data found for dataset {dataset_name}. Skipping.")
            continue 

        print(f"  Calculating score for {dataset_name} with {len(predictions)} samples.")

        if args.e:
            score = scorer_e(dataset_name, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset_name, predictions, answers, all_classes)
        scores[dataset_name] = score
    if args.e:
        out_path = os.path.join(args.save_dir, "pred_e", model_name, "result.json")
    else:
        out_path = os.path.join(args.save_dir, "pred", model_name, "result.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)