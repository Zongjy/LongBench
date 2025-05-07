# pred_v1.py 
import os
from datasets import load_dataset
import json
import time
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import multiprocessing

USER = os.getenv("USER")
URL = "http://0.0.0.0:11451/v1"
API_KEY = "None"
model2path = json.load(open(f'/home/{USER}/LongBench/LongBenchv1/config/model2path.json', "r"))
model2maxlen = json.load(open(f'/home/{USER}/LongBench/LongBenchv1/config/model2maxlen.json', "r"))
dataset2prompt = json.load(open(f"/home/{USER}/LongBench/LongBenchv1/config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open(f"/home/{USER}/LongBench/LongBenchv1/config/dataset2maxlen.json", "r"))

def query_llm(prompt, model, tokenizer, client=None, temperature=0.6, max_new_tokens=128, stop=None):
    # truncate
    max_len = model2maxlen.get(model, 2048)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

    model_path = model2path.get(model)
    if not model_path:
        print(f"Model {model} not found in model2path.")
        return ''

    tries = 0
    while tries < 5:
        tries += 1
        try:
            # print(prompt)
            # exit(1)
            completion = client.chat.completions.create(
                model=model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=stop
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"PID {os.getpid()} Error Occurs: \"{str(e)}\"        Retry ...")
            time.sleep(1)
    else:
        print(f"PID {os.getpid()} Max tries. Failed.")
        return ''

def get_pred(data, prompt_format, max_new_tokens, out_path, args, lock, rank, world_size):
    model = args.model

    tokenizer = AutoTokenizer.from_pretrained(model2path[model], trust_remote_code=True)

    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )

    data_subset = data[rank::world_size]
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) processing {len(data_subset)} samples on {URL} and writing to {out_path}...")

    tqdm_position = rank % 8
    for json_obj in tqdm(data_subset, desc=f"Rank {rank} Progress", unit="sample", position=tqdm_position):
        try:
            prompt = prompt_format.format(**json_obj)
        except KeyError as e:
             print(f"Rank {rank} (PID: {os.getpid()}) Skipped sample due to missing key in json_obj for prompt format: {e}")
             continue
        # with open("prompt.txt", "a", encoding="utf-8") as f:
        #     f.write(str(len(tokenizer.encode(prompt, add_special_tokens=False))) + "\n")
        #     f.write(prompt + "\n")

        output = query_llm(prompt, model, tokenizer, client, temperature=0.8, max_new_tokens=max_new_tokens)
        if output == '':
            print(f"Rank {rank} (PID: {os.getpid()}) query_llm returned empty string for a sample.")

        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump(
                    {
                        "pred": output,
                        "answers": json_obj.get("answers", "N/A"),
                        "all_classes": json_obj.get("all_classes", "N/A"),
                        "length": json_obj.get("length", "N/A"),
                        "rank": rank,
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

def seed_everything(seed):
    """Seeds the random number generators for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    # torch seeding commented out as per original script's comment
    # import torch
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True


def main(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid RANK ({rank}) or WORLD_SIZE ({world_size}). Ensure script is launched correctly with a distributed launcher.")

    print(f"Process Rank {rank}/{world_size} (PID: {os.getpid()}): Starting with arguments: {args}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_name = args.model + ("_" + args.desc if args.desc else "")

    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
        pred_dir = os.path.join(args.save_dir, "pred_e") # LongBench-E results directory
    else:
        if args.datasets == "all":
             datasets = list(dataset2prompt.keys())
             # Filter out _e datasets if not evaluating LongBench-E
             datasets = [d for d in datasets if not d.endswith('_e')]
        else:
            datasets = args.datasets.split(",")
        pred_dir = os.path.join(args.save_dir, "pred")

    os.makedirs(pred_dir, exist_ok=True) 

    lock = multiprocessing.Lock()

    if rank == 0:
        dataset_iterator = tqdm(datasets, desc="Processing Datasets", unit="dataset")
    else:
        dataset_iterator = datasets

    for dataset in dataset_iterator:
        print(f"Rank {rank} (PID: {os.getpid()}): Loading dataset {dataset}...")
        # Construct the single output file path for this dataset
        out_dir = os.path.join(pred_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dataset}.jsonl") 

        prompt_format = dataset2prompt.get(dataset)
        max_new_tokens = dataset2maxlen.get(dataset)

        try:
            data = load_dataset(
                f"/home/{USER}/LongBench/data/LongBench.py",
                f"{dataset}_e" if args.e else dataset,
                split="test",
                trust_remote_code=True
            )
            data_all = list(data)
        except Exception as e:
            print(f"Rank {rank} (PID: {os.getpid()}): Failed to load dataset {dataset}: {e}. Skipping.")
            continue


        get_pred(
            data_all,
            prompt_format,
            max_new_tokens,
            out_path=out_path,
            args=args,
            lock=lock,
            rank=rank,
            world_size=world_size
        )
        print(f"Rank {rank} (PID: {os.getpid()}): Finished dataset {dataset}.")


    print(f"Rank {rank} (PID: {os.getpid()}): All assigned datasets done.")

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results", help="Save directory")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., internlm2-7b-sft)")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--desc", type=str, default=None, help="Optional description for model name directory")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated list of datasets to evaluate on, or 'all'")

    args = parser.parse_args()

    main(args)