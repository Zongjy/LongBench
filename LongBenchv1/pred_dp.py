# pred.py
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

URLS = [
    "http://0.0.0.0:11451/v1",
    "http://0.0.0.0:11452/v1",
]
API_KEY = "EMPTY"
model2path = json.load(open('/home/liyi/LongBench/LongBenchv1/config/model2path.json', "r"))
model2maxlen = json.load(open('/home/liyi/LongBench/LongBenchv1/config/model2maxlen.json', "r"))
dataset2prompt = json.load(open("/home/liyi/LongBench/LongBenchv1/config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("/home/liyi/LongBench/LongBenchv1/config/dataset2maxlen.json", "r"))

def query_llm(prompt, model, tokenizer, client=None, temperature=0.8, max_new_tokens=128, stop=None):
    # truncate
    max_len = model2maxlen[model]
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    model = model2path[model]

    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                stop=stop
            )
            return completion.choices[0].message.content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

def get_pred(data, prompt_format, max_new_tokens, out_path, args):
    model = args.model
    rank = args.rank
    world_size = args.world_size
    if rank >= len(URLS):
        raise ValueError("Rank is larger than or equal to the number of URLs.")

    tokenizer = AutoTokenizer.from_pretrained(model2path[model], trust_remote_code=True)
    base_url = URLS[rank]
    client = OpenAI(
        base_url=base_url,
        api_key=API_KEY
    )
    data_subset = data[rank::world_size]
    print(f"Rank {rank} processing {len(data_subset)} samples on {base_url}...")
    rank_out_path = out_path.replace(".jsonl", f"_{rank}.jsonl")

    for json_obj in tqdm(data_subset, desc=f"Rank {rank} Progress", unit="sample", position=rank):
        prompt = prompt_format.format(**json_obj)
        output = query_llm(prompt, model, tokenizer, client, temperature=0.8, max_new_tokens=max_new_tokens)
        if output == '':
            continue
        with open(rank_out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": output,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                    "rank": rank,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.cuda.manual_seed_all(seed)

def main(args):
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
    else:
        datasets = args.datasets.split(",")


    pred_dir = os.path.join(args.save_dir, "pred")
    pred_e_dir = os.path.join(args.save_dir, "pred_e")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(pred_e_dir, exist_ok=True)

    if args.rank == 0:
        dataset_iterator = tqdm(datasets, desc="Processing Datasets", unit="dataset")
    else:
        # 其他 ranks 只正常遍历数据集列表，不显示外层 tqdm
        dataset_iterator = datasets
    for dataset in dataset_iterator:
        print(f"Rank {args.rank}: Loading dataset {dataset}...")
        if args.e:
            data = load_dataset("/home/liyi/LongBench/LongBenchv1/LongBench.py", f"{dataset}_e", split="test", trust_remote_code=True)
            out_dir = os.path.join(pred_e_dir, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{dataset}.jsonl")
        else:
            data = load_dataset("/home/liyi/LongBench/LongBenchv1/LongBench.py", dataset, split="test", trust_remote_code=True)
            out_dir = os.path.join(pred_dir, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{dataset}.jsonl")

        prompt_format = dataset2prompt[dataset]
        max_new_tokens = dataset2maxlen[dataset]

        data_all = list(data)

        get_pred(
            data_all,
            prompt_format,
            max_new_tokens,
            out_path=out_path,
            args=args
        )

    print("All datasets done.")

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--desc", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="all", help="Datasets to evaluate on")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")), help="Rank of the current process")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")), help="Total number of processes")
    args = parser.parse_args()
    print(args)

    main(args)
