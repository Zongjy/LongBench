# pred.py
import os
from datasets import load_dataset
import torch
import json
import time
from openai import OpenAI
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse

URL = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

def query_llm(prompt, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    max_len = maxlen_map[model]
    if model in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0
    if model in model_map:
        model = model_map[model]
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
                presence_penalty=0.5,
                frequency_penalty=0.5,
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
    tokenizer = AutoTokenizer.from_pretrained(model_map[model], trust_remote_code=True)
    client = OpenAI(
        base_url=URL,
        api_key=API_KEY
    )

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        output = query_llm(prompt, model, tokenizer, client, temperature=0.1, max_new_tokens=max_new_tokens)
        if output == '':
            continue
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": output,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    model_name = args.model

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
        datasets = [
            # "narrativeqa",
            # "qasper",
            "multifieldqa_en",
            # "multifieldqa_zh",
            "hotpotqa",
            # "2wikimqa",
            # "musique",
            # "dureader",
            # "gov_report",
            "qmsum",
            # "multi_news",
            # "vcsum",
            # "trec",
            "triviaqa",
            # "samsum",
            # "lsht",
            # "passage_count",
            "passage_retrieval_en",
            # "passage_retrieval_zh",
            # "lcc",
            "repobench-p",
        ]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    pred_dir = os.path.join(args.save_dir, "pred")
    pred_e_dir = os.path.join(args.save_dir, "pred_e")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(pred_e_dir, exist_ok=True)

    for dataset in datasets:
        if args.e:
            data = load_dataset("~/exp/LongBench/LongBench/LongBench.py", f"{dataset}_e", split="test", trust_remote_code=True)
            if args.sp:
                out_dir = os.path.join(pred_e_dir, model_name+"-sparse")
            else:
                out_dir = os.path.join(pred_e_dir, model_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{dataset}.jsonl")
        else:
            data = load_dataset("~/exp/LongBench/LongBench/LongBench.py", dataset, split="test", trust_remote_code=True)
            if args.sp:
                out_dir = os.path.join(pred_dir, model_name + "-sparse")
            else:
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
    parser.add_argument("--sp", action="store_true" ,help="Sparse")
    args = parser.parse_args()
    main()