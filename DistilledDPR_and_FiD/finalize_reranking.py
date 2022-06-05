import os
import json
import argparse
from glob import glob
from tqdm import tqdm
import math
from functools import partial

from multiprocessing import Pool

def process(lines, n_ctxs):
    result = []
    for line in tqdm(lines, desc='processing', total=len(lines)):
        line = json.loads(line)
        ori_ranks = set()
        for _, val in line['clusters'].items():
            for item in val[:n_ctxs]:
                ori_ranks.add(item[0])
        ori_rank2new_rank = {}
        ctxs = []
        for ori_rank in ori_ranks:
            ori_rank2new_rank[ori_rank] = len(ctxs)
            ctxs.append(line['ctxs'][ori_rank])
        line['ctxs'] = ctxs
        for pred, val in line['clusters'].items():
            line['clusters'][pred] = [(ori_rank2new_rank[item[0]], item[1], item[2]) for item in val[:n_ctxs]]
        result.append(json.dumps(line))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str)
    parser.add_argument("--suffix", type=str, default=".raw")
    parser.add_argument("--n_ctxs", type=int, default=10)
    args = parser.parse_args()

    for split in ['train', 'dev', 'test']:
        raw_fnames = []
        for fname in glob(os.path.join(args.dirname, "{}*jsonl*".format(split))):
            if not fname.endswith(args.suffix):
                raw_fname = "{}{}".format(fname, args.suffix)
                os.rename(fname, raw_fname)
            else:
                raw_fname = fname
            raw_fnames.append(raw_fname)
        n_workers = 60
        workers = Pool(n_workers)
        fn = partial(process, n_ctxs=args.n_ctxs)
        tar_fname = os.path.join(args.dirname, "{}.jsonl".format(split))
        with open(tar_fname, "w") as file:
            for fname in raw_fnames:
                with open(fname, "r") as src:
                    lines = src.readlines()
                    sz_per_worker = math.ceil(len(lines) / n_workers)
                    tasks = []
                    for s in range(0, len(lines), sz_per_worker):
                        tasks.append(lines[s: s + sz_per_worker])
                    for results in workers.map(fn, tasks):
                        for line in results:
                            print(line.strip(), file=file, flush=True)

if __name__ == '__main__':
    main()
