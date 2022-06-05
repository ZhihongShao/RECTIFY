import os
from subprocess import Popen
import argparse
import json
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_data", type=str)
    parser.add_argument("--cluster_data", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--passages", type=str)
    parser.add_argument("--passages_embeddings", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--n_reranked_docs", type=int, default=500)
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--use_question", action='store_true')
    parser.add_argument("--add_missed_answers", type=str, choices=['false', 'true'], default='false')
    args = parser.parse_args()

    cuda_device_cnt = torch.cuda.device_count()
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        CUDA_VISIBLE_DEVICES = list(map(int, os.getenv('CUDA_VISIBLE_DEVICES').split(",")))
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    else:
        CUDA_VISIBLE_DEVICES = list(range(cuda_device_cnt))

    if os.path.exists(args.output_path):
        os.remove(args.output_path)
    for r in range(args.n_rounds):
        cmds = []
        paths = []
        for split_id, gpu_id in enumerate(CUDA_VISIBLE_DEVICES):
            output_path = args.output_path + ".{}".format(split_id)
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python passage_retrieval.py " \
                f"--src_data {args.src_data} " \
                f"--cluster_data {args.cluster_data} " \
                f"--model_path {args.model_path} " \
                f"--passages {args.passages} " \
                f"--passages_embeddings \"{args.passages_embeddings}\" " \
                f"--output_path {output_path} " \
                f"--n_reranked_docs {args.n_reranked_docs} " \
                f"--num_splits {cuda_device_cnt * args.n_rounds} " \
                f"--split_id {split_id + r * cuda_device_cnt} "
            if args.use_question:
                cmd += " --use_question"
            if args.add_missed_answers == 'true':
                cmd += " --add_missed_answers"
            cmds.append(cmd)
            paths.append(output_path)
        procs = [Popen(cmd, shell=True) for cmd in cmds]
        for proc in procs:
            proc.wait()

        assert all(os.path.exists(path) for path in paths)

        with open(args.output_path, "a") as file:
            for path in paths:
                with open(path, "r") as src:
                    for line in src:
                        print(line.strip(), file=file)
                os.remove(path)

if __name__ == '__main__':
    main()
