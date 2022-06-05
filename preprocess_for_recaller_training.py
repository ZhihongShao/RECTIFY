from genericpath import exists
import os

import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_result_path", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--n_contexts", type=int, default=10)
    args = parser.parse_args()

    for path in [args.retrieval_result_path, args.target_path]:
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
    data = json.load(open(args.retrieval_result_path, "r", encoding='utf-8'))
    for item in data:
        item['ctxs'] = item['ctxs'][:args.n_contexts]
    json.dump(data, open(args.target_path, "w", encoding='utf-8'))

if __name__ == '__main__':
    main()
