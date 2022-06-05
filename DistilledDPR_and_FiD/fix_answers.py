import json
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_fname", type=str)
    parser.add_argument("--tar_fname", type=str)
    args = parser.parse_args()

    q2ans = {}
    q2id = {}
    with open(args.src_fname, "r") as file:
        for line in tqdm(file, desc='reading'):
            line = json.loads(line)
            q2ans[line['question']] = line['answers']
            q2id[line['question']] = line['id']

    data = []
    with open(args.tar_fname, "r") as file:
        for line in tqdm(file, desc='fixing'):
            line = json.loads(line)
            ans = q2ans.get(line['question'], None)
            i = q2id.get(line['question'], None)
            assert ans is not None and i is not None
            line['answers'] = ans
            line['id'] = i
            data.append(line)
    with open(args.tar_fname, "w") as file:
        for line in tqdm(data, desc='flushing'):
            print(json.dumps(line), file=file)

if __name__ == '__main__':
    main()
