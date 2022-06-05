# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path
from tqdm import tqdm
import regex as re
import regex
from functools import partial
import multiprocessing
from multiprocessing import Pool
import unicodedata
import math
import string

import numpy as np
import torch
import transformers

import src.slurm
import src.util
import src.model
import src.data

from torch.utils.data import DataLoader

from itertools import chain

import copy
from tqdm import tqdm

logger = logging.getLogger(__name__)

def embed_questions(opt, data, model, tokenizer):
    batch_size = opt.per_gpu_batch_size * opt.world_size
    dataset = src.data.Dataset(data)
    collator = src.data.Collator(opt.question_maxlength, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=10, collate_fn=collator)
    model.eval()
    embedding = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (idx, _, _, question_ids, question_mask) = batch
            output = model.embed_text(
                text_ids=question_ids.to(opt.device).view(-1, question_ids.size(-1)), 
                text_mask=question_mask.to(opt.device).view(-1, question_ids.size(-1)), 
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            embedding.append(output)

    embedding = torch.cat(embedding, dim=0)
    logger.info(f'Questions embeddings shape: {embedding.size()}')

    return embedding

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    return unicodedata.normalize("NFD", text)

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extend_alternative_answers(answers):
    lst = []
    for answer in answers:
        if re.search('\d', answer):
            lst.append(answer)
        elif ', and' in answer:
            lst.extend(extend_alternative_answers(answer.split(", and")))
        elif '|' in answer:
            lst.extend(extend_alternative_answers(answer.split("|")))
        elif ',' in answer:
            lst.extend(extend_alternative_answers(answer.split(",")))
        elif 'and' in answer:
            lst.extend(extend_alternative_answers(answer.split("and")))
        else:
            lst.append(answer)
    lst = list(set([answer.strip() for answer in lst if answer.strip()]))
    return lst

def has_answer(answer, passage, tokenizer, uncased=True, ignore_puncts=False, ignore_stop_words=False):
    def remove_punc(words):
        exclude = set(string.punctuation)
        lst = []
        for word in words:
            word = "".join(ch for ch in word if ch not in exclude)
            if word:
                lst.append(word)
        return lst

    def remove_stop_words(words):
        stop_words = set(['the', 'of', 'on', 'to', 'for', 'mr', 'miss', 'mrs', 'dr'])
        return [word for word in words if word not in stop_words]

    if isinstance(passage, str):
        passage_tokens = tokenizer.tokenize(passage, uncased=uncased)
    else:
        assert isinstance(passage, (list, tuple))
        passage_tokens = passage
    if ignore_puncts:
        _passage_tokens = remove_punc(passage_tokens)
        if _passage_tokens:
            passage_tokens = _passage_tokens
    if ignore_stop_words:
        _passage_tokens = remove_stop_words(passage_tokens)
        if _passage_tokens:
            passage_tokens = _passage_tokens
    for text in answer:
        tokens = tokenizer.tokenize(text, uncased=uncased)
        if ignore_puncts:
            _tokens = remove_punc(tokens)
            if _tokens:
                tokens = _tokens
        if ignore_stop_words:
            _tokens = remove_stop_words(tokens)
            if _tokens:
                tokens = _tokens
        token_str = " ".join(tokens)
        for s in range(len(passage_tokens) - len(tokens) + 1):
            if token_str == " ".join(passage_tokens[s: s + len(tokens)])[:len(token_str)]:
                return True
    return False

class Index:
    def __init__(self, embedding_files, max_pool_size):
        id2block_id2idx = {}
        embedding_blocks = []
        for i, file_path in enumerate(embedding_files):
            logger.info(f'Loading file {file_path}')
            with open(file_path, 'rb') as fin:
                ids, embeddings = pickle.load(fin)
                for idx, i in enumerate(ids):
                    id2block_id2idx[i] = (len(embedding_blocks), idx)
                embedding_blocks.append(embeddings)
        self.id2block_id2idx = id2block_id2idx
        self.embedding_blocks = embedding_blocks

        self.embedding_dim = self.embedding_blocks[0].shape[1]
        self.passages_embeddings = torch.zeros((max_pool_size, self.embedding_dim), dtype=torch.float16).cuda()
        self.max_pool_size = max_pool_size
        self.pool_size = max_pool_size

    def gather_passage_embeddings(self, passage_ids):
        pool_size = len(passage_ids)
        assert pool_size <= self.max_pool_size
        self.pool_size = pool_size
        self.passage_ids = passage_ids
        for rid, i in enumerate(passage_ids):
            block_id, idx = self.id2block_id2idx[i]
            embedding = torch.tensor(self.embedding_blocks[block_id][idx]).half()
            self.passages_embeddings[rid] = embedding

    def query(self, query_embeddings):
        scores = torch.matmul(query_embeddings, torch.transpose(self.passages_embeddings[: self.pool_size], 1, 0))
        values, indices = torch.sort(scores, descending=True)
        return values, indices

def fn(inputs, consider_title, ignore_puncts, ignore_stop_words):
    simple_tokenizer = SimpleTokenizer()
    id2pred_rank = {}
    for (ctxs, sample_data) in tqdm(inputs, desc='answer checking', total=len(inputs)):
        tokenized_ctxs = [simple_tokenizer.tokenize(_normalize(ctx['title'] + " " + ctx['text'] if consider_title else ctx['text']), uncased=True) for ctx in ctxs]
        pred_rank = set()
        for sample in sample_data:
            pred = _normalize(sample['prediction'])
            for rank in range(len(ctxs)):
                if has_answer([pred], tokenized_ctxs[rank], simple_tokenizer, uncased=True, ignore_puncts=ignore_puncts, ignore_stop_words=ignore_stop_words):
                    pred_rank.add((pred, rank))
        id2pred_rank[sample_data[0]['id']] = pred_rank
    return id2pred_rank

def main(opt):
    src.util.init_logger(is_main=True)
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    data = src.data.load_data(opt.src_data, global_rank=opt.split_id, world_size=opt.num_splits)
    id2idx = {sample['id']: idx for idx, sample in enumerate(data)}
    q2idx = {sample['question']: idx for idx, sample in enumerate(data)}
    with open(opt.cluster_data, "r") as file:
        for line in file:
            line = json.loads(line)
            if opt.use_question:
                idx = q2idx.get(line['question'], None)
                if idx is not None:
                    data[idx]['clusters'] = line['clusters']
            else:
                idx = id2idx.get(line['id'], None)
                if idx is not None:
                    data[idx]['clusters'] = line['clusters']
    del id2idx
    del q2idx
    model_class = src.model.Retriever
    model = model_class.from_pretrained(opt.model_path)

    model.cuda()
    model.eval()
    if not opt.no_fp16:
        model = model.half()

    # index all passages
    input_paths = glob.glob(args.passages_embeddings)
    input_paths = sorted(input_paths)
    index = Index(input_paths, opt.n_reranked_docs)

    import time
    all_questions = []
    for sample in tqdm(data, desc='collecting questions', total=len(data)):
        added_preds = set()
        if args.add_missed_answers:
            norm_preds = set()
            for pred in sample['clusters']:
                norm_preds.add(_normalize_answer(pred))
            for answer in sample['answers']:
                na = _normalize_answer(answer)
                if na not in norm_preds:
                    norm_preds.add(na)
                    sample['clusters'][answer] = []
                    added_preds.add(answer)
        sample['added_preds'] = added_preds
        sample_data = []
        for pred in sample['clusters']:
            new_sample = {
                'id': sample['id'],
                'question': sample['question'] + " " + pred,
                'answers': sample['answers'],
                'prediction': pred,
            }
            sample_data.append(new_sample)
        all_questions.append(sample_data)

    all_question_embeddings = embed_questions(opt, list(chain(*all_questions)), model, tokenizer)
    for sample, sample_data in tqdm(zip(data, all_questions), desc='reranking', total=len(data)):
        passage_ids = [doc['id'] for doc in sample['ctxs'][: opt.n_reranked_docs]]
        index.gather_passage_embeddings(passage_ids)
        question_embeddings = all_question_embeddings[: len(sample_data)]
        all_question_embeddings = all_question_embeddings[len(sample_data):]
        values, indices = index.query(question_embeddings)
        clusters = {}
        for new_sample, scores, ranks in zip(sample_data, values.cpu().numpy().tolist(), indices.cpu().numpy().tolist()):
            clusters[new_sample['prediction']] = list(zip(ranks, scores))
        sample['clusters'] = clusters

    n_workers = 6
    _tasks = [(sample['ctxs'][: opt.n_reranked_docs], sample_data) for sample, sample_data in zip(data, all_questions)]
    sz_per_worker = int(math.ceil(len(_tasks) / n_workers))
    tasks = []
    for s in range(0, len(_tasks), sz_per_worker):
        tasks.append(_tasks[s : s + sz_per_worker])
    workers = Pool(n_workers)
    _fn = partial(
        fn,
        consider_title=not opt.ignore_title,
        ignore_puncts=not opt.consider_puncts,
        ignore_stop_words=not opt.consider_stop_words,
    )
    id2pred_rank = {}
    for item in workers.map(_fn, tasks):
        id2pred_rank.update(item)
    for sample in tqdm(data, desc='finalizing', total=len(data)):
        clusters = {}
        added_preds = sample.pop('added_preds')
        for pred, val in sample['clusters'].items():
            clusters[pred] = sorted([(item[0], int((pred, item[0]) in id2pred_rank[sample['id']]), item[1]) for item in val], key=lambda x: x[1:], reverse=True)
            if pred in added_preds and clusters[pred][0][1] == 0:
                clusters.pop(pred)
        sample['clusters'] = clusters

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, 'w') as fout:
        for sample in data:
            print(json.dumps(sample), file=fout, flush=True)
    logger.info(f'Saved results to {args.output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_data', required=True, type=str, default=None)
    parser.add_argument('--cluster_data', required=True, type=str, default=None, 
                        help=".jsonl file containing question and answers, similar format to reader data")
    parser.add_argument('--passages_embeddings', type=str, default=None, help='Glob path to encoded passages')
    parser.add_argument('--output_path', type=str, default=None, help='Results are written to output_path')
    parser.add_argument('--per_gpu_batch_size', type=int, default=64, help="Batch size for question encoding")
    parser.add_argument('--model_path', type=str, help="path to directory containing model weights and config file")
    parser.add_argument('--no_fp16', action='store_true', help="inference in fp32")
    parser.add_argument('--passage_maxlength', type=int, default=200, help="Maximum number of tokens in a passage")
    parser.add_argument('--question_maxlength', type=int, default=40, help="Maximum number of tokens in a question")

    parser.add_argument("--num_splits", type=int, default=8)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--n_reranked_docs", type=int, default=100, help="evidences are retrieved from top-n passages relevant to the original question")
    parser.add_argument("--add_missed_answers", action='store_true')

    parser.add_argument("--use_question", action='store_true')

    parser.add_argument("--ignore_title", action='store_true')
    parser.add_argument("--consider_puncts", action='store_true')
    parser.add_argument("--consider_stop_words", action='store_true')


    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
