#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for the Reader model related data processing tasks
"""

import glob
import json
import math
import multiprocessing
import os
import pickle
import torch

import numpy as np
from functools import partial
from typing import Tuple, List, Dict, Iterable, Optional
from torch import Tensor as T
from tqdm import tqdm
import copy
from collections import defaultdict

from src.utils import read_serialized_data_from_files, log
from src.utils.rerank_evaluate_script import get_f1_prec_recall, get_exact_match, SimpleTokenizer2, extend_alternative_answers, locate_answer


# configuration for preprocessing
class PreprocessingCfg:
    def __init__(self):
        self.multi_answers = True
        self.n_contexts = 10
        self.irrelevant_token_id = 1786
        self.relevant_token_id = 269
        self.num_neg_per_pos = 10
        self.train_split_answers = True


DEFAULT_PREPROCESSING_CFG_TRAIN = PreprocessingCfg()


def set_default_preprocessing_cfg(**kwargs):
    global DEFAULT_PREPROCESSING_CFG_TRAIN
    for key, val in kwargs.items():
        assert hasattr(DEFAULT_PREPROCESSING_CFG_TRAIN, key)
        setattr(DEFAULT_PREPROCESSING_CFG_TRAIN, key, val)


def _get_cached_data_cfgs_repr(is_train):
    global DEFAULT_PREPROCESSING_CFG_TRAIN
    cfg_str = "prepro_"
    for key in sorted(DEFAULT_PREPROCESSING_CFG_TRAIN.__dict__):
        val = getattr(DEFAULT_PREPROCESSING_CFG_TRAIN, key)
        cfg_str += "{}-{}_".format(key, val)
    cfg_str += "is-train-{}".format(is_train)
    return cfg_str[:-1]


class Passage(object):
    def __init__(
        self,
        id=None,
        title: str = None,
        text: str = None,
        score=None,
        rank=None,
        has_answer: bool = None
    ):
        self.id = id
        self.title = title
        self.passage_text = text
        self.score = float(score) if score is not None else None
        self.rank = rank
        self.has_answer = has_answer

        self.passage_tokens = None
        self.sequence_ids = None

    def on_serialize(self):
        self.passage_tokens = None

    def on_deserialize(self):
        pass


class Cluster(object):
    def __init__(
        self,
        id,
        question: str,
        prediction: str,
        relevance_tag: int,
        passages: List[Passage] = [],
        passage_ranks: List = [],
    ):
        self.id = id
        self.question = question
        self.prediction = prediction
        self.relevance_tag = relevance_tag
        self.passages = passages
        self.passage_ranks = passage_ranks

    def on_serialize(self):
        self.question = None
        self.passages = None

    def on_deserialize(self, question, passages):
        self.question = question
        self.passages = [passages[rank] for rank in self.passage_ranks]


class Sample(object):
    """
    Container to collect all Q&A passages data per singe question
    """

    def __init__(
        self,
        id,
        question: str,
        answers: List,
        passages: List[Passage] = [],
        clusters: List[Cluster] = [],
        positive_cluster_indices: List[int] = [],
        negative_cluster_indices: List[int] = [],
    ):
        self.id = id
        self.question = question
        self.answers = answers
        self.passages = passages
        self.clusters = clusters
        self.positive_cluster_indices = positive_cluster_indices
        self.negative_cluster_indices = negative_cluster_indices

    def on_serialize(self):
        for passage in self.passages:
            passage.on_serialize()
        for cluster in self.clusters:
            cluster.on_serialize()

    def on_deserialize(self):
        for passage in self.passages:
            passage.on_deserialize()
        for cluster in self.clusters:
            cluster.on_deserialize(self.question, self.passages)


def _deserialize_samples(samples):
    for sample in samples:
        sample.on_deserialize()
    return samples


class VerifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        files: str,
        is_train: bool,
        tokenizer,
        run_preprocessing: bool,
        num_workers: int,
    ):
        self.files = files
        self.data = []
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.run_preprocessing = run_preprocessing
        self.num_workers = num_workers

    def __getitem__(self, index):
        return self.train_data[index]

    def __len__(self):
        return len(self.train_data)

    def load_data(
        self,
        num_shards,
        shard_id,
        cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    ):
        data_files = glob.glob(self.files)
        log("Data files: {}".format(data_files))
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = sorted(self._get_preprocessed_files(data_files, cfg))
        self.data = read_serialized_data_from_files(preprocessed_data_files, num_shards, shard_id)

        # _num_workers = self.num_workers
        _num_workers = 1 # NOTE: overrule self.num_workers to avoid memory allocation error
        if _num_workers > 1:
            workers = multiprocessing.Pool(_num_workers)
            ds_size = len(self.data)
            step = max(math.ceil(ds_size / _num_workers), 1)
            chunks = [self.data[i : i + step] for i in range(0, ds_size, step)]
        else:
            chunks = [self.data]
        data = []
        for samples in (workers.map(_deserialize_samples, chunks) if _num_workers > 1 else map(_deserialize_samples, chunks)):
            data.extend(samples)
        self.data = data
        self.train_data = self.gen_train_samples(epoch=0, cfg=cfg, verbose=True) if self.is_train else self.data

    def gen_train_samples(
        self,
        epoch: int,
        shuffle_seed: int = 0,
        cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
        verbose: bool = False,
    ):
        g = torch.Generator()
        g.manual_seed(shuffle_seed + epoch)
        positive_clusters = []
        negative_clusters = []
        for sample in self.data:
            for cid in sample.positive_cluster_indices:
                positive_clusters.append(sample.clusters[cid])
            for cid in sample.negative_cluster_indices:
                negative_clusters.append(sample.clusters[cid])
        if verbose:
            log("Num original positive clusters = {}; num original negative clusters = {}".format(len(positive_clusters), len(negative_clusters)))
        if cfg.num_neg_per_pos:
            num_neg = int(cfg.num_neg_per_pos * len(positive_clusters))
            indices = torch.randperm(len(negative_clusters), generator=g).tolist()  # type: ignore
            negative_indices = indices * (num_neg // len(negative_clusters)) + indices[:num_neg % len(negative_clusters)]
            # negative_indices = np.random.choice(indices, num_neg, replace=(num_neg > len(indices)))
            negative_clusters = [negative_clusters[idx] for idx in negative_indices]
            if verbose:
                log("Num trained negative clusters = {}".format(len(negative_clusters)))
        clusters = positive_clusters + negative_clusters
        # indices = torch.randperm(len(clusters), generator=g).tolist()  # type: ignore
        # clusters = [clusters[idx] for idx in indices]
        return clusters

    def apply(self, func):
        for item in self.data:
            func(item)

    def _get_preprocessed_files(
        self,
        data_files: List,
        cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    ):

        serialized_files = [file for file in data_files if file.endswith(".pkl")]
        if serialized_files:
            return serialized_files
        assert len(data_files) == 1, "Only 1 source file pre-processing is supported."

        # data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            # data with different preprocessing cfgs are cached into different dirs
            cached_cfgs_str = _get_cached_data_cfgs_repr(self.is_train)
            dir_path = os.path.join(dir_path, cached_cfgs_str)
            os.makedirs(dir_path, exist_ok=True)
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        input_file = data_files[0]
        if not self.is_train or (self.run_preprocessing and not serialized_files):
            with open(input_file, "r", encoding="utf-8") as f:
                samples = []
                for line in f:
                    sample = json.loads(line)
                    if self.run_preprocessing and not serialized_files:
                        pred2passages = sample['clusters']
                        for pred, passages in pred2passages.items():
                            pred2passages[pred] = passages[: cfg.n_contexts] # NOTE: to reduce memory usage
                        sample['clusters'] = pred2passages
                    else:
                        sample.pop('clusters')
                    samples.append(sample)
        if not self.is_train:
            log(
                "Loaded {} questions + retrieval results from {}".format(len(samples), input_file)
            )
            self._id2raw_sample = {sample['id']: sample for sample in samples}

        if serialized_files:
            log("Found preprocessed files. {}".format(serialized_files))
            return serialized_files

        log("Data are not preprocessed for reader training. Start pre-processing ...")

        # start pre-processing and save results
        def _run_preprocessing(tokenizer):
            # temporarily disable auto-padding to save disk space usage of serialized files
            serialized_files = format_samples(
                self.is_train,
                samples,
                out_file_prefix,
                tokenizer,
                num_workers=self.num_workers,
            )
            return serialized_files

        if self.run_preprocessing:
            serialized_files = _run_preprocessing(self.tokenizer)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            serialized_files, _ = _find_cached_files(data_files[0])
        return serialized_files


def preprocess_data(
    samples: List[Dict],
    tokenizer,
    cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    is_train_set: bool = True,
) -> Iterable[Sample]:
    """
    Preprocess.
    :param samples: samples from the retriever's json file results
    :param tokenizer: transformers tokenizer object for text to model input tensors conversions
    :param cfg: PreprocessingCfg object with positive and negative passage selection parameters
    :param is_train_set: if the data should be processed as a train set
    :return: iterable of Sample objects which can be consumed by the reader model
    """

    tokenizer = SimpleTokenizer2()
    ret = []
    for sample in samples:
        if not sample['clusters']:
            if is_train_set:
                continue
            else:
                sample['clusters'] = {
                    'no predictions': [(i, 0) for i in range(len(sample['ctxs']))]
                }
        if cfg.multi_answers:
            answers = []
            for anno in sample.get('annotations', []):
                if anno['type'] == 'singleAnswer':
                    answers.append([anno['answer']])
                else:
                    answers.append([qa['answer'] for qa in anno['qaPairs']])
        else:
            answers = [[sample['answers']]] if 'answers' in sample else []
        passages = []
        for rank, ctx in enumerate(sample['ctxs']):
            passages.append(
                Passage(
                    id=ctx['id'],
                    title=ctx['title'],
                    text=ctx['text'],
                    score=ctx['score'],
                    rank=rank,
                    has_answer=ctx.get('hasanswer', False),
                )
            )
        pred2cluster = sample['clusters']
        sample = Sample(
            id=sample['id'],
            question=sample['question'],
            answers=answers,
            passages=passages,
        )
        predictions = list(pred2cluster.keys())
        best_annotation = None
        best_recall = None
        if is_train_set and cfg.train_split_answers:
            answers = [[extend_alternative_answers(answer) for answer in annotated_answers] for annotated_answers in answers]
        for annotated_answers in answers:
            _, _, recall = get_f1_prec_recall(annotated_answers, predictions)
            if best_recall is None or best_recall < recall:
                best_recall = recall
                best_annotation = annotated_answers
        clusters = []
        positive_cluster_indices = []
        negative_cluster_indices = []
        for pred, passage_cluster in pred2cluster.items():
            if is_train_set:
                relevance_tag = cfg.irrelevant_token_id
                negative_cluster_indices.append(len(clusters))
            else:
                relevance_tag = cfg.relevant_token_id # just a placeholder for testing
            clusters.append(
                Cluster(
                    sample.id,
                    sample.question,
                    pred,
                    relevance_tag,
                    passage_ranks=[item[0] for item in passage_cluster[:cfg.n_contexts]]
                )
            )
        sample.clusters = clusters
        sample.positive_cluster_indices = positive_cluster_indices
        sample.negative_cluster_indices = negative_cluster_indices
        sample.answers = best_annotation if is_train_set else answers
        ret.append(sample)
    return ret


def format_samples(
    is_train_set: bool,
    samples: List[Dict],
    out_file_prefix: str,
    tokenizer,
    num_workers: int = 8,
) -> List[str]:
    """
    Converts the file with dense retriever(or any compatible file format) results into the reader input data and
    serializes them into a set of files.
    Conversion splits the input data into multiple chunks and processes them in parallel. Each chunk results are stored
    in a separate file with name out_file_prefix.{number}.pkl
    :param is_train_set: if the data should be processed for a train set (i.e. with answer span detection)
    :param input_file: path to a json file with data to convert
    :param out_file_prefix: output path prefix.
    :param tokenizer: transformers tokenizer object for text to model input tensors conversions
    :param num_workers: the number of parallel processes for conversion
    :return: names of files with serialized results
    """
    if num_workers > 1:
        workers = multiprocessing.Pool(num_workers)
        ds_size = len(samples)
        step = max(math.ceil(ds_size / num_workers), 1)
        chunks = [samples[i : i + step] for i in range(0, ds_size, step)]
    else:
        chunks = [samples]
    chunks = [(i, chunks[i]) for i in range(len(chunks))]

    log("Split data into {} chunks".format(len(chunks)))

    processed = 0
    _parse_batch = partial(
        _preprocess_samples_chunk,
        out_file_prefix=out_file_prefix,
        tokenizer=tokenizer,
        is_train_set=is_train_set,
    )
    serialized_files = []
    for file_name in (workers.map(_parse_batch, chunks) if num_workers > 1 else map(_parse_batch, chunks)):
        processed += 1
        serialized_files.append(file_name)
        log("Chunks processed {}".format(processed))
        log("Data saved to {}".format(file_name))
    log("Preprocessed data stored in {}".format(serialized_files))
    return serialized_files


def _preprocess_samples_chunk(
    samples: List,
    out_file_prefix: str,
    tokenizer,
    is_train_set: bool,
) -> str:
    chunk_id, samples = samples
    log("Start batch {}".format(len(samples)))
    iterator = preprocess_data(
        samples,
        tokenizer,
        is_train_set=is_train_set,
    )

    results = []

    iterator = tqdm(iterator)
    for i, r in enumerate(iterator):
        r.on_serialize()
        results.append(r)

    out_file = out_file_prefix + "." + str(chunk_id) + ".pkl"
    with open(out_file, mode="wb") as f:
        log("Serialize {} results to {}".format(len(results), out_file))
        pickle.dump(results, f)
    return out_file
