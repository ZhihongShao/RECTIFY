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

from src.utils.rerank_evaluate_script import SimpleTokenizer2, extend_alternative_answers, locate_answer


class Passage(object):
    def __init__(
        self,
        id=None,
        title: str = None,
        text: str = None,
        score=None,
        rank=None,
        covered_index2answer: Dict = {},
        has_answer: bool = None
    ):
        self.id = id
        self.title = title
        self.passage_text = text
        self.score = float(score) if score is not None else None
        self.rank = rank
        self.covered_index2answer = covered_index2answer 
        self.has_answer = has_answer

        self.passage_tokens = None
        self.sequence_ids = None

    def on_serialize(self):
        if self.sequence_ids is not None and isinstance(self.sequence_ids, torch.Tensor):
            self.sequence_ids = self.sequence_ids.numpy()
        # self.passage_text = None
        # self.title = None
        self.passage_tokens = None
        self.covered_index2answer = None

    def on_deserialize(self):
        if self.sequence_ids is not None and not isinstance(self.sequence_ids, torch.Tensor):
            self.sequence_ids = torch.tensor(self.sequence_ids)


class TestSample(object):
    """
    Container to collect all Q&A passages data per singe question
    """

    def __init__(
        self,
        id,
        question: str,
        answers: List,
        passages: List[Passage] = [],
        positive_passage_rank_answer_pairs: List = [],
        negative_passage_ranks: List[int] = [],
    ):
        self.id = id
        self.question = question
        self.answers = answers
        self.passages = passages
        self.positive_passage_rank_answer_pairs = positive_passage_rank_answer_pairs
        self.negative_passage_ranks = negative_passage_ranks

    def on_serialize(self):
        for passage in self.passages:
            passage.on_serialize()

    def on_deserialize(self):
        for passage in self.passages:
            passage.on_deserialize()


class TrainSample(object):
    """
    Container to collect all Q&A passages data per singe question
    Only used for training recaller
    THIS WILL NOT BE SAVED AS A PICKLE FILE!!!
    """

    def __init__(
        self,
        id,
        question: str,
        answers: List[List],
        passage: Passage,
    ):
        self.id = id
        self.question = question
        self.answers = answers
        self.passage = passage


# configuration for preprocessing
class PreprocessingCfg:
    def __init__(self):
        self.multi_answers = True
        self.n_contexts = 100
        self.num_neg_per_pos = 2 # can be a float; no neg if < 0; negs are sampled every epoch
        self.top_k_pos = 20 # only the top k pos (cover as many diverse answers as possible) will be used for training
        self.irrelevant_answer = "irrelevant"
        # recognizing positive passages, only for training
        self.consider_title = True
        self.split_answers = True
        self.ignore_puncts = True
        self.ignore_stop_words = True


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
    return cfg_str


class RecallerDataset(torch.utils.data.Dataset):
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
    ):
        data_files = glob.glob(self.files)
        log("Data files: {}".format(data_files))
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = sorted(self._get_preprocessed_files(data_files))
        self.data = read_serialized_data_from_files(preprocessed_data_files, num_shards, shard_id)
        self.train_data = self.gen_train_samples(epoch=0, verbose=True) if self.is_train else self.data

    def apply(self, func):
        for item in self.data:
            func(item)

    def _get_preprocessed_files(
        self,
        data_files: List,
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
            base_name = base_name.replace(".json", "")
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        input_file = data_files[0]
        if not self.is_train or (self.run_preprocessing and not serialized_files):
            with open(input_file, "r", encoding="utf-8") as f:
                samples = json.loads("".join(f.readlines()))
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

    def gen_train_samples(
        self,
        epoch: int,
        shuffle_seed: int = 0,
        cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
        verbose: bool = False,
    ):
        g = torch.Generator()
        g.manual_seed(shuffle_seed + epoch)
        samples = []
        num_pos_cases = 0
        num_neg_cases = 0
        for sample in self.data:
            passages = sample.passages
            for (rank, answers) in sample.positive_passage_rank_answer_pairs:
                train_sample = TrainSample(
                    id=sample.id,
                    question=sample.question,
                    answers=answers,
                    passage=passages[rank]
                )
                samples.append(train_sample)
                num_pos_cases += 1
            if cfg.num_neg_per_pos:
                num_negs = min(int(cfg.num_neg_per_pos * len(sample.positive_passage_rank_answer_pairs)), len(sample.negative_passage_ranks))
                indices = torch.randperm(len(sample.negative_passage_ranks), generator=g).tolist()  # type: ignore
                for idx in indices[:num_negs]:
                    rank = sample.negative_passage_ranks[idx]
                    train_sample = TrainSample(
                        id=sample.id,
                        question=sample.question,
                        answers=[[(cfg.irrelevant_answer, 0)]],
                        passage=passages[rank]
                    )
                    samples.append(train_sample)
                    num_neg_cases += 1
        if verbose:
            log("Num positive cases = {}; num negative cases = {}".format(num_pos_cases, num_neg_cases))
        return samples


def get_positive_passages(
    tokenizer,
    sorted_passages: List[Passage],
    answers: List[List[str]],
    consider_title: bool,
    ignore_puncts: bool,
    ignore_stop_words: bool,
    locate_top_positive_per_answer: bool = True,
):
    """
    :return: top_positive_ranks (List[int]), (postive_ranks (List[int]), covered_answers (List[Dict: covered_index2answer]))
    """
    top_positive_ranks = []
    positive_ranks = []
    covered_answers = []
    covered_indices = set()
    for rank, passage in enumerate(sorted_passages):
        if passage.passage_tokens is None:
            passage.passage_tokens = tokenizer.tokenize(
                passage.title + " " + passage.passage_text if consider_title else passage.passage_text, 
                uncased=True
            )
        covered = {}
        covered_new_indices = set()
        for idx, answer in enumerate(answers):
            if locate_top_positive_per_answer and idx in covered_indices:
                continue
            covered_texts = []
            for text in answer:
                ret = locate_answer([text], passage.passage_tokens, tokenizer, uncased=True, ignore_puncts=ignore_puncts, ignore_stop_words=ignore_stop_words)
                if ret:
                    assert len(ret) == 1
                    covered_texts.append(ret[0])
            if covered_texts:
                if idx not in covered_indices:
                    covered_new_indices.add(idx)
                covered[idx] = covered_texts
                covered_indices.add(idx)
        if covered:
            covered_answers.append(covered)
            positive_ranks.append(rank)
        if covered_new_indices:
            top_positive_ranks.append(rank)
    return top_positive_ranks, (positive_ranks, covered_answers)


def preprocess_data(
    samples: List[Dict],
    tokenizer,
    cfg: PreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    is_train_set: bool = True,
) -> Iterable[TestSample]:
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
        for rank, ctx in enumerate(sample['ctxs'][: cfg.n_contexts]):
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
        sample = TestSample(
            id=sample['id'],
            question=sample['question'],
            answers=answers,
            passages=passages,
        )
        if is_train_set:
            best_annotation = None
            best_covered_ratio = None
            best_av_rank = None
            best_positive_ranks = None
            best_covered_answers = None
            for annotated_answers in answers:
                top_positive_ranks, (positive_ranks, covered_answers) = get_positive_passages(tokenizer, passages, annotated_answers, consider_title=cfg.consider_title, ignore_puncts=cfg.ignore_puncts, ignore_stop_words=cfg.ignore_stop_words, locate_top_positive_per_answer=False)
                if cfg.split_answers:
                    all_covered_indices = []
                    for covered in covered_answers:
                        all_covered_indices.extend(list(covered.keys()))
                    all_covered_indices = set(all_covered_indices)
                    if len(all_covered_indices) < len(annotated_answers):
                        split_answers = []
                        for i in range(len(annotated_answers)):
                            if i not in all_covered_indices:
                                extended_answers = extend_alternative_answers(annotated_answers[i])
                                new_answers = set(extended_answers).difference(set(annotated_answers[i]))
                                if new_answers:
                                    for answer in new_answers:
                                        split_answers.append([answer])
                                else:
                                    split_answers.append(annotated_answers[i])
                        annotated_answers = [annotated_answers[i] for i in all_covered_indices] + split_answers
                        top_positive_ranks, (positive_ranks, covered_answers) = get_positive_passages(tokenizer, passages, annotated_answers, consider_title=cfg.consider_title, ignore_puncts=cfg.ignore_puncts, ignore_stop_words=cfg.ignore_stop_words, locate_top_positive_per_answer=False)
                if positive_ranks:
                    av_rank = np.mean(top_positive_ranks)
                    all_covered_indices = []
                    for covered in covered_answers:
                        all_covered_indices.extend(list(covered.keys()))
                    covered_ratio = len(set(all_covered_indices)) / len(annotated_answers)
                    if best_annotation is None or best_covered_ratio < covered_ratio or (best_covered_ratio == covered_ratio and best_av_rank > av_rank):
                        best_annotation = annotated_answers
                        best_covered_ratio = covered_ratio
                        best_av_rank = av_rank
                        best_positive_ranks = positive_ranks
                        best_covered_answers = covered_answers

            if best_positive_ranks is not None:
                answer_index2covered_cnt = {}
                positive_passage_priority_rank_answer_triples = []
                for rank, covered in zip(best_positive_ranks, best_covered_answers):
                    passage = passages[rank]
                    passage.covered_index2answer = covered
                    for answer_index, answer in covered.items():
                        covered_cnt = answer_index2covered_cnt.get(answer_index, 0)
                        answer_index2covered_cnt[answer_index] = answer_index2covered_cnt.get(answer_index, 0) + 1
                        positive_passage_priority_rank_answer_triples.append(
                            (rank + len(passages) * covered_cnt, rank, answer_index)
                        )
                # positive_passage_ranks = []
                # for item in sorted(positive_passage_priority_rank_answer_triples, key=lambda x: x[0]):
                #     if item[1] not in positive_passage_ranks:
                #         positive_passage_ranks.append(item[1])
                # positive_passage_rank_answer_pairs = [[rank, list(passages[rank].covered_index2answer.values())] for rank in positive_passage_ranks]
                positive_passage_rank_answer_pairs = []
                for item in sorted(positive_passage_priority_rank_answer_triples, key=lambda x: x[0]):
                    rank = item[1]
                    passage = passages[rank]
                    covered_index2answer = passage.covered_index2answer
                    positive_passage_rank_answer_pairs.append(
                        [
                            rank,
                            [covered_index2answer[item[2]]] + [covered_index2answer[i] for i in covered_index2answer if i != item[2]]
                        ]
                    )
                negative_passage_ranks = [rank for rank, passage in enumerate(passages) if not passage.has_answer]
                sample.answers = best_annotation
                sample.positive_passage_rank_answer_pairs = positive_passage_rank_answer_pairs[:cfg.top_k_pos]
                sample.negative_passage_ranks = negative_passage_ranks
                ret.append(sample)
        else:
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
