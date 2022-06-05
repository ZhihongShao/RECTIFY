# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import torch
import random
import json
import numpy as np
import collections
from typing import Dict, List, Tuple

from dpr.utils.data_utils import (
    read_data_from_json_files,
)
from dpr.data.biencoder_data import (
    get_dpr_files,
)

logger = logging.getLogger(__name__)

RetrieverBatch = collections.namedtuple(
    "RetrieverBatch",
    [
        "index",
        "question_ids",
        "question_mask",
        "question_token_type_ids",
        "passage_ids",
        "passage_mask",
        "passage_token_type_ids",
        "positives",
        "hard_negatives",
    ],
)

class RetrieverCollator(object):
    def __init__(
        self,
        tokenizer,
        passage_maxlength: int = 200,
        question_maxlength: int = 40,
    ):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(
        self,
        batch,
        num_hard_negatives: int = 1,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
    ):
        self.num_hard_negatives = num_hard_negatives
        self.num_other_negatives = num_other_negatives
        self.shuffle = shuffle
        self.shuffle_positives = shuffle_positives
        self.hard_neg_fallback = hard_neg_fallback

        all_ctxs = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in batch:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if self.shuffle and self.shuffle_positives:
                positive_ctxs = sample['positive_passages']
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample['positive_passages'][0]

            neg_ctxs = sample['negative_passages']
            hard_neg_ctxs = sample['hard_negative_passages']

            if self.shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if self.hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0: self.num_hard_negatives]

            neg_ctxs = neg_ctxs[0: self.num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0: self.num_hard_negatives]

            hard_negatives_start_idx = 1 + len(neg_ctxs)
            hard_negatives_end_idx = hard_negatives_start_idx + len(hard_neg_ctxs)

            positive_ctx_indices.append(len(all_ctxs))
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        len(all_ctxs) + hard_negatives_start_idx,
                        len(all_ctxs) + hard_negatives_end_idx,
                    )
                ]
            )
            all_ctxs.extend([positive_ctx] + neg_ctxs + hard_neg_ctxs)

        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question_ids, question_mask, question_token_type_ids = self.convert_question_to_ids_and_mask(question)

        passage_ids, passage_masks, passage_token_type_ids = self.convert_passage_to_ids_and_mask(all_ctxs)

        return RetrieverBatch(
            index,
            question_ids,
            question_mask,
            question_token_type_ids,
            passage_ids,
            passage_masks,
            passage_token_type_ids,
            positive_ctx_indices,
            hard_neg_ctx_indices,
        )

    def convert_question_to_ids_and_mask(self, questions):
        questions = self.tokenizer.batch_encode_plus(
            questions,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        return questions['input_ids'], questions['attention_mask'].bool(), questions['token_type_ids']

    def convert_passage_to_ids_and_mask(self, passages):
        passages = self.tokenizer.batch_encode_plus(
            passages,
            max_length=self.passage_maxlength,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        return passages['input_ids'], passages['attention_mask'].bool(), passages['token_type_ids']


class JsonQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        question_prefix: str = "question:",
        title_prefix: str = "title:",
        passage_prefix: str = "context:",
        shuffle_positives: bool = False,
    ):
        self.file = file
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.shuffle_positives = shuffle_positives
        self.data_files = []
        self.data = []
        self.ctx_id2index = {}
        self.all_ctxs = []
        self.load_data()
        logger.info("Data files: %s", self.data_files)

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        data = read_data_from_json_files(self.data_files)
        # filter those without positive ctx
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0]
        has_score = ('score' in self.data[0]['positive_ctxs'][0])
        for idx, item in enumerate(self.data):
            if 'id' not in item:
                item['id'] = idx
            for key in ['positive_ctxs', 'negative_ctxs', 'negative_ctxs']:
                if key in item:
                    if has_score:
                        item[key] = sorted(item[key], key=lambda x: -float(x['score']))
                    for ctx in item[key]:
                        if ctx['id'] not in self.ctx_id2index:
                            self.ctx_id2index[ctx['id']] = len(self.all_ctxs)
                            self.all_ctxs.append(ctx)
        logger.info("Total cleaned data size: {}, all ctxs size: {}".format(len(self.data), len(self.all_ctxs)))

    def _create_question(self, question):
        return "{} {}".format(self.question_prefix, question)

    def _create_passage(self, ctx: dict):
        return (self.title_prefix + " {} " + self.passage_prefix + " {}").format(
            ctx["title"], ctx['text']
        )

    def get_sharded_ranking_data(self, shard_id, num_shards):
        samples = []
        all_ctxs = []
        questions = []
        ctxs = []
        ctx_id2pos = {}
        num_samples_per_shard = math.ceil(len(self.data) / num_shards)
        for sample in self.data[shard_id * num_samples_per_shard: (shard_id + 1) * num_samples_per_shard]:
            samples.append(sample)
            questions.append(self._create_question(sample['question']))
            tmp = []
            for key in ['positive_ctxs', 'other_positive_ctxs', 'negative_ctxs', 'hard_negative_ctxs']:
                if key not in sample:
                    continue
                for ctx in sample[key]:
                    if ctx['id'] not in ctx_id2pos:
                        ctx_id2pos[ctx['id']] = len(all_ctxs)
                        all_ctxs.append(self.all_ctxs[self.ctx_id2index[ctx['id']]])
                        ctxs.append(self._create_passage(all_ctxs[-1]))
        return samples, all_ctxs, questions, ctxs

    def __getitem__(self, index) -> Dict:
        json_sample = self.data[index]
        question = self._create_question(json_sample['question'])

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = (
            json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        )
        hard_negative_ctxs = (
            json_sample["hard_negative_ctxs"]
            if "hard_negative_ctxs" in json_sample
            else []
        )

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = ""

        positive_passages = [self._create_passage(ctx) for ctx in positive_ctxs]
        negative_passages = [self._create_passage(ctx) for ctx in negative_ctxs]
        hard_negative_passages = [self._create_passage(ctx) for ctx in hard_negative_ctxs]
        return {
            'index': index,
            'question': question,
            'positive_passages': positive_passages,
            'negative_passages': negative_passages,
            'hard_negative_passages': hard_negative_passages,
        }

    def __len__(self):
        return len(self.data)
