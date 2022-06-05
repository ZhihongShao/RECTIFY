#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import collections
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as T
from torch.nn import CrossEntropyLoss

from dpr.data.reader_data import ReaderSample, ReaderPassage
from dpr.utils.model_utils import init_weights
from dpr.models.extractive_reader_loss_helper import (
    compute_logprob,
    document_level_loss,
    paragraph_level_loss,
)

logger = logging.getLogger()

ReaderBatch = collections.namedtuple('ReaderBatch', ['input_ids', 'start_positions', 'end_positions', 'answers_mask', 'switches_label'])

# configuration for reader model loss
ExtractiveReaderLossCfg = collections.namedtuple(
    "ExtractiveReaderLossCfg",
    [
        "use_answer_indicator",
        "train_sigmoid_answer_indicator",
        "global_loss_type",
        "global_loss_coeff",
        "local_loss_type",
        "local_loss_coeff",
        "anneal_steps"
    ],
)

DEFAULT_LOSS_CFG_TRAIN = ExtractiveReaderLossCfg(
    use_answer_indicator=True,
    train_sigmoid_answer_indicator=False,
    global_loss_type=None,
    global_loss_coeff=0.0,
    local_loss_type='h2-span-mml',
    local_loss_coeff=1.0,
    anneal_steps=-1
)

def set_default_extractive_reader_loss_cfg(
    _use_answer_indicator=True,
    _train_sigmoid_answer_indicator=False,
    _global_loss_type=None,
    _global_loss_coeff=0.0,
    _local_loss_type='h2-span-mml',
    _local_loss_coeff=1.0,
    _anneal_steps=-1
):
    global DEFAULT_LOSS_CFG_TRAIN
    DEFAULT_LOSS_CFG_TRAIN = DEFAULT_LOSS_CFG_TRAIN._replace(
        use_answer_indicator=_use_answer_indicator,
        train_sigmoid_answer_indicator=_train_sigmoid_answer_indicator,
        global_loss_type=_global_loss_type,
        global_loss_coeff=_global_loss_coeff,
        local_loss_type=_local_loss_type,
        local_loss_coeff=_local_loss_coeff,
        anneal_steps=_anneal_steps
    )

class Reader(nn.Module):

    def __init__(self, encoder: nn.Module, hidden_size, use_answer_indicator, loss_cfg=DEFAULT_LOSS_CFG_TRAIN):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.use_answer_indicator = use_answer_indicator
        self.loss_cfg=loss_cfg
        self.qa_outputs = nn.Linear(hidden_size, 2)
        weights_to_init = [self.qa_outputs]
        if use_answer_indicator:
            self.qa_classifier = nn.Linear(hidden_size, 1)
            weights_to_init.append(self.qa_classifier)
        init_weights(weights_to_init)

    def forward(self, input_ids: T, attention_mask: T, token_type_ids=None, start_positions=None, end_positions=None, answer_mask=None, switches_label=None, global_steps: int = -1):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(input_ids.view(N * M, L), 
                                token_type_ids.view(N * M, L) if token_type_ids is not None else None,
                                attention_mask.view(N * M, L))
        if self.training:
            return compute_loss(start_positions, end_positions, answer_mask, switches_label, start_logits, end_logits, relevance_logits, N, M, self.loss_cfg, global_steps)

        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M) if self.use_answer_indicator else None

    def _forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, token_type_ids, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :]) if self.use_answer_indicator else None
        return start_logits, end_logits, rank_logits


def compute_loss(start_positions, end_positions, answer_mask, switches_label, start_logits, end_logits, relevance_logits, N, M, loss_cfg, global_steps):
    answer_mask = answer_mask.type(torch.FloatTensor).cuda()

    span_loss = 0.0
    if loss_cfg.global_loss_coeff > 0:
        span_loss = loss_cfg.global_loss_coeff * document_level_loss(start_logits, start_positions, end_logits, end_positions, answer_mask, loss_cfg.global_loss_type, global_steps=global_steps, anneal_steps=loss_cfg.anneal_steps)
    if loss_cfg.local_loss_coeff > 0:
        span_loss = loss_cfg.local_loss_coeff * paragraph_level_loss(start_logits, start_positions, end_logits, end_positions, answer_mask, cfg.local_loss_type, global_steps=global_steps, anneal_steps=loss_cfg.anneal_steps) + span_loss

    # compute switch loss
    if loss_cfg.use_answer_indicator:
        if loss_cfg.train_sigmoid_answer_indicator:
            switch_loss = torch.mean(torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(relevance_logits, switches_label), dim=1))
        else:
            masked_switch_log_probs = compute_logprob(relevance_logits, 1) + (switches_label.float() - 1) * 1e10
            switch_loss = -torch.mean(torch.max(switches_label, dim=1)[0].float() * torch.logsumexp(masked_switch_log_probs, dim=1))
    else:
        switch_loss = 0.0
    return span_loss + switch_loss


def create_reader_input(pad_token_id: int,
                        samples: List[ReaderSample],
                        passages_per_question: int,
                        single_positive_per_question: bool,
                        max_length: int,
                        max_n_answers: int,
                        is_train: bool,
                        use_answer_indicator: bool,
                        shuffle: bool,
                        ) -> ReaderBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s
    :param pad_token_id: id of the padding token
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a batch
    :param max_length: max model input sequence length
    :param max_n_answers: max num of answers per single question
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: ReaderBatch instance
    """
    input_ids = []
    start_positions = []
    end_positions = []
    answers_masks = []
    switches_labels = [] if use_answer_indicator else None
    
    empty_sequence = torch.Tensor().new_full((max_length,), pad_token_id, dtype=torch.long)

    for sample in samples:
        positive_ctxs = sample.positive_passages
        negative_ctxs = sample.negative_passages if is_train else sample.passages

        sample_tensors = _create_question_passages_tensors(positive_ctxs,
                                                           negative_ctxs,
                                                           passages_per_question,
                                                           single_positive_per_question,
                                                           empty_sequence,
                                                           max_n_answers,
                                                           pad_token_id,
                                                           is_train,
                                                           is_random=shuffle)
        if not sample_tensors:
            logger.warning('No valid passages combination for question=%s ', sample.question)
            continue
        sample_input_ids, starts_tensor, ends_tensor, answer_mask, switches_label = sample_tensors
        input_ids.append(sample_input_ids)
        if is_train:
            start_positions.append(starts_tensor)
            end_positions.append(ends_tensor)
            answers_masks.append(answer_mask)
            if use_answer_indicator:
                switches_labels.append(switches_label)
    input_ids = torch.cat([ids.unsqueeze(0) for ids in input_ids], dim=0)

    if is_train:
        start_positions = torch.stack(start_positions, dim=0)
        end_positions = torch.stack(end_positions, dim=0)
        answers_masks = torch.stack(answers_masks, dim=0)
        if use_answer_indicator:
            switches_labels = torch.stack(switches_labels, dim=0)

    return ReaderBatch(input_ids, start_positions, end_positions, answers_masks, switches_labels)


def _pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)


def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    positive_a_spans = positives[idx].answers_spans
    return [span for span in positive_a_spans if (span[0] < max_len and span[1] < max_len)]


def _get_positive_idx(positives: List[ReaderPassage], max_len: int, is_random: bool):
    # select just one positive
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    if not _get_answer_spans(positive_idx, positives, max_len):
        # question may be too long, find the first positive with at least one valid span
        positive_idx = next((i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)),
                            None)
    return positive_idx


def _create_question_passages_tensors(positives: List[ReaderPassage], negatives: List[ReaderPassage], total_size: int,
                                      single_positive_per_question,
                                      empty_ids: T,
                                      max_n_answers: int,
                                      pad_token_id: int,
                                      is_train: bool,
                                      use_answer_indicator: bool,
                                      is_random: bool = True):
    max_len = empty_ids.size(0)
    if is_train:
        if single_positive_per_question:
            # select just one positive
            positive_idx = _get_positive_idx(positives, max_len, is_random)
            positive_indices = [positive_idx]
        else:
            valid_positive_indices = [i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)]
            if valid_positive_indices:
                indices = valid_positive_indices + [idx + len(positives) for idx in range(len(negatives))]
                positive_indices = []
                while True:
                    choices = np.random.choice(indices, min(total_size, len(indices)), replace=False)
                    positive_indices = np.intersect1d(choices, valid_positive_indices)
                    if positive_indices.size > 1:
                        break
                positive_indices = positive_indices.tolist()
            else:
                positive_indices = [None]
        # if any([idx is None for idx in positive_indices]):
        #     return None
        positive_indices = [idx for idx in positive_indices if idx is not None]

        positive_a_spans = []
        positives_selected = []
        answer_starts_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_ends_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_mask = torch.zeros((total_size, max_n_answers), dtype=torch.long) if use_answer_indicator else None
        switches_label = torch.zeros((total_size,)).long()
        for num_positives, positive_idx in enumerate(positive_indices):
            curr_positive_a_spans = _get_answer_spans(positive_idx, positives, max_len)[0: max_n_answers]
            positive_a_spans.extend(curr_positive_a_spans)

            answer_starts = [span[0] for span in curr_positive_a_spans]
            answer_ends = [span[1] for span in curr_positive_a_spans]

            assert all(s < max_len for s in answer_starts)
            assert all(e < max_len for e in answer_ends)

            positive_input_ids = _pad_to_len(positives[positive_idx].sequence_ids, pad_token_id, max_len)
            positives_selected.append(positive_input_ids)

            answer_starts_tensor[num_positives, 0:len(answer_starts)] = torch.tensor(answer_starts)

            answer_ends_tensor[num_positives, 0:len(answer_ends)] = torch.tensor(answer_ends)

            answer_mask[num_positives, 0:len(answer_starts)] = torch.tensor([1 for _ in range(len(answer_starts))])

            if use_answer_indicator:
                switches_label[num_positives] = 1
        if not use_answer_indicator:
            answer_mask[:, 0] = 1
    else:
        positives_selected = []
        answer_starts_tensor = None
        answer_ends_tensor = None
        answer_mask = None
        switches_label = None

    positives_num = len(positives_selected)
    negative_idxs = np.random.permutation(range(len(negatives))) if is_random else range(
        len(negatives) - positives_num)

    negative_idxs = negative_idxs[:total_size - positives_num]

    negatives_selected = [_pad_to_len(negatives[i].sequence_ids, pad_token_id, max_len) for i in negative_idxs]

    while len(negatives_selected) < total_size - positives_num:
        negatives_selected.append(empty_ids.clone())

    input_ids = torch.stack([t for t in positives_selected + negatives_selected], dim=0)
    return input_ids, answer_starts_tensor, answer_ends_tensor, answer_mask, switches_label
