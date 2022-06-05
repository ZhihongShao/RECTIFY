#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

from functools import reduce
import types
import collections
from typing import List
import types
import deepspeed
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import T5ForConditionalGeneration, T5Config
from transformers import TrainingArguments
from transformers.integrations import DeepSpeedConfigHF

from src.data.verifier_data import Sample, Cluster
from src.utils import log

InputBatch = collections.namedtuple(
    'InputBatch', 
        [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_input_len",
            "relevance_tag",
        ]
)


class Verifier(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_input_len, relevance_tag=None, **kwargs):
        # inputs might have already be resized in the generate method
        if input_ids.dim() == 3:
            self.encoder.n_passages = input_ids.size(1)
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        lm_logits = output[0]
        relevance_logits = torch.gather(lm_logits, 1, (decoder_input_len - 1).unsqueeze(1).unsqueeze(2).expand(-1, -1, lm_logits.size(2)))
        relevance_logits = relevance_logits.squeeze(1)
        output = (relevance_logits, ) + output[1:]
        if relevance_tag is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(relevance_logits, relevance_tag)
            return (loss, ) + output
        else:
            return output

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, decoder_input_ids, decoder_input_len):
        # gradient_checkpointing = self.decoder.config.gradient_checkpointing
        # self.set_checkpoint(False)
        self.encoder.n_passages = input_ids.size(1)
        output = self.forward(input_ids, attention_mask, decoder_input_ids, decoder_input_len)
        relevance_logits = output[0]
        # self.set_checkpoint(gradient_checkpointing)
        relevant_token_id = getattr(self.config, 'relevant_token_id', 2193)
        irrelevant_token_id = getattr(self.config, 'irrelevant_token_id', 26213)
        # return torch.argmax(relevance_logits[:, [irrelevant_token_id, relevant_token_id]], dim=1)
        return torch.softmax(relevance_logits[:, [irrelevant_token_id, relevant_token_id]], dim=1)

    def wrap(self):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder.forward = types.MethodType(get_wrapped_encoder_forward(self.encoder.forward), self.encoder)

    @classmethod
    def get_init_model(cls, pretrained_model_name_or_path, deepspeed_config, model_path=None):
        t5_config = T5Config.from_pretrained(pretrained_model_name_or_path)
        hf_args = TrainingArguments(output_dir=None, deepspeed=deepspeed_config)
        hf_ds_cfg = DeepSpeedConfigHF(hf_args)
        model = cls.from_pretrained(model_path or pretrained_model_name_or_path, config=t5_config)
        model.hf_ds_cfg = hf_ds_cfg
        model.wrap()
        return model

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """

        self.encoder.config.use_cache = False
        self.encoder.config.gradient_checkpointing = use_checkpoint
        self.decoder.config.gradient_checkpointing = use_checkpoint
        if use_checkpoint:
            self.config.use_cache = False
            self.decoder.config.use_cache = False
        else:
            self.config.use_cache = True
            self.decoder.config.use_cache = True

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask, score_at_output_position):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        passage_len = context_mask.size(2)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        bsz, n_heads, _, _ = scores[0].size()
        n_layers = len(scores)
        score_at_output_position = score_at_output_position.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(bsz, n_heads, 1, n_passages * passage_len)
        scores = [
            torch.gather(_scores, 2, score_at_output_position) \
                for _scores in scores
        ]
        scores = torch.cat(scores, dim=2)
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None].bool(), 0.).float()
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        output = []
        values, indices = torch.sort(scores, descending=True)
        for _indices, _values in zip(indices.cpu().numpy().tolist(), values.cpu().numpy().tolist()):
            output.append(
                [(pid, score) for pid, score in zip(_indices, _values)]
            )
        return output

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


def get_wrapped_encoder_forward(encoder_forward):
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = encoder_forward(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * passage_length, -1)
        return outputs
    return forward


def add_fns_to_deepspeed_model_engine(model_engine):
    model_engine.generate = types.MethodType(generate, model_engine)
    model_engine.reset_score_storage = types.MethodType(reset_score_storage, model_engine)
    model_engine.get_crossattention_scores = types.MethodType(get_crossattention_scores, model_engine)
    model_engine.overwrite_forward_crossattention = types.MethodType(overwrite_forward_crossattention, model_engine)


def generate(self, input_ids, attention_mask, decoder_input_ids, decoder_input_len):
    assert not self.module.training

    if self.zero_optimization_partition_weights():
        # Enable automated discovery of external parameters by indicating that
        # we are in a forward pass.
        for module in self.module.modules():
            module._parameters._in_forward = True
            pass

    output = self.module.generate(input_ids, attention_mask, decoder_input_ids, decoder_input_len)

    if self.zero_optimization_partition_weights():
        # Reset the ZeRO-3 state if we are only doing forward-passes (ie evaluation).
        if not torch._C.is_grad_enabled():
            self.optimizer.param_coordinator.reset_step()

        # Disable automated discovery of external parameters
        for module in self.module.modules():
            module._parameters._in_forward = False

    return output


def reset_score_storage(self):
    self.module.reset_score_storage()


def get_crossattention_scores(self, context_mask, score_at_output_position):
    return self.module.get_crossattention_scores(context_mask, score_at_output_position)


def overwrite_forward_crossattention(self):
    self.module.overwrite_forward_crossattention()


def cross_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    int_seq_length = int(seq_length)

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.training and self.gradient_checkpointing:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -int_seq_length:, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn_weights = F.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = F.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


def create_model_input(
    tokenizer,
    samples: List,
    n_contexts: int,
    max_context_length: int,
    max_answer_length: int,
    answer_in_encoder: bool,
    answer_in_decoder: bool,
    decoder_start_token_id: int,
    is_train: bool,
    batch_size: int,
    reduce_memory_usage: bool = True,
):
    """
    Creates a model batch instance out of a list of Sample-s
    :param samples: list of samples to create the batch for
    :param is_train: if the samples are for a train set
    :return: InputBatch instance
    """

    trigger_token_id = tokenizer.encode("is")[0]
    clusters = samples if is_train else list(chain(*[sample.clusters for sample in samples]))
    batches = []
    for start in range(0, len(clusters), batch_size):
        _clusters = clusters[start: start + batch_size]
        input_ids = []
        output_ids = []
        relevance_tag = [] if is_train else None
        for cluster in _clusters:
            assert isinstance(cluster, Cluster)
            for passage in cluster.passages[: n_contexts]:
                if passage.sequence_ids is None:
                    if answer_in_encoder:
                        sequence = "question: {} prediction: {} title: {} context: {}".format(cluster.question, cluster.prediction, passage.title, passage.passage_text)
                    else:
                        sequence = "question: {} title: {} context: {}".format(cluster.question, passage.title, passage.passage_text)
                    passage.sequence_ids = tokenizer.encode(
                        sequence,
                        add_special_tokens=True,
                        pad_to_max_length=True,
                        max_length=max_context_length,
                        truncation=True,
                        return_tensors='pt',
                    )
            input_ids.append(
                torch.cat([passage.sequence_ids for passage in cluster.passages[: n_contexts]], dim=0)
            )
            if reduce_memory_usage:
                for passage in cluster.passages[: n_contexts]:
                    passage.sequence_ids = None
            if answer_in_decoder:
                prefix_ids = ([decoder_start_token_id] + tokenizer.encode(cluster.prediction, add_special_tokens=False))[: max_answer_length - 1]
                output_ids.append(
                    torch.tensor(prefix_ids + [trigger_token_id], dtype=torch.long)
                )
            else:
                output_ids.append(
                    torch.tensor([decoder_start_token_id], dtype=torch.long)
                )
            if is_train:
                relevance_tag.append(cluster.relevance_tag)

        input_ids = torch.stack(input_ids, dim=0)
        decoder_input_ids = torch.empty((input_ids.size(0), max_answer_length), dtype=torch.long).fill_(tokenizer.pad_token_id)
        decoder_input_len = []
        for idx, ids in enumerate(output_ids):
            decoder_input_ids[idx, :len(ids)] = ids
            decoder_input_len.append(len(ids))
        decoder_input_len = torch.tensor(decoder_input_len, dtype=torch.long)
        if is_train:
            relevance_tag = torch.tensor(relevance_tag, dtype=torch.long)
        batches.append(
            InputBatch(
                input_ids=input_ids,
                attention_mask=input_ids.ne(tokenizer.pad_token_id).float(),
                decoder_input_ids=decoder_input_ids,
                decoder_input_len=decoder_input_len,
                relevance_tag=relevance_tag,
            )
        )

    if not hasattr(create_model_input, "log_io_format"):
        create_model_input.log_io_format = {
            'train': False,
            'test': False
        }
    key = 'train' if is_train else 'test'
    if not create_model_input.log_io_format[key]:
        batch = batches[0]
        for i in range(min(5, len(batch.input_ids))):
            for j, ctx in enumerate(batch.input_ids[i]):
                log("encoder input context {}: {}".format(j + 1, tokenizer.decode(ctx)))
            log("decoder input tokens: {}".format(tokenizer.decode(batch.decoder_input_ids[i])))
            log("decoder input length: {}".format(batch.decoder_input_len[i]))
            if is_train:
                log("relevance tag: {}".format(tokenizer.decode([batch.relevance_tag[i]])))
        create_model_input.log_io_format[key] = True

    return batches
