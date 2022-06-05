#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import time
import types
import collections
from typing import List
import types
import deepspeed
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from itertools import chain

from transformers import T5ForConditionalGeneration, T5Config
from transformers import TrainingArguments
from transformers.integrations import DeepSpeedConfigHF

from src.data.recaller_data import TrainSample, TestSample
from src.utils import log

InputBatch = collections.namedtuple(
    'InputBatch', 
        [
            "input_ids",
            "attention_mask",
            "labels",
        ]
)


class Recaller(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def generate(self, input_ids, attention_mask, max_length, tokenizer, cluster=False):
        """
        * Arguments:
            * input_ids (torch.LongTensor): [batch_size, n_passages, max_passage_length]
            * attention_mask (torch.FloatTensor): [batch_size, n_passages, max_passage_length]
            * max_length (int): max output length
            * tokenizer :
            * cluster (bool): whether to cluster passages
        * Returns:
            * if cluster == True, returns ^batch_size^ dict as follows
                {
                    "<answer:str>": [[<passage_id:int>, ]]
                }
            * if cluster == False, returns ^batch_size^ dict as follows
                {
                    "<answer:str>": [[<passage_id:int>, prob of the answer]]
                }
        """
        # gradient_checkpointing = self.decoder.config.gradient_checkpointing
        # self.set_checkpoint(False)
        batch_size, n_passages, passage_length = input_ids.size()
        input_ids = input_ids.view(batch_size * n_passages, passage_length)
        attention_mask = attention_mask.view(batch_size * n_passages, passage_length)

        # start = time.time()
        torch.cuda.empty_cache()
        outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, output_hidden_states=cluster, return_dict_in_generate=True)
        answer_cand_txts = [tokenizer.decode(answer, skip_special_tokens=False) for answer in outputs['sequences']]
        # print("generation elapse time : {}".format(time.time() - start), flush=True)

        irrelevant_answer = getattr(self.config, 'irrelevant_answer', 'irrelevant')
        answer_separator = getattr(self.config, 'answer_separator', '<pad>')
        _sample_idx2answer_cand_txts = []
        for all_answers in np.reshape(answer_cand_txts, [batch_size, n_passages]):
            tmp = {}
            for pid, answers in enumerate(all_answers):
                if irrelevant_answer in answers:
                    continue
                answers = [answer.strip() for answer in answers.split(answer_separator) if answer.strip() and answer.strip() != tokenizer.eos_token]
                for aid, answer in enumerate(answers):
                    if len(answers) > 1 and aid == len(answers) - 1 and not answer.endswith(tokenizer.eos_token):
                        continue
                    answer = re.sub(tokenizer.eos_token, "", answer)
                    if answer not in tmp:
                        tmp[answer] = []
                    tmp[answer].append([pid])
            if not tmp:
                tmp[irrelevant_answer] = [[0]]
            _sample_idx2answer_cand_txts.append(tmp)
        if not cluster:
            return _sample_idx2answer_cand_txts
        else:
            encoder_hidden_states = outputs['encoder_hidden_states'][-1].cpu()
            del outputs
            torch.cuda.empty_cache()
            sample_idx2answer_cand_txts = []
            tile_answer_cand_txts = []
            for item in _sample_idx2answer_cand_txts:
                sample_idx2answer_cand_txts.append(list(item.keys()))
                tile_answer_cand_txts.extend([sample_idx2answer_cand_txts[-1]] * n_passages)
            flat_answer_cand_txts = list(chain(*tile_answer_cand_txts))

            target = tokenizer.batch_encode_plus(
                flat_answer_cand_txts,
                add_special_tokens=True,
                # max_length=max_length,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True,
            )
            _max_length = target['input_ids'].size(-1)
            target_ids = target["input_ids"]
            target_mask = (target["attention_mask"].bool() & target_ids.ne(tokenizer.eos_token_id)) # not counting the eos_token
            target_ids = target_ids.masked_fill(~target_mask, -100)

            _target_ids = []
            for sid, answer_cand_txts in enumerate(sample_idx2answer_cand_txts):
                if answer_cand_txts:
                    block = target_ids[: n_passages * len(answer_cand_txts)].view(n_passages, len(answer_cand_txts), _max_length)
                    target_ids = target_ids[n_passages * len(answer_cand_txts):]
                    for plane in block:
                        _target_ids.append(plane)
                else:
                    for _ in range(n_passages):
                        _target_ids.append([])

            # start = time.time()
            answer_cands_nll = []
            cluster_batch_size = getattr(self.config, 'cluster_batch_size', 10)
            for s in range(0, input_ids.size(0), cluster_batch_size):
                answer_cands_nll.append(
                    self.score_answer_candidates2(
                        input_ids=input_ids[s: s + cluster_batch_size],
                        attention_mask=attention_mask[s: s + cluster_batch_size],
                        encoder_outputs=[encoder_hidden_states[s: s + cluster_batch_size]],
                        labels=[_target_ids[i] for i in range(s, s + cluster_batch_size) if i < len(_target_ids)],
                    )
                )
            if len(answer_cands_nll) == 1:
                answer_cands_nll = answer_cands_nll[0]
            else:
                answer_cands_nll = torch.cat(answer_cands_nll, 0)
            # answer_cands_nll = answer_cands_nll.view(batch_size, n_passages, n_passages)
            answer_cands_log_prob = -answer_cands_nll
            # print("score elapse time : {}".format(time.time() - start), flush=True)

            # start = time.time()
            output = []
            for sid, all_answers in enumerate(sample_idx2answer_cand_txts):
                tmp = {}
                if len(all_answers):
                    all_answers_log_prob = answer_cands_log_prob[:n_passages * len(all_answers)].view(n_passages, len(all_answers))
                    answer_cands_log_prob = answer_cands_log_prob[n_passages * len(all_answers): ]
                    for aid, answer in enumerate(all_answers):
                        answer_log_prob = all_answers_log_prob[:, aid]
                        values, indices = torch.sort(answer_log_prob, descending=True)
                        tmp[answer] = []
                        for pid, lp in zip(indices.cpu().numpy().tolist(), values.cpu().numpy().tolist()):
                            tmp[answer].append([pid, lp])
                if not tmp:
                    tmp[irrelevant_answer] = [[0, 0.0]]
                output.append(tmp)
            # print("output elapse time : {}".format(time.time() - start), flush=True)
        # self.set_checkpoint(gradient_checkpointing)
        return output

    @torch.no_grad()
    def score_answer_candidates(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0]

        assert labels is not None
        batch_size, n_answers, max_length = labels.size()
        _labels = labels.view(batch_size * n_answers, max_length)

        # start = time.time()
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(_labels)
        decoder_input_ids = decoder_input_ids.view(batch_size, n_answers * max_length)
        causal_mask = (torch.arange(max_length).unsqueeze(1) >= torch.arange(max_length).unsqueeze(0)).float().cuda()
        decoder_attention_mask = torch.zeros((n_answers * max_length, n_answers * max_length), dtype=torch.float32).cuda()
        for chunk_id in range(n_answers):
            decoder_attention_mask[chunk_id * max_length: (chunk_id + 1) * max_length, chunk_id * max_length: (chunk_id + 1) * max_length] = causal_mask
        decoder_attention_mask = decoder_attention_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        labels = labels.view(batch_size, n_answers * max_length)
        labels_mask = (labels > -100).float()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        nll = torch.sum((loss_fct(lm_logits.transpose(2, 1), labels) * labels_mask).view(batch_size, n_answers, max_length), dim=2)
        # print("decoding score elapse time : {}".format(time.time() - start), flush=True)
        return nll

    @torch.no_grad()
    def score_answer_candidates2(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        assert labels is not None

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        hidden_states = encoder_outputs[0].cuda()
        _, encoder_seq_len, encoder_hidden_dim = hidden_states.size()

        hidden_states_indices = []
        _labels = []
        for idx, answers in enumerate(labels):
            hidden_states_indices.extend([idx] * len(answers))
            if len(answers):
                _labels.append(answers)
        _labels = torch.cat(_labels, dim=0)
        _labels = _labels.cuda()
        hidden_states = hidden_states[hidden_states_indices]
        if attention_mask is not None:
            attention_mask = attention_mask[hidden_states_indices]

        # start = time.time()
        # get decoder inputs from shifting lm labels to the right
        max_length = _labels.size(-1)
        decoder_input_ids = self._shift_right(_labels)
        causal_mask = (torch.arange(max_length).unsqueeze(1) >= torch.arange(max_length).unsqueeze(0)).float().cuda()

        decoder_attention_mask = causal_mask.unsqueeze(0).expand(len(_labels), -1, -1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        labels_mask = (_labels > -100).float()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        nll = torch.sum((loss_fct(lm_logits.transpose(2, 1), _labels) * labels_mask), dim=1)
        # print("decoding score elapse time : {}".format(time.time() - start), flush=True)
        return nll

    @classmethod
    def get_init_model(cls, pretrained_model_name_or_path, deepspeed_config, model_path=None):
        t5_config = T5Config.from_pretrained(pretrained_model_name_or_path)
        hf_args = TrainingArguments(output_dir=None, deepspeed=deepspeed_config)
        hf_ds_cfg = DeepSpeedConfigHF(hf_args)
        model = cls.from_pretrained(model_path or pretrained_model_name_or_path, config=t5_config)
        model.hf_ds_cfg = hf_ds_cfg
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


def add_generation_fn_to_deepspeed_model_engine(model_engine):
    model_engine.generate = types.MethodType(generate, model_engine)


def generate(self, input_ids, attention_mask, max_length, tokenizer, cluster=False):
    assert not self.module.training

    if self.zero_optimization_partition_weights():
        # Enable automated discovery of external parameters by indicating that
        # we are in a forward pass.
        for module in self.module.modules():
            module._parameters._in_forward = True
            pass

    output = self.module.generate(input_ids, attention_mask, max_length, tokenizer, cluster)

    if self.zero_optimization_partition_weights():
        # Reset the ZeRO-3 state if we are only doing forward-passes (ie evaluation).
        if not torch._C.is_grad_enabled():
            self.optimizer.param_coordinator.reset_step()

        # Disable automated discovery of external parameters
        for module in self.module.modules():
            module._parameters._in_forward = False

    return output


def create_model_input(
    tokenizer,
    samples: List,
    max_context_length: int,
    max_answer_length: int,
    is_train: bool,
    shuffle_answers: bool,
    max_num_answers: bool,
    answer_separator: str
) -> InputBatch:
    """
    Creates a model batch instance out of a list of Sample-s
    :param samples: list of samples to create the batch for
    :param is_train: if the samples are for a train set
    :return: InputBatch instance
    """

    input_ids = []
    output_texts = [] if is_train else None
    for sample in samples:
        if is_train:
            assert isinstance(sample, TrainSample)
            passages = [sample.passage]
        else:
            assert isinstance(sample, TestSample)
            passages = sample.passages
        for passage in passages:
            if passage.sequence_ids is None:
                sequence = "question: {} title: {} context: {}".format(sample.question, passage.title, passage.passage_text)
                passage.sequence_ids = tokenizer.encode(
                    sequence,
                    add_special_tokens=True,
                    pad_to_max_length=True,
                    max_length=max_context_length,
                    truncation=True,
                    return_tensors='pt',
                )
        if is_train:
            input_ids.append(sample.passage.sequence_ids)
            # indices = list(range(len(sample.answers)))
            # np.random.shuffle(indices)
            indices = list(range(len(sample.answers)))[1:]
            np.random.shuffle(indices)
            indices = [0] + indices
            num_out_answers = len(indices) if max_num_answers == -1 else max_num_answers
            answers = []
            for idx in indices[:num_out_answers]:
                choice = np.random.choice(range(len(sample.answers[idx])))
                answers.append(sample.answers[idx][choice])
            if not shuffle_answers:
                answers = [item[0] for item in sorted(answers, key=lambda x: x[1])]
            else:
                answers = [item[0] for item in answers]
            output_texts.append(answer_separator.join(answers))
        else:
            input_ids.append(
                torch.cat([passage.sequence_ids for passage in sample.passages], dim=0)
            )
        for passage in passages:
            passage.sequence_ids = None

    if is_train:
        input_ids = torch.cat(input_ids, dim=0)
        target = tokenizer.batch_encode_plus(
            output_texts,
            add_special_tokens=True, # T5Tokenizer does not add eos token or bos token
            padding='max_length',
            max_length=max_answer_length,
            truncation=True,
            return_tensors='pt',
        )
        labels = target['input_ids']
        labels = labels.masked_fill(~target['attention_mask'].bool(), -100)
    else:
        input_ids = torch.stack(input_ids, dim=0)
        labels = None

    batch = InputBatch(
        input_ids=input_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id).float(),
        labels=labels,
    )
    if not hasattr(create_model_input, "log_io_format"):
        create_model_input.log_io_format = {
            'train': False,
            'test': False
        }
    key = 'train' if is_train else 'test'
    if not create_model_input.log_io_format[key]:
        if is_train:
            output_labels = batch.labels.masked_fill((batch.labels < 0), tokenizer.pad_token_id)
        for i in range(min(5, len(batch.input_ids))):
            input_ids = batch.input_ids[i] if is_train else batch.input_ids[i][0]
            log("encoder input context: {}".format(tokenizer.decode(input_ids)))
            if is_train:
                log("decoder input tokens: {}".format(tokenizer.decode(output_labels[i])))
        create_model_input.log_io_format[key] = True
    return batch
