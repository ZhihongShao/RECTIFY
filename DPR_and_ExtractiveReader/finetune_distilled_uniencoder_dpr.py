#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import logging
import math
import os
import random
import sys
import time
from typing import Tuple

import numpy as np
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
from torch._C import device
from torch.nn.functional import normalize

from transformers import BertTokenizer
from transformers.file_utils import is_tf_available

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoderNllLoss
from dpr.models.hf_models import get_optimizer

from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.utils.conf_utils import BiencoderDatasetsCfg
from dpr.utils.data_utils import (
    ShardedDataIterator,
    MultiSetDataIterator,
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)

from dpr.data.distill_uniencoder_dpr_data import (
    RetrieverBatch,
    RetrieverCollator,
)

distilled_dpr_code_base = os.path.join(os.getcwd(), "../DistilledDPR_and_FiD")
sys.path.append(distilled_dpr_code_base)
from src.model import (
    Retriever,
    RetrieverConfig,
)
from src.util import (
    set_dropout,
    set_optim,
    load,
)
from src.index import Indexer
from passage_retrieval import add_embeddings
from rerank_evaluate_script import calculate_matches

logger = logging.getLogger()
setup_logger(logger)


class RetrieverTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, cfg: DictConfig):
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1
        self.retriever = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)
        self.collator = RetrieverCollator(
            self.tokenizer,
            cfg.passage_maxlength,
            cfg.question_maxlength,
        )

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_class = Retriever
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        if model_file:
            retriever_config = RetrieverConfig(
                indexing_dimension=cfg.indexing_dimension,
                apply_question_mask=not cfg.no_question_mask,
                apply_passage_mask=not cfg.no_passage_mask,
                extract_cls=cfg.extract_cls,
                projection=not cfg.no_projection,
            )
            self.retriever = model_class(retriever_config, initialize_wBERT=True)
            set_dropout(self.retriever, cfg.dropout)

            saved_state = load_states_from_checkpoint(model_file)
            # set_cfg_params_from_state(saved_state.encoder_params, cfg)
            self._load_saved_state(saved_state)
        else:
            self.retriever = model_class.from_pretrained(cfg.init_checkpoint_path)
            logger.info(f"Model loaded from {cfg.init_checkpoint_path}")

        self.optimizer = get_optimizer(
            self.retriever,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )

        self.retriever, self.optimizer = setup_for_distributed_mode(
            self.retriever,
            self.optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.retriever = (self.retriever.module if hasattr(self.retriever, 'module') else self.retriever)

        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):

        hydra_datasets = (
            self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        )
        sampling_rates = self.ds_cfg.sampling_rates

        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names
            if is_train_set
            else self.ds_cfg.dev_datasets_names,
        )

        # randomized data loading to avoid file system congestion
        datasets_list = [ds for ds in hydra_datasets]
        rnd = random.Random(rank)
        rnd.shuffle(datasets_list)
        [ds.load_data() for ds in datasets_list]

        sharded_iterators = [
            ShardedDataIterator(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
                strict_batch_size=True,
                drop_last=False,
            )
            for ds in hydra_datasets
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            rank=rank,
        )

    def run_train(self):
        cfg = self.cfg

        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            # TODO: ideally we'd want to just call
            # scheduler.load_state_dict(self.scheduler_state)
            # but it doesn't work properly as of now

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(
                self.optimizer, warmup_steps, total_updates
            )

        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.local_rank in [-1, 0]

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        new_best = False
        if not cfg.dev_datasets:
            validation_loss = 0
        else:
            if epoch >= cfg.val_av_rank_start_epoch:
                def _is_better(curr_validation_result, prev_validation_result):
                    if prev_validation_result is None:
                        return True
                    keys = []
                    for top_k in [100, 40, 30, 20, 10, 5]:
                        for prefix in ['all', 'multi']:
                            keys.append('{}-MRECALL@{}'.format(prefix, top_k))
                    cur_metrics = [curr_validation_result[key] for key in keys]
                    prev_metrics = [prev_validation_result[key] for key in keys]
                    return cur_metrics > prev_metrics
                validation_loss = self.validate_average_rank()
                new_best = _is_better(validation_loss, self.best_validation_result)
            else:
                validation_loss = self.validate_nll()
                new_best = validation_loss < (self.best_validation_result or validation_loss + 1)

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration, new_best)
            logger.info("Saved checkpoint to %s", cp_name)

            if new_best:
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_nll(self) -> float:
        logger.info("(Local rank = %d) NLL validation ...", self.cfg.local_rank)
        cfg = self.cfg
        self.retriever.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        log_result_step = cfg.train.log_batch_step
        batches = 0
        dataset = 0

        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            # logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank)
            retriever_input = self.collator(
                samples_batch,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.dev_datasets[dataset]

            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.retriever,
                retriever_input,
                self.tokenizer,
                cfg,
            )
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "(Local rank = %d) Eval step: %d , used_time=%f sec., loss=%f ",
                    cfg.local_rank,
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "(Local rank = %d) NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            cfg.local_rank,
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("(Local rank = %d) Average rank validation ...", self.cfg.local_rank)

        cfg = self.cfg
        self.retriever.eval()

        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        dev_dataset = self.ds_cfg.dev_datasets[0] # assume multi-dataset iterator contains only one dataset

        q_represenations = []
        ctx_represenations = []

        samples, all_ctxs, questions, ctxs = dev_dataset.get_sharded_ranking_data(max(cfg.local_rank, 0), self.distributed_factor)
        for start_batch in range(0, len(questions), cfg.train.val_av_rank_bsz):
            q_ids, q_attn_mask, _ = self.collator.convert_question_to_ids_and_mask(questions[start_batch: start_batch + cfg.train.val_av_rank_bsz])
            inp = move_to_device(
                {
                    'ids': q_ids,
                    'mask': q_attn_mask,
                },
                cfg.device
            )
            with torch.no_grad():
                q_dense = self.retriever.embed_text(
                    inp['ids'],
                    inp['mask'],
                    apply_mask=self.retriever.config.apply_question_mask,
                    extract_cls=self.retriever.config.extract_cls,
                )
            q_represenations.extend(q_dense.cpu().split(1, dim=0))
        for start_batch in range(0, len(ctxs), cfg.train.val_av_rank_bsz):
            ctx_ids_batch, ctx_attn_mask, _ = self.collator.convert_passage_to_ids_and_mask(ctxs[start_batch: start_batch + cfg.train.val_av_rank_bsz])
            inp = move_to_device(
                {
                    'ids': ctx_ids_batch,
                    'mask': ctx_attn_mask,
                },
                cfg.device
            )
            with torch.no_grad():
                ctx_dense = self.retriever.embed_text(
                    inp['ids'],
                    inp['mask'],
                    apply_mask=self.retriever.config.apply_passage_mask,
                    extract_cls=self.retriever.config.extract_cls,
                )

            ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

        index = Indexer(self.retriever.config.indexing_dimension)

        ctx_represenations = torch.cat(ctx_represenations, dim=0).cpu().numpy()
        ctx_ids = list(range(len(ctx_represenations)))
        q_represenations = torch.cat(q_represenations, dim=0).cpu().numpy()

        while ctx_represenations.shape[0] > 0:
            ctx_represenations, ctx_ids = add_embeddings(index, ctx_represenations, ctx_ids, 50000)

        logger.info(
            "(Local rank = %d) Av.rank validation: total q_vectors size=%s",
            cfg.local_rank, len(q_represenations)
        )
        logger.info(
            "(Local rank = %d) Av.rank validation: total ctx_vectors size=%s",
            cfg.local_rank, len(ctx_represenations)
        )

        top_ids_and_scores = index.search_knn(q_represenations, 100) 

        for sample, ranks in zip(samples, top_ids_and_scores):
            indices = list(map(int, ranks[0]))
            ctxs = [all_ctxs[idx] for idx in indices]
            sample['ctxs'] = ctxs
        local_validation_result = calculate_matches(samples, workers_num=-1, verbose=False)

        def log_validation_results(validation_result, is_local):
            prefix = "rank-{}".format(cfg.local_rank) if is_local else "global"
            for key, value in validation_result.items():
                logger.info("{} > {} = {}".format(prefix, key, value))

        if self.distributed_factor > 1:
            log_validation_results(local_validation_result, is_local=True)
            gathered_results = all_gather_list(local_validation_result, max_size=cfg.global_loss_buf_sz)
            global_validation_result = {}
            for i, item in enumerate(gathered_results):
                global_validation_result['n_samples'] = global_validation_result.get('n_samples', 0) + item['n_samples']
                global_validation_result['n_samples (multi)'] = global_validation_result.get('n_samples (multi)', 0) + item['n_samples (multi)']
                for key, val in item.items():
                    if key.startswith('all-'):
                        global_validation_result[key] = global_validation_result.get(key, 0) + val * item['n_samples']
                    elif key.startswith('multi-'):
                        global_validation_result[key] = global_validation_result.get(key, 0) + val * item['n_samples (multi)']
            for key, val in global_validation_result.items():
                if key.startswith('all-'):
                    global_validation_result[key] /= max(global_validation_result['n_samples'], 1)
                elif key.startswith('multi-'):
                    global_validation_result[key] /= max(global_validation_result['n_samples (multi)'], 1)
        else:
            global_validation_result = local_validation_result
        if cfg.local_rank in [-1, 0]:
            log_validation_results(global_validation_result, is_local=False)
        return global_validation_result

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):

        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        seed = cfg.seed
        self.retriever.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        dataset = 0
        if epoch == 0:
            self.retriever.eval()
            self.validate_and_save(epoch, 0, scheduler)
            self.retriever.train()
        for i, samples_batch in enumerate(
            train_data_iterator.iterate_ds_data(epoch=epoch)
        ):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            ds_cfg = self.ds_cfg.train_datasets[dataset]
            shuffle_positives = ds_cfg.shuffle_positives

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            retriever_batch = self.collator(
                samples_batch,
                num_hard_negatives,
                num_other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
            )

            loss_scale = (
                cfg.loss_scale_factors[dataset] if cfg.loss_scale_factors else None
            )
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.retriever,
                retriever_batch,
                self.tokenizer,
                cfg,
                loss_scale=loss_scale,
            )

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if cfg.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(self.optimizer), cfg.train.max_grad_norm
                    )
            else:
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.retriever.parameters(), cfg.train.max_grad_norm
                    )

            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.retriever.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "(Local rank = %d) Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("(Local rank = %d) Train batch %d", cfg.local_rank, data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "(Local rank = %d) Avg. loss per last %d batches: %f",
                    cfg.local_rank,
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                logger.info(
                    "(Local rank = %d) Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(
                    epoch, train_data_iterator.get_iteration(), scheduler
                )
                self.retriever.train()

        logger.info("Epoch finished on %d", cfg.local_rank)
        if data_iteration % eval_step != 0:
            self.validate_and_save(epoch, data_iteration, scheduler)
            self.retriever.train()

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("(Local rank = %d) Av Loss per epoch=%f", cfg.local_rank, epoch_loss)
        logger.info("(Local rank = %d) epoch total correct predictions=%d", cfg.local_rank, epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int, is_new_best: bool) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.retriever)
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch) + "-" + str(offset))
        meta_params = get_encoder_params_state_from_cfg(cfg)
        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        if is_new_best:
            best_ckpt_dir = os.path.join(cfg.output_dir, "best_ckpt")
            model_to_save.save_pretrained(best_ckpt_dir)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        if self.cfg.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.retriever)
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state)

        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                logger.info("Loading saved optimizer state ...")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(
    cfg,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = cfg.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
        normalize=True,
    )

    return loss, is_correct


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: RetrieverBatch,
    tokenizer,
    cfg,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]:

    input = RetrieverBatch(**move_to_device(input._asdict(), cfg.device))

    if model.training:
        local_q_vector = model.embed_text(
            input.question_ids,
            input.question_mask,
            apply_mask=model.config.apply_question_mask,
            extract_cls=model.config.extract_cls,
        )
        local_ctx_vectors = model.embed_text(
            input.passage_ids,
            input.passage_mask,
            apply_mask=model.config.apply_passage_mask,
            extract_cls=model.config.extract_cls
        )
    else:
        with torch.no_grad():
            local_q_vector = model.embed_text(
                input.question_ids,
                input.question_mask,
                apply_mask=model.config.apply_question_mask,
                extract_cls=model.config.extract_cls,
            )
            local_ctx_vectors = model.embed_text(
                input.passage_ids,
                input.passage_mask,
                apply_mask=model.config.apply_passage_mask,
                extract_cls=model.config.extract_cls
            )

    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(
        cfg,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.positives,
        input.hard_negatives,
        loss_scale=loss_scale,
    )

    is_correct = is_correct.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.gradient_accumulation_steps
    return loss, is_correct


@hydra.main(config_path="conf", config_name="distill_uniencoder_dpr_train_cfg")
def main(cfg: DictConfig):
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = RetrieverTrainer(cfg)

    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    elif cfg.model_file and cfg.dev_datasets:
        logger.info(
            "No train files are specified. Run 2 types of validation for specified model file"
        )
        trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
