#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import math
from glob import glob
import collections
from functools import reduce
from itertools import chain
import json
import sys
import shutil

import hydra
import numpy as np
import os
import torch
from tqdm import tqdm

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict

import deepspeed
from deepspeed.utils import init_distributed
import argparse
from transformers import T5TokenizerFast

from src.data.verifier_data import (
    set_default_preprocessing_cfg,
    VerifierDataset,
)
from src.models import init_verifier_components
from src.models.verifier import create_model_input, InputBatch

from src.model_utils import (
    save_checkpoint,
    move_to_device,
)
from src.utils import (
    all_gather_list,
    reduce_losses,
    set_seed,
    make_data_loader,
    config_logger,
    log,
    log_on_tensorboard,
    ds_config_check,
)
from src.utils.rerank_evaluate_script import (
    calculate_matches,
    _normalize_answer,
)

ModelPredictions = collections.namedtuple(
    "ModelPredictions", ["id", "predictions"]
)


class Trainer(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        log("***** Initializing components *****")

        self.start_epoch = 0
        self.start_batch = 0
        self.step = 0
        self.num_worse_validation = 0
        self.tokenizer = T5TokenizerFast.from_pretrained(cfg.plm_name_or_path)

        if cfg.train_files is not None:
            _, cfg.train.max_steps = self.get_train_data()
        # self.model, self.optimizer, self.scheduler, client_state = init_model_components(cfg, init_optimizer_and_lr_scheduler=(cfg.train_files is not None))
        self.model, self.optimizer, self.scheduler, client_state = init_verifier_components(cfg, init_optimizer_and_lr_scheduler=True)

        if cfg.train_files is not None:
            self.train_iterator, _ = self.get_train_data()

        self.best_validation_result = None

        self.thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1000]

        if client_state is not None and cfg.train.continue_training:
            self.load_training_cfgs(client_state)


    def get_train_data(self):
        cfg = self.cfg
        train_iterator = self.get_data_iterator(
            cfg.train_files,
            is_train=True,
            shuffle_seed=cfg.seed,
        )
        iterations_per_epoch = len(train_iterator)
        updates_per_epoch = (
            iterations_per_epoch // cfg.train.gradient_accumulation_steps
        )
        total_updates = (
            iterations_per_epoch * cfg.train.num_train_epochs // cfg.train.gradient_accumulation_steps
        )
        log("Total iterations per epoch={}".format(iterations_per_epoch))
        log("Updates per epoch={}".format(updates_per_epoch))
        log("Total updates={}".format(total_updates))
        if cfg.train.max_steps is None:
            return train_iterator, total_updates
        else:
            return train_iterator, cfg.train.max_steps


    def get_data_iterator(
        self,
        path: str,
        is_train: bool,
        shuffle_seed: int = 0,
    ):
        cfg = self.cfg

        run_preprocessing = (self.cfg.distributed.rank == 0)

        set_default_preprocessing_cfg(
            multi_answers=cfg.dataset.multi_answers,
            n_contexts=cfg.dataset.n_contexts,
            irrelevant_token_id=cfg.dataset.irrelevant_token_id,
            relevant_token_id=cfg.dataset.relevant_token_id,
            num_neg_per_pos=cfg.dataset.num_neg_per_pos,
            train_split_answers=cfg.dataset.extend_alternative_answers_for_training,
        )
        dataset = VerifierDataset(
            path,
            is_train,
            self.tokenizer,
            run_preprocessing,
            self.cfg.num_workers,
        )

        dataset.load_data(
            num_shards=cfg.dataset.eval_num_shards if not is_train else 1,
            shard_id=cfg.dataset.eval_shard_id if not is_train else 0,
        )

        batch_size = cfg.train.batch_size if is_train else cfg.train.dev_batch_size

        iterator = make_data_loader(
            dataset,
            is_train=is_train,
            batch_size=batch_size,
            drop_last=False,
            shuffle_seed=shuffle_seed,
        )

        return iterator


    def run_train(self):
        cfg = self.cfg

        eval_step = cfg.train.eval_step
        log("  Eval step = {}".format(eval_step))
        log("***** Training *****")

        for epoch in range(self.start_epoch, cfg.train.num_train_epochs):
            if cfg.train.max_steps is not None and self.step > cfg.train.max_steps * cfg.train.gradient_accumulation_steps:
                break
            if cfg.train.early_stopping is not None and self.num_worse_validation >= cfg.train.early_stopping:
                break
            log("***** Epoch {} *****".format(epoch))
            self._train_epoch(
                epoch, eval_step, self.train_iterator
            )
            self.train_iterator.dataset.gen_train_samples(epoch + 1)

        log("Training finished.")


    def validate_and_save(self, epoch: int, iteration: int):
        validation_result = self.validate()

        def _is_better(curr_validation_result, prev_validation_result):
            if prev_validation_result is None:
                return True
            else:
                keys = ['all-ans-f1', 'multi-ans-f1']
                prev_best_thres = None
                curr_best_thres = None
                prev_best_f1_all_plus_multi = None
                curr_best_f1_all_plus_multi = None
                for thres in self.thresholds:
                    key = 'all-ans-f1 (thres={})'.format(thres)
                    if prev_best_f1_all_plus_multi is None or prev_best_f1_all_plus_multi < sum([prev_validation_result["{} (thres={})".format(key, thres)] for key in keys]):
                        prev_best_f1_all_plus_multi = sum([prev_validation_result["{} (thres={})".format(key, thres)] for key in keys])
                        prev_best_thres = thres
                    if curr_best_f1_all_plus_multi is None or curr_best_f1_all_plus_multi < sum([curr_validation_result["{} (thres={})".format(key, thres)] for key in keys]):
                        curr_best_f1_all_plus_multi = sum([curr_validation_result["{} (thres={})".format(key, thres)] for key in keys])
                        curr_best_thres = thres
                prev_metrics = [sum([prev_validation_result["{} (thres={})".format(key, prev_best_thres)] for key in keys]), prev_validation_result["multi-ans-f1 (thres={})".format(prev_best_thres)], prev_validation_result["all-ans-num (thres={})".format(prev_best_thres)]]
                curr_metrics = [sum([curr_validation_result["{} (thres={})".format(key, curr_best_thres)] for key in keys]), curr_validation_result["multi-ans-f1 (thres={})".format(curr_best_thres)], curr_validation_result["all-ans-num (thres={})".format(curr_best_thres)]]
                return curr_metrics > prev_metrics

        save_best = False
        self.num_worse_validation += 1

        if _is_better(validation_result, self.best_validation_result):
            save_best = True
            self.best_validation_result = validation_result
            self.num_worse_validation = 0
        cp_path = self._save_checkpoint(epoch, iteration, save_best)
        if os.path.exists(self.cfg.prediction_results_file) and self.cfg.distributed.rank in [0, -1]:
            shutil.move(self.cfg.prediction_results_file, os.path.join(cp_path, os.path.split(self.cfg.prediction_results_file)[-1]))


    def validate(self):
        log("Validation ...")
        cfg = self.cfg
        self.model.eval()

        if hasattr(self, "dev_iterator"):
            data_iterator = self.dev_iterator
        else:
            data_iterator = self.get_data_iterator(
                cfg.dev_files,
                is_train=False,
            )
            self.dev_iterator = data_iterator

        all_results = []
        processed_ids = set()

        for samples_batch in tqdm(data_iterator, desc='Validation', total=len(data_iterator), disable=(cfg.distributed.rank != 0)):
            inputs = create_model_input(
                self.tokenizer,
                samples_batch,
                cfg.dataset.n_contexts,
                cfg.dataset.max_context_length,
                cfg.dataset.max_answer_length,
                cfg.dataset.answer_in_encoder,
                cfg.dataset.answer_in_decoder,
                self.model.module.config.decoder_start_token_id,
                is_train=False,
                batch_size=cfg.train.dev_batch_size,
            )

            relevance_scores = []
            for input in inputs:
                input = InputBatch(**move_to_device(input._asdict(), torch.device(cfg.distributed.device)))
                output = self.model.generate(input.input_ids, input.attention_mask, input.decoder_input_ids, input.decoder_input_len)
                relevance_scores.append(output)
            
            relevance_scores = torch.cat(relevance_scores, dim=0).cpu().numpy().tolist()
            clusters = list(chain(*[sample.clusters for sample in samples_batch]))
            assert len(relevance_scores) == len(clusters)
            for cluster, score in zip(clusters, relevance_scores):
                cluster.relevance_tag = score
            for sample in samples_batch:
                if sample.id in processed_ids:
                    continue
                processed_ids.add(sample.id)
                predictions = sorted(
                    [
                        (cluster.prediction, cluster.relevance_tag) for cluster in sample.clusters
                    ],
                    key=lambda x: -x[1][1]
                )
                nws = set()
                thres2predictions = {}
                for pred in predictions:
                    nw = _normalize_answer(pred[0])
                    if nw not in nws:
                        for thres in self.thresholds:
                            if thres not in thres2predictions:
                                thres2predictions[thres] = []
                            if (thres >= 0 and pred[1][1] > thres) or (thres == -1 and pred[1][1] > pred[1][0]) or (thres == 1000 and len(nws) == 0):
                                thres2predictions[thres].append(pred)
                        nws.add(nw)
                for thres in thres2predictions:
                    if not thres2predictions[thres]:
                        thres2predictions[thres].append(predictions[0])
                all_results.append(
                    ModelPredictions(
                        id=sample.id,
                        predictions=thres2predictions
                    )
                )

        validation_result = {}
        for thres in self.thresholds:
            raw_samples = []
            for result in all_results:
                raw_samples.append(data_iterator.dataset._id2raw_sample[result.id])
            _validation_result = calculate_matches(
                raw_samples,
                workers_num=-1,
                predictions=[[item[0] for item in result.predictions[thres]] for result in all_results],
                verbose=False,
            )
            for key, val in _validation_result.items():
                if key in ['n_samples', 'n_samples (multi)'] or 'MRECALL' in key:
                    assert val == validation_result.get(key, val)
                else:
                    key = key + " (thres={})".format(thres)
                validation_result[key] = val

        def log_validation_results(validation_result, is_local):
            prefix = "rank-{}".format(cfg.distributed.rank) if is_local else "global"
            for key, value in validation_result.items():
                log("{} > {} = {}".format(prefix, key, value), log_on_rank_0_only=(not is_local))
            log_on_tensorboard(
                self.step // cfg.train.gradient_accumulation_steps,
                **{"eval/{}/".format(prefix) + key: val for key, val in validation_result.items()}
            )

        if cfg.distributed.world_size > 1:
            log_validation_results(validation_result, is_local=True)
            gathered_results = all_gather_list(validation_result, max_size=cfg.distributed.global_loss_buf_sz)
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
            global_validation_result = validation_result
        log_validation_results(global_validation_result, is_local=False)

        if cfg.prediction_results_file:
            self._save_predictions(cfg.prediction_results_file, all_results)

        return global_validation_result

    def _train_epoch(
        self,
        epoch,
        eval_step,
        train_data_iterator,
    ):
        cfg = self.cfg
        log_result_step = cfg.train.log_batch_step
        gradient_accumulation_steps = cfg.train.gradient_accumulation_steps

        self.model.train()
        train_data_iterator.sampler.set_epoch(epoch)

        tr_loss = 0
        tr_loss_breakdown = {}

        for i, samples_batch in tqdm(enumerate(train_data_iterator), total=len(train_data_iterator), desc='Training', disable=(cfg.distributed.rank != 0)):
            if epoch == self.start_epoch and i < self.start_batch:
                continue

            inputs = create_model_input(
                self.tokenizer,
                samples_batch,
                cfg.dataset.n_contexts,
                cfg.dataset.max_context_length,
                cfg.dataset.max_answer_length,
                cfg.dataset.answer_in_encoder,
                cfg.dataset.answer_in_decoder,
                self.model.module.config.decoder_start_token_id,
                is_train=True,
                batch_size=cfg.train.batch_size,
            )
            assert len(inputs) == 1
            input = inputs[0]

            loss, reduced_loss_breakdown_dict = self._calc_loss(input)
            self.model.backward(loss)
            self.model.step()

            self.step += 1

            tr_loss += loss.clone().detach()
            for key, loss in reduced_loss_breakdown_dict.items():
                key = "train/{}".format(key)
                tr_loss_breakdown[key] = tr_loss_breakdown.get(key, 0) + loss

            if self.step % (log_result_step * gradient_accumulation_steps) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                tr_loss /= (gradient_accumulation_steps * log_result_step)
                for key, loss in tr_loss_breakdown.items():
                    tr_loss_breakdown[key] = loss / gradient_accumulation_steps / log_result_step
                log(
                    "Rank = {} > Epoch: {}: Step: {}/{}, global_step={}, lr={}, loss = {}".format(
                        cfg.distributed.rank,
                        epoch,
                        i + 1,
                        len(train_data_iterator),
                        self.step // gradient_accumulation_steps,
                        lr,
                        tr_loss,
                    )
                )
                log_on_tensorboard(
                    self.step // gradient_accumulation_steps,
                    **tr_loss_breakdown,
                )
                tr_loss = 0
                tr_loss_breakdown = {}

            if epoch >= cfg.train.start_eval_epoch and self.step >= (cfg.train.start_eval_step * gradient_accumulation_steps) and self.step % (eval_step * gradient_accumulation_steps) == 0:
                self.validate_and_save(epoch, i + 1)
                self.model.train()

                if cfg.train.early_stopping is not None and self.num_worse_validation >= cfg.train.early_stopping:
                    return

            if cfg.train.max_steps is not None and self.step > (cfg.train.max_steps * gradient_accumulation_steps):
                return

    def _save_checkpoint(self, epoch: int, offset: int, save_best: bool) -> str:
        cfg = self.cfg
        client_state = {
            'epoch': epoch,
            'offset': offset,
            'step': self.step,
            'num_worse_validation': self.num_worse_validation,
            'best_validation_result': self.best_validation_result,
        }
        global_step = self.step // cfg.train.gradient_accumulation_steps
        ckpt_tag = "{}_{}".format(cfg.ckpt_tag_prefix, global_step) if cfg.ckpt_tag_prefix else str(global_step)
        return save_checkpoint(
            cfg,
            self.model,
            save_dir=cfg.output_dir,
            ckpt_tag=ckpt_tag,
            client_state=client_state,
            max_to_keep=cfg.train.max_to_keep,
            save_best=save_best,
        )


    def load_training_cfgs(self, client_state):
        self.start_epoch = client_state['epoch']
        self.start_batch = client_state['offset']
        self.step = client_state['step']
        self.num_worse_validation = client_state['num_worse_validation']
        self.best_validation_result = client_state['best_validation_result']
        log("Loading checkpoint @ batch={} and epoch={}".format(self.start_batch, self.start_epoch))


    def _calc_loss(self, input: InputBatch, calc_reduced_losses: bool = True):
        cfg = self.cfg
        input = InputBatch(**move_to_device(input._asdict(), torch.device(cfg.distributed.device)))

        loss = self.model(
            input_ids=input.input_ids,
            attention_mask=input.attention_mask,
            decoder_input_ids=input.decoder_input_ids,
            decoder_input_len=input.decoder_input_len,
            relevance_tag=input.relevance_tag,
        )[0]

        if calc_reduced_losses:
            _loss = reduce_losses([loss], max_size=cfg.distributed.global_loss_buf_sz)[0]
        else:
            _loss = loss.clone().detach()
        reduced_loss_breakdown_dict = {
            'loss': _loss,
        }

        return loss, reduced_loss_breakdown_dict


    def _save_predictions(
        self, out_file: str, prediction_results: List[ModelPredictions]
    ):
        cfg = self.cfg
        if cfg.distributed.world_size > 1:
            local_out_file = out_file + ".rank{}".format(cfg.distributed.rank)
        else:
            local_out_file = out_file
        with open(local_out_file, "w", encoding="utf-8") as output:
            save_results = {}
            for r in prediction_results:
                save_results[r.id] = r.predictions
            json.dump(save_results, output, indent=4)
        # To merge local_out_files, why not use all_gather_list
        if cfg.distributed.world_size > 1:
            torch.distributed.barrier()
            if cfg.distributed.rank == 0:
                local_rank_files = glob(out_file + ".rank*")
                all_data = {}
                for local_rank_file in local_rank_files:
                    with open(local_rank_file, "r", encoding="utf-8") as src:
                        all_data.update(json.load(src))
                    os.remove(local_rank_file)
                with open(out_file, "w", encoding="utf-8") as file:
                    json.dump(all_data, file, indent=4)
            torch.distributed.barrier()
        log("Save prediction results to {}".format(out_file))


@hydra.main(config_path="conf", config_name="model_cfg")
def main(cfg: DictConfig):
    ds_config_check(cfg)
    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg.distributed.world_size = int(os.getenv("WORLD_SIZE", 1))
    cfg.distributed.rank = int(os.getenv("RANK", -1))
    cfg.distributed.local_rank = int(os.getenv("LOCAL_RANK", -1))
    assert cfg.distributed.local_rank != -1, \
        "This script should be launched via distributed launcher"
    cfg.distributed.device = cfg.distributed.local_rank

    init_distributed("nccl")
    set_seed(cfg)
    config_logger(log_on_rank_0_only=cfg.log_on_rank_0_only)

    if cfg.distributed.local_rank == 0:
        log("CFG (after gpu configuration):")
        log("{}".format(OmegaConf.to_yaml(cfg)))

    trainer = Trainer(cfg)

    if cfg.train_files is not None:
        trainer.run_train()
    elif cfg.dev_files:
        log("No train files are specified. Run validation.")
        trainer.validate()
    else:
        log(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )


if __name__ == "__main__":
    log("Sys.argv: {}".format(sys.argv))
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    args, unparse = parser.parse_known_args()
    cs = ConfigStore.instance()
    cs.store(name='model_cfg', node=args.__dict__)
    hydra_formatted_args = [__file__]
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in unparse:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    log("Hydra formatted Sys.argv: {}".format(hydra_formatted_args))
    sys.argv = hydra_formatted_args
    main()
