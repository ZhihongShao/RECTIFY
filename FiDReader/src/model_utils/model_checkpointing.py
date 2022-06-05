from genericpath import exists
import os
import shutil
from glob import glob

import torch

import deepspeed

from src.utils import log

def get_best_ckpt_tag(save_dir):
    best_ckpt_tag = ""
    if os.path.exists(os.path.join(save_dir, "best")):
        with open(os.path.join(save_dir, "best"), "r", encoding='utf-8') as file:
            best_ckpt_tag = file.readline().strip()
    return best_ckpt_tag

def save_best_ckpt_tag(save_dir, ckpt_tag):
    with open(os.path.join(save_dir, "best"), "w", encoding='utf-8') as file:
        file.write(ckpt_tag)

def get_latest_ckpt_tag(save_dir):
    best_ckpt_tag = ""
    if os.path.exists(os.path.join(save_dir, "latest")):
        with open(os.path.join(save_dir, "latest"), "r", encoding='utf-8') as file:
            best_ckpt_tag = file.readline().strip()
    return best_ckpt_tag

def save_latest_ckpt_tag(save_dir, ckpt_tag):
    with open(os.path.join(save_dir, "latest"), "w", encoding='utf-8') as file:
        file.write(ckpt_tag)

def _rotate_checkpoints(save_dir, max_to_keep=-1):
    if max_to_keep <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted(
        filter(os.path.isdir, glob(os.path.join(save_dir, "*"))),
        key=os.path.getmtime
    )
    # skip best checkpoint
    best_ckpt_tag = get_best_ckpt_tag(save_dir)
    if best_ckpt_tag is not None:
        checkpoints_sorted = [ckpt for ckpt in checkpoints_sorted if not ckpt.endswith(best_ckpt_tag)]

    if len(checkpoints_sorted) <= max_to_keep:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - max_to_keep)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        log("Deleting older checkpoint [{}] due to `max_to_keep`={}".format(checkpoint, max_to_keep))
        shutil.rmtree(checkpoint)

def get_fp32_model_path(args):
    load_dir = args.ckpt_dir if args.ckpt_dir is not None else os.path.join(args.output_dir, get_latest_ckpt_tag(args.output_dir))
    model_path = os.path.join(load_dir, "pytorch_model.bin")
    if not os.path.exists(model_path) and args.distributed.local_rank == 0:
        ckpt_dir = os.path.join(load_dir, get_latest_ckpt_tag(load_dir))
        if ckpt_dir != load_dir and os.path.exists(ckpt_dir):
            # TODO: recover ZeRO 2 or 3 models from optimizer states
            # if os.path.e  xists(os.path.join(load_dir, "zero_to_fp32.py")):
            #     log("Converting zero model to fp32 model")
            #     convert_script = os.path.join(os.path.abspath(os.path.dirname(__file__)), "zero_to_fp32.py")
            #     os.system(f"python {convert_script} {ckpt_dir} {model_path}")
            if not os.path.exists(model_path):
                log("Extracting non-zero model")
                state_dict = torch.load(list(filter(lambda x: x.endswith("model_states.pt"), glob(os.path.join(ckpt_dir, "*"))))[0], map_location='cpu')['module']
                for name in state_dict.keys():
                    param = state_dict[name]
                    if param.dtype == torch.half:
                        state_dict[name] = param.float()
                torch.save(state_dict, model_path)
    torch.distributed.barrier()
    if not os.path.exists(model_path):
        model_path = None
    return model_path

def load_checkpoint(args, model_engine, load_optimizer_states, load_lr_scheduler_states):
    load_dir = args.ckpt_dir if args.ckpt_dir is not None else os.path.join(args.output_dir, get_latest_ckpt_tag(args.output_dir))
    log("Loading checkpoint from {}".format(load_dir))
    # avoid loading optimizer states if not `load_optimizer_states`
    # TODO: not so helpful for memory usage reduction
    optim_cache_dir = None
    if not load_optimizer_states:
        if args.distributed.local_rank == 0:
            ckpt_dir = os.path.join(load_dir, get_latest_ckpt_tag(load_dir))
            optim_state_files = glob(os.path.join(ckpt_dir, "*optim_states.pt"))
            if optim_state_files:
                optim_cache_dir = os.path.join(ckpt_dir, "optim_caches")
                os.makedirs(optim_cache_dir, exist_ok=True)
                for fname in optim_state_files:
                    shutil.move(fname, optim_cache_dir)

        torch.distributed.barrier()

    load_path, client_state = model_engine.load_checkpoint(
        load_dir,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )
    if args.distributed.local_rank == 0 and optim_cache_dir is not None:
        for fname in glob(os.path.join(optim_cache_dir, "*")):
            shutil.move(fname, os.path.dirname(optim_cache_dir))
        shutil.rmtree(optim_cache_dir)
    torch.distributed.barrier()
    return load_path, client_state

def save_checkpoint(args, model_engine, save_dir, ckpt_tag, client_state={}, max_to_keep=-1, save_best=False):
    cp_path = os.path.join(save_dir, ckpt_tag)
    os.makedirs(cp_path, exist_ok=True)
    # TODO: recover ZeRO 2 or 3 model state dict from optimizer states
    # if model_engine.zero_optimization():
    #     param_name_groups = []
    #     param_groups = model_engine.optimizer._get_param_groups()
    #     if all('_param_names' in group for group in param_groups):
    #         world_size = torch.distributed.get_world_size()
    #         for group in param_groups:
    #             partition_param_names = {}
    #             for i, name in enumerate(group['_param_names']):
    #                 j = i % world_size
    #                 if not j in partition_param_names:
    #                     partition_param_names[j] = []
    #                 partition_param_names[j].append(name)
    #             reordered_names = []
    #             for _, names in partition_param_names.items():
    #                 reordered_names.extend(names)
    #             param_name_groups.append(reordered_names)
    #         client_state['param_name_groups'] = param_name_groups
    model_engine.save_checkpoint(cp_path, client_state=client_state, save_latest=True)
    if args.distributed.local_rank == 0:
        if save_best:
            save_best_ckpt_tag(save_dir, ckpt_tag)
        save_latest_ckpt_tag(save_dir, ckpt_tag)
        _rotate_checkpoints(save_dir, max_to_keep)
    torch.distributed.barrier()
    log("Saved checkpoint to {}".format(cp_path))
    if save_best:
        log("New best checkpoint {}".format(cp_path))
    return cp_path
