import os
import argparse
import logging
from time import localtime
import torch
from subprocess import Popen

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="AmbigQA")
args, _ = parser.parse_known_args()

def setup_logger(logger):
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_formatter = logging.Formatter(
        "[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(log_formatter)
    logger.addHandler(console)

logger = logging.getLogger()
setup_logger(logger)

root_dir = os.path.abspath(os.path.dirname(__file__))
args.data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
os.makedirs(args.data_dir, exist_ok=True)

def main():
    os.chdir(root_dir)
    logger.info("Preparing data for finetuning distilled dpr ...")
    for split in ['train', 'dev', 'test']:
        src_path = os.path.join(args.data_dir, "retrieval_results", "distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
        tar_path = os.path.join(args.data_dir, "retriever", "finetune_distilled_dpr", args.dataset, "{}.json".format(split))
        if os.path.exists(tar_path):
            continue
        cmd = f"python preprocess_for_distilled_dpr_finetuning.py " \
              f"--dataset {src_path} " \
              f"--out_file {tar_path} "
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
    logger.info("Done data preparation.")

    cuda_device_cnt = torch.cuda.device_count()
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        CUDA_VISIBLE_DEVICES = list(map(int, os.getenv('CUDA_VISIBLE_DEVICES').split(",")))
    else:
        CUDA_VISIBLE_DEVICES = list(range(cuda_device_cnt))

    logger.info("Finetuning distilled dpr ...")
    target_dir = os.path.join(root_dir, "DPR_and_ExtractiveReader")
    hydra_train_record_dir = os.path.join(target_dir, "outputs", args.dataset)
    os.chdir(target_dir)
    init_checkpoint_path = os.path.join(args.data_dir, "retriever/finetune_distilled_dpr/init_checkpoints/nq_retriever")
    cmd = f"python -m torch.distributed.launch --nproc_per_node={cuda_device_cnt} --master_port=10000 finetune_distilled_uniencoder_dpr.py " \
          f"hydra.run.dir={hydra_train_record_dir} " \
          f"train=biencoder_dpr_nq " \
          f"train_datasets=[{args.dataset.lower()}_train] " \
          f"dev_datasets=[{args.dataset.lower()}_dev] " \
          f"output_dir=retriever_checkpoints " \
          f"init_checkpoint_path={init_checkpoint_path} " \
          f"checkpoint_file_name=distill_uniencoder_dpr " \
          f"train.num_train_epochs=40 " \
          f"train.batch_size=16 " \
          f"train.hard_negatives=2 " \
          f"train.other_negatives=0 " \
          f"train.eval_per_epoch=2 " \
          f"val_av_rank_start_epoch=0 "
    logger.info("Applying cmd: {}".format(cmd))
    os.system(cmd)
    logger.info("Done finetuning distilled dpr.")

    best_ckpt_dir = os.path.join(hydra_train_record_dir, "retriever_checkpoints", "best_ckpt")

    logger.info("Running retrieval with finetuned distill dpr ...")
    target_dir = os.path.join(root_dir, "DistilledDPR_and_FiD")
    os.chdir(target_dir)

    logger.info("Generating embeddings ...")
    num_shards = 50
    embeds_dir = os.path.join(hydra_train_record_dir, "wikipedia_embeddings")
    passages_file = os.path.join(args.data_dir, "wikipedia_split/psgs_w100.tsv")
    embed_file_prefix = os.path.join(embeds_dir, "wiki")
    shard_id = 0
    while shard_id < num_shards:
        cmds = []
        for idx, gpu_id in enumerate(CUDA_VISIBLE_DEVICES):
            if shard_id + idx >= num_shards:
                continue
            cmd = f"python generate_passage_embeddings.py " \
                  f"--model_path {best_ckpt_dir} " \
                  f"--passages {passages_file} " \
                  f"--output_path {embed_file_prefix} " \
                  f"--shard_id {shard_id + idx} " \
                  f"--num_shards {num_shards} " \
                  f"--per_gpu_batch_size 1000 " \
                  f"--gpu_ids {gpu_id} "
            cmds.append(cmd.split())
        procs = [Popen(cmd) for cmd in cmds]
        for proc in procs:
            proc.wait()
        shard_id += cuda_device_cnt

    logger.info("Running retrieval ...")
    embed_files = os.path.join(embeds_dir, "wiki_*")
    for split in ['dev', 'test', 'train']:
        output_path = os.path.join(args.data_dir, "retrieval_results", "finetuned_distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
        data_file = os.path.join(args.data_dir, "datasets/{}/{}.json".format(args.dataset, split))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES[0]} python passage_retrieval.py " \
                f"--model_path {best_ckpt_dir} " \
                f"--passages {passages_file} " \
                f"--data {data_file} " \
                f"--passages_embeddings \"{embed_files}\" " \
                f"--validation_workers 1 " \
                f"--output_path {output_path} " \
                f"--n-docs 100 "
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
    logger.info("Done retrieval.")

    logger.info("Evaluating retrieved results ...")
    for split in ['train', 'dev', 'test']:
        output_path = os.path.join(args.data_dir, "retrieval_results", "finetuned_distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
        cmd = f"python rerank_evaluate_script.py " \
                f"--src_filename {output_path} "
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
    logger.info("Done evaluation.")


if __name__ == '__main__':
    main()
