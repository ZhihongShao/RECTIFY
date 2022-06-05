import os
import argparse
import logging
import glob
import shutil
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--job_name", type=str, default="verifier")
parser.add_argument("--recaller_job_name", type=str, default="recaller")
parser.add_argument("--dataset", type=str, default="AmbigQA")
parser.add_argument("--plm_name_or_path", type=str, default="")
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--do_eval", action='store_true')
parser.add_argument("--zero_stage", type=int, default=2)
parser.add_argument("--master_port", type=str, default="10000")
args, unparse = parser.parse_known_args()

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
for attr in dir(args):
    if attr.endswith("_dir"):
        os.makedirs(getattr(args, attr), exist_ok=True)

def add_unparsed_args(cmd, unparse):
    assert len(unparse) % 2 == 0, "Parse error"
    for i in range(0, len(unparse), 2):
        cmd += " {}={}".format(
            unparse[i][2:] if unparse[i].startswith("--") else unparse[i],
            unparse[i + 1]
        )
    return cmd

def main():
    cuda_device_cnt = torch.cuda.device_count()
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        CUDA_VISIBLE_DEVICES = list(map(int, os.getenv('CUDA_VISIBLE_DEVICES').split(",")))
        os.environ.pop('CUDA_VISIBLE_DEVICES')
    else:
        CUDA_VISIBLE_DEVICES = list(range(cuda_device_cnt))
    gpus_str = ",".join(map(str, CUDA_VISIBLE_DEVICES))
    ds_args = "--include=localhost:{} ".format(gpus_str)
    if args.master_port:
        ds_args += "--master_port={}".format(args.master_port)

    hydra_job_record_dir = None
    target_dir = os.path.join(root_dir, "FiDReader")

    def get_best_ds_ckpt_tag(save_dir):
        return open(os.path.join(save_dir, "best"), "r", encoding='utf-8').readline().strip()

    def get_ds_config_path(conf_dir, dataset, zero_stage):
        if zero_stage <= 1:
            ds_config_path = os.path.join(conf_dir, "{}.json".format(dataset.lower()))
        else:
            ds_config_path = os.path.join(conf_dir, "{}_zero{}.json".format(dataset.lower(), zero_stage))
        return ds_config_path

    ds_config_path = get_ds_config_path(os.path.join(target_dir, "conf", "deepspeed_configs", "verifier"), args.dataset, args.zero_stage)
    ds_config = json.load(open(ds_config_path, "r", encoding='utf-8'))
    batch_size = ds_config['train_micro_batch_size_per_gpu']

    os.chdir(target_dir)
    if args.do_train:
        logger.info("Training verifier on {} ...".format(args.dataset))
        train_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "aggregation_results", args.recaller_job_name, "train.jsonl")
        dev_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "aggregation_results", args.recaller_job_name, "dev.jsonl")
        hydra_train_record_dir = os.path.join(target_dir, "outputs", args.dataset, "{}_with_{}".format(args.job_name, args.recaller_job_name))
        cmd = f"deepspeed {ds_args} train_verifier.py " \
            f"hydra.run.dir={hydra_train_record_dir} " \
            f"train_files={train_file} " \
            f"dev_files={dev_file} " \
            f"output_dir=verifier_{args.dataset}_checkpoints " \
            f"prediction_results_file=verifier_{args.dataset}_checkpoints/verifier_dev.json " \
            f"plm_name_or_path={args.plm_name_or_path} " \
            f"ckpt_tag_prefix=verifier " \
            f"train.batch_size={batch_size} " \
            f"train.dev_batch_size=8 " \
            f"distributed.zero_stage={args.zero_stage} " \
            f"dataset={args.dataset.lower()} " \
            f"--deepspeed " \
            f"--deepspeed_config {ds_config_path} "
        cmd = add_unparsed_args(cmd, unparse)
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
        logger.info("{}Done training verifier on {}.".format(args.dataset))

    if args.do_eval:
        logger.info("Inference & evaluation with trained verifier ...")
        save_dir = os.path.join(target_dir, "outputs", args.dataset, "{}_with_{}".format(args.job_name, args.recaller_job_name), "verifier_{}_checkpoints".format(args.dataset))
        best_ckpt_path = os.path.join(save_dir, get_best_ds_ckpt_tag(save_dir))
        output_dir = os.path.join(args.data_dir, "RECTIFY", args.dataset, "verification_results", args.recaller_job_name)
        for split in ['dev', 'test']:
            test_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "aggregation_results", args.recaller_job_name, "{}.jsonl".format(split))
            hydra_eval_record_dir = os.path.join(target_dir, "outputs", args.dataset, "{}_with_{}_eval_{}".format(args.job_name, args.recaller_job_name, split))
            cmd = f"deepspeed {ds_args} train_verifier.py " \
                f"hydra.run.dir={hydra_eval_record_dir} " \
                f"dev_files={test_file} " \
                f"ckpt_dir={best_ckpt_path} " \
                f"output_dir={output_dir} " \
                f"prediction_results_file={output_dir}/verified_{split}.json " \
                f"plm_name_or_path={args.plm_name_or_path} " \
                f"train.batch_size={batch_size} " \
                f"train.dev_batch_size=8 " \
                f"dataset={args.dataset.lower()} " \
                f"distributed.zero_stage={args.zero_stage} " \
                f"--deepspeed " \
                f"--deepspeed_config {ds_config_path} "
            cmd = add_unparsed_args(cmd, unparse)
            logger.info("Applying cmd: {}".format( cmd))
            os.system(cmd)
        logger.info("Done inference & evaluation.")

if __name__ == '__main__':
    main()
