import os
import argparse
import logging
import glob
import shutil
import json
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--job_name", type=str, default="recaller")
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

    target_dir = os.path.join(root_dir, "FiDReader")

    def get_best_ds_ckpt_tag(save_dir):
        return open(os.path.join(save_dir, "best"), "r", encoding='utf-8').readline().strip()

    def get_ds_config_path(conf_dir, dataset, zero_stage):
        if zero_stage <= 1:
            ds_config_path = os.path.join(conf_dir, "{}.json".format(dataset.lower()))
        else:
            ds_config_path = os.path.join(conf_dir, "{}_zero{}.json".format(dataset.lower(), zero_stage))
        return ds_config_path

    ds_config_path = get_ds_config_path(os.path.join(target_dir, "conf", "deepspeed_configs", "recaller"), args.dataset, args.zero_stage)
    ds_config = json.load(open(ds_config_path, "r", encoding='utf-8'))
    batch_size = ds_config['train_micro_batch_size_per_gpu']

    if args.do_train:
        logger.info("Prepare data for recaller training ...")
        for split in ['train', 'dev', 'test']:
            if args.dataset == 'NQ':
                retrieval_result_path = os.path.join(args.data_dir, "retrieval_results", "distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
            else:
                retrieval_result_path = os.path.join(args.data_dir, "retrieval_results", "finetuned_distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
            target_path = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recaller", "{}.json".format(split))
            cmd = f"python preprocess_for_recaller_training.py " \
                f"--retrieval_result_path={retrieval_result_path} " \
                f"--target_path={target_path} " \
                f"--n_contexts=100 "
            logger.info("Applying cmd: {}".format(cmd))
            os.system(cmd)
        logger.info("Done preparation.")
        os.chdir(target_dir)
        logger.info("Training recaller on {} ...".format(args.dataset))
        train_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recaller", "train.json")
        dev_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recaller", "dev.json")
        hydra_train_record_dir = os.path.join(target_dir, "outputs", args.dataset, args.job_name)
        cmd = f"deepspeed {ds_args} train_recaller.py " \
            f"hydra.run.dir={hydra_train_record_dir} " \
            f"train_files={train_file} " \
            f"dev_files={dev_file} " \
            f"output_dir=recaller_{args.dataset}_checkpoints " \
            f"prediction_results_file=recaller_{args.dataset}_checkpoints/recaller_dev.jsonl " \
            f"plm_name_or_path={args.plm_name_or_path} " \
            f"ckpt_tag_prefix=recaller " \
            f"train.batch_size={batch_size} " \
            f"train.dev_batch_size=3 " \
            f"distributed.zero_stage={args.zero_stage} " \
            f"dataset={args.dataset.lower()} " \
            f"--deepspeed " \
            f"--deepspeed_config {ds_config_path} "
        cmd = add_unparsed_args(cmd, unparse)
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
        logger.info("Done training recaller on {}.".format(args.dataset))

    if args.do_eval:
        split2num_shards = {'train': 4}
        save_dir = os.path.join(target_dir, "outputs", args.dataset, args.job_name, "recaller_{}_checkpoints".format(args.dataset))
        best_ckpt_path = os.path.join(save_dir, get_best_ds_ckpt_tag(save_dir))
        output_dir = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recalling_results", args.job_name)
        for split in ['dev', 'test', 'train']:
            test_file = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recaller", "{}.json".format(split))
            num_shards = split2num_shards.get(split, 1)
            for shard_id in range(num_shards):
                predict_file = os.path.join(output_dir, "{}.jsonl.shard_{}".format(split, shard_id))
                hydra_eval_record_dir = os.path.join(target_dir, "outputs", args.dataset, args.job_name + "_eval_{}_{}".format(split, shard_id))
                cmd = f"deepspeed {ds_args} train_recaller.py " \
                    f"hydra.run.dir={hydra_eval_record_dir} " \
                    f"dev_files={test_file} " \
                    f"ckpt_dir={best_ckpt_path} " \
                    f"output_dir={output_dir} " \
                    f"prediction_results_file={predict_file} " \
                    f"plm_name_or_path={args.plm_name_or_path} " \
                    f"train.batch_size={batch_size} " \
                    f"train.dev_batch_size=3 " \
                    f"dataset={args.dataset.lower()} " \
                    f"dataset.eval_num_shards={num_shards} " \
                    f"dataset.eval_shard_id={shard_id} " \
                    f"distributed.zero_stage={args.zero_stage} " \
                    f"--deepspeed " \
                    f"--deepspeed_config {ds_config_path} "
                cmd = add_unparsed_args(cmd, unparse)
                logger.info("Applying cmd: {}".format(cmd))
                os.system(cmd)
            if num_shards <= 1:
                shutil.move(os.path.join(output_dir, "{}.jsonl.shard_0".format(split)), \
                    os.path.join(output_dir, "{}.jsonl".format(split)))
            else:
                predict_files = glob.glob(os.path.join(output_dir, "{}.jsonl.shard_*".format(split)))
                with open(os.path.join(output_dir, "{}.jsonl".format(split)), "w", encoding='utf-8') as file:
                    for fname in predict_files:
                        with open(fname, "r", encoding='utf-8') as src:
                            for line in src:
                                print(line.strip(), file=file)
                        os.remove(fname)
        logger.info("Done evaluation.")

if __name__ == '__main__':
    main()
