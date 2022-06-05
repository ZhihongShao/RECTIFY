import os
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--recaller_job_name", type=str, default="recaller")
parser.add_argument("--dataset", type=str, default="AmbigQA")
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
    target_dir = os.path.join(root_dir, "DistilledDPR_and_FiD")
    os.chdir(target_dir)

    for split in ['train', 'dev', 'test']:
        logger.info("Aggregating evidence for {}-{}".format(args.dataset, split))
        if args.dataset == 'NQ':
            src_data_path = os.path.join(args.data_dir, "retrieval_results", "distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
        else:
            src_data_path = os.path.join(args.data_dir, "retrieval_results", "finetuned_distilled_dpr", args.dataset, "retrieved_{}.json".format(split))
        cluster_data_path = os.path.join(args.data_dir, "RECTIFY", args.dataset, "recalling_results", args.recaller_job_name, "{}.jsonl".format(split))
        passages_path = os.path.join(args.data_dir, "wikipedia_split", "psgs_w100.tsv")
        model_path = os.path.join(root_dir, "DPR_and_ExtractiveReader", "outputs", args.dataset, "retriever_checkpoints", "best_ckpt")
        embeds_dir = os.path.join(root_dir, "DPR_and_ExtractiveReader", "outputs", args.dataset, "wikipedia_embeddings")
        output_dir = os.path.join(args.data_dir, "RECTIFY", args.dataset, "aggregation_results", args.recaller_job_name)
        output_path = os.path.join(output_dir, "{}.jsonl".format(split))
        n_rounds = 25 if args.dataset == 'NQ' and split == 'train' else 5
        cmd = f"python rerank.py " \
            f"--src_data {src_data_path} " \
            f"--cluster_data {cluster_data_path} " \
            f"--model_path {model_path} " \
            f"--passages {passages_path} " \
            f"--passages_embeddings \"{embeds_dir}/wiki_*\" " \
            f"--output_path {output_path} " \
            f"--n_reranked_docs 100 " \
            f"--n_rounds {n_rounds} "
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
        cmd = f"python finalize_reranking.py " \
            f"--dirname {output_dir} "
        logger.info("Applying cmd: {}".format(cmd))
        os.system(cmd)
        if args.dataset == 'NQ':
            cmd = f"python fix_answers.py " \
                f"--src_fname {cluster_data_path} " \
                f"--tar_fname {output_path} "
            logger.info("Applying cmd: {}".format(cmd))
            os.system(cmd)

if __name__ == '__main__':
    main()
