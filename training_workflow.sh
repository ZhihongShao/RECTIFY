export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

t5-base=<abs-path-to-your-cached-t5-base>
t5-3b=<abs-path-to-your-cached-t5-3b>
root_dir=<abs-path-to-this-directory>

# Answer recaller
# Pre-train a recaller on NQ
python train_recaller.py \
    --job_name recaller-t5-base \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-base} \
    --train.num_train_epochs 10 \
    --dataset NQ \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0 \
    --dataset.max_context_length 240 \
python train_recaller.py \
    --job_name recaller-t5-3b \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --train.num_train_epochs 10 \
    --dataset NQ \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0 \
    --dataset.max_context_length 240
# finetune the pre-trained recaller on AmbigQA
python train_recaller.py \
    --job_name recaller-neg-0.1-t5-base \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-base} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/recaller-t5-base/recaller_NQ_checkpoints/recaller_17000 \
    --train.num_train_epochs 20 \
    --dataset AmbigQA \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0.1 \
    --dataset.max_context_length 240
python train_recaller.py \
    --job_name recaller-neg-0.1-t5-3b \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/recaller-t5-3b/recaller_NQ_checkpoints/recaller_4000 \
    --train.num_train_epochs 20 \
    --train.early_stopping 6 \
    --dataset AmbigQA \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0.1 \
    --dataset.max_context_length 240
# finetune the pre-trained recaller on WebQSP
python train_recaller.py \
    --job_name recaller-neg-0.1-t5-base \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-base} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/recaller-t5-base/recaller_NQ_checkpoints/recaller_17000 \
    --train.num_train_epochs 80 \
    --train.early_stopping 6 \
    --dataset WebQSP \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0.1 \
    --dataset.max_context_length 240
python train_recaller.py \
    --job_name recaller-neg-0.1-t5-3b \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/recaller-t5-3b/recaller_NQ_checkpoints/recaller_4000 \
    --train.num_train_epochs 80 \
    --train.start_eval_epoch 20 \
    --train.early_stopping 6 \
    --dataset WebQSP \
    --dataset.n_contexts 100 \
    --dataset.num_neg_per_pos 0.1 \
    --dataset.max_context_length 240


# Evidence aggregator
# aggregate evidence for candidates on NQ
python aggregate.py \
    --dataset NQ \
    --recaller_job_name recaller-t5-base
# aggregate evidence for candidates on AmbigQA
python aggregate.py \
    --dataset AmbigQA \
    --recaller_job_name recaller-neg-0.1-t5-base
python aggregate.py \
    --dataset AmbigQA \
    --recaller_job_name recaller-neg-0.1-t5-3b
# aggregate evidence for candidates for WebQSP
python aggregate.py \
    --dataset WebQSP \
    --recaller_job_name recaller-neg-0.1-t5-base
python aggregate.py \
    --dataset WebQSP \
    --recaller_job_name recaller-neg-0.1-t5-3b


# Answer verifier
# Pre-train a verifier on NQ
python train_verifier.py \
    --job_name verifier \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --recaller_job_name recaller-t5-base \
    --train.num_train_epochs 3 \
    --train.start_eval_epoch 2 \
    --train.early_stopping 4 \
    --dataset NQ \
    --dataset.n_contexts 10 \
    --dataset.num_neg_per_pos 10 \
    --dataset.max_context_length 280
# finetune the pre-trained verifier on AmbigQA
python train_verifier.py \
    --job_name verifier \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/verifier_with_recaller-t5-base/verifier_NQ_checkpoints/verifier_38000 \
    --recaller_job_name recaller-neg-0.1-t5-base \
    --train.num_train_epochs 10 \
    --train.eval_step 500 \
    --dataset AmbigQA \
    --dataset.n_contexts 10 \
    --dataset.num_neg_per_pos 10 \
    --dataset.max_context_length 280
python train_verifier.py \
    --job_name verifier \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/verifier_with_recaller-t5-base/verifier_NQ_checkpoints/verifier_38000 \
    --recaller_job_name recaller-neg-0.1-t5-3b \
    --train.num_train_epochs 10 \
    --train.eval_step 500 \
    --dataset AmbigQA \
    --dataset.n_contexts 10 \
    --dataset.num_neg_per_pos 10 \
    --dataset.max_context_length 280
# finetune the pre-trained verifier on WebQSP
python train_verifier.py \
    --job_name verifier \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/verifier_with_recaller-t5-base/verifier_NQ_checkpoints/verifier_38000 \
    --recaller_job_name recaller-neg-0.1-t5-base \
    --train.num_train_epochs 30 \
    --train.start_eval_epoch 20 \
    --train.eval_step 500 \
    --dataset WebQSP \
    --dataset.n_contexts 10 \
    --dataset.num_neg_per_pos 10 \
    --dataset.max_context_length 280
python train_verifier.py.py \
    --job_name verifier \
    --do_train \
    --do_eval \
    --plm_name_or_path ${t5-3b} \
    --ckpt_dir ${root_dir}/FiDReader/outputs/NQ/verifier_with_recaller-t5-base/verifier_NQ_checkpoints/verifier_38000 \
    --recaller_job_name recaller-neg-0.1-t5-3b \
    --train.num_train_epochs 30 \
    --train.start_eval_epoch 20 \
    --train.eval_step 500 \
    --dataset WebQSP \
    --dataset.n_contexts 10 \
    --dataset.num_neg_per_pos 10 \
    --dataset.max_context_length 280
