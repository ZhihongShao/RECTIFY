## Dependencies

1. Set `prefix` to your own path in `RECTIFY.yml`
2. Create the conda environment with the following command
   ```bash
    conda env create -f RECTIFY.yml
    ```

# Train a Recall-then-Verify System

* Suppose the dataset used for pre-training is called `<pretrain_dataset>`, such as `NQ`, and the target multi-answer dataset is called `<dataset>`, such as `AmbigQA`, `WebQSP`, or a name of your own dataset.

* Configuration files for training on this two dataset should be placed under `FiDReader/conf/dataset` and `FiDReader/conf/deepspeed_configs`; we have created examples there.

## Answer Recalling

### Pre-train a recaller on `<pretrain_dataset>`

* The following command will pre-train a recaller called `<pretrained_recaller>` on `<pretrain_dataset>`, save checkpoints under `FiDReader/outputs/<pretrain_dataset>/<pretrained_recaller>/recaller_<pretrain_dataset>_checkpoints`, and save answer recalling results with the best checkpoint under `data/<pretrain_dataset>/recalling_results/<pretrained_recaller>`.

    ```bash
    python train_recaller.py \
        --job_name <pretrained_recaller> \ # this can be any string that uniquely tags the recaller, such as `recaller_neg-0.1_t5-base` which denotes a recaller trained with t5-base and $\alpha_{neg}=0.1$
        --do_train \
        --do_eval \
        --plm_name_or_path <plm_name_or_path> \ # this is the path to your cached pre-trained LM which will be used to initialize the recaller
        --dataset <pretrain_dataset> \
        ...
    ```

### Finetune the pre-trained recaller on `<dataset>`

* The following command will finetune the pre-trained checkpoint called `<pretrained_recaller>` on `<dataset>`, save checkpoints under `FiDReader/outputs/<dataset>/<finetuned_recaller>/recaller_<dataset>_checkpoints`, and save answer recalling results with the best checkpoint under `data/<dataset>/recalling_results/<finetuned_recaller>`.

    ```bash
    python train_recaller.py \
        --job_name <finetuned_recaller> \
        --do_train \
        --do_eval \
        --plm_name_or_path <plm_name_or_path> \
        --ckpt_dir <absolute_path_to_the_best_checkpoint> \ # specify the best pre-trained recaller here which will be used for initialization
        --dataset <dataset> \
        ...
    ```

## Evidence Aggregation

* Please switch to the `DPR` environment for evidence aggregation

### Aggregate evidence for recalled candidates on `<pretrain_dataset>`

* The following command will aggregate evidence for answer candidates recalled by `<pretrained_recaller>` on `<pretrain_dataset>`, and save aggregation results under `data/<pretrain_dataset>/aggregation_results/<pretrained_recaller>`.

    ```bash
    python aggregate.py \
        --dataset <pretrain_dataset> \
        --recaller_job_name <pretrained_recaller>
    ```

### Aggregate evidence for recalled candidates on `<dataset>`

* The following command will aggregate evidence for answer candidates recalled by `<finetuned_recaller>` on `<dataset>`, and save aggregation results under `data/<dataset>/aggregation_results/<recaller>`.

    ```bash
    python aggregate.py \
        --dataset <dataset> \
        --recaller_job_name <finetuned_recaller>
    ```

## Answer Verification

* Switch back to the `RECTIFY` environment

### Pre-train a verifier on `<pretrain_dataset>`

* The following command will pre-train a verifier on `<pretrain_dataset>`, save checkpoints under `FiDReader/outputs/<pretrain_dataset>/<pretrained_verifier>_with_<pretrained_recaller>/verifier_<pretrain_dataset>_checkpoints`, and save answer verification results with the best checkpoint under `data/<pretrain_dataset>/verification_results/<pretrained_recaller>`.

    ```bash
    python train_verifier.py \
        --job_name <pretrained_verifier> \ # this can be any string that uniquely tags the verifier
        --recaller_job_name <pretrained_recaller> \ # this is the pre-trained recaller paired with the verifier
        --do_train \
        --do_eval \
        --plm_name_or_path <plm_name_or_path> \
        --dataset <pretrain_dataset> \
        ...
    ```

### Finetune the pre-trained verifier on `<dataset>`

* The following command will pre-train a verifier on `<dataset>`, save checkpoints under `FiDReader/outputs/<dataset>/<finetuned_verifier>_with_<finetuned_recaller>/verifier_<dataset>_checkpoints`, and save answer verification results with the best checkpoint under `data/<dataset>/verification_results/<finetuned_recaller>`.

    ```bash
    python train_verifier.py \
        --job_name <finetuned_verifier> \
        --recaller_job_name <finetuned_recaller> \
        --do_train \
        --do_eval \
        --plm_name_or_path <plm_name_or_path> \
        --ckpt_dir <absolute_path_to_the_best_checkpoint> \ # specify the best pre-trained verifier here which will be used for initialization
        --dataset <dataset> \
        ...
    ```

Please refer to `training_workflow.sh` for the detailed commands we used for training on `AmbigQA` and `WebQSP`.
