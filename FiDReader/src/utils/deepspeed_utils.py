import json

def ds_config_check(args):
    if hasattr(args, 'deepspeed_config'):
        ds_config = json.load(open(args.deepspeed_config, "r", encoding='utf-8'))
        assert ds_config.get("train_micro_batch_size_per_gpu", 1) == args.train.batch_size, \
            "Training batch size mismatch"
        assert ds_config.get("gradient_accumulation_steps", 1) == args.train.gradient_accumulation_steps, \
            "Gradient accumulation steps mismatch"
        if "zero_optimization" in ds_config:
            assert ds_config["zero_optimization"].get("stage", 0) == args.distributed.zero_stage, \
                "ZeRO stage mismatch"
        