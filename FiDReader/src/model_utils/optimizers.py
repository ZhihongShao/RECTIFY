import torch
from transformers import AdamW
# from apex.optimizers import FusedAdam

def get_params_for_weight_decay_optimization(module):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    no_decay = ["bias", "layernorm.weight", "layer_norm.weight"]
    weight_decay_named_params = list(zip(*[(n, p) for n, p in module.named_parameters() if not any(nd in n.lower() for nd in no_decay)]))
    weight_decay_params = {"params": weight_decay_named_params[1], "_param_names": weight_decay_named_params[0]}
    no_weight_decay_named_params = list(zip(*[(n, p) for n, p in module.named_parameters() if any(nd in n.lower() for nd in no_decay)]))
    no_weight_decay_params = {"params": no_weight_decay_named_params[1], "_param_names": no_weight_decay_named_params[0], "weight_decay": 0.0}
    return weight_decay_params, no_weight_decay_params

def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    param_groups = get_params_for_weight_decay_optimization(model)

    if args.train.cpu_optimizer:
        if args.train.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.train.lr,
                                       weight_decay=args.train.weight_decay)
    else:
        # torch AdamW has bugs for ZeRO 3
        optimizer = torch.optim.AdamW(param_groups,
        # transformers AdamW is untested for ZeRO 3
        # optimizer = AdamW(param_groups,
        # optimizer = FusedAdam(param_groups,
                        lr=args.train.lr,
                        weight_decay=args.train.weight_decay,
                        betas=eval(args.train.adam_betas),
                        eps=args.train.adam_eps)

    return optimizer