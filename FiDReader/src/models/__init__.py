import deepspeed
from torch.optim import lr_scheduler

from src.models import recaller, verifier
from src.model_utils import get_optimizer
from src.model_utils import get_learning_rate_scheduler
from src.model_utils import load_checkpoint, get_fp32_model_path

def init_model_components(model_cls, args, init_optimizer_and_lr_scheduler):
    model_path = get_fp32_model_path(args) if not args.train.continue_training else None
    with deepspeed.zero.Init(remote_device=args.distributed.remote_device,
                             config=args.deepspeed_config,
                             pin_memory=args.distributed.use_pin_memory,
                             enabled=args.distributed.zero_stage==3):
        model = model_cls.get_init_model(args.plm_name_or_path, args.deepspeed_config, model_path)
        model.set_checkpoint(args.train.use_checkpoint)
    if init_optimizer_and_lr_scheduler:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer = None
        lr_scheduler = None
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    if args.train.continue_training:
        load_path, client_state = load_checkpoint(
            args,
            model_engine,
            load_optimizer_states=args.train.continue_training,
            load_lr_scheduler_states=args.train.continue_training,
        )
    else:
        client_state = {}
    return model_engine, model_engine.optimizer, model_engine.lr_scheduler, client_state

def init_recaller_components(args, init_optimizer_and_lr_scheduler):
    model_engine, optimizer, lr_scheduler, client_state = init_model_components(recaller.Recaller, args, init_optimizer_and_lr_scheduler)
    model_engine.module.config.irrelevant_answer = args.dataset.irrelevant_answer
    model_engine.module.config.answer_separator = args.dataset.answer_separator
    model_engine.module.config.cluster_batch_size = args.cluster_batch_size
    recaller.add_generation_fn_to_deepspeed_model_engine(model_engine)
    return model_engine, optimizer, lr_scheduler, client_state

def init_verifier_components(args, init_optimizer_and_lr_scheduler):
    model_engine, optimizer, lr_scheduler, client_state = init_model_components(verifier.Verifier, args, init_optimizer_and_lr_scheduler)
    model_engine.module.config.relevant_token_id = args.dataset.relevant_token_id
    model_engine.module.config.irrelevant_token_id = args.dataset.irrelevant_token_id
    verifier.add_fns_to_deepspeed_model_engine(model_engine)
    return model_engine, optimizer, lr_scheduler, client_state
