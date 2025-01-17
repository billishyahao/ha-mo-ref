#!/usr/bin/env python
# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, print_stats, is_hpu, hpu_mark_step
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...',
                        required=True)
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument("--dropout",
                        type=float,
                        default=None,
                        help="If provided, use as model dropout. "
                             "Both --dropout and --disable_dropout are not allowed.")
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')

    # tensorboard
    parser.add_argument("--tb_output_dir",
                        type=str,
                        default=None,
                        help="Tensorboard output files root dir.")
    parser.add_argument("--tb_job_name",
                        type=str,
                        default=None,
                        help="Tensorboard job name. If not provided, tensorboard is disabled.")

    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument('--bf16',
                        action='store_true',
                        help='Use bf16.')
    parser.add_argument('--no_bf16_to_fp32_loss',
                        action='store_false',
                        dest='bf16_to_fp32_loss',
                        help='Relevant only with --bf16 argument. '
                             'If specified, loss is calculated in bf16. Otherwise, calculated in fp32.')
    parser.add_argument("--eval_interval",
                        type=int,
                        default=0,
                        help="If > 0, perform evaluation at this interval")
    parser.add_argument("--eval_iters",
                        type=int,
                        default=100,
                        help="Maximum evaluation iterations")
    parser.add_argument("--optimized_reward_loss_calc",
                        action='store_true',
                        help="Whether to use an optimized approach for RM loss calculation, or legacy flow")
    parser.add_argument('--no_fused_kernels',
                        action='store_true',
                        help='Do not use cuda fused kernels.')
    parser.add_argument("--add_eot_token",
                        action='store_true',
                        help="Add <|endoftext|> as additional special token to tokenizer")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    assert args.dropout is None or not args.disable_dropout, \
        'Not allowed to use both --dropout and --disable_dropout'

    return args


def main():
    if is_hpu():
        import habana_frameworks.torch.core as htcore
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        if is_hpu():
            get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(args.local_rank))
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    bf16=args.bf16,
                                    tb_output_path=args.tb_output_dir,
                                    tb_job_name=args.tb_job_name)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    dropout = 0. if args.disable_dropout else args.dropout
    loss_to_fp32 = args.bf16 and args.bf16_to_fp32_loss
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   dropout=dropout,
                                   loss_to_fp32=loss_to_fp32,
                                   optimized_reward_loss_calc=args.optimized_reward_loss_calc,
                                   seed=args.seed)

    # Model bigscience/bloom-560m has large variance at ln_f.weight parameter
    # This makes bf16 finetuning hard.
    # In general, since we are replacing the model head, it makes sense to reset
    # the LN that precedes it.
    force_optimize_params = []
    if "bigscience/bloom-" in args.model_name_or_path:
        zero_init_enabled = (args.zero_stage == 3)
        params = [rm_model.rwtranrsformer.ln_f.weight, rm_model.rwtranrsformer.ln_f.bias]
        with deepspeed.zero.GatheredParameters(params,
                                               modifier_rank=0,
                                               enabled=zero_init_enabled):
            if deepspeed.comm.get_rank() == 0 or not zero_init_enabled:
                torch.nn.init.ones_(rm_model.rwtranrsformer.ln_f.weight)
                torch.nn.init.zeros_(rm_model.rwtranrsformer.ln_f.bias)
        force_optimize_params.extend(
            ['rwtranrsformer.ln_f.weight', 'rwtranrsformer.ln_f.bias'])

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            force_optimize_params.append('v_head.weight')
            rm_model = only_optimize_lora_parameters(rm_model, force_optimize_params)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    # TODO SW-146776: remove this WA once SW-141762 is resolved
    if is_hpu():
        import habana_frameworks.torch.core as htcore
        rm_model.to(dtype=torch.bfloat16, device=device)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, dataloader, eval_iters):
        model.eval()
        _correct_predictions = 0
        _total_predictions = 0
        _chosen_scores = 0.
        _rejected_scores = 0.
        for _step, _batch in enumerate(dataloader):
            _batch = to_device(_batch, device)
            with torch.no_grad():
                _outputs = model(**_batch)

            chosen = _outputs["chosen_mean_scores"]
            rejected = _outputs["rejected_mean_scores"]
            _correct_predictions += (chosen > rejected).sum()
            _total_predictions += chosen.shape[0]
            _chosen_scores += _outputs["chosen_mean_scores"].mean().float()
            _rejected_scores += _outputs["rejected_mean_scores"].mean().float()
            hpu_mark_step()
            if (_step + 1) == eval_iters:
                break
        model.train()
        _acc = _correct_predictions / _total_predictions
        _chosen_scores = _chosen_scores / (_step + 1)
        _rejected_scores = _rejected_scores / (_step + 1)
        try:
            _acc = get_all_reduce_mean(_acc).item()
            _chosen_scores = get_all_reduce_mean(_chosen_scores).item()
            _rejected_scores = get_all_reduce_mean(_rejected_scores).item()
        except:
            pass
        return _chosen_scores, _rejected_scores, _acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    # TODO SW-146129: change the file to use HPEX optimizer instead of AdamW on hpu
    if args.offload:
        AdamOptimizer = DeepSpeedCPUAdam
    elif args.no_fused_kernels or is_hpu():
        AdamOptimizer = torch.optim.AdamW
    else:
        AdamOptimizer = FusedAdam
    print(f'Using {AdamOptimizer.__name__} optimizer')

    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    steps_per_print = ds_config['steps_per_print']
    train_bs = ds_config['train_batch_size']
    gas = args.gradient_accumulation_steps
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, reject_score, acc = evaluation_reward(rm_model, eval_dataloader, args.eval_iters)
    print_rank_0(f"chosen_last_scores (higher is better) : {reward_score}, "
                 f"rejected_last_scores (lower is better) : {reject_score}, "
                 f"acc (higher is better) : {acc}", args.global_rank)

    total_micro_steps = 0
    for epoch in range(args.num_train_epochs):
        last_time = time.time()
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        loss_sum = None
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            hpu_mark_step()
            rm_model.step()
            hpu_mark_step()
            loss_ = loss.detach()
            last_time, loss_sum = print_stats(epoch, step, steps_per_print, last_time, train_bs, gas, loss_, loss_sum, args.global_rank)
            mean_loss += loss.item()

            total_micro_steps += 1
            gas_boundary = (total_micro_steps % args.gradient_accumulation_steps == 0)
            total_steps = total_micro_steps // args.gradient_accumulation_steps
            if args.eval_interval and gas_boundary and (total_steps % args.eval_interval == 0):
                print_rank_0(f"Iter {total_steps}: Evaluating reward", args.global_rank)
                reward_score, reject_score, acc = evaluation_reward(rm_model, eval_dataloader, args.eval_iters)
                print_rank_0(f"Iter {total_steps}: c_scores: {reward_score}, r_scores: {reject_score}, "
                             f"diff: {reward_score - reject_score}, acc: {acc}", args.global_rank)

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, reject_score, acc = evaluation_reward(rm_model, eval_dataloader, args.eval_iters)
        print_rank_0(f"chosen_last_scores (higher is better) : {reward_score}, "
                     f"rejected_last_scores (lower is better) : {reject_score}, "
                     f"acc (higher is better) : {acc}", args.global_rank)
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
