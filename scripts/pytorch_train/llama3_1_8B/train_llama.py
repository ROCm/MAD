# lint as: python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
import logging
import random
import sys
import json
import time
import numpy as np
import argparse
from dataclasses import asdict
from contextlib import nullcontext
from functools import partial
from tqdm import tqdm
import logging

import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformer_engine.pytorch import fp8_autocast
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch.nn.functional as F
import torch.optim as optim

from accelerate import Accelerator, FullyShardedDataParallelPlugin
import accelerate
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model import ModelConfig, FP8Llama, FP8TransformerBlock



def load_model(config):
    model_config = ModelConfig(**config)
    model = FP8Llama(**asdict(model_config)) 
    return model_config, model

class DummyIndexDataset(IterableDataset):
    def __init__(self, seq_len=4096, vocab_size=128256):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __iter__(self):
        while True:
            x = torch.randint(self.vocab_size, [self.seq_len], dtype=torch.int64)
            y = torch.roll(x, shifts=-1)
            y[-1] = 0
            yield x, y

def train(args):
    
    # Load the configuration file
    with open(args.config) as f:
        config = json.load(f)

    # Setup random seed
    accelerate.utils.set_seed(config['training']['seed'])
    
    # Initialize Accelerator with FSDP
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={FP8TransformerBlock}),
        state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=False, rank0_only=False),
    )
    accelerator = Accelerator(
        fsdp_plugin = fsdp_plugin,
    )
    world_size = accelerator.num_processes
    global_rank = accelerator.process_index
    local_rank = accelerator.local_process_index

    # Setup the logger
    if accelerator.is_main_process:
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()
    else:
        logger = None 

    # Initialize the model
    model_config, model = load_model(config['model'])
    model_config.calculate_llama_token_flops()
    model.train()
    
    # Initialize the optimizer and scheduler
    optimizer_class = getattr(optim, config['optimizer']['type'])
    optimizer = optimizer_class(model.parameters(), **config['optimizer']['params'])
    scheduler_class = getattr(optim.lr_scheduler, config['scheduler']['type'])
    scheduler = scheduler_class(optimizer, **config['scheduler']['params'])

    # Calculate the memory and flops
    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{local_rank}") * 1e-6
    tokens_per_batch = config['training']['batch_size'] * config['model']['max_seq_len']
    flops_per_iter = model_config.flops_per_token * tokens_per_batch
    if accelerator.is_main_process:
        print(f"GPU memory use = {pre_mem_use:.2f}MB")
        print(f"TFLOP per iteration: {flops_per_iter/1e12:.2f}")
    
    # Generate dataloader
    dataset = DummyIndexDataset(config['model']['max_seq_len'], config['model']['vocab_size'])
    data_loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'],
        num_workers=world_size, 
        pin_memory=True, 
        shuffle=False
    )
    
    optimizer, scheduler, data_loader, model = accelerator.prepare(optimizer, scheduler, data_loader, model)
    data_loader_iter = iter(data_loader)
    
    total_time = 0.0
    progress_bar = tqdm(range(config['training']['total_steps']), disable=not accelerator.is_main_process)

    last_time = time.time()
    
    for step in progress_bar:

        input, labels = next(data_loader_iter)
        
        logits = model(input, is_first_microbatch=step % config['training']['grad_acc_steps'] == 0)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss /= config['training']['grad_acc_steps']

        accelerator.backward(loss)

        if accelerator.sync_gradients and (step + 1) % config['training']['grad_acc_steps'] == 0:
            model.clip_grad_norm_(1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process:
            current_time = time.time()
            iteration_time = current_time-last_time
            token_per_sec = tokens_per_batch/iteration_time
            
            if step < config['training']['total_steps'] // 2:
                msg = "[Warmup] "
            else:
                msg = "[Test] "
                total_time += iteration_time
            
            msg += f"Step: {step}/{config['training']['total_steps']}; "
            msg += f"TFLOP/s: {flops_per_iter/iteration_time/1e12:.2f}; "
            msg += f"iter time: {iteration_time:.2f}; "
            msg += f"tokens/s: {token_per_sec:.2f}"

            logger.info(msg)
            progress_bar.set_description(msg)
            
            last_time = current_time

    if accelerator.is_main_process:
        avg_iter_time = total_time / (config['training']['total_steps'] // 2)
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") * 1e-6

        accelerator.print(f"Avg token per second: {tokens_per_batch/avg_iter_time:.2f}")
        accelerator.print(f"Avg iter time: {avg_iter_time:.4f}")
        accelerator.print(f"TFLOP per iteration: {flops_per_iter/1e12:.2f}")
        accelerator.print(f"Avg TFLOP/s: {flops_per_iter/avg_iter_time/1e12:.2f}")
        accelerator.print(f"Peak memory use = {peak_memory:.2f}MB")

        logger.info(f"Avg token per second: {tokens_per_batch/avg_iter_time:.2f}")
        logger.info(f"Avg iter time: {avg_iter_time:.4f}")
        logger.info(f"TFLOP per iteration: {flops_per_iter/1e12:.2f}")
        logger.info(f"Avg TFLOP/s: {flops_per_iter/avg_iter_time/1e12:.2f}")
        logger.info(f"Peak memory use = {peak_memory:.2f}MB")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Configuration file of the model and exp")
    parser.add_argument('--log_file', type=str, help="Output log file")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
