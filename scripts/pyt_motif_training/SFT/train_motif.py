import argparse
import os
from functools import partial

import torch
from datasets import load_dataset
from loguru import logger
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator


def collate_fn(samples, tokenizer):
    inp, attn_mask = [], []
    for x in samples:
        message = [{"role": "system", "content": "you are an helpful assistant"}] + x['context']
        chat = tokenizer.apply_chat_template(message, tokenize=False)
        single_batch = tokenizer(
            chat,
            padding="max_length",
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        inp.append(single_batch["input_ids"])
        attn_mask.append(single_batch["attention_mask"])

    return torch.concatenate(inp, dim=0), torch.concatenate(attn_mask, dim=0)


def main(args):
    accelerator = Accelerator()

    # this demo will use 100 samples of origin data
    # downloading the dataset will consume about 2.5 gb of your storage
    train_dataset = load_dataset("nvidia/HelpSteer3", split="train[:100]")
    total_iters = len(train_dataset) // args.batchsize

    # loading model
    # due to tie-weights, you may see logs like below(which can be ignored, in progress)
    #    Some weights of MotifForCausalLM were not initialized from the model checkpoint at Motif-Technologies/Motif-2.6B and are newly initialized: ['lm_head.weight']
    #    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    model = AutoModelForCausalLM.from_pretrained(
        "Motif-Technologies/Motif-2.6B",
        trust_remote_code=True,
        _attn_implementation="eager",  # also supports flash_attention_2, install if interested
        torch_dtype="bfloat16",  # used bfloat16 for 1-gpu MI250 budget, but you are free to use float32
    ).bfloat16()
    model.train()

    # loading tokenizer
    # maybe you want to apply your own chat template here, for example
    # tokenizer.chat_template = "some_jinja_template"
    tokenizer = AutoTokenizer.from_pretrained(
        "Motif-Technologies/Motif-2.6B",
        trust_remote_code=True,
    )

    # defining dataloader, optimizer and scheduler
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        drop_last=True,
        pin_memory=True,
        shuffle=False,
        num_workers=accelerator.num_processes,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = LinearLR(optimizer=optimizer, total_iters=total_iters, last_epoch=-1)

    # wrap everything with accelerator
    # use accelerate config when running!
    # adopted from train_llama.py

    local_rank = accelerator.local_process_index

    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{local_rank}") * 1e-6
    logger.info(f"GPU {local_rank} memory use = {pre_mem_use:.2f}MB")

    optimizer, lr_scheduler, dataloader, model = accelerator.prepare(
        optimizer, lr_scheduler, dataloader, model
    )

    # train loop starts
    if accelerator.is_main_process:
        logger.info("=====TRAIN START=====")

    for epoch in range(args.epochs):
        for idx, batch in enumerate(dataloader):
            loss = model(
                input_ids=batch[0],
                labels=batch[0],
                attention_mask=batch[1],
            ).loss

            accelerator.backward(loss)
            model.clip_grad_norm_(1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                logger.info(
                    f"TRAIN | {epoch + 1}/{args.epochs + 1} epochs | \
                        {idx + 1}/{total_iters} steps | loss: {loss.item()} | lr: {lr_scheduler.get_lr()[0]}"
                )

            accelerator.wait_for_everyone()
    accelerator.end_training()

    # save trained model & tokenizer
    # using AutoModel-compatible api
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, "./exp01")
    os.mkdir(save_path)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(save_path)
    logger.info("=====TRAIN COMPLETE=====")


if __name__ == "__main__":
    # define argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=1)
    parser.add_argument("--batchsize", "-b", type=int, default=4)
    parser.add_argument("--lr", "-l", type=float, default=5e-5)
    args = parser.parse_args()

    main(args)
