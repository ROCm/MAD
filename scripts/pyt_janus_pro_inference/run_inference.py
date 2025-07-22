###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
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
'''
Code adapted from https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
and https://github.com/deepseek-ai/Janus/blob/main/inference.py
thanks to the authors of Janus-Pro.
'''
import argparse
import os
import random
import sys
import torch
import time
import numpy as np

from math import ceil
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from janus.models import VLChatProcessor, MultiModalityCausalLM
from typing import List
from utils import batcher, get_parser_args


MODEL_PATH = 'deepseek-ai/Janus-Pro-7B'
DATASET_PATH = 'lmarena-ai/vision-arena-bench-v0.1'
PROMPTS_FILE_PATH = './img_gen_prompts.txt'


def load_model():
    '''
    Loads VLM model and pre-processor
    '''
    vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_PATH)
    vl_gpt = MultiModalityCausalLM.from_pretrained(MODEL_PATH).to(torch.bfloat16).cuda().eval()

    return vl_gpt, vl_chat_processor


def load_data_understanding(args):
    '''
    Loads image understanding dataset from Huggingface, shuffles and returns specified number in array
    '''
    ds = load_dataset(DATASET_PATH)['train']
    seed = args.seed
    N_samples = args.nsamples
    ds_shuffled = ds.shuffle(seed=seed).select(range(N_samples))

    return ds_shuffled


def build_inputs_understanding(batch):
    '''
    Build conversation and PIL image arrays for input to image_understanding task
    '''
    convs, pil_imgs = [], []
    for example in batch:
        convs.append([
            {
                'role': 'User',
                'content': '<image_placeholder>\n' + example['turns'][0][0]['content'],
                'images': ['<image_placeholder>'],
            },
            {'role': 'Assistant', 'content': ''},
        ])
        pil_imgs.append([example['images'][0].convert('RGB')])

    return convs, pil_imgs


@torch.inference_mode()
def run_img_understanding(args, vl_gpt, vl_chat_processor, ds_shuffled):
    '''
    Run "image understanding" task of the VLM
    '''
    tokenizer = vl_chat_processor.tokenizer

    n_batch, etimes = 0, []
    out_lines = [] if args.save_log else None

    # Start batch run
    for batch in batcher(ds_shuffled, args.batch_sz):

        print(f'Image understanding sample batch ( {n_batch + 1} / {ceil(args.nsamples / args.batch_sz)} ) ...')
        # Build conversations and images lists
        convs, pil_imgs = build_inputs_understanding(batch)
        
        # Process each example individually
        single_preps = []
        for conv, imgs in zip(convs, pil_imgs):
            prep = vl_chat_processor(
                conversations=conv,
                images=imgs,
                return_tensors='pt',
                force_batchify=False
            )
            single_preps.append(prep)

        # Batchfy preps into one BatchFeature
        proc = vl_chat_processor.batchify(single_preps).to(vl_gpt.device)

        # warm-up run
        if n_batch == 0:
            embeds = vl_gpt.prepare_inputs_embeds(**proc)
            _ = vl_gpt.language_model.generate(inputs_embeds=embeds, attention_mask=proc.attention_mask,
                    pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, 
                    eos_token_id=tokenizer.eos_token_id, max_new_tokens=args.max_new_token, do_sample=False, use_cache=True)
        
        # Measure times only in the device
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Get embeddings and generate
        embeds = vl_gpt.prepare_inputs_embeds(**proc)

        outputs = vl_gpt.language_model.generate(
            inputs_embeds=embeds,
            attention_mask=proc.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_token,
            do_sample=False,
            use_cache=True,
        )

        # Only time in device
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        etimes.append((len(batch), dt))

        # decode and collect
        for idx, (ex, gen) in enumerate(zip(batch, outputs)):

            ans = tokenizer.decode(gen.cpu().tolist(), skip_special_tokens=True)
            sft = proc.sft_format[idx]

            if args.save_log:
                out_lines.append(f'question_id: {ex["question_id"]}, cluster_name: {ex["cluster_name"]}')
                out_lines.append(20 * '----')
                out_lines.append(sft)
                out_lines.append(ans)
                out_lines.append(20 * '====')

        n_batch += 1

    # Write log file at once
    if args.save_log:
        with open(args.log_filename, 'w') as f:
            f.write('\n'.join(out_lines))

    print(f'Number of samples: {args.nsamples}, Number of batches: {n_batch}')
    print(f'No. images   Elapsed(sec)   Avg. samples/sec')
    [print(f'{len:>8}\t{elap:.4f}\t\t{len/elap:.4f}') for len, elap in etimes]

    total_time = sum([e[1] for e in etimes])
    print(f'\nTotal elapsed time: {total_time:.4f} sec')
    print(f'Avg throughput: {args.nsamples / total_time:.4f} samples/sec')


def _make_conversation(text: str):
    '''
    Convert request text into multi-turn conversation
    '''
    return [
        {
            "role": "<|User|>",
            "content": text,
        },
        {
            "role": "<|Assistant|>",
            "content": ""
        }
    ]


def _load_prompts(args):
    '''
    Loads image generation prompts from local file and shuffles them befor returning in an array
    '''
    with open(PROMPTS_FILE_PATH, 'r', encoding='utf-8') as f:
        # strip trailing newlines and ignore empty lines and comments
        prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    assert len(prompts) >= args.n_prompts, ValueError(f'Requested more prompts, {args.n_prompts}, than available: {len(prompts)}')
    
    random.seed(args.seed)
    random.shuffle(prompts)
    prompts = prompts[:args.n_prompts]

    print(f'Number of image generation prompts = {len(prompts)}')
    return prompts


def _convert_prompt(user_input: str, vl_chat_proc: VLChatProcessor):
    '''
    Convert user input string to prompt in Supervised Fine-tuning (SFT) format
    '''
    sft_format = vl_chat_proc.apply_sft_template_for_multi_turn_prompts(
        conversations=_make_conversation(user_input),
        sft_format=vl_chat_proc.sft_format,
        system_prompt="",
    )

    return sft_format + vl_chat_proc.image_start_tag


def write_images(args, imgs_array, prompts):
    '''
    Write multi-dimensional array of generated images into file
    '''
    (n_prompts, n_images, img_size, img_size, _) = imgs_array.shape

    for p in range(n_prompts):
        for i in range(n_images):
            img = Image.fromarray(imgs_array[p, i, :, :, :])
            prompt_txt = prompts[p].split(':')[1].strip().replace(' ', '_')[:30]
            out_path = os.path.join(
                args.img_gen_dir, f'prom{p}_img{i}_{prompt_txt}.png'
            )
            img.save(out_path)
            print(f'Saved image for prompt[{p}]_img[{i}] to {out_path}')


@torch.inference_mode()
def generate_image_from_text(args, mmgpt: MultiModalityCausalLM, vl_chat_processor: VLChatProcessor, prompt: str):
    '''
    function to generate image(s) from text prompt by vision model of Multimodal GPT
    '''
    os.makedirs(args.img_gen_dir, exist_ok=True)

    # clear CUDA cache before generating
    torch.cuda.empty_cache()

    # unpack parameters
    temperature = args.temperature
    parallel_size = args.n_images
    cfg_weight = args.weight
    image_token_num_per_image = args.token_per_img
    img_size = args.image_sz
    patch_size = args.patch_sz

    # start timer
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    # send data to the device
    tokens = tokens.cuda()
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image),
                                   dtype=torch.int).cuda()

    # main generation loop
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                             shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
                                            )

    # convert to visual image
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    # end timer
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return visual_img, elapsed
    

def run_img_generation(args, vl_gpt, vl_chat_processor):
    '''
    Main function to run text-to-image generation
    '''
    # prep input/output arrays
    prompts = [_convert_prompt(p, vl_chat_processor) for p in _load_prompts(args)]

    n_prompts, parallel_size, image_sz = args.n_prompts, args.n_images, args.image_sz
    imgs_array = np.zeros((n_prompts, parallel_size, image_sz, image_sz, 3), dtype=np.uint8)
    elapsed_gen = []

    # warmup function with single prompt
    _, _ = generate_image_from_text(args, vl_gpt, vl_chat_processor, prompts[0])

    # main measured performance run
    for idx, prompt in enumerate(prompts):
        images, elapsed = generate_image_from_text(args, vl_gpt, vl_chat_processor, prompt)
        imgs_array[idx, :, :, :, :] = images
        elapsed_gen.append((len(prompt), elapsed))

    # write generated outputs
    write_images(args, imgs_array, prompts)

    print('\nPrompt length, Elapsed (sec):')
    [print(f'{len:>10}\t{elap:.4f}') for len, elap in elapsed_gen]

    total_time = sum([e[1] for e in elapsed_gen])
    print(f'\nTotal elapsed time: {total_time:.4f} sec')
    print(f'Avg throughput: {args.n_prompts / total_time:.4f} prompts/sec')


def main():
    '''
    Main CLI function for running inference tasks
    '''
    args = get_parser_args()
    assert args.command and (args.command in ['under', 'gen']), 'Subcommand should be "under" or "gen"'

    vl_gpt, vl_chat_processor = load_model()

    if args.command == 'under':
        # image understanding inference
        torch.cuda.empty_cache()

        # prep data
        ds_shuffled = load_data_understanding(args)

        # run function
        run_img_understanding(args, vl_gpt, vl_chat_processor, ds_shuffled)

        print('\nDONE Inference: Image understanding')

    else:
        # image generation inference
        torch.cuda.empty_cache()

        # run function
        run_img_generation(args, vl_gpt, vl_chat_processor)

        print('\nDONE Inference: Text-2-image generation')


if __name__ == '__main__':
    sys.exit(main())
