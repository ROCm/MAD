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

import argparse
from itertools import islice


def batcher(iterator, batch_size):
    it = iter(iterator)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def get_parser_args():
    parser = argparse.ArgumentParser(
        description='Janus-Pro CLI: image_understanding ("under") or img2txt_generation ("gen")'
    )
    shared = argparse.ArgumentParser(add_help=False)

    shared.add_argument('--n_warmup', default=1, type=int, help='Number of warmup runs')
    shared.add_argument('--n_repeat', default=5, type=int, help='Number of repeated runs for average')
    shared.add_argument('--seed', default=123, type=int, help='Random seed for multimodal understanding and generation')
    shared.add_argument('--save_log', default=False, type=bool, help='Save inference log file if activated')
    shared.add_argument('--tunable_ops', default='off', type=str, help='Controls PyTorch Tunable Ops tuning')
    shared.add_argument('--compile', action='store_true', help='Enables PyTorch compile on the model (for image generation only)')

    subparsers = parser.add_subparsers(dest='command', required=True)
    under = subparsers.add_parser('under', parents=[shared], help='Run the vision-arena image understanding benchmark')

    under.add_argument('--nsamples', default=64, type=int, help='Number of samples to run for image understanding')
    under.add_argument('--batch_sz', default=32, type=int, help='Batch size for image understanding')
    under.add_argument('--max_new_token', default=512, type=int, help='Max new token for image understanding')
    under.add_argument('--log_filename', default='janus_infer_under.log', type=str, help='Name of the log file generated for image understanding')

    gen = subparsers.add_parser('gen', parents=[shared], help='Run txt2img generation benchmark')
    gen.add_argument('--n_prompts', default=14, type=int, help='Number of image generation prompts')
    gen.add_argument('--n_images', default=4, type=int, help='Number of images generated per prompt')
    gen.add_argument('--temperature', default=1, type=float, help='Temperature for image generation')
    gen.add_argument('--weight', default=5, type=float, help='Config weight for image generation')
    gen.add_argument('--token_per_img', default=576, type=int, help='Tokens per image for image generation')
    gen.add_argument('--image_sz', default=384, type=int, help='Size of image generated: (size x size)')
    gen.add_argument('--patch_sz', default=16, type=int, help='Size of a patch as image token: (size x size)')
    gen.add_argument('--img_gen_dir', default='generated_samples', type=str, help='Directory where generated images are stored')
    gen.add_argument('--log_filename', default='janus_infer_gen.log', type=str, help='Name of the log file generated for image generation')

    args = parser.parse_args()

    # post-parse validation
    if args.command == 'gen':
        lhs = args.image_sz * args.image_sz
        rhs = args.patch_sz * args.patch_sz * args.token_per_img
        assert lhs == rhs, f'Image_size^2 ({lhs} != patch_size^2 * tokens_per_image ({rhs})'
        # 384^2 = 16^2 * 576

    return args
