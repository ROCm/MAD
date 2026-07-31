###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

import pandas as pd
import argparse
import csv
import os

parser = argparse.ArgumentParser(description='Convert pytorch train output format to MAD csv output format')
parser.add_argument("--mode", type=str, help="pretrain or posttrain")
parser.add_argument("--model", type=str, help="model name")
parser.add_argument("--input", type=str, help="path to input file")
parser.add_argument("--output", type=str, help="path to output file")
parser.add_argument("--precision", type=str, help="training precision")
parser.add_argument("--batch_size", type=int, help="batch size used for training")
parser.add_argument("--seq_len", type=int, help="sequence length used for training")
parser.add_argument("--device", type=str, help="device architecture (e.g., MI300X, MI355X)")
parser.add_argument("--num_gpus", type=int, help="number of GPUs used for training")

args = parser.parse_args()
input_file = args.input
output_file = args.output
precision = args.precision
print("Input file path: ", input_file)
print("Output file path: ", output_file)
print("Precision: ", precision)

SUPPORTED_DIFFUSION_MODELS = ["Flux", "Stable-Diffusion-XL", "Mochi-1", "Hunyuan-video", "Wan2_1-i2v"]

if args.mode == "pretrain" and args.model == "DLRM":
    df = pd.read_csv(input_file)
    recs_per_s_mean = (df.iloc[-1]['Recommendations/s (mean)']).item()
    recs_per_s_cv = (df.iloc[-1]['Recommendations/s (std/mean)']).item()
    data = [
        {'model': args.model, 'performance': recs_per_s_mean, 'metric': 'recs_per_s_mean', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
        {'model': args.model, 'performance': recs_per_s_cv, 'metric': 'recs_per_s_cv', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
    ]

elif args.mode == "posttrain" and args.model in SUPPORTED_DIFFUSION_MODELS:
    df = pd.read_csv(input_file)
    FPS_per_GPU = float(df.iloc[-1]['avg_fps_gpu'])
    TFLOPS_per_GPU = float(df.iloc[-1]['avg_tflops'])
    data = [
        {'model': args.model, 'performance': FPS_per_GPU, 'metric': 'FPS_per_GPU', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus},
        {'model': args.model, 'performance': TFLOPS_per_GPU, 'metric': 'TFLOPS_per_GPU', 'mode': args.mode, 'precision': precision, 'batch_size': args.batch_size, 'seq_len': args.seq_len, 'device': args.device, 'num_gpus': args.num_gpus}
    ]

else:
    print(f"Error: Unsupported model '{args.model}' with mode '{args.mode}'.")
    print(f"Supported models: DLRM (pretrain), {', '.join(SUPPORTED_DIFFUSION_MODELS)} (posttrain)")
    exit(1)

if not os.path.exists(output_file) or os.stat(output_file).st_size == 0:
    mode = 'w'
else:
    mode = 'a'
with open(output_file, mode=mode, newline='') as file:
    print("Preparing to write performance data...")
    print("Data: ", data)
    writer = csv.DictWriter(file, fieldnames=['model','performance','metric', 'mode', 'precision', 'batch_size', 'seq_len', 'device', 'num_gpus'])
    if mode == 'w':
        writer.writeheader()
    writer.writerows(data)
    print("Completed writing to output file")
