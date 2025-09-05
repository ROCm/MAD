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

import argparse

# parse arguments
parser = argparse.ArgumentParser(description='Convert input script parameters to torchtune script parameters')
parser.add_argument("--mode",
                        type=str,
                        help="finetune_fw, finetune_lora, finetune_qlora")
parser.add_argument("--model",
                        type=str,
                        help="Supported models: Llama-2-70B, Llama-2-13B, Llama-2-7B, \
                                Llama-3-70B, Llama-3-8B, Llama-3.1-405B, \
                                Llama-3.2-3B, Llama-3.2-1B, \
                                Llama-3.2-vision-11B, Llama-3.2-vision-90B, \
                                Llama-3.3-70B, Llama-4-scout-17B-16E, \
                                Qwen2-1.5B, Qwen2-7B, Qwen2.5-32B, \
                                Qwen2.5-72B, Qwen3-8B, Qwen3-32B")

# read arguments
args = parser.parse_args()
mode = args.mode 
model = args.model 

# Parse finetuning mode
if mode == "finetune_fw":
    method = "full"
elif mode == "finetune_lora":
    method = "lora"
elif mode == "finetune_qlora":
    method = "qlora"
else:
    print("Invalid finetuning mode selected.")

# Input string
parts = model.split('-')
model_family = parts[0].lower() + parts[1].replace('.', '_')
# Llama-3.2-vision-11B => llama3_2_vision, 11B_full
if model == "Llama-3.2-vision-11B" or model == "Llama-3.2-vision-90B":
    model_family += '_' + parts[2]
    model_size = parts[3]
# Llama-4-scout-17B-16E => llama4, 17B_16E
elif model == "Llama-4-scout-17B-16E":
    model_size = parts[3] + '_' + parts[4]
# Qwen models => qwen2, qwen2_5, qwen3
elif model.startswith("Qwen"):
    model_size = parts[1]
    if model.startswith("Qwen2.5"):
        model_family = "qwen2_5"
    elif model.startswith("Qwen2"):
        model_family = "qwen2"
    elif model.startswith("Qwen3"):
        model_family = "qwen3"
# Llama-3.1-70B => llama3_1, 70B_lora
else:
    model_size = parts[2]

# Print the results
print(f"model: {model}")
print(f"model_family: {model_family}")
print(f"model_size: {model_size}")
print(f"method: {method}")
