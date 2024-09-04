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
import csv, sys, json
import argparse
import os

# parse arg
parser = argparse.ArgumentParser(description='Convert vllm csv output format to DLM csv output format')
parser.add_argument("--mode",
                        type=str,
                        help="latency or throughput")
parser.add_argument("--model",
                        type=str,
                        help="model name")
parser.add_argument("--tp",
                        type=str,
                        help="tensorparallel size")
parser.add_argument("--batch-size",
                        type=str,
                        help="batch size")
parser.add_argument("--num-prompts",
                        type=str,
                        help="input requests")
parser.add_argument("--input-len",
                        type=str,
                        help="input seq length")
parser.add_argument("--output-len",
                        type=str,
                        help="output seq length")
parser.add_argument("--dtype",
                        type=str,
                        help="data_type")
parser.add_argument("--input-json",
                        help="path to the input_json file")
parser.add_argument("--output-csv",
                        help="path to the output_csv file")

# read args
args = parser.parse_args()

def extract_val(dirty_list, key):
    for i,x in enumerate(dirty_list):
        if x == key:
            return dirty_list[i+1]

if args.mode == "latency":
    with open(args.input_json, newline='') as inpf:
        header_write = 0 if os.path.exists(args.output_csv) else 1
        with open(args.output_csv,'a+',newline='') as outf:
            writer = csv.writer(outf, delimiter=',')
            if header_write:
                writer.writerow(['model', 'latency (ms)', 'latency_per_tkn (ms)','tp', 'batch_size', 'input_len', 'output_len', 'dtype']) if header_write else None
            
            # workaround to vllm's dirty json output from multi-gpu cases
            dirty_json = inpf.read()
            dirty_list = dirty_json.replace(",","").replace(":","").replace("\"","").split()
            avg_latency = float(extract_val(dirty_list, "avg_latency"))
            try:
                latency_per_tkn = str(avg_latency / int(args.output_len) * 1000)
                model_details = args.model                       ,\
                                str(avg_latency * 1000),\
                                latency_per_tkn                  ,\
                                args.tp                          ,\
                                args.batch_size                  ,\
                                args.input_len                   ,\
                                args.output_len                  ,\
                                args.dtype
                writer.writerow(model_details)
            except csv.Error as e:
                sys.exit('file {}: {}'.format(args.input_json, e))

elif args.mode == "throughput":
    with open(args.input_json, newline='') as inpf:
        header_write = 0 if os.path.exists(args.output_csv) else 1
        with open(args.output_csv,'a+',newline='') as outf:
            writer = csv.writer(outf, delimiter=',')
            if header_write:
                writer.writerow(['model', 'throughput_tot (tok/sec)', 'throughput_gen (tok/sec)', 'tp', 'requests', 'input_len', 'output_len', 'dtype']) if header_write else None

            # workaround to vllm's dirty json output from multi-gpu cases
            dirty_json = inpf.read()
            dirty_list = dirty_json.replace(",","").replace(":","").replace("\"","").split()
            tokens_per_second = float(extract_val(dirty_list, "tokens_per_second"))
            elapsed_time = float(extract_val(dirty_list, "elapsed_time"))

            try:
                gen_throughput = str(int(int(args.num_prompts) * int(args.output_len) / elapsed_time))
                model_details = args.model                            ,\
                                str(int(tokens_per_second)) ,\
                                gen_throughput                        ,\
                                args.tp                               ,\
                                args.num_prompts                      ,\
                                args.input_len                        ,\
                                args.output_len                       ,\
                                args.dtype
                writer.writerow(model_details)
            except csv.Error as e:
                sys.exit('file {}: {}'.format(args.input_json, e))
