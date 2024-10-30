# vllm benchmark script

## Usage
#### Command

```sh
./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype
```

-   Note: The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

-   Note: If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
```sh
OSError: You are trying to access a gated repo.

# pass your HF_TOKEN
export HF_TOKEN=$your_personal_hf_token
```

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  | meta-llama/Meta-Llama-3.1-8B-Instruct   | Llama 3.1 8B                                     |
| (float16)    | meta-llama/Meta-Llama-3.1-70B-Instruct  | Llama 3.1 70B                                    |
|              | meta-llama/Meta-Llama-3.1-405B-Instruct | Llama 3.1 405B                                   |
|              | meta-llama/Llama-2-7b-chat-hf           | Llama 2 7B                                       |
|              | meta-llama/Llama-2-70b-chat-hf          | Llama 2 70B                                      |
|              | mistralai/Mixtral-8x7B-Instruct-v0.1    | Mistral 8x7B                                     |
|              | mistralai/Mixtral-8x22B-Instruct-v0.1   | Mistral 8x22B                                    |
|              | mistralai/Mistral-7B-Instruct-v0.3      | Mistral 7B                                       |
|              | Qwen/Qwen2-7B-Instruct                  | Qwen2 7B                                         |
|              | Qwen/Qwen2-72B-Instruct                 | Qwen2 72B                                        |
|              | core42/jais-13b-chat                    | JAIS 13B                                         |
|              | core42/jais-30b-chat-v3                 | JAIS 30B                                         |
| $model_repo  | amd/Meta-Llama-3.1-8B-Instruct-FP8-KV   | Llama 3.1 8B                                     |
| (float8)     | amd/Meta-Llama-3.1-70B-Instruct-FP8-KV  | Llama 3.1 70B                                    |
|              | amd/Meta-Llama-3.1-405B-Instruct-FP8-KV | Llama 3.1 405B                                   |
|              | amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV   | Mistral 8x7B                                     |
|              | amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV  | Mistral 8x22B                                    |
| $num_gpu     | 1 or 8                                  | Number of GPUs.                                  |
| $datatype    | float16, float8                         |                                                  |

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

-   Benchmark example - latency
 
Use this command to benchmark the latency of the Llama 3.1 8B model on one GPU with the float16 and float8 data type.

```sh
./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
./vllm_benchmark_report.sh -s latency -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8
```

You can find the latency report at *./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_latency_report.csv*.
You can find the latency report at *./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_latency_report.csv*.

-   Benchmark example - throughput

Use this command to benchmark the throughput of the Llama 3.1 8B model on one GPU with the float16 and float8 data type.

```sh
./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
./vllm_benchmark_report.sh -s throughput -m amd/Meta-Llama-3.1-8B-Instruct-FP8-KV -g 1 -d float8
```

You can find the throughput report at *./reports_float16/summary/Meta-Llama-3.1-8B-Instruct_throughput_report.csv*.
You can find the throughput report at *./reports_float8/summary/Meta-Llama-3.1-8B-Instruct-FP8-KV_throughput_report.csv*.

-   throughput\_tot = requests \* (**input lengths + output lengths**) / elapsed\_time

-   throughput\_gen = requests \* **output lengths** / elapsed\_time
