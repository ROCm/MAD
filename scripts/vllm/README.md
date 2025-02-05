# vllm benchmark script

## Usage
#### Command

```sh
./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype
```

>[!NOTE]
>The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

>[!NOTE]
>If you encounter this error, pass your access-authorized Hugging Face token to the gated models.
>```sh
>OSError: You are trying to access a gated repo.
>
># pass your HF_TOKEN
>export HF_TOKEN=$your_personal_hf_token
>```

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  | meta-llama/Llama-3.1-8B-Instruct   | [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| (float16)    | meta-llama/Llama-3.1-70B-Instruct  | [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)                            |
|              | meta-llama/Llama-3.1-405B-Instruct | [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)                           |
|              | meta-llama/Llama-3.2-11B-Vision-Instruct| [Llama 3.2 11B Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)                     |
|              | meta-llama/Llama-2-7b-chat-hf           | [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)                                |
|              | meta-llama/Llama-2-70b-chat-hf          | [Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)                               |
|              | mistralai/Mixtral-8x7B-Instruct-v0.1    | [Mixtral MoE 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)                         |
|              | mistralai/Mixtral-8x22B-Instruct-v0.1   | [Mixtral MoE 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)                        |
|              | mistralai/Mistral-7B-Instruct-v0.3      | [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)                           |
|              | Qwen/Qwen2-7B-Instruct                  | [Qwen2 7B](https://huggingface.co/Qwen/Qwen2-7B-Instruct)                                       |
|              | Qwen/Qwen2-72B-Instruct                 | [Qwen2 72B](https://huggingface.co/Qwen/Qwen2-72B-Instruct)                                      |
|              | core42/jais-13b-chat                    | [JAIS 13B](https://huggingface.co/core42/jais-13b-chat)                                         |
|              | core42/jais-30b-chat-v3                 | [JAIS 30B](https://huggingface.co/core42/jais-30b-chat-v3)                                      |
|              | databricks/dbrx-instruct                | [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)                                     |
|              | google/gemma-2-27b                      | [Gemma 2 27B](https://huggingface.co/google/gemma-2-27b)                                           |
|              | CohereForAI/c4ai-command-r-plus-08-2024 | [C4AI Command R+ 08-2024](https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024)                      |
|              | deepseek-ai/deepseek-moe-16b-chat       | [DeepSeek MoE 16B](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)                            |
| $model_repo  | amd/Llama-3.1-70B-Instruct-FP8-KV  | [Llama 3.1 70B](https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV)                            |
| (float8)     | amd/Llama-3.1-405B-Instruct-FP8-KV | [Llama 3.1 405B](https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)                           |
|              | amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV   | [Mixtral MoE 8x7B](https://huggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV)                        |
|              | amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV  | [Mixtral MoE 8x22B](https://huggingface.co/amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV)                       |
|              | amd/Mistral-7B-v0.1-FP8-KV              | [Mistral 7B](https://huggingface.co/amd/Mistral-7B-v0.1-FP8-KV)                                   |
|              | amd/dbrx-instruct-FP8-KV                | [DBRX Instruct](https://huggingface.co/amd/dbrx-instruct-FP8-KV)                                     |
|              | amd/c4ai-command-r-plus-FP8-KV          | [C4AI Command R+ 08-2024](https://huggingface.co/amd/c4ai-command-r-plus-FP8-KV)                               |
| $num_gpu     | 1 or 8                                  | Number of GPUs                                   |
| $datatype    | float16, float8                         | Data type                                        |

#### Run the benchmark tests on the MI300X accelerator 🏃

Here are some examples and the test results:

- Benchmark example - latency

  Use this command to benchmark the latency of the Llama 3.1 70B model on 8 GPUs with the float16 and float8 data type.

  ```sh
  ./vllm_benchmark_report.sh -s latency -m meta-llama/Llama-3.1-70B-Instruct -g 8 -d float16
  ./vllm_benchmark_report.sh -s latency -m amd/Llama-3.1-70B-Instruct-FP8-KV -g 8 -d float8
  ```

  The latency reports are available at:

  - `./reports_float16/summary/Llama-3.1-70B-Instruct_latency_report.csv`
  - `./reports_float8/summary/Llama-3.1-70B-Instruct-FP8-KV_latency_report.csv`

- Benchmark example - throughput

  Use this command to benchmark the throughput of the Llama 3.1 70B model on one GPU with the float16 and float8 data type.

  ```sh
  ./vllm_benchmark_report.sh -s throughput -m meta-llama/Llama-3.1-70B-Instruct -g 8 -d float16
  ./vllm_benchmark_report.sh -s throughput -m amd/Llama-3.1-70B-Instruct-FP8-KV -g 8 -d float8
  ```

  The throughput reports are available at:

  - `./reports_float16/summary/Llama-3.1-70B-Instruct_throughput_report.csv`
  - `./reports_float8/summary/Llama-3.1-70B-Instruct-FP8-KV_throughput_report.csv`

>[!NOTE]
>Throughput is calculated as:
>-   `throughput_tot = requests * (input lengths + output lengths) / elapsed_time`
>-   `throughput_gen = requests * output lengths / elapsed_time`
