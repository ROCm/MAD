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
|              | serving                                 | Measure online serving throughput                |
|              | all                                     | Measure all                                      |
|              | meta-llama/Llama-2-70b-chat-hf          | [Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)                               |
| $model_repo  | meta-llama/Llama-3.1-8B-Instruct   | [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| (float16)    | meta-llama/Llama-3.1-70B-Instruct  | [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)                            |
|              | meta-llama/Llama-3.1-405B-Instruct | [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)                           |
|              | mistralai/Mixtral-8x7B-Instruct-v0.1    | [Mixtral MoE 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)                         |
|              | mistralai/Mixtral-8x22B-Instruct-v0.1   | [Mixtral MoE 8x22B](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)                        |
| $model_repo  | amd/Llama-3.1-70B-Instruct-FP8-KV  | [Llama 3.1 70B](https://huggingface.co/amd/Llama-3.1-70B-Instruct-FP8-KV)                            |
| (float8)     | amd/Llama-3.1-405B-Instruct-FP8-KV | [Llama 3.1 405B](https://huggingface.co/amd/Llama-3.1-405B-Instruct-FP8-KV)                           |
|              | amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV   | [Mixtral MoE 8x7B](https://huggingface.co/amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV)                        |
|              | amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV  | [Mixtral MoE 8x22B](https://huggingface.co/amd/Mixtral-8x22B-Instruct-v0.1-FP8-KV)                       |
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

- Benchmark example - serving

  Use this command to benchmark the online serving throughput of the Llama 3.1 70B model on 8 GPUs with the float16 and float8 data type.

  ```sh
  ./vllm_benchmark_report.sh -s serving -m meta-llama/Llama-3.1-70B-Instruct -g 8 -d float16
  ./vllm_benchmark_report.sh -s serving -m amd/Llama-3.1-70B-Instruct-FP8-KV -g 8 -d float8
  ```

  The serving reports are available at:

  - `./reports_float16/summary/Llama-3.1-70B-Instruct_serving_report.csv`
  - `./reports_float8/summary/Llama-3.1-70B-Instruct-FP8-KV_serving_report.csv`