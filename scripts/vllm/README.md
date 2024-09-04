# vllm benchmark script

## Usage
#### Command

```sh
./vllm_benchmark_report.sh -s $test_option -m $model_repo -g $num_gpu -d $datatype
```

Note: The input sequence length, output sequence length, and tensor parallel (TP) are already configured. You don't need to specify them with this script.

#### Variables

| Name         | Options                                 | Description                                      |
| ------------ | --------------------------------------- | ------------------------------------------------ |
| $test_option | latency                                 | Measure decoding token latency                   |
|              | throughput                              | Measure token generation throughput              |
|              | all                                     | Measure both throughput and latency              |
| $model_repo  | meta-llama/Meta-Llama-3.1-8B-Instruct   | Llama 3.1 8B                                     |
|              | meta-llama/Meta-Llama-3.1-70B-Instruct  | Llama 3.1 70B                                    |
|              | meta-llama/Meta-Llama-3.1-405B-Instruct | Llama 3.1 405B                                   |
|              | meta-llama/Llama-2-7b-chat-hf           | Llama 2 7B                                       |
|              | mistralai/Mistral-7B-Instruct-v0.3      | Mistral 7B                                       |
|              | Qwen/Qwen2-7B-Instruct                  | Qwen2 7B                                         |
|              | core42/jais-13b-chat                    | JAIS 13B                                         |
|              | core42/jais-30b-chat-v3                 | JAIS 30B                                         |
| $num_gpu     | 1 or 8                                  | Number of GPUs.                                  |
| $datatype    | float16                                 |                                                  |

## example
#### latency + throughput
```sh
./vllm_benchmark_report.sh -s all -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
```
#### latency 
```sh
./vllm_benchmark_report.sh -s latency -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
```
#### throughput
```sh
./vllm_benchmark_report.sh -s throughput -m meta-llama/Meta-Llama-3.1-8B-Instruct -g 1 -d float16
```
