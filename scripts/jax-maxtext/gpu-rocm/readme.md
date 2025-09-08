# ROCM benchmarking
Scripts under this folder are used to benchmark rocm docker for maxtext-jax with different models. They will launch docker and run the benchmark. **Please run them on host instead of inside any docker**

All the scripts without the _multinode suffix can be launched on single node like:
```
IMAGE="rocm/jax-maxtext-training:xxx" HF_HOME=/home/amd-shared-home/.cache/huggingface bash ./deepseek_v2_16b.sh
```
Please adjust the $HF_HOME and $IMAGE to your environment. 

HF_HOME is where huggingface_hub will store local data, please refer to [Huggingface cli Document](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-download) on how to download the data.

For the multinode one, they were written for AMD internal cluster, and will need to be adjusted for other cluster setting. They can be launched via slurm like:
```
sbatch -N <num_nodes> llama3_70b_multinode.sh
```
## Tokenizer download
For single node scripts, they will use the $HF_HOME folder on the host. The script will mount the host HF folder to the docker. Please make sure that the data already got downloaded to $HF_HOME folder / your HF token is saved in the config file before running the script. The tokenizer of corresponding models will be used for the training.

|  Model tag | Huggingface webpage  |
|---|---|
| meta-llama/Llama-2-7b  |  https://huggingface.co/meta-llama/Llama-2-7b |
| meta-llama/Llama-2-70b  | https://huggingface.co/meta-llama/Llama-2-70b  |
| meta-llama/Meta-Llama-3-8B  | https://huggingface.co/meta-llama/Meta-Llama-3-8B  |
| meta-llama/Meta-Llama-3-70B  |  https://huggingface.co/meta-llama/Meta-Llama-3-70B |

Example command for downloading the llama model tokenizer
```
huggingface-cli login --token=hf_xxxx
huggingface-cli download meta-llama/Llama-2-7b  --include "**token**" 
huggingface-cli download meta-llama/Llama-2-70b  --include "**token**" 
huggingface-cli download meta-llama/Meta-Llama-3-8B  --include "**token**" 
huggingface-cli download meta-llama/Meta-Llama-3-70B  --include "**token**" 
```
## Dataset download
Please run this command for downloading the c4 dataset
```
huggingface-cli download legacy-datasets/c4 --include "*.parquet" --repo-type dataset  --revision refs/convert/parquet
```
Please check this path and see if data got downloaded to $HF_HOME/hub/datasets--legacy-datasets--c4/snapshots/5abe0d085aa23dd9db2a6c1e86cfce4e4db6f0c3/en/partial-train/000*.parquet