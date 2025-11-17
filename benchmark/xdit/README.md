# xDiT Diffusion inference

rocm/pytorch-xdit images support several diffusion model inference workloads on gfx942
and gfx950 series (AMD Instinct™ MI300X, MI308X, MI325X and MI350X, MI355X) GPUs. The
image has ROCm 7.9.0 (preview, based on [TheRock](https://github.com/ROCm/TheRock)) and uses
[xDiT](https://github.com/xdit-project/xDiT) distributed diffusion model inference
framework for high-performance text and video generation.

## Setup

Use

```sh
git clone https://github.com/ROCm/MAD
cd MAD
pip install -r requirements.txt
```

to clone the ROCm Model Automation and Dashboarding (MAD) repository to a local directory
and install the required packages on the host machine.

## Run MAD benchmarks

Execute benchmarks with

```sh
MAD_SECRETS_HFTOKEN=HFTOKEN madengine run --tags TAG --live-output
```

where `HFTOKEN` is a valid Hugging Face token (note that some models are gated) and TAG
a supported MAD model tag. See Available models section for more details. The inference
latencies can be found from `results.csv` once the benchmark runs have finished.

## Available models

| MAD model TAG                  | Model repository                                              |
| -------------------------------| --------------------------------------------------------------|
| pyt_xdit_hunyuanvideo          | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)   |
| pyt_xdit_wan_2_1               | [Wan 2.1](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)  |
| pyt_xdit_wan_2_2               | [Wan 2.2](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B)      |
| pyt_xdit_flux                  | [Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
