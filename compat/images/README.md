# Named image specs

Each `<name>.yaml` here is an image spec (the format of `scripts/<dir>/image_spec.yaml`)
that several workloads share. A card selects one through madengine's OmegaConf
interpolation and may override any part of it:

```json
"image_spec": "${images.<name>}"
"image_spec": {"extends": "${images.<name>}", "gpu_arch": "gfx950"}
```

A spec may give its base per GPU architecture (`base: {gfx942: ..., gfx950: ...}`), select
other builds on one GPU (`variants.<arch>`), extend another spec (`extends: ${images.<name>}`),
put files from `docker/` into the image (`copy`), and run checks after its components
(`finish`). Its components are builds authorized in `compat/rocm.yaml`.

| Spec | What it is |
|---|---|
| `di-vllm` | what every vLLM distributed-inference stack shares; the others extend it |
| `vllm-mori121` | the validated mori121 stack (gfx942): the DeepSeek-V3, Llama and most vLLM DI cards |
| `vllm-kimi-k3` | the Kimi-K3 stack (gfx942, gfx950) |
| `vllm-glm51` | the GLM-5.1 stack (gfx942) |
| `vllm-per-gpu` | the stack of each GPU, composed from `vllm-mori121` (gfx942) and the Kimi-K3 stack's builds (gfx950); no build of its own |
| `vllm-dsv4` | the DeepSeek-V4 stack (gfx942): the upstream vLLM ROCm 7.2 nightly as base, vllm-router and the DeepSeek-V4 patches baked in; develop's `docker/vllm_disagg_gfx942_inference.dsv4.ubuntu.amd.Dockerfile` as a spec |
| `sglang-disagg` | the SGLang release image per GPU, with MoRI on gfx942 |

A build can generalize an image into this form: `madengine build --emit-config DIR` places the
image under the stack it is closest to (`extends` and only what differs) -- see
[docs/context_as_code_developer_guide.md](../../docs/context_as_code_developer_guide.md#from-a-build-to-the-matrix).
