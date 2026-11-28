This directory contains the inference examples of using Numeric Doctor.

| Example | What it is |
| --- | --- |
| [`qwen_hf_generate_nan/`](qwen_hf_generate_nan/) | Qwen2.5-0.5B fp32 inference (TF32 enabled) with Hugging Face `transformers` `generate()`, 3 processes per gfx950 GPU: some processes produce NaN logits after some requests; self-contained, one command |
| [`qwen_vllm_serve_nan/`](qwen_vllm_serve_nan/) | Qwen2.5-0.5B bf16 served by vLLM `LLM.generate()` with its defaults (`torch.compile` + HIP graphs): `Logprobs are not finite` on a request, deterministically; MI350X |
| [`qwen_sglang_serve_nan/`](qwen_sglang_serve_nan/) | The same model served by SGLang `Engine.generate()` with its defaults: `Logprobs are not finite` on a request, deterministically; MI350X |
| [`deepseek_mla_decode_wrong/`](deepseek_mla_decode_wrong/) | A DeepSeek-V3-shaped MLA batched greedy decode (bf16, 128 heads, page size 1): **no NaN** -- the model's fp32 parity check reports wrong attention outputs on some calls, and greedy tokens differ between identical passes; MI350X |
| [`qwen_vllm_tp2_nan/`](qwen_vllm_tp2_nan/) | A Qwen2.5-1.5B-shaped bf16 model served by vLLM `LLM.generate()` at tensor parallel 2 (defaults: `torch.compile` + HIP graphs): `Logprobs are not finite` on long requests; MI350X |
| [`deepseek_sparse_mla_decode_nan/`](deepseek_sparse_mla_decode_nan/) | A DeepSeek-V3.2 / GLM-5-shaped sparse-MLA greedy decode (one TP8 rank: 16 heads, top-2048 indexer, page size 1) on vLLM's public ROCm image: `Logits are not finite` from the first decode step; MI350X |
