---
name: mad-generate-dockerfile
description: Standalone Dockerfile generator for MAD models using the framework-specific Jinja2 templates in framework_templates/. Use when the user asks to generate, create, or scaffold a Dockerfile for a MAD model without running the full mad-add-model workflow.
disable-model-invocation: true
---

# mad-generate-dockerfile Skill Instructions

## Purpose
Generate a MAD-compliant Dockerfile from a framework template without going
through the full model-addition workflow (see the `mad-add-model` skill for
that).

## Paths

All paths in this skill are relative to the **repository root**; run every
command from there.

- Templates: `.claude/skills/mad-generate-dockerfile/framework_templates/{framework}_base.Dockerfile.jinja`
- Renderer: `.claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py`
  (requires `jinja2`; `pip install jinja2` if it is missing)

Always render with the script rather than filling in a template by hand, so
the output is identical whichever agent runs this skill.

## Available Templates

| `--framework` | Use for | Reference Dockerfile (supplies the default `base_image`) |
|---|---|---|
| `pytorch` | Generic PyTorch workloads (incl. diffusion models needing ROCm flash-attention) | `docker/pyt_wan2.1_inference.ubuntu.amd.Dockerfile` |
| `jax` | JAX / MaxText | `docker/primus_maxtext.ubuntu.amd.Dockerfile` |
| `vllm` | vLLM inference | `docker/pyt_vllm.ubuntu.amd.Dockerfile` |
| `sglang` | SGLang inference | `docker/pyt_sglang.ubuntu.amd.Dockerfile` |
| `atom` | ATOM inference | `docker/pyt_atom.ubuntu.amd.Dockerfile` |
| `xdit` | xDiT diffusion inference | `docker/pyt_xdit.ubuntu.amd.Dockerfile` |
| `sglang_disagg` | SGLang disaggregated prefill-decode (RDMA + MoRI) | `docker/sglang_disagg_inference.ubuntu.amd.Dockerfile` |
| `vllm_disagg` | vLLM disaggregated prefill-decode (MoRI, AITER, NIXL/RIXL, DeepEP) | `docker/vllm_disagg_inference.ubuntu.amd.Dockerfile` |

`render_dockerfile.py --list` prints the context variables each template
accepts and the default base image it would use. The renderer reads that
default from the reference Dockerfile's `ARG BASE_DOCKER=` line on every run,
so templates never pin a stale image. Each template requires the `# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}`
header (already present).

### Common context variables (all templates)

- `year` — copyright year (the renderer defaults it to the current year)
- `base_image` — overrides the default `ARG BASE_DOCKER=` taken from the
  reference Dockerfile
- `apt_packages`, `pip_packages` — lists of extra packages
- `git_repos` — list of `{url, path, checkout}` dicts
- `custom_entrypoint` — `true` to emit `ENTRYPOINT [""]` (pytorch, jax,
  atom; vllm always emits it, vllm_disagg always emits `ENTRYPOINT []`)

### Template-specific context variables

- **pytorch_base**
  - `build_flash_attention` — `true` to build ROCm flash-attention from source
    for `MAD_SYSTEM_GPU_ARCHITECTURE` (the pattern used by wan2.1 / janus_pro)
  - `fa_repo` (default `https://github.com/ROCm/flash-attention.git`),
    `fa_branch` (default `v3.0.0.r1-cktile`)
- **xdit_base** — always installs `lshw`; keeps the upstream image's WORKDIR
  (the validator's `WORKSPACE_DIR not defined` warning is expected)
- **sglang_disagg_base**
  - `gpu_arch` (default `gfx942`)
  - `install_mori` — set `false` to default `INSTALL_MORI=0`
  - `mori_commit` — MoRI commit checked out in `/sgl-workspace/mori`
- **vllm_disagg_base**
  - `gfx_arch` (default `gfx942`), `nic_arch` (default `cx7`), `max_jobs` (default `32`)
  - `with_nixl` — set `false` to default `WITH_NIXL=0` (skips UCX/RIXL/rocSHMEM/DeepEP)
  - `build_mori` — set `false` to keep the base image's MoRI;
    `mori_repo`, `mori_ref` (default `v1.2.1`)
  - `aiter_wheel_url` — if set, replaces AITER with this prebuilt wheel
    (put companion pins such as `flydsl==…` in `pip_packages`)
  - `vllm_repo` + `vllm_ref` — if both set, compiles vLLM from source
  - `router_repo` + `router_ref` (+ `rust_toolchain`, default `1.88.0`) —
    if both set, builds `vllm-router` into the image
  - `ucx_ref`, `rixl_ref` — pin the NIXL transport sources

## Workflow

Steps that say **ask the user** mean: ask in the conversation (using your
structured-question tool if you have one) and wait for the answer — do not
assume defaults and continue on your own.

1. **Ask the user** which framework (pytorch, jax, vllm, sglang, atom, xdit,
   sglang_disagg, vllm_disagg) and model name they need a Dockerfile for.
2. Read the matching template above.
3. Run `render_dockerfile.py --list` to see the default `base_image` (read
   from the reference Dockerfile). Pass `base_image` only when the user wants
   a different one. For the disagg templates, also read the reference
   Dockerfile for the current MoRI/AITER/vLLM pins.
4. **Ask the user** for any extra `apt`/`pip` packages or git repos to bake
   in, plus any template-specific options above (e.g. flash-attention for
   PyTorch diffusion models).
5. Render a preview to stdout with the gathered context as a JSON object:
   ```bash
   python3 .claude/skills/mad-generate-dockerfile/scripts/render_dockerfile.py \
     --framework vllm \
     --context '{"pip_packages": ["foo==1.0"]}'
   ```
   Fix any `ERROR`/`WARNING` lines (a warning means a context key the
   template does not use, usually a typo).
6. Show the rendered Dockerfile to the user and **ask the user** to confirm.
7. On confirmation, re-run the same command with
   `--output docker/{model_name}.ubuntu.amd.Dockerfile` (it refuses to
   overwrite an existing file unless `--force` is given).
8. Validate with `python3 tools/validation/dockerfile_validator.py docker/{model_name}.ubuntu.amd.Dockerfile`.

## Agent Notes

- This skill writes files, so it runs only when invoked explicitly
  (`disable-model-invocation` for Claude Code and Cursor,
  `agents/openai.yaml` for Codex).
- This skill works in Claude Code, Cursor and Codex. It only needs file
  read/write and a shell; use whichever tools your agent provides for those.
- In a sandboxed agent (e.g. Codex), `pip install jinja2` needs network
  access and may require the user's approval.
