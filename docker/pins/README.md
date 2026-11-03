# Pin sets

A pin set says what a distributed-inference image is built from: the base image per
GPU architecture (by digest), and each component -- MoRI, AITER, vLLM, vllm-router,
the UCX / RIXL / rocSHMEM / DeepEP transport stack, SGLang's extra packages -- at a
full commit or an exact version, each with why it is pinned there. One Dockerfile per
engine builds any pin set:

| Engine | Dockerfile | Pin sets |
|---|---|---|
| vLLM | `docker/di_vllm.ubuntu.amd.Dockerfile` | `vllm-mori121`, `vllm-glm51`, `vllm-kimi-k3` |
| SGLang | `docker/di_sglang.ubuntu.amd.Dockerfile` | `sglang-disagg` |

```bash
python3 docker/pins/pins.py validate                       # all pin sets
docker build -f docker/di_vllm.ubuntu.amd.Dockerfile \
  $(python3 docker/pins/pins.py build-args vllm-kimi-k3 --arch gfx950) \
  -t <registry>/di-vllm:vllm-kimi-k3-gfx950 .
```

`build-args` prints `PINS`, the base image for the architecture, and
`MAD_SYSTEM_GPU_ARCHITECTURE`. The build stops on an architecture the set is not for or
another base image. The image records `/app/versions.txt` (every component's commit)
and `/app/pins.json`, and its last step checks them against the set.

## Patches instead of forks

Changes that are not upstream yet are patch files in `docker/patches/<component>/`,
applied after the upstream commit is checked out; the pin set names each with its
sha256 and the upstream PR it waits on, and keeps the fork commit it reproduces as
`fork_commit`. Each patch is `git diff <upstream merge-base> <fork commit>`, so applying
it to the upstream commit gives the fork commit's tree exactly; `docker/patches/verify.py`
checks that against GitHub. When a change lands upstream, move the pin set to an
upstream commit that has it and delete the patch.

| Patch | Component | Reproduces |
|---|---|---|
| `vllm/wide-ep-write-0629.patch` | vLLM | the Wide-EP WRITE-mode branch of the mori121 stack (15 commits) |
| `vllm/kimi-k3.patch` | vLLM | the Kimi-K3 vLLM (67 commits) |
| `vllm/glm51.patch` | vLLM | the GLM-5.1 vLLM (10 commits) |
| `router/dp-rank-kv-notify.patch` | vllm-router | DP-rank round-robin + 2P2D KV-notify fix (vllm-project/router#181) |
| `aiter/gqa64.patch` | AITER | the gqa64 fix (ROCm/aiter#4957) |

## Rules

- A component is a full 40-hex commit of its upstream repository, plus patches. No
  personal fork is a source. A branch, tag or short SHA may be kept in
  `ref` as a note; it is never what is built.
- A base image is a digest (`repo:tag@sha256:...`).
- A pip package has an exact version. One that was unpinned in the image a set
  replaces is listed with `"version": null` and a `pending` reason until it is pinned
  from a built image; `validate` reports it and `validate --strict` refuses it.

## The images these replace

Each set reproduces one per-model Dockerfile; `tests/test_pins.py` checks that it
builds the same sources, switches and settings (the refs those files named are
resolved to commits in its `RESOLVED` table):

| Pin set | Replaces | Differs only in |
|---|---|---|
| `vllm-mori121` | `vllm_disagg_inference.ubuntu.amd.Dockerfile` | DeepEP pinned (was its default branch) |
| `vllm-glm51` | `vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile` | rocSHMEM pinned (was `develop`, which no longer builds); DeepEP pinned |
| `vllm-kimi-k3` | `vllm_kimi_k3.ubuntu.amd.Dockerfile` | DeepEP pinned |
| `sglang-disagg` | `sglang_disagg_inference.ubuntu.amd.Dockerfile` | none (its pip packages are pending pins) |

The cards still name the per-model Dockerfiles until each pin set has been built and
its `versions.txt` compared with the image it replaces.

Run the tests from the MAD root: `python3 -m unittest discover -s docker/pins/tests -v`.
