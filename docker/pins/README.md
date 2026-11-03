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

## Per pin set

**The image is the contract.** MAD's scripts are catalog and configuration only and
patch nothing in the image at start, so every source fix a model needs is in its pin
set: a commit or a patch. Serving a model from an image built off an older pin is
unsupported and can fail silently (it boots, then returns garbage or stalls the KV
transfer). To take a fix, move the pin and rebuild.

- `vllm-mori121`: the validated v1.2.1 MoRI stack for DeepSeek-V3/R1 and the other
  shared-image cards, gfx942. The MoRIIO disagg fixes (notify, LL split, DP-rank
  hash fail-safe) are in its vLLM patch. The rixl transport stack is on (all
  connectors); the moriio paths are the validated ones.
- `vllm-glm51`: GLM-5.1-FP8 (MLA + DeepSeek sparse attention) over MoRI-EP wide-EP,
  gfx942. 2P/2D EP16 and 4P/4D EP32 were validated on these pins (NIAH 2k/8k at
  9.3-9.7/10 over three seeds; no failed requests or GPU faults in the sweeps). The
  vLLM patch carries the DSA indexer KV transfer over MoRIIO (vLLM #47766) and the
  EP32 combine-topk fix; AITER carries the gfx942 gqa64 decode fix. MoRI-EP only.
- `vllm-kimi-k3`: every Kimi-K3 vLLM card, colocated and disaggregated, built for
  one architecture at a time. gfx942 (MI300X) is the validated stack (single-needle
  NIAH 10K-900K on 2 prefill + 2 decode nodes). gfx950 (MI355X) is bring-up, not
  validated: the vLLM patch is the gfx942 line, and the gfx950 GLM work needed EP16
  start-up gates and a MoRI combine fix that are not in it. AITER 68e42f5f carries the
  Kimi-K3 tuned MoE configs, `aiter.ops.triton.conv` and flydsl 0.2.4. Runtime
  differences between the architectures are recipe settings in `models.yaml`.
- `sglang-disagg`: the SGLang release image per architecture; MoRI 158c7e8 is added on
  gfx942 only (gfx950's image ships its MoRI).

The ROCm 7.2.3 GPU-RDMA settings (`expandable_segments:False`,
`HSA_ENABLE_IPC_MODE_LEGACY=0`) are not in any image: they are in
`scripts/vllm_dissag/connectors/<connector>.env`, applied at launch.

## Rules

- A component is a full 40-hex commit of its upstream repository, plus patches. No
  personal fork is a source. A branch, tag or short SHA may be kept in
  `ref` as a note; it is never what is built.
- A base image is a digest (`repo:tag@sha256:...`).
- A pip package has an exact version. One that was unpinned in the image a set
  replaces is listed with `"version": null` and a `pending` reason until it is pinned
  from a built image; `validate` reports it and `validate --strict` refuses it.

## The images these replace

Each set reproduces one per-model Dockerfile (removed; in git at 13cfcac); `tests/test_pins.py` checks that it
builds the same sources, switches and settings (the refs those files named are
resolved to commits in its `RESOLVED` table):

| Pin set | Replaces | Differs only in |
|---|---|---|
| `vllm-mori121` | `vllm_disagg_inference.ubuntu.amd.Dockerfile` | DeepEP pinned (was its default branch) |
| `vllm-glm51` | `vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile` | rocSHMEM pinned (was `develop`, which no longer builds); DeepEP pinned |
| `vllm-kimi-k3` | `vllm_kimi_k3.ubuntu.amd.Dockerfile` | DeepEP pinned |
| `sglang-disagg` | `sglang_disagg_inference.ubuntu.amd.Dockerfile` | none (its pip packages are pending pins) |

The cards name `docker/di_vllm` or `docker/di_sglang` and their pin set
(`"docker_build_arg": {"PINS": "<name>"}`, `"dockercontext": "."`); madengine adds the
base image for the build's architecture. The per-model Dockerfiles were removed after
commit 13cfcac; the test reads them from there. Before relying on a set, build it and
compare its `/app/versions.txt` with the image it replaces.

Run the tests from the MAD root: `python3 -m unittest discover -s docker/pins/tests -v`.
