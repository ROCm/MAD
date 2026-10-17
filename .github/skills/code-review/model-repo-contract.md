# The model-repository authoring contract

What a model repository must get right for madengine to build, run, and report its models.
Use this when reviewing changes to `models.json`, Dockerfiles, run scripts, `data.json`, or
CI config in a model repo — and when reviewing engine changes that would alter any of these
contracts, since model repos cannot be updated in lockstep.

Every rule here is enforced by the engine; the source file is named so a reviewer can check
the current behavior rather than trusting this summary.

---

## 1. `models.json`

An array of model objects, at the repo root or at `scripts/<dir>/models.json`. The
authoritative field list is the `CustomModel` dataclass in
`utils/discover_models.py`.

| Field | Type | Default | Controls |
| --- | --- | --- | --- |
| `name` | string | **required** | Identifier. In `scripts/<dir>/models.json` the engine prefixes it to `<dir>/<name>`. Drives the image tag `ci-<name>_<dockerfile>` and log filenames. |
| `dockerfile` | string | `""` | **Path prefix, not a filename.** `"docker/dummy"` globs `docker/dummy.*`. |
| `dockercontext` | string | `""` | Overrides the build context directory. |
| `scripts` | string | `""` | Run script path relative to the repo root. |
| `n_gpus` | **string** | `"-1"` | GPUs to request; `"-1"` is all. Requesting more than the host has raises at run time. |
| `timeout` | int (seconds) | `7200` | Per-model timeout. Only applied when the CLI `--timeout` is left at its default. |
| `tags` | list of strings, or comma string | `[]` | Selection labels for `--tags`. `all` matches everything. |
| `args` | string | `""` | Extra arguments appended to the run script invocation. |
| `training_precision` | string | `""` | Free-form label (`fp16`, `bf16`, `fp32`); lands in `perf.csv`. |
| `owner` | string | `""` | Contact. Metadata only. |
| `url` | string | `""` | Upstream model source. Metadata only. |
| `data` | string | `""` | Comma-separated dataname(s) resolved against `data.json`. |
| `cred` | string | `""` | Credential block name; injects `<cred>_<KEY>` as build args and run env vars. |
| `multiple_results` | string | `""` | Filename of a CSV the script writes instead of a single performance line. |
| `skip_gpu_arch` | string (comma list) | `""` | Architectures to skip, e.g. `"gfx908, gfx90a, A100"`. Overridable with `--disable-skip-gpu-arch`. |
| `deprecated` | bool | `false` | Skipped unless `madengine run --ignore-deprecated`. |
| `env_vars` | object | `{}` | Per-model env. `env_vars.DOCKER_IMAGE_NAME` lets `--use-image auto` find a prebuilt image. |
| `distributed` | object | `{}` | `launcher`, `nnodes`, `nproc_per_node`, `backend`, `port`. |
| `slurm` | object | `{}` | Per-model SLURM defaults (`partition`, `nodes`, `gpus_per_node`, `time`, ...). |
| `additional_docker_run_options` | string | `""` | Appended verbatim to `docker run`. |

**Review checks**

- `n_gpus` written as a number rather than a string.
- `dockerfile` given as a full filename (`docker/x.ubuntu.amd.Dockerfile`) instead of the
  prefix `docker/x` — this silently matches nothing.
- `scripts` path that does not exist, or is not relative to the repo root.
- `timeout` too low for the workload; the model will be killed mid-run and recorded as a
  failure.
- A new model with no `tags`, which makes it unselectable except by name.
- `skip_gpu_arch` listing an architecture that no longer exists, or omitting one the model
  genuinely cannot run on.

**Dynamic discovery.** `scripts/<dir>/get_models_json.py` must define `list_models()`
returning `CustomModel` instances. A directory may **not** contain both `models.json` and
`get_models_json.py`. Overriding `name` or `tags` inside `update_model()` is unsupported —
set them in the constructor.

---

## 2. The run script

**Success is signalled by output, not exit code.** A run that exits 0 but prints no
performance data is recorded as `FAILURE`.

### Single result

```bash
echo "performance: 14164 samples_per_second"
```

Parsed by `PERFORMANCE_LOG_PATTERN` in `deployment/base.py`. The value accepts a sign, a
decimal, and scientific notation; the metric is any token containing no whitespace or
comma. These are all valid:

```
performance: 14164 samples_per_second
performance: 1.23e+4 throughput
performance: 14164/s, samples_per_second
performance: 0.87 accuracy
```

There is a fallback for HuggingFace Trainer output (`train_samples_per_second`), recorded
with metric `samples_per_second`, tried only if the primary pattern fails.

### Multiple results

Set `multiple_results` on the model and write that CSV. Required columns are `performance`
at minimum, and `model,performance,metric` for superset reporting — a missing column
raises. The engine passes the filename in as `MAD_OUTPUT_CSV`.

### Environment available to the script

| Variable | Meaning |
| --- | --- |
| `MAD_MODEL_NAME` | The model's `name` |
| `MAD_GPU_VENDOR` | `AMD` or `NVIDIA` |
| `MAD_SYSTEM_NGPUS` | GPUs on the host |
| `MAD_RUNTIME_NGPUS` | GPUs actually granted to this run — **use this, not `n_gpus`** |
| `MAD_SYSTEM_GPU_ARCHITECTURE` | e.g. `gfx942` |
| `MAD_SYSTEM_HIP_VERSION`, `MAD_SYSTEM_GPU_PRODUCT_NAME` | Host GPU detail |
| `MAD_GUEST_OS` | `UBUNTU` or `CENTOS` — pick a package manager from this |
| `MAD_MULTI_NODE_RUNNER` | The launcher command string, e.g. `torchrun --standalone --nproc_per_node=8`. Empty for self-managing launchers (vllm, sglang, primus) |
| `MAD_OUTPUT_CSV` | Set when `multiple_results` is set |
| `MAD_DATAHOME`, `MAD_DATANAME` | Set when the model declares `data` |
| `ROCM_PATH` | In-container ROCm root (AMD) |

Distributed runs additionally get `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`,
`NODE_RANK`, `NNODES`, `NPROC_PER_NODE`, `MAD_COLLECT_METRICS` (`true` on rank 0),
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`, and the visible-device variables. SLURM adds
`MAD_DEPLOYMENT_TYPE=slurm`, `MAD_SLURM_JOB_ID`, `MAD_NODE_RANK`, `MAD_TOTAL_NODES`,
`MAD_IN_SLURM_JOB=1`, `MAD_LAUNCHER_TYPE`. Kubernetes adds `MAD_K8S_POD_NAME`,
`MAD_K8S_NAMESPACE`, `MAD_K8S_JOB=true`, `MAD_DEPLOYMENT_TYPE=kubernetes`,
`JOB_COMPLETION_INDEX`.

**Review checks**

- A script that hardcodes a GPU count instead of reading `MAD_RUNTIME_NGPUS`.
- A script under `set -u` that reads `MAD_MULTI_NODE_RUNNER` without a default — it is set
  to the empty string for self-managing launchers.
- A performance line built with a comma or space inside the metric name.
- Output written somewhere other than the working directory or the script directory when
  `multiple_results` is used.

### Log error scanning

After the run, the log is scanned for these substrings
(`execution/container_runner_helpers.py::DEFAULT_LOG_ERROR_PATTERNS`):

```
OutOfMemoryError, HIP out of memory, CUDA out of memory, RuntimeError:,
AssertionError:, ValueError:, SystemExit, failed (exitcode:,
Traceback (most recent call last), FAILED, Exception:, ImportError:,
ModuleNotFoundError:
```

A match fails the run **only when no performance data was extracted**. Three keys tune
this, settable on the model entry or in `additional_context` (context wins):

- `log_error_pattern_scan` (bool, default true) — set false to disable scanning
- `log_error_benign_patterns` (list of literal substrings) — exclude matching lines
- `log_error_patterns` (non-empty list) — replace the default list entirely

Prefer `log_error_benign_patterns` over disabling the scan.

---

## 3. Dockerfiles

Live under `docker/` by default, named `<prefix>.<guest_os>.<gpu_vendor>.Dockerfile` —
for example `docker/dummy.ubuntu.amd.Dockerfile`. The `.amd.` / `.nvidia.` infix is how
the engine infers the image's GPU vendor (falling back to scanning `FROM` for
`rocm`/`amd` versus `nvidia`/`cuda`).

### The `# CONTEXT` marker is mandatory

`DockerBuilder` reads a `# CONTEXT ` comment from the **first five lines** of each
candidate Dockerfile and evaluates it as a Python dict literal (`Context.filter`). Keys
present must equal the current context; an empty dict matches everything.

```dockerfile
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=rocm/pytorch
FROM $BASE_DOCKER
```

**A Dockerfile with no `# CONTEXT` line resolves to no Dockerfiles for that model, and the
model silently never builds.** This is the single most common model-repo authoring
mistake — check for it on every new Dockerfile.

### Base image

Declare `ARG BASE_DOCKER=<image>` and `FROM $BASE_DOCKER`. The engine greps
`^ARG BASE_DOCKER=` to record `base_docker` and `docker_sha` in `perf.csv`, and this is
what lets a caller override the base with
`--additional-context "{'docker_build_arg':{'BASE_DOCKER':'...'}}"`.

### Build context and `tools=`

Default context is `./docker`; a `dockerfile` path containing `primus` uses the repo root;
`dockercontext` overrides both. If `docker/common/` exists, the engine adds
`--build-context tools=docker/common`, so a Dockerfile may use:

```dockerfile
COPY --from=tools <file> <dest>
```

The full build is `docker build --network=host [--build-context tools=...] -t <image>
--pull -f <dockerfile> <build-args> <context>`. Note `--pull`: the base image is always
refreshed.

### GPU architecture args

`execution/dockerfile_utils.py` parses these from `ARG`/`ENV` lines for `--target-archs`
validation: `MAD_SYSTEM_GPU_ARCHITECTURE`, `PYTORCH_ROCM_ARCH`, `GPU_TARGETS`,
`GFX_COMPILATION_ARCH`, `GPU_ARCHS`. Declaring
`ARG MAD_SYSTEM_GPU_ARCHITECTURE` with **no default** warns at build time and forces every
caller to supply it — give it a default (`ARG MAD_SYSTEM_GPU_ARCHITECTURE=gfx942`).

---

## 4. `data.json`

An object keyed by dataname, each mapping provider type to config:

```json
{
  "my_dataset": {
    "local": { "path": "/shared/datasets/my_dataset" },
    "minio": { "path": "s3://bucket/my_dataset" }
  }
}
```

Provider types are `local`, `custom`, `nas`, `minio`, `aws`. When several are listed for
one dataname they are tried in the order **custom → local → minio → nas → aws**, and the
first reachable one wins. Every provider takes a required `path`; all but `local` accept
`mirrorlocal` (`local` rejects it). Remote credentials come from `core/constants.py` and
`credential.json`, never from `data.json`.

The engine sets `MAD_DATAHOME` (default `/data_dlm`, suffixed with the provider index) and
records `dataname`, `data_provider_type`, `data_size`, and `data_download_duration` into
`perf.csv`. An unknown dataname raises at run time.

---

## 5. `additional_context` keys

Recognized top-level keys, from `cli/validators.py::validate_additional_context_structure`
and `docs/configuration.md`:

| Key | Type | Purpose |
| --- | --- | --- |
| `docker_build_arg` | object | `--build-arg` values |
| `docker_env_vars` | object | Env vars into the container |
| `docker_mounts` | object | `{"/container": "/host"}` → `-v` |
| `docker_gpus` | string | GPU subset, e.g. `"0,2-4,7"` |
| `env_vars` | object | Deployment-config env vars |
| `tools` | array | Profiling tool selection |
| `pre_scripts`, `post_scripts` | array | Extra scripts around the run |
| `k8s` / `kubernetes` | object | Kubernetes config — **infers the target** |
| `slurm` | object | SLURM config — **infers the target** |
| `distributed` | object | `launcher`, `nnodes`, `nproc_per_node`, `master_port` |
| `vllm` | object | vLLM-specific config |
| `MAD_CONTAINER_IMAGE` | string | Use a prebuilt image and skip the build |
| `gpu_vendor` | string | `AMD` or `NVIDIA` (defaults to `AMD` at build time) |
| `guest_os` | string | `UBUNTU` or `CENTOS` (defaults to `UBUNTU`) |
| `MAD_ROCM_PATH` | string | Host ROCm root override |
| `timeout` | number | Timeout override |
| `log_error_pattern_scan` / `log_error_benign_patterns` / `log_error_patterns` | see §2 | Log scan tuning |

Remember the value is a Python dict literal or JSON — single quotes are fine.

---

## 6. Repo layout

```
<model-repo-root>/            # run madengine from here
├── models.json               # required
├── data.json                 # optional
├── credential.json           # optional
├── docker/
│   ├── <name>.<os>.<vendor>.Dockerfile   # needs the `# CONTEXT` line
│   └── common/               # optional; becomes `--build-context tools=`
└── scripts/
    ├── <model>/
    │   ├── run.sh            # prints `performance: ...`
    │   ├── models.json       # optional, scoped models
    │   └── get_models_json.py  # optional, mutually exclusive with models.json
    └── common/               # POPULATED BY madengine at run time
```

`scripts/common/` is filled from the madengine package during a run and removed afterwards.
**Never commit files there.** A PR that adds content under `scripts/common/` will have it
deleted on the next run.

Setting `MODEL_DIR` to a different path makes madengine copy `docker/`, `scripts/`,
`models.json`, `credential.json`, and `data.json` from there into the working directory
before discovery.

Runs write `perf.csv`, `perf_entry.csv`, and any `multiple_results` CSV to the repo root.
