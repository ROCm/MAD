# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
ARG BASE_DOCKER=vllm/vllm-openai-rocm:v0.27.1
FROM ${BASE_DOCKER}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Re-declared so the LABEL at the end of the file can still see it.
ARG BASE_DOCKER

ARG MLPERF_WORKSPACE=/workspace
ARG MLPERF_INFERENCE_REPO=https://github.com/mlcommons/inference.git
ARG MLPERF_INFERENCE_REF=1b9bc00344640a412adf65d3db4ab44e3085aea6

ENV WORKSPACE_DIR=${MLPERF_WORKSPACE}
ENV MLPERF_INFERENCE_DIR=${MLPERF_WORKSPACE}/inference
ENV MLPERF_HARNESS_DIR=${MLPERF_WORKSPACE}/inference/language/llama3.1-8b
ENV NLTK_DATA=${MLPERF_WORKSPACE}/nltk_data
ENV HF_MODULES_CACHE=${MLPERF_WORKSPACE}/hf_modules
ENV PIP_NO_CACHE_DIR=1

# No submodules: they belong to other benchmarks (bert, deepseek-r1, wan-2.2),
# while loadgen and the llama3.1-8b harness are plain in-tree sources.
RUN git clone "${MLPERF_INFERENCE_REPO}" "${MLPERF_INFERENCE_DIR}" && \
    cd "${MLPERF_INFERENCE_DIR}" && \
    git checkout "${MLPERF_INFERENCE_REF}" && \
    git rev-parse HEAD

# The harness targets vLLM 0.6.3, where `generate()` still took raw token ids.
# Modern vLLM only accepts prompt objects — which the Server path in this very
# file already builds, so the Offline path is simply stale.
RUN python3 - <<'PY'
from pathlib import Path
import os

path = Path(os.environ["MLPERF_HARNESS_DIR"]) / "SUT_VLLM.py"
source = path.read_text()
old = """            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor, sampling_params=self.sampling_params
            )"""
new = """            outputs = self.model.generate(
                [TokensPrompt(prompt_token_ids=ids) for ids in input_ids_tensor],
                sampling_params=self.sampling_params,
            )"""
assert source.count(old) == 1, "upstream SUT_VLLM.py changed; recheck the vLLM call"
path.write_text(source.replace(old, new))
print("patched", path)
PY

RUN pip install "${MLPERF_INFERENCE_DIR}/loadgen" && \
    python3 -c "import mlperf_loadgen; print('loadgen', mlperf_loadgen.__file__)"

# The harness `requirements.txt` pins `vllm==0.6.3` and `transformers==4.46.2`;
# installing it would replace the ROCm vLLM build that this base image exists
# for. Only the imports the base image is actually missing are added here.
RUN pip install nltk rouge-score absl-py

# Pre-seed what the accuracy pass would otherwise fetch mid-run: nltk's sentence
# tokenizer and the `rouge` metric module from the Hub. HF_MODULES_CACHE is set
# explicitly so a run-time HF_HOME override cannot invalidate the warm cache.
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='${NLTK_DATA}'); nltk.download('punkt_tab', download_dir='${NLTK_DATA}')" && \
    python3 -c "import evaluate; print('rouge metric', evaluate.load('rouge').name)"

RUN cd "${MLPERF_HARNESS_DIR}" && \
    python3 -c "import vllm, torch, transformers, mlperf_loadgen, nltk, evaluate, pandas, datasets; \
print('vllm', vllm.__version__); \
print('torch', torch.__version__, 'hip', torch.version.hip); \
print('transformers', transformers.__version__)" && \
    python3 -c "import dataset, evaluation; print('harness modules import ok')"

LABEL mlperf_base="${BASE_DOCKER}"
LABEL mlperf_inference_ref="${MLPERF_INFERENCE_REF}"

# madengine keeps the container alive with `docker run -t -d <image> cat`, which
# the base image's `vllm serve` entrypoint would swallow.
ENTRYPOINT []
CMD []

WORKDIR ${WORKSPACE_DIR}
