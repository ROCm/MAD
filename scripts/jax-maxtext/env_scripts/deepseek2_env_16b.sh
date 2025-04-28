###############################################################################
#
# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################

export XLA_FLAGS=" --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_latency_hiding_scheduler=TRUE --xla_gpu_enable_cublaslt=True --xla_gpu_graph_level=0  --xla_gpu_autotune_level=5 --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_memory_limit_slop_factor=95"
export HSA_FORCE_FINE_GRAIN_PCIE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.967
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
