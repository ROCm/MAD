# Shared DI on MI355X + AMD Pensando AI NIC (ionic): MoRI GPU-atomic-free, GLM-5.3-Flash and Kimi K3

Report: [`shared_DI_gpu_atomics_free_MI355X_ionic.html`](shared_DI_gpu_atomics_free_MI355X_ionic.html) (open locally in a browser).

What it covers:

- **MoRI-EP GPU-atomic-free mode (MoRI PR #736).** On virtualized MI355X nodes where the hypervisor enables PCIe ACS, RDMA atomics from the NIC to GPU memory never complete, so MoRI-EP's atomic signalling hangs at warm-up. The atomic-free mode signals with RDMA WRITEs instead. Enable it by building `ROCm/mori` main at or after `6731ee1` (or `amirakb89/mori@e8e412be`) and setting `MORI_EP_DISABLE_RDMA_ATOMICS=1`. Measured EP16 / EP32 bandwidth, accuracy and stress, atomic-free vs atomic, ACS on vs off.
- **Platform and driver versions** used: ionic `26.08.18.001.51.g8be56c6` (firmware `1.117.5-a-196`), amdgpu `6.16.13`, ROCm 7.2, libionic GDA builds.
- **Step-by-step MoRI-EP EP16 / EP32 test instructions** (bench, accuracy, stress) with pass criteria.
- **GLM-5.3-Flash-MXFP4 disaggregated serving** (EP8 vs EP16 vs per-node EP8 instances; static and AgentX conformance_256k / 512k).
- **Kimi K3 MXFP4 disaggregated serving** (TP8 + DCP8, TEP8, DEP8, EP16; AgentX conformance_256k / 512k / 750k capability matrix; decode-KV-bound 128K / 8K test) and when EP16 pays off.
- **Open issues / upstream asks** (MoRI-EP InterNode deadlock with ionic CCQE on — workaround `MORI_DISABLE_IONIC_CCQE=1`; MoRI-EP 2 GiB buffer limit; MoRI heap vs HIP free-memory accounting; MoRIIO limits for TP8 prefill).
