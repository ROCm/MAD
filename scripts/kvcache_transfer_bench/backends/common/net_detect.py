"""
Network defaults for clusters that do not pin them: the RDMA device and the
socket interface. run_node.sh resolves both once per node and exports them.

Usage: net_detect.py rdma-device [CANDIDATES] | socket-ifname
"""

import os
import sys
from pathlib import Path

SYS_IB = Path("/sys/class/infiniband")
KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")
PCI_DEVICES = Path("/sys/bus/pci/devices")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def socket_ifname() -> str:
    """Interface carrying the default route."""
    for line in _read(Path("/proc/net/route")).splitlines()[1:]:
        fields = line.split()
        if len(fields) > 1 and fields[1] == "00000000":
            return fields[0]
    return ""


def _gpu_pci_paths() -> list:
    """sysfs PCI paths of the GPUs in HIP device order, which is KFD's node order."""
    try:
        nodes = sorted((p for p in KFD_NODES.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    except OSError:
        return []
    paths = []
    for node in nodes:
        props = dict(line.split()[:2] for line in _read(node / "properties").splitlines() if len(line.split()) > 1)
        if int(props.get("simd_count", 0)) == 0:
            continue
        loc, domain = int(props.get("location_id", 0)), int(props.get("domain", 0))
        bdf = f"{domain:04x}:{loc >> 8:02x}:{(loc >> 3) & 0x1f:02x}.{loc & 7:x}"
        paths.append(os.path.realpath(PCI_DEVICES / bdf))
    return paths


def _numa(pci_path: str) -> int:
    try:
        return int(_read(Path(pci_path) / "numa_node"))
    except ValueError:
        return -1


def _active_gbps(dev: str) -> int:
    """Summed rate of the ACTIVE ports (MoRI-IO's RdmaDevice::TotalActiveGbps)."""
    total = 0
    for port in (SYS_IB / dev / "ports").glob("*"):
        rate = _read(port / "rate").split()
        if _read(port / "state").endswith("ACTIVE") and rate:
            try:
                total += int(float(rate[0]))
            except ValueError:
                pass
    return total


def _pci_hops(a: str, b: str) -> int:
    """PCIe nodes strictly between two sysfs device paths, as MoRI-IO's TopoPathPci::Hops.

    Root complexes meet at a virtual root, so devices under different ones are
    connected through it.
    """
    pa, pb = a.split("/"), b.split("/")
    shared = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        shared += 1
    return len(pa) + len(pb) - 2 * shared - 1


def _mori_rank(dev: str, gpu: str) -> tuple:
    """Sort key of MoRI-IO's TopoSystem::MatchGpuAndNic: speed, NUMA, PCIe hops, name."""
    nic = os.path.realpath(SYS_IB / dev / "device")
    return (-_active_gbps(dev), _numa(nic) != _numa(gpu), _pci_hops(gpu, nic), dev)


def rdma_device(candidates: str = "") -> str:
    """The RDMA device MoRI-IO would assign to GPU 0, which every backend allocates on.

    Like MoRI-IO, prefer the fastest active link, then a NIC on the GPU's NUMA node,
    then the fewest PCIe hops, then the name. Unlike it, skip devices with no active
    port, the device behind the default-route interface (the frontend NIC), and bonds
    while any other device qualifies. ``candidates`` (comma-separated) narrows the
    choice, and a single name is returned as given.
    """
    names = [c.strip() for c in candidates.split(",") if c.strip()]
    if len(names) == 1:
        return names[0]
    try:
        devices = sorted(p.name for p in SYS_IB.iterdir())
    except OSError:
        return ""
    if names:
        devices = [d for d in devices if d in names]
    frontend = socket_ifname()

    def usable(dev: str) -> bool:
        if not any(_read(p).endswith("ACTIVE") for p in (SYS_IB / dev / "ports").glob("*/state")):
            return False
        try:
            return frontend not in os.listdir(SYS_IB / dev / "device" / "net")
        except OSError:
            return True

    found = [d for d in devices if usable(d)]
    found = [d for d in found if "bond" not in d] or found
    if not found:
        return ""
    gpus = _gpu_pci_paths()
    if not gpus:
        return min(found, key=lambda d: (-_active_gbps(d), d))
    return min(found, key=lambda d: _mori_rank(d, gpus[0]))


if __name__ == "__main__":
    if sys.argv[1:2] == ["rdma-device"]:
        print(rdma_device(sys.argv[2] if len(sys.argv) > 2 else ""))
    elif sys.argv[1:2] == ["socket-ifname"]:
        print(socket_ifname())
    else:
        sys.exit(__doc__.strip().splitlines()[-1])
