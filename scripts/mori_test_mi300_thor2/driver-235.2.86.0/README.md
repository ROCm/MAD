# Broadcom bnxt driver 235.2.86.0 (host prerequisite)

The MoRI-EP/MoRI-IO test suite requires the Broadcom **bnxt_en / bnxt_re `235.2.86.0`** driver on each
host (with firmware `238.1.138.0`, kernel `5.15.0-177-generic`). This is the driver version that fixes
the `bnxt_re_dv_create_cq` EIO seen with the 237/238 drivers on this firmware.

## The driver binaries are intentionally NOT redistributed here
To keep this repository free of third-party vendor binaries, the actual driver files are **not** committed:

- `bnxt_en.ko`, `bnxt_re.ko` (prebuilt for 5.15.0-177-generic)
- `bnxt-dkms-src-235.2.86.0.tar.gz` (DKMS source, rebuilds `.ko` for any kernel)
- `bnxt-rocelib-235.2.86.0.tar.gz` (userspace `libbnxt_re-rdmav34.so` provider)

## Where to get them
Download the `235.2.86.0` debs / DKMS packages from Broadcom's public repository:

- **https://packages.broadcom.com/artifactory/ethernet-nic-debian-public/pool/main/**

Look for `bnxt-en-dkms_1.10.3.235.2.86.0_*.deb`, `bnxt-re-dkms_235.2.86.0_*.deb`,
`bnxt-re-conf_235.2.86.0_*.deb`, and `bnxt-rocelib_235.2.86.0_*.deb`.

## Install
Use `../scripts/install_driver_235.sh` (deb or DKMS-tarball path), then **reboot**. Verify:

```bash
modinfo bnxt_re | grep ^version        # -> 235.2.86.0
ibv_devinfo -d rdma3 | grep PORT_ACTIVE
```

`rocelib-README.TXT` (kept in this folder) is Broadcom's own userspace/rocelib readme for reference.
