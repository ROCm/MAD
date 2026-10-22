#!/bin/bash
# Install Broadcom bnxt 235.2.86.0 RoCE driver (kernel + userspace) on an MI300X/Thor2 host.
# This clears the MoRI-EP DV-CQ blocker (bnxt_re_dv_create_cq errno 5 on 237/238 drivers).
# Firmware is LEFT UNTOUCHED (stays 238.1.138.x) — this is a DRIVER-ONLY change.
#
# Run as root on EACH node, then REBOOT. Verify NFS-over-RDMA survives (benic2/rdma1).
#
# Source of truth for the packages:
#   Public Broadcom repo (no login):
#   https://packages.broadcom.com/artifactory/ethernet-nic-debian-public/pool/main/
#   Packages needed (235.2.86.0):
#     bnxt-en-dkms_1.10.3.235.2.86.0_all.deb
#     bnxt-re-dkms_235.2.86.0_all.deb
#     bnxt-re-conf_235.2.86.0-1_all.deb
#     bnxt-rocelib_235.2.86.0-1_all.deb   (this build is glibc-2.34 => works on Ubuntu 22.04)
#
# This repo's driver-235.2.86.0/ folder ALSO ships:
#   - bnxt-dkms-src-235.2.86.0.tar.gz   (the DKMS source trees; rebuilds .ko for any kernel)
#   - bnxt-rocelib-235.2.86.0.tar.gz    (the built userspace libbnxt_re-rdmav34.so provider)
#   - bnxt_re.ko / bnxt_en.ko           (prebuilt for kernel 5.15.0-177-generic)
#
set -euo pipefail
PKGDIR="${1:-.}"   # dir containing the .deb files, OR use the tarballs below

echo "=== Option A: install from .deb (preferred, if you have them) ==="
if ls "$PKGDIR"/bnxt-*235.2.86.0*.deb >/dev/null 2>&1; then
  dpkg -i "$PKGDIR"/bnxt-en-dkms_1.10.3.235.2.86.0_all.deb \
          "$PKGDIR"/bnxt-re-dkms_235.2.86.0_all.deb \
          "$PKGDIR"/bnxt-re-conf_235.2.86.0-1_all.deb \
          "$PKGDIR"/bnxt-rocelib_235.2.86.0-1_all.deb
else
  echo "=== Option B: install from the DKMS source tarball in this repo ==="
  tar xzf "$PKGDIR"/bnxt-dkms-src-235.2.86.0.tar.gz -C /usr/src
  dkms add    bnxt_en/1.10.3.235.2.86.0 || true
  dkms add    bnxt_re/235.2.86.0        || true
  dkms build  bnxt_en/1.10.3.235.2.86.0
  dkms build  bnxt_re/235.2.86.0
  dkms install bnxt_en/1.10.3.235.2.86.0
  dkms install bnxt_re/235.2.86.0
  # userspace lib
  tar xzf "$PKGDIR"/bnxt-rocelib-235.2.86.0.tar.gz -C /usr/local/lib
  ln -sf /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so
fi

echo "=== CRITICAL: remove stale .ko from prior 237/238 installs ==="
# DKMS copies only bnxt_re.ko to updates/dkms, NOT bnxt_en.ko -> copy it manually,
# and purge any stale updates/*.ko or you get "disagrees about version of symbol
# bnxt_ulp_get_stats / bnxt_en_ulp_dcqcn_flow_create" and bnxt_re refuses to load.
KV=$(uname -r)
cp -f /var/lib/dkms/bnxt_en/1.10.3.235.2.86.0/$KV/x86_64/module/bnxt_en.ko \
      /lib/modules/$KV/updates/dkms/bnxt_en.ko 2>/dev/null || true
depmod -a
update-initramfs -u

echo "=== DONE. REBOOT this node now, then verify: ==="
echo "  modinfo bnxt_re | grep ^version   # expect 235.2.86.0"
echo "  ibv_devinfo -d rdma3 | grep -E 'fw_ver|PORT_ACTIVE'"
echo "  # confirm NFS still mounted (proto=rdma over benic2) before/after"
