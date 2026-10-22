#!/bin/bash
# Standardize a node to bnxt 235.2.86.0 from ANY prior state (236 in-box / 237 / 238 DKMS).
# Run as root ON the node. Reads the driver package from NFS (/mnt/nfs/cookbook/bnxt-235).
# Handles the logbook gotchas: purge prior-version DKMS + stale updates/*.ko, copy bnxt_en.ko,
# match the userspace libbnxt_re provider. Does NOT reboot (caller reboots).
set -uo pipefail
PKG=/mnt/nfs/cookbook/bnxt-235
KV=$(uname -r)
log(){ echo "[std-235 $(hostname)] $*"; }

log "start; current bnxt_re=$(modinfo bnxt_re 2>/dev/null | awk '/^version/{print $2}')"

# 1. Remove ALL prior bnxt DKMS modules (any version != 235)
for mod in bnxt_en bnxt_re; do
  for ver in $(dkms status 2>/dev/null | sed -n "s#^${mod}/\([^,]*\),.*#\1#p" | sort -u); do
    if [ "$ver" != "1.10.3.235.2.86.0" ] && [ "$ver" != "235.2.86.0" ]; then
      log "dkms remove ${mod}/${ver}"
      dkms remove "${mod}/${ver}" --all 2>/dev/null || true
    fi
  done
done

# 2. Purge dpkg bnxt packages (DKMS debs) so old versions don't relink on update
for p in bnxt-re-dkms bnxt-en-dkms bnxt-re-conf bnxt-rocelib; do
  dpkg -l 2>/dev/null | grep -q "^ii  $p " && { log "purge $p"; apt-get -y remove --purge "$p" 2>/dev/null || dpkg --purge --force-all "$p" 2>/dev/null || true; }
done

# 3. Purge stale updates/*.ko (the "disagrees about symbol" trap)
#    IMPORTANT: prior installs leave bnxt modules in BOTH updates/dkms/ AND
#    updates/drivers/infiniband/hw/bnxt_re/ (+ .../ethernet/broadcom/bnxt/). The latter
#    path SHADOWS the dkms one at boot, so it must be removed or the node boots the OLD driver.
rm -f /lib/modules/$KV/updates/dkms/bnxt_en.ko /lib/modules/$KV/updates/dkms/bnxt_re.ko 2>/dev/null
rm -f /lib/modules/$KV/updates/bnxt_en.ko /lib/modules/$KV/updates/bnxt_re.ko 2>/dev/null
# remove any non-dkms bnxt .ko anywhere under updates/ (the driver-path shadow)
find /lib/modules/$KV/updates -name 'bnxt_re.ko' -o -name 'bnxt_en.ko' 2>/dev/null | grep -v '/updates/dkms/' | xargs -r rm -f
rm -rf /lib/modules/$KV/updates/drivers/infiniband/hw/bnxt_re 2>/dev/null

# 4. Build+install 235 from the NFS DKMS source tarball
rm -rf /usr/src/bnxt_en-1.10.3.235.2.86.0 /usr/src/bnxt_re-235.2.86.0 2>/dev/null
tar xzf "$PKG"/bnxt-dkms-src-235.2.86.0.tar.gz -C /usr/src
dkms add    bnxt_en/1.10.3.235.2.86.0 2>/dev/null || true
dkms add    bnxt_re/235.2.86.0        2>/dev/null || true
dkms build  bnxt_en/1.10.3.235.2.86.0 || { log "FAIL build bnxt_en"; exit 1; }
dkms build  bnxt_re/235.2.86.0        || { log "FAIL build bnxt_re"; exit 1; }
dkms install --force bnxt_en/1.10.3.235.2.86.0
dkms install --force bnxt_re/235.2.86.0

# 5. Ensure BOTH .ko are in updates/dkms (DKMS often copies only bnxt_re.ko)
mkdir -p /lib/modules/$KV/updates/dkms
cp -f /var/lib/dkms/bnxt_en/1.10.3.235.2.86.0/$KV/x86_64/module/bnxt_en.ko /lib/modules/$KV/updates/dkms/bnxt_en.ko 2>/dev/null || true
cp -f /var/lib/dkms/bnxt_re/235.2.86.0/$KV/x86_64/module/bnxt_re.ko       /lib/modules/$KV/updates/dkms/bnxt_re.ko 2>/dev/null || true

# 6. Userspace provider: install the 235 libbnxt_re-rdmav34.so
tar xzf "$PKG"/bnxt-rocelib-235.2.86.0.tar.gz -C /usr/local/lib 2>/dev/null || true
ln -sf /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/local/lib/libbnxt_re-rdmav34.so 2>/dev/null || true
# also refresh the libibverbs provider dir copy if present
if [ -e /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so ]; then
  cp -f /usr/local/lib/x86_64-linux-gnu/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so 2>/dev/null || true
fi
ldconfig 2>/dev/null || true

# 7. Rebuild module deps + initramfs so 235 loads at boot
depmod -a
update-initramfs -u 2>/dev/null || true

log "built 235; dkms status:"; dkms status 2>/dev/null | grep -i bnxt
log "DONE — REBOOT required to load matched 235 modules + clear any FW cmdq stall."
