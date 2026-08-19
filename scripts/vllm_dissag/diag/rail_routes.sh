#!/bin/bash
# Enable cross-RAIL RoCEv2 reachability on skyRiver.  add | del | show [node ...]
#
# ============================================================================
# ROOT CAUSE (confirmed by measurement 2026-08-15 -- read this before editing)
# ============================================================================
# MoRI builds an unconditional full QP mesh (context.cpp:408,:464), so EP16 across
# 2 nodes forms QPs between DIFFERENT rails, e.g. <nodeA-fabric-ip> -> <nodeB-fabric-ip>.
# Those failed with:
#   bnxt_re_resolve_eth_dmac: Failed to resolve gid dmac: -110
#   -> mori bnxt.cpp:417 ModifyInit2Rtr: Assertion `!status' failed
#
# TWO independent host-side faults had to be fixed. Either one alone still fails:
#
#   (1) NO ROUTE on the SENDER.  Nodes carry only the eight on-link scope-link
#       routes, so a cross-rail peer is unroutable -> nothing to ARP -> -110.
#       The switch DOES route: every 192.168.20X.0/24 has a live gateway .254,
#       all answering with one MAC (98:4a:6b:6c:e8:9a) = one router, SVI per rail.
#
#   (2) rp_filter ON THE RECEIVER.  net.ipv4.conf.all.rp_filter=1 and the kernel
#       takes max(all, per-dev), so strict RPF applied on every bond even though
#       each per-dev knob read 0. A packet from <nodeA-fabric-ip> arriving on bond5
#       reverse-resolves to bond0, not bond5 -> silently dropped.
#       PROOF: nstat TcpExtIPReversePathFilter on the receiver incremented 1:1
#       with pings sent (11->16 for 5 pings, 16->25 for 9). The frames were
#       arriving the whole time; the host was discarding them.
#
#   After BOTH: node1:<nodeA-fabric-ip> -> node2:<nodeB-fabric-ip> = 0% loss, 0.220 ms.
#
# SUPERSEDED THEORIES (do not re-derive):
#   - "rails are isolated L2, cross-rail impossible"          -- WRONG, it routes.
#   - "the .254 SVIs answer ARP/ICMP but do not forward"      -- WRONG, they forward.
#     That came from a bad test: `ping -I bond0 <dst>` uses an INTERFACE NAME, which
#     is SO_BINDTODEVICE. It bypasses the policy rule and makes the kernel ARP for the
#     DESTINATION on-link (neigh -> INCOMPLETE). Always test with the SOURCE ADDRESS
#     form: `ping -I <nodeA-fabric-ip> <dst>`.
#   - MORI_RDMA_TC/SL 104/3 vs 41/0 -- QoS marking, cannot create reachability. No effect.
#
# WHY POLICY TABLES AND NOT `main`:
#   main keeps one route per prefix, so eight bonds cannot each hold a `via` route to
#   the same remote rail. Each source address gets its own table (id = rail number,
#   200..207; verified free -- rt_tables has only local/main/default/unspec) saying
#   "to reach any other rail, exit via MY rail's gateway".
#
# SAFETY:
#   - Routing + rp_filter only. No PFC/ETS/DCB, no addresses, no link state, no bond
#     params. Nothing the no-host-fabric-QoS rule covers.
#   - `del` restores exactly: rules/tables removed, rp_filter written back from the
#     per-node baseline snapshot saved by `add`.
#   - Runtime only. ifcfg-bond* carry no GATEWAY= and there are no route-* files, so a
#     reboot wipes all of this regardless -- re-run `add` after any reboot.
# ============================================================================
set -uo pipefail
ACT=${1:?usage: rail_routes.sh add|del|show <node> [node ...]}; shift || true
# Nodes are REQUIRED -- no default list. This writes routes and a sysctl on every node
# it is handed, so it must never guess which machines those are.
NODES=("$@")
[ ${#NODES[@]} -eq 0 ] && { echo "ERROR: name the nodes, e.g. rail_routes.sh $ACT n01 n02 n03 n04"; exit 2; }

BASE=/var/tmp/rail_routes_rpfilter.baseline   # per-node revert snapshot
PRIO=1200                                     # well below main (32766)

for n in "${NODES[@]}"; do
  echo "########## $n ($ACT) ##########"
  case "$ACT" in
  show)
    ssh -n "$n" '
      echo "--- rules ---"; ip rule show | grep -v "^0:\|^32766:\|^32767:" || echo "  (none)"
      echo "--- rail tables ---"
      for t in 200 201 202 203 204 205 206 207; do
        r=$(ip route show table $t 2>/dev/null)
        [ -n "$r" ] && { echo "  table $t:"; echo "$r" | sed "s/^/    /"; }
      done
      echo "--- rp_filter ---"
      echo "  all=$(cat /proc/sys/net/ipv4/conf/all/rp_filter) default=$(cat /proc/sys/net/ipv4/conf/default/rp_filter) bond0=$(cat /proc/sys/net/ipv4/conf/bond0/rp_filter)"
    ' ;;

  add)
    ssh -n "$n" "
      # Snapshot rp_filter once, so del can restore byte-exact.
      [ -f $BASE ] || for f in /proc/sys/net/ipv4/conf/*/rp_filter; do echo \"\$f=\$(cat \$f)\"; done > $BASE

      # Loose RPF (2): accept a packet if the source is reachable via ANY interface.
      # Required because cross-rail arrivals are legitimately asymmetric.
      sysctl -qw net.ipv4.conf.all.rp_filter=2 net.ipv4.conf.default.rp_filter=2
      for d in /proc/sys/net/ipv4/conf/*/rp_filter; do echo 2 > \$d 2>/dev/null; done

      # Derive rail<->bond<->src live from sysfs; never hardcode (devices move on reboot).
      for b in \$(ip -br -4 addr show | awk '\$1 ~ /^bond[0-7]\$/ && \$3 ~ /^192\.168\.20/ {print \$1}'); do
        src=\$(ip -br -4 addr show \$b | awk '{print \$3}' | cut -d/ -f1)
        rail=\$(echo \$src | cut -d. -f3)          # 200..207, doubles as the table id
        gw=192.168.\$rail.254
        ip rule del from \$src lookup \$rail 2>/dev/null   # idempotent: no duplicate rules
        ip rule add from \$src lookup \$rail priority $PRIO
        ip route flush table \$rail 2>/dev/null
        for r in 200 201 202 203 204 205 206 207; do
          [ \"\$r\" = \"\$rail\" ] && continue
          ip route add 192.168.\$r.0/24 via \$gw dev \$b src \$src table \$rail
        done
        # Same-rail traffic must stay on-link, not detour through the router.
        ip route add 192.168.\$rail.0/24 dev \$b scope link src \$src table \$rail
      done
      echo \"  rules=\$(ip rule show | grep -c 'lookup 20') rp_filter_all=\$(cat /proc/sys/net/ipv4/conf/all/rp_filter)\"
    " ;;

  del)
    ssh -n "$n" "
      for b in \$(ip -br -4 addr show | awk '\$1 ~ /^bond[0-7]\$/ && \$3 ~ /^192\.168\.20/ {print \$1}'); do
        src=\$(ip -br -4 addr show \$b | awk '{print \$3}' | cut -d/ -f1)
        rail=\$(echo \$src | cut -d. -f3)
        while ip rule del from \$src lookup \$rail 2>/dev/null; do :; done
        ip route flush table \$rail 2>/dev/null
      done
      if [ -f $BASE ]; then
        while IFS='=' read -r f v; do [ -w \"\$f\" ] && echo \"\$v\" > \"\$f\"; done < $BASE
        rm -f $BASE; echo '  rp_filter restored from baseline'
      else
        echo '  WARN: no baseline file; rp_filter left as-is'
      fi
      echo \"  rules_left=\$(ip rule show | grep -c 'lookup 20') rp_filter_all=\$(cat /proc/sys/net/ipv4/conf/all/rp_filter)\"
    " ;;
  *) echo "unknown action: $ACT"; exit 2 ;;
  esac
done
