#!/usr/bin/env python3
"""placement.py — compute replica placement from (TP, DP) across cluster nodes.

Core abstraction for scalable serving of a model that FITS IN ONE NODE:
each replica is node-local (TP <= gpus_per_node), and we place
(DP per node) replicas on each node, replicated across all nodes.

  replicas_per_node = DP            (caller picks DP; DP*TP must be <= gpus_per_node)
  total_replicas    = len(nodes) * DP
  full utilization  <=> DP * TP == gpus_per_node

Emits one line per replica:  NODE  GPUS  PORT  REPLICA_GLOBAL_IDX
e.g.  node-01  0,1,2,3  8000  0
"""
import sys, yaml, os

def load(p):
    with open(p) as f: return yaml.safe_load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--dp", type=int, default=None,
                    help="replicas per node; default = gpus_per_node // tp (full utilization)")
    ap.add_argument("--nodes", default=None,
                    help="comma-separated node override (default: cluster.yaml nodes)")
    args = ap.parse_args()

    c = load(args.cluster)
    G = int(c["gpus_per_node"]); port_base = int(c.get("port_base", 8000))
    nodes = args.nodes.split(",") if args.nodes else c["nodes"]
    tp = args.tp
    dp = args.dp if args.dp is not None else (G // tp)

    if tp > G:
        sys.exit(f"ERROR: TP={tp} > gpus_per_node={G} (model must fit in one node; no multi-node TP)")
    if dp * tp > G:
        sys.exit(f"ERROR: DP*TP = {dp}*{tp} = {dp*tp} > gpus_per_node={G}")

    for ni, node in enumerate(nodes):
        for r in range(dp):                      # r = replica index on this node
            gpus = ",".join(str(g) for g in range(r*tp, (r+1)*tp))
            port = port_base + r
            gidx = ni * dp + r                   # global replica index
            print(f"{node}\t{gpus}\t{port}\t{gidx}")

if __name__ == "__main__":
    main()
