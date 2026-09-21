#!/usr/bin/env python3
"""topo.py — resolve a DSv4 disagg topology into a node-role placement.

Usage:
  topo.py <cluster.yaml> <model.yaml> <topo_name> roles
      -> TSV lines: role<TAB>worker_idx<TAB>node<TAB>port
         role in {prefill, decode}; node 0 of prefill also hosts the router.
  topo.py <model.yaml> <topo_name> field <key>
      -> scalar/list field of the topology (xP, yD, shapes, conc, ...)

Topology drives node count: nodes_needed = xP*pf_nodes_per_worker + yD*dec_nodes_per_worker,
with nodes_per_worker = ceil(TP / gpus_per_node) (TP8 on 8-GPU node => 1 node/worker).
"""
import sys, yaml, math

def load(f): return yaml.safe_load(open(f))

def roles(cluster_f, model_f, topo_name):
    c = load(cluster_f); m = load(model_f)
    gpn = c["gpus_per_node"]; nodes = c["nodes"]
    t = m["topologies"][topo_name]
    xP, yD = t["xP"], t["yD"]
    pf_tp = t["prefill"]["tp"]; dec_tp = t["decode"]["tp"]
    pf_npw = math.ceil(pf_tp / gpn); dec_npw = math.ceil(dec_tp / gpn)
    need = xP * pf_npw + yD * dec_npw
    if need > len(nodes):
        sys.exit(f"ERROR: topo {topo_name} needs {need} nodes, cluster has {len(nodes)}")
    pf_port = c["prefill_port"]; dec_port = c["decode_port"]
    out = []; i = 0
    for w in range(xP):
        out.append(f"prefill\t{w}\t{nodes[i]}\t{pf_port}"); i += pf_npw
    for w in range(yD):
        out.append(f"decode\t{w}\t{nodes[i]}\t{dec_port}"); i += dec_npw
    return out

def main():
    a = sys.argv
    if a[2:3] == ["roles"] or (len(a) >= 4 and a[3] == "roles"):
        # form: topo.py cluster model topo roles  -> but support both orderings
        pass
    if len(a) >= 5 and a[4] == "roles":
        print("\n".join(roles(a[1], a[2], a[3]))); return
    if len(a) >= 5 and a[3] == "field":
        m = load(a[1]); t = m["topologies"][a[2]]; v = t.get(a[4])
        if isinstance(v, list): print(" ".join(str(x) for x in v))
        elif isinstance(v, dict): print(" ".join(f"{k}={vv}" for k, vv in v.items()))
        else: print("" if v is None else v)
        return
    sys.exit("usage: topo.py <cluster> <model> <topo> roles | topo.py <model> <topo> field <key>")

if __name__ == "__main__":
    main()
