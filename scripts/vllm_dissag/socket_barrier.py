import os
import socket
import sys
import time
import threading
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Optionally open and close a port on the local node.")
parser.add_argument("--local-ip", required=False, help="Local IP address to bind the server.")
parser.add_argument("--local-port", type=int, required=False, help="Port number to bind the server.")
parser.add_argument("--enable-port", action="store_true", help="Enable opening and closing of local port.")
parser.add_argument("--node-ips", required=True, help="Comma-separated list of node IPs.")
parser.add_argument("--node-ports", required=True, help="Comma-separated list of ports to check.")
# Both default to the old behaviour (wait forever, nothing to watch). Without them a node
# whose peer had already given up waited here until the job's wall clock: in SLURM job
# 442891 the prefill master timed out at 17:55 and the other three nodes sat in this loop
# until the job was cancelled by hand.
parser.add_argument("--timeout", type=float, default=0,
                    help="Give up (exit 1) after this many seconds; 0 waits forever.")
parser.add_argument("--abort-file",
                    help="Exit 1 as soon as this file exists: a peer has declared the job failed.")
args = parser.parse_args()

# Parse node IPs and ports from command-line arguments
NODE_IPS = [ip.strip() for ip in args.node_ips.split(",") if ip.strip()]
NODE_PORTS = [int(port.strip()) for port in args.node_ports.split(",") if port.strip()]

# Ensure port list matches node list or default to using the same port for all nodes
if len(NODE_PORTS) == 1:
    NODE_PORTS *= len(NODE_IPS)
elif len(NODE_PORTS) != len(NODE_IPS):
    print("Error: Number of ports must match number of node IPs or only one port should be given for all.")
    exit(1)

server_socket = None  # Global server socket reference

def is_port_open(ip, port):
    """Check if a given IP and port are accessible."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)  # Avoid long wait times
        return s.connect_ex((ip, port)) == 0

def _give_up(reason):
    print(f"Barrier failed: {reason}", flush=True)
    sys.exit(1)

def wait_for_all_ports():
    """Wait until all nodes have opened the specified ports."""
    start = time.monotonic()
    while True:
        all_open = all(is_port_open(ip, port) for ip, port in zip(NODE_IPS, NODE_PORTS))
        if all_open:
            break
        if args.abort_file and os.path.exists(args.abort_file):
            try:
                why = open(args.abort_file).read().strip()
            except OSError:
                why = ""
            _give_up(f"a peer aborted the job ({args.abort_file}): {why or 'no reason recorded'}")
        if args.timeout and time.monotonic() - start >= args.timeout:
            closed = [f"{ip}:{port}" for ip, port in zip(NODE_IPS, NODE_PORTS) if not is_port_open(ip, port)]
            _give_up(f"timed out after {args.timeout:.0f}s; still closed: {', '.join(closed)}")
        print("Waiting for nodes. . .", flush=True)
        time.sleep(5)

def bind_port():
    """Bind the local barrier port, or fail the barrier.

    A bind that fails must not be survivable. It used to fail inside the accept thread,
    which printed a traceback and died while this node carried on waiting -- and every
    node's check then connected to whatever else was listening on that port and counted
    it as this node being ready. On OCI port 5000 is held on the hosts by something the
    job cannot kill, so the rixl/TP barrier passed on both nodes without either one
    having opened it.
    """
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((args.local_ip, args.local_port))
    except OSError as e:
        _give_up(f"cannot open port {args.local_port} on {args.local_ip}: {e}. Something else on "
                 f"this host holds it; set BARRIER_PORT to a free port.")
    server_socket.listen(5)
    print(f"Port {args.local_port} is now open on {args.local_ip}.")

def open_port():
    """Accept (and drop) connections on the bound barrier port."""
    while True:
        conn, addr = server_socket.accept()
        conn.close()

def close_port():
    """Close the opened port."""
    global server_socket
    if server_socket:
        server_socket.close()
        print(f"Port {args.local_port} has been closed on {args.local_ip}.")

if __name__ == "__main__":
    if not NODE_IPS:
        print("Error: NODE_IPS argument is empty or not set.")
        exit(1)

    if args.enable_port:
        bind_port()
        threading.Thread(target=open_port, daemon=True).start()

    wait_for_all_ports()

    if args.enable_port:
        time.sleep(30)
        close_port()