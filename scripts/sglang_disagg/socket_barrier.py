import argparse
import os
import socket
import sys
import threading
import time

DEFAULT_TIMEOUT_SECONDS = 3600


def _default_timeout():
    """Read BARRIER_TIMEOUT_SECONDS, rejecting garbage with a readable message.

    This runs on every node of every run, so a typo here would otherwise surface as
    a bare ValueError traceback on all of them at import time, before argparse can
    say anything useful. Unset or empty means "not configured" and falls back to the
    default; anything else set but unparseable is an operator mistake and is
    reported rather than silently replaced -- quietly substituting a different
    timeout than the one asked for is how you get a run that behaves nothing like
    its configuration.
    """
    raw = os.environ.get("BARRIER_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return int(raw)
    except ValueError:
        print(
            f"ERROR: BARRIER_TIMEOUT_SECONDS must be an integer number of seconds "
            f"(0 = wait forever), got {raw!r}.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Optionally open and close a port on the local node.")
parser.add_argument("--local-ip", required=False, help="Local IP address to bind the server.")
parser.add_argument("--local-port", type=int, required=False, help="Port number to bind the server.")
parser.add_argument("--enable-port", action="store_true", help="Enable opening and closing of local port.")
parser.add_argument("--node-ips", required=True, help="Comma-separated list of node IPs.")
parser.add_argument("--node-ports", required=True, help="Comma-separated list of ports to check.")
parser.add_argument(
    "--timeout",
    type=int,
    default=_default_timeout(),
    help=f"Give up after this many seconds (0 = wait forever). Default "
    f"{DEFAULT_TIMEOUT_SECONDS}, override with BARRIER_TIMEOUT_SECONDS.",
)
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

def wait_for_all_ports():
    """Wait until all nodes have opened the specified ports.

    Bounded, because an unbounded wait turns one node's failure into a whole
    allocation held until the SLURM wall clock. Observed on a 4-node run: rank 0's
    readiness wait timed out and tore its servers down, and every other rank sat
    here printing "Waiting for nodes. . ." for the remaining two hours because
    nothing told it the peer was gone. Failing here instead lets SLURM reclaim the
    nodes. Pass --timeout 0 (or BARRIER_TIMEOUT_SECONDS=0) for the old behaviour.
    """
    deadline = None if args.timeout <= 0 else time.monotonic() + args.timeout
    while True:
        missing = [
            f"{ip}:{port}"
            for ip, port in zip(NODE_IPS, NODE_PORTS)
            if not is_port_open(ip, port)
        ]
        if not missing:
            return
        if deadline is not None and time.monotonic() >= deadline:
            print(
                f"ERROR: barrier timed out after {args.timeout}s; "
                f"{len(missing)}/{len(NODE_IPS)} peer(s) never opened their port: "
                + ", ".join(missing),
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        print(f"Waiting for nodes. . . ({len(missing)} not ready)", flush=True)
        time.sleep(5)

def open_port():
    """Open a listening socket on the current node."""
    global server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.local_ip, args.local_port))
    server_socket.listen(5)
    print(f"Port {args.local_port} is now open on {args.local_ip}.")
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
        threading.Thread(target=open_port, daemon=True).start()

    wait_for_all_ports()

    if args.enable_port:
        time.sleep(30)
        close_port()