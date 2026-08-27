import html
import datetime
import socket
from typing import Dict, List

class RDMAHtmlReporter:
    def __init__(self, mapper):
        self.mapper = mapper

    def _esc(self, x):
        return html.escape(str(x))

    def generate(self, output_file: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        host = socket.gethostname()

        devices = self.mapper.print_table()
        fw_map = self.mapper.get_firmware_report()
        env_variables = self.mapper.get_framework_env_variables()
        docker_cmds = self.mapper.get_docker_recommendation()

        html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RDMA Cluster Report</title>
<style>
body {{ font-family: Arial; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; }}
th {{ background: #f0f0f0; }}
code, pre {{ background: #f8f8f8; padding: 10px; display: block; }}
.warn {{ color: #c0392b; font-weight: bold; }}
</style>
</head>
<body>

<h1>RDMA Cluster Report</h1>
<p><b>Host:</b> {self._esc(host)}<br>
<b>Generated:</b> {self._esc(ts)}</p>

<h2>RDMA Device Summary</h2>
<table>
<tr>
<th>RDMA</th><th>PCI</th><th>Netdev</th>
<th>Firmware</th><th>GID IDX</th><th>GID</th><th>Vendor</th>
</tr>
"""

        for d in devices:
            html_doc += f"""
<tr>
<td>{self._esc(d['rdma'])}</td>
<td>{self._esc(d['pci'])}</td>
<td>{self._esc(d['netdev'])}</td>
<td>{self._esc(d['firmware'])}</td>
<td>{self._esc(d['gid_index'])}</td>
<td>{self._esc(d['gid_value'])}</td>
<td>{self._esc(d['vendor'])}</td>
</tr>
"""

        html_doc += "</table>"

        # ---------------- Firmware ----------------
        html_doc += "<h2>Firmware Report</h2>"
        for fw, devs in fw_map.items():
            html_doc += f"""
<p><b>{self._esc(fw)}</b>: {self._esc(", ".join(devs))}</p>
"""

        if len(fw_map) > 1:
            html_doc += "<p class='warn'>Multiple firmware versions detected</p>"

        # ---------------- Env Vars ----------------
        html_doc += "<h2>Recommended Environment Variables</h2>"
        html_doc += "<h3>NCCL / GLOO</h3><pre>"
        for v in nccl_env:
            html_doc += self._esc(v) + "\n"
        html_doc += "</pre>"

        html_doc += """
<h3>rocSHMEM</h3>
<pre>
export ROCSHMEM_HEAP_SIZE=7524589824
export ROCSHMEM_MAX_NUM_CONTEXTS=256
</pre>
"""

        # ---------------- Docker ----------------
        html_doc += "<h2>Docker Launch Commands</h2>"
        for vendor, cmd in docker_cmds.items():
            html_doc += f"""
<h3>{self._esc(vendor)}</h3>
<pre>{self._esc(cmd)}</pre>
"""

        html_doc += "</body></html>"

        with open(output_file, "w") as f:
            f.write(html_doc)

