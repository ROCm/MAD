#!/bin/bash
# setup_deps.sh — sourced by server_atom.sh at startup.
#
# In the InferenceX CI this installs/patches runtime deps into the container. The M3 ATOM
# disagg image (rocm/atom-dev:nightly_202607011530) is expected to ship everything needed:
# atomesh (/usr/local/bin/atomesh), mooncake (py3.12 site-packages +
# /usr/local/bin/mooncake_{master,client}), atom.entrypoints.openai_server.
#
# So this is intentionally a NO-OP stub. If a future image is missing a dep, add the
# install here. Kept as a separate file so server_atom.sh runs unmodified (verbatim vendor).
:
