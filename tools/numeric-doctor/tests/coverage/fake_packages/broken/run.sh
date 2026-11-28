#!/bin/bash
# Test only (tests/coverage G4): a package that fails before printing a result -- verify_nodes.sh must count it as
# ERROR, never as NOT_REPRODUCED.
echo "fake package: missing image" >&2
exit 3
