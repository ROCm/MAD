#!/bin/bash
# Test only (tests/coverage G4): a "reproducer" that reproduces only on hosts whose name ends in -08 -- the shape of
# a node-dependent bug, without a GPU. verify_nodes.sh must report NODE_DEPENDENT when it gets one such host and one
# other.
h=$(hostname -s)
case $h in *-08) echo "LADDER_RESULT=REPRODUCED at concurrency 1 -- fake, host $h" ;;
            *)    echo "LADDER_RESULT=NOT_REPRODUCED -- fake, host $h" ;; esac
