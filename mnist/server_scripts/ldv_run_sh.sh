#!/bin/bash


COMMAND = "
sh ldv-3090-ws01.sh
"

# ldv-3090-ws01
USER="ga63vuf"
HOST="ldv-3090-ws01"
scp ../*.py ${USER}@${HOST}:/nas/netstore/ldv/ga63vuf/WorkSpace/essl_isometry/mnist/
scp ../server_scripts/ ${USER}@${HOST}:/nas/netstore/ldv/ga63vuf/WorkSpace/essl_isometry/mnist/server_scripts/

ssh -l ${USER} ${HOST} ${COMMAND}