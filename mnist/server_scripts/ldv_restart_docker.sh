#!/bin/bash


# ldv-3090-ws01
USER="ga63vuf"
HOST="ldv-3090-ws01"
ssh -l ${USER} ${HOST} "docker restart ga63vuf"

