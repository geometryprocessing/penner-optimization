#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh disk_quadratic &
bash ${SCRIPTPATH}/pipeline.sh disk_slim &