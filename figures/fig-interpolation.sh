#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh interpolation_log_length

# Need to run sequentially for interpolation
wait

bash ${SCRIPTPATH}/pipeline.sh interpolation_log_scale
