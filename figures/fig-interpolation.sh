#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh interpolation_log_length &
bash ${SCRIPTPATH}/pipeline.sh interpolation_log_scale &
