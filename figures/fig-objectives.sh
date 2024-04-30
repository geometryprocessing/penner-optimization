#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh objectives_log_length &
bash ${SCRIPTPATH}/pipeline.sh objectives_log_length_p4 &
bash ${SCRIPTPATH}/pipeline.sh objectives_log_scale &
