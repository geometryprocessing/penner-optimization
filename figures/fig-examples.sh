#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh examples_closed &
bash ${SCRIPTPATH}/pipeline.sh examples_open &
bash ${SCRIPTPATH}/pipeline.sh examples_cut &
