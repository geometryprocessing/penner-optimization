#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh initial_0_5 &
bash ${SCRIPTPATH}/pipeline.sh initial_1_5 &
bash ${SCRIPTPATH}/pipeline.sh initial_2_5 &
