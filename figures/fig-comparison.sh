#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/pipeline.sh comparison_fixed &
bash ${SCRIPTPATH}/pipeline.sh comparison_quadratic &
