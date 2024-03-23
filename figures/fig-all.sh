#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

bash ${SCRIPTPATH}/fig-comparison.sh &
bash ${SCRIPTPATH}/fig-disk.sh &
bash ${SCRIPTPATH}/fig-examples.sh &
bash ${SCRIPTPATH}/fig-initial.sh &
bash ${SCRIPTPATH}/fig-interpolation.sh &
bash ${SCRIPTPATH}/fig-objectives.sh &
bash ${SCRIPTPATH}/fig-teaser.sh &
