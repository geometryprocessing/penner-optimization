#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")

python3 scripts/script_pipeline.py ${SCRIPTPATH}/standard_deviation_0_5/_pipeline.json &
python3 scripts/script_pipeline.py ${SCRIPTPATH}/standard_deviation_1_5/_pipeline.json &
python3 scripts/script_pipeline.py ${SCRIPTPATH}/standard_deviation_2_5/_pipeline.json &