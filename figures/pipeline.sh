#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
pipeline=$1

output_dir=${SCRIPTPATH}/../output/${pipeline}
mkdir -p ${output_dir}
cp ${SCRIPTPATH}/pipelines/${pipeline}.json ${output_dir}/_pipeline.json
python3 ${SCRIPTPATH}/../scripts/pipeline.py ${output_dir}/_pipeline.json
