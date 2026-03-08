#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
data_dir=$1
meshname=$2
num_quads=$3
podman run \
    -v ${data_dir}:${data_dir} \
    --replace \
    --rm \
    --name ${meshname} \
    localhost/bearquad:latest \
    /bin/sh -c "
        cd ${data_dir};
        BearQuad \
        --type uv \
        --quant arc \
        --objective rel \
        --save-intermediates false \
        --out-json metadata.json \
        --uv-scale-mode nquads \
        --param-max-iter 100 \
        --uv-scale-arg ${num_quads} \
        -i ${meshname}.obj
    " &> ${data_dir}/${meshname}_bearquad.log

