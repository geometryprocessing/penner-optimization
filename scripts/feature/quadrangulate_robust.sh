#! /bin/bash
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
data_dir=$1
meshname=$2
#num_quads=$3
scale=$3
test_name=$4
allow_zero_arcs=$5

# --strategy flow-eager is faster, but less tested
# --strategy QGP is slower, but more well tested
#        obj_import \
#            --input ${meshname}.obj \
#            --output-om ${meshname}.om \
#            --uv-scale-mode nquads \
#            --uv-scale-arg ${num_quads}
#    localhost:5000/quantmesh:latest \
#    localhost:5000/quantmesh:fastsanit \
#            --final-param ${final_param} \
#            --strategy QGP
#            --strategy flow-eager
#    localhost:5000/quantmesh:turbo_featvert \

podman run \
    -v ${data_dir}:${data_dir} \
    --rm \
    --replace \
    --name ${meshname}_${test_name} \
    localhost:5000/quantmesh:latest \
    /bin/sh -c "
        cd ${data_dir};
        igmtest \
            -i ${meshname}.obj \
            -o ${meshname}_quad.obj \
            --json-output metadata.json \
            --recompute-cut-graph 0 \
            --allow-zero-arcs ${allow_zero_arcs} \
            -u${scale} -f1  \
            --strategy flow-eager
    " &> ${data_dir}/${meshname}_igmtest.log

