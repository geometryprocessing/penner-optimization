#! /bin/bash

input_dir=$1
uv_dir=$2
output_dir=$3

build_dir="build"
suffix="refined_with_uv"
model_list=($(ls ${input_dir}))
mkdir -p ${output_dir}

for file in ${model_list[@]}
do
  if [[ "$file" = *"_output" ]]; then
    mesh_name=${file%_output}
		./${build_dir}/bin/glue_mesh \
		    --mesh_name ${mesh_name} \
		    --mesh ${input_dir}/${mesh_name}_output/${mesh_name}_${suffix}.obj \
		    --uv_mesh ${uv_dir}/${mesh_name}_output/${mesh_name}_optimized_uv.obj \
			--output ${output_dir}/ &
  fi
done
wait

