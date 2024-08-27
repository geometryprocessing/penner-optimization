#! /bin/bash

input_dir=$1
output_dir=$2

build_dir="build"
sym_opt_dir="../wildmeshing-toolkit/"
sym_opt_build_dir="${sym_opt_dir}/build/"
cut_mesh_dir="output/cut_meshes"
suffix="refined_with_uv"
mkdir -p ${cut_mesh_dir}
mkdir -p ${cut_mesh_dir}/EE
mkdir -p ${output_dir}
model_list=($(ls ${input_dir}))

for file in ${model_list[@]}
do
  if [[ "$file" = *"_output" ]]; then
    mesh_name=${file%_output}
		./${build_dir}/bin/cut_mesh \
		    --mesh ${input_dir}/${mesh_name}_output/${mesh_name}_${suffix}.obj \
		    --mesh_name ${mesh_name} \
			  --output ${cut_mesh_dir}/ &
  fi
done
wait

for file in ${model_list[@]}
do
  if [[ "$file" = *"_output" ]]; then
    mesh_name=${file%_output}
		${sym_opt_build_dir}/app/extreme_opt/extreme_opt \
		  -i ${cut_mesh_dir} \
			-j ${sym_opt_dir}/json/example.json \
			-o ${output_dir} \
			-m ${mesh_name} &
  fi
done
wait

for file in ${model_list[@]}
do
  if [[ "$file" = *"_output" ]]; then
    mesh_name=${file%_output}
    mkdir -p ${output_dir}/${mesh_name}_output
    mv ${output_dir}/${mesh_name}_out.obj ${output_dir}/${mesh_name}_output/${mesh_name}_optimized_uv.obj
    mv ${output_dir}/${mesh_name}.json ${output_dir}/${mesh_name}_output/
  fi
done
wait
