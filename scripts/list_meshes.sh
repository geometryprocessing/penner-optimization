#! /bin/bash

input_dir=$1
file_list=($(ls ${input_dir}))
output_dir=$2

for file in ${file_list[@]}
do
  if [[ "$file" = *".obj" ]]; then
    "${file}", > ${output_dir}/mesh_list.txt
  fi
done
