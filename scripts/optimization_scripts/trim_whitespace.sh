#! /bin/bash

# Optimization 
input_dir=$1
file_list=($(ls ${input_dir}))
output_dir=$2
mkdir -p ${output_dir}

for file in ${file_list[@]}
do
  if [[ "$file" = *".png" ]]; then
    frame=${file%.*}
    convert ${input_dir}/${frame}.png -trim ${output_dir}/${frame}.png
  fi
done
