#!/bin/bash

#define the variable here
container_image_name=$1
yaml_file_fullpath=$2

# Extract filename
input_yaml_file_name=$(basename "$yaml_file_fullpath")

# Extract directory path
user_workspace=$(dirname "$yaml_file_fullpath")

echo User Work Directory: $user_workspace
echo Input Yaml Filename: $input_yaml_file_name

compiler_root_dir=bin 
executor_root_dir=/usr/local/executor_app/
executor_work_dir=executor_input_workspace

workspace_dir=$executor_root_dir$executor_work_dir


#first we need to build the docker
docker build -t $container_image_name ./ --build-arg compiler_root=$compiler_root_dir --build-arg yaml_file=$input_yaml_file_name --build-arg root_dir=$executor_root_dir --build-arg work_dir=$executor_work_dir

#and then run it using the docker run command with the commands as needed
docker run -v $user_workspace:$workspace_dir $container_image_name "./$executor_work_dir/$input_yaml_file_name"
