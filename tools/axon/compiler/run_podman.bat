::define the variable here
@echo off
set container_image_name=%1
set yaml_file_fullpath=%2

:: Extract the directory path
for %%I in ("%yaml_file_fullpath%") do set "user_workspace=%%~dpI"
echo User Work Directory: %user_workspace%

:: Extract the filename with extension
for %%I in ("%yaml_file_fullpath%") do set "input_yaml_file_name=%%~nxI"
echo Input Yaml Filename: %input_yaml_file_name%

set compiler_root_dir=bin
set executor_root_dir=/usr/local/executor_app/
set executor_work_dir=executor_input_workspace

set workspace_dir=%executor_root_dir%%executor_work_dir%


podman machine start

::#first we need to build the docker
podman build -t %container_image_name% ./ --build-arg compiler_root=%compiler_root_dir% --build-arg yaml_file=%input_yaml_file_name% --build-arg root_dir=%executor_root_dir% --build-arg work_dir=%executor_work_dir%

::#and then run it using the docker run command with the commands as needed
podman run -v %user_workspace%:%workspace_dir% %container_image_name% "./%executor_work_dir%/%input_yaml_file_name%"

podman machine stop