# A dockerfile must always start by importing the base image.
# We use the keyword 'FROM' to do that.
# For our executor, we want import the python image.
# So we write 'python' for the image name and '3.11.8-slim' for the version.

FROM python:3.11.8-slim

ARG compiler_root
ARG yaml_file
ARG work_dir
ARG root_dir

WORKDIR ${root_dir}

ENV COMPILER_ROOT_FOLDER=./app

COPY $compiler_root ./app
COPY include/nrf_axon_nn_compiler_types.h include/nrf_axon_nn_compiler_types.h
COPY scripts/utility ./scripts/utility
COPY scripts/requirements.txt ./requirements.txt
COPY scripts/tflite_converter.py ./scripts/
COPY scripts/user_handler_functions.py ./scripts/
COPY scripts/compare_models.py ./scripts/
COPY scripts/compile_model_helper.py ./scripts/
COPY scripts/axons_ml_nn_compiler_executor.py ./scripts/
COPY scripts/model_wrapper.py ./scripts/

RUN echo ${compiler_root}
RUN echo ${root_dir}
RUN echo ${work_dir}
RUN echo ${yaml_file}

RUN pip install --upgrade pip
RUN pip install --no-compile --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python3", "./scripts/axons_ml_nn_compiler_executor.py"]

CMD [ "python3", "./scripts/axons_ml_nn_compiler_executor.py", "./executor_input_workspace/input.yaml"]
# CMD ["python3","--version"]