.. _nn_compiler:

AXON NPU TFLITE COMPILER
########################

.. contents::
   :local:
   :depth: 3

Introduction
************
The Axon TFlite Compiler's primary task is to compile a .tflite file into object code that is consumed by the Axon NPU.
As part of this task, the compiler can also optionally perform inference on a test data set to confirm that the accuracy matches that of tflite.

There are two major components to the process, an executor script and a compiler shared library.
The executor script (`<axons_ml_nn_compiler_executor.py>`_) is responsible for pre-processing the .tflite file and translating it to a format known by the compiler.
The executor script then invokes the compiler library (`<../bin/Windows/axons_ml_nn_compiler_lib_amd64.dll>`_ or `<../bin/Linux/libaxons_ml_nn_compiler_lib_amd64.so>`_ or `<../bin/Darwin/libaxons_ml_nn_compiler_lib_arm64.dylib>`_). The compiler library is responsible for generating the object code and writing the results to a header that is then included in the application build.

Input parameters are conveyed to the executor script via a .yaml file. A template of this file is provided in `<compiler_input_yaml_template.yaml>`_. Parameters are described in detail below.

Output is placed in an output folder underneath the folder containing the .yaml file.
The output is a c header file containing the compiled axons object and referenced constants. If a test data set is provided, a test vectors header file is also generated that can be used to verify the model's performance on-target.

Directories
***********
workspace
-------------------
The workspace directory is where the yaml file is located. All paths in the .yaml file are relative to this directory (unless absolute paths are provided). The output folder is directly below the workspace folder.

tools
------
Users should not modify the directory structure. The executor expects the header files it needs in the include folder parallel to scripts, and the library to also be in a folder parallel to scripts. (The user will be able to build the library in a future release that supports user created operations).

The conda environment must be executed from this directory in order to execute the scripts.

Setting up the Executor
***********************
The current python projects support python version 3.11.8. 

NOTE:  The various commands in the following explanation are used on windows command line. If you are using a different command line interface like PS or Linux bash, modify the commands accordingly.

Setting up the Virtual environment
------------------------------------------
It is advised to set up a virtual environment to run python projects. The virtual environment provides an abstraction for the different python packages installed for running executor successfully. 

There are different ways you can set up an environment. An example using the miniconda is explained below.

Install Miniforge3
=========================================
Miniforge is a lightweight and community driven command line tool for package and environment management.
Follow the link to install miniforge3.

- `Miniforge <https://conda-forge.org/miniforge/#latest-release>`_

NOTE: Make sure to add the conda scripts path to SYSTEM Path. This may look something like, **PATH = C:/Users/<user_name>/AppData/Local/miniforge3/Scripts**

Create an environment
======================
Use the following link to create an environment and set it up with the required python version.
- `Miniforge : Usage <https://github.com/conda-forge/miniforge/blob/main/README.md#usage>`_

Create the environment and install the exact python version supported:
::
    conda create -n <environment_name> python=<_version_>

Activate the environment
========================
All the commands to install packages and to run the executor scripts must be performed from within the activated environment.
::
    conda activate <environment_name>



Installing Python Packages
------------------------------
Use the requirements.txt file and pip to install the necessary packages.

Windows:
(from the root directory of the project.)
::
    pip install -r requirements.txt

Unix-like (Linux and MacOS):
(from the root directory of the project.)
::
    pip install -r requirements.txt

On MacOS, the error "ERROR: No matching distribution found for tensorflow==2.15.1" may occur.
Run this command in response:
::
    conda install -c conda-forge tensorflow=<##.##.##>
<##.##.##> is the tensorflow version specificed in requirements.txt
for example:
::
    conda install -c conda-forge tensorflow=2.15.1



Input Parameters
****************
The input to the python executor is a yaml file specified as a command line argument. This file specifies one or more models to compile, and the parameters to compile each model with.

The .yaml template (`<compiler_input_yaml_template.yaml>`) file can be used as a starting point of a custom file.

NOTE : It is advised to use forward slashes (rather than back slashes) in file paths as they are compatible in both windows and linux systems.

Primary Parameters
------------------
These parameters affect the basic behavior of the compiler. Most are mandatory. Others are optional, but enable core behaviour like running test vectors to get performance estimates.

.. list-table:: Primary Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - MODEL_NAME
     - STR
     - Short-hand name of the model. This is incorporated into the output file names as well as c symbols, so must not contain any character that are not allowed by either file systems or c symbols. C symobls are more restrictive, so the name can only have alpha-numerics and underscores, and cannot start with a number. 
        
       Always mandatory.

   * - TFLITE_MODEL
     - STR
     - Path and name of the int8 quantized .tflite or .lite file to compile for axon. When TFLITE_MODEL is provided instead of FLOAT_MODEL+TRAIN_DATA+TEST_DATA,
       the float model accuracy is not calculated, consequently the quantization loss is also not calculated.

       Mandatory unless FLOAT_MODEL along with TRAIN_DATA is provided.
   * - FLOAT_MODEL
     - STR
     - The path and name to the floating-point model file to compile for axon. Used to calculate the floating-point model accuracy when TEST_DATA is provided. 
       TRAIN_DATA must be provided if TFLITE_MODEL is not provided (TRAIN_DATA is needed to calculate quantization parameters.)
       Must be a .h5 model file (i.e a keras model.)

       Mandatory (along with TRAIN_DATA) if TFLITE_MODEL is not provided.
   * - TRAIN_DATA
     - STR 
     - The path and name of the train dataset file in floating point. Must be a numpy file and in the format supported by the respective model. Used only to convert a floating-point model into a tflite model when a TFLITE_MODEL is not provided.

       Mandatory if TFLITE_MODEL is not provided.

   * - TEST_DATA
     - STR
     - The path and name of the test dataset file in float (Directory + file name). Must be a numpy file and in the format supported by the model. 
     
       Optional, needed if the user wants accuracy results and test vectors header file in the output.

   * - TEST_LABELS
     - STR
     - The path and name of test dataset's labels file. 
       Note that text translation of the labels is specifed separately in CLASSIFICATION_LABELS.
       Must be in numpy format.

       Optional, unless TEST_DATA is provided. Needed for calculating accuracy results from the test data. 
   * - TEST_LABELS_FORMAT
     - STR
     - Specifies how the TEST_LABELS file is to be interpreted. Supported options are:

       **just_labels** : Each label is a single number indicating the index of the classification (0 based).

       **last_layer_vector** : test labels are the full output vector of the last layer of the model. The classification index is the index of the maximum value in this vector.

       **edge_impulse_labels** : Format of the test labels for models generated from edge impulse. Labels are in the first column followed by 3 more values. The executor handles this by reading the first column values and subtracting 1 from the labels as the labels are indexed from 1 instead of 0.

       If the test label is not in one of the above formats, the user can provide a custom format along with a handler function that can perform the required operation on the test labels. The user handler function is provided using the parameter user_handle_test_labels.
   * - CLASSIFICATION_LABELS
     - LIST
     - Text representation of each classification index, in order of the index. Shows what each number in test labels represents. This list is conveyed to the target and allows inference code to translate a result index into meaningful text.

       For example, the famous KWS 12 classifications are:

       ``["silence", "unknown", "yes", "no","up", "down" ,"left" ,"right" ,"on", "off" ,"stop","go"]``

       Optional.

   * - TEST_VECTORS
     - LIST or ``all``

     - List of indices and/or ranges of indices of TEST_DATA on which to run inference and get accuracy results. 
       
       For example, ``[0,2-6,10,34]`` specifies that only vectors 0, 2,3,4,5,6,10, and 34 will be inferenced. Useful to limit the test data set to speed execution or to focus on a particular test vector result.

       The literal text ``all`` specifies the complete test data set for mass inferencing. 

   * - HEADER_FILE_TEST_VECTOR_CNT
     - INT
     - Count of test vectors the compiler will generate in the test_vectors_header file, default value is zero. 

       The first HEADER_FILE_TEST_VECTOR_CNT number of vectors from the specified test_vectors will be populated in the test_vectors header output file. These are then used to sanity test model inference on the target device.

       A larger value will increase the size of the file, too large of a number will not fit in the devices available memory.

       Optional (default is 0).

Variant Parameters
------------------
These parameters affect the output of the compiler for a given model. They can be assigned a single valid, or a set of values.

When a set of values is specified, the model will be compiled in all permutations of that set (and other sets), and incorporate the variant value into the output file names and symbols. A comparison table is generated that shows the memory footprint, accuracy, and estimated performance of each variant permuation. The designer can then choose the the variant permuation that best fits their needs.

.. list-table:: Variant Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - TRANSPOSE_KERNEL
     - BOOL
     - IF True, transposes the kernel/filters of operations to generate output that has height and width swapped. Can be used if the model has a smaller width value than the height. Might improve performance on the axon hardware.

       default : ``False``

Advanced Parameters
------------------

.. list-table:: Advanced Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - Name
     - Type
     - Description


   * - INTERLAYER_BUFFER_SIZE
     - INT
     - The interlayer buffer is the buffer used to store intermediate layer results. This is provisioned on the device through the build system. It must be sized large enough to accomodate the largest layer of any model that will run in the build.
       The compiler will report the interlayer buffer size needed for all models and variants configured to run.
       This number is a threshold that will generate an error message if the size needed is in excess of the threshold. This will not interfere with compilation nor inference in the compiler.

       default : 125000. 
   * - PSUM_BUFFER_SIZE
     - INT
     - The psum buffer is similar to the interlayer buffer, but is used only for 2D convolutions when conv2d_settings is not local_psum, and psum_buffer_placement is dedicated_buffer. Similar to interlayer_buffer_size, this is a threshold value that will generate an error message if the needed psum buffer size is in excess of this threshold.
   * - OP_RADIX
     - INT
     - Sets the output radix to be used when disabling output quantization.

       Output radix should is based on quantization range to mazimize precision (maximum radix) with some margin to prevent saturation.
   * - USER_HANDLE_ACCURACY_RESULTS
     - STR
     - The executor can determine accuracy for classification models. Vector models that use a calculated distance or other technique to determine accuarcy require a separate handler.
       This maps a user handling function name to get the model accuracy along with module.

       e.g *user_handler_functions.<user_handler_function_name>*

       The user adds their function to `<user_handler_functions.py>`_, which will be loaded and executed by the executor.

   * - USER_HANDLE_TEST_LABELS
     - STR
     - Add user handling function name handle the test labels as needed to match the expected format.

       The test labels are expected to simply contain the true labels of the test data set. The true labels are indexed from 0 to the (number of classification – 1). 

       For example, for an image classification model, we may have 10 labels as follows, ["airplane" ,"automobile" ,"bird" ,"cat" ,"deer" ,"dog", "frog", "horse", "ship" ,"truck"] and therefore the test labels must have the values 0 – 9 where 0:airplane and 9:truck. 

       The test labels should be in a numpy format to be loaded into the compiler. This function is only called when the user sets the test_label_format as “custom”.

       e.g *user_handler_functions.<user_handler_function_name>*

   * - LOG_LEVEL
     - STR
     - Specified the log levels to the logs which are created and stored in the user workspace logs folder. 
       Options are: ``debug, info, warn, error, critical``
       Default : ``info``

   * - PRECISION_THRESHOLD
     - FLOAT
     - Specifies a confidence threshold that classifications must meet in order for the classification to not be considered "inconclusive".
       "inconclusive" is a meta classification that means none-of-the-above. This allows the user to increase precision (reduced false positives) at the cost of accuracy.
       
       This can only be used when softmax is the final operation on the model.
       
       Must be a number between 0 and 1. 0 disables the feature.
   * - PRECISION_MARGIN
     - FLOAT
     - Similar to PRECISION_THRESHOLD, specifies a minimum marge between 1st and 2nd place classifications.
       If the threshold is not met, the classification is deemed in conclusive, and is not counted against the precision. 

       Only used when softmax is enabled as it provides a probability for the last layer.
   * - RESHAPE_INPUT
     - BOOL
     - When true, reshapes the test data input to match the input shape of the model if the only transformation needed on the test data is a simple reshaping of the test input. Checks for the total length of the input needed from the shape and reshapes it to match the model input shape.

       default : ``False``

       NOTE : This only solves the mismatch in the shape of the test data and model input. Any other transformation on the test data before feeding into the model apart from a simple reshape will lead to unexpected results. The user needs to be aware of such transformation if present beforehand.
       e.g, if the model expects an input image with shape 1x96x96x3 and the test data is simply flattened to be 1x27648, the reshape_input flag will enable reshaping the test data to match the shape of the model input.
   
Running the Executor
*********************
The compiler is run by launching the python executor from the command line.
::
    python <path_to_scripts>\axons_ml_nn_compiler_executor.py <path_and_name_of_the_yaml_file>.yaml

Paths can be relative or absolute.

Sample TinyML Models
*********************
TinyML Commons sample models are provided as examples for the user to run the executor. Instructions on getting the model files are present in the `models/tinyml <models/tinyml>`_ folder.

A sample compiler input yaml file *compiler_sample_<model_name>_input.yml* file is provided for each of the models for the user to run the executor, e.g `<models/tinyml/image_classification/compiler_sample_ic_input.yaml>`_ for the image classification model.

The user will have to get the model artifacts needed to run the executor by carefully following the README description which explains how to get the model artifacts and place them in the expected location.

Information on getting the models and the test/train data is provided in the individual README files of the models.

NOTE : 
 The *compiler_sample_<model_name>_input.yml* files for the models can only be used if the instructions to get the data set and model files are followed.
 The user must place the model artifacts like the tflite model file, keras model file and the train/test numpy files in the exact locations the yaml file expects them to be, in order to successfully run the executor.

A general description about the TinyML Commons model repository can be found at the following `TinyML Models README <models/tinyml/README.rst>`_  

An example for getting the data and training the kws model can be found in `kws_model_script <kws_model_script.py>`_ . 

The script has examples on getting the data, generating csv files to get features using the axon_fe exe or generating the features directly using the dll.

The `README file <models/tinyml/kws/README.rst>`_ for the kws model explains this in more detail.

The user can write their own scripts for getting data and train models, to use the axon feature extractor and the executor to run their models using axon by referencing the *kws_model_script.py* .

Using Docker
******************

Docker is an optional way to set up and run the compiler.

Follow the links to build and run a simple docker container which runs a simple python application.
You will have to install docker on your system and then create a Dockerfile, build and run it to test that docker is up and running.

* `A beginner’s guide to Docker — how to create your first Docker application (freecodecamp.org) <https://www.freecodecamp.org/news/a-beginners-guide-to-docker-how-to-create-your-first-docker-application-cc03de9b639f/>`_

Extra links for guidance on creating and defining a Dockerfile using the best practices-

* `Creating the Dockerfile <https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/#creating-the-dockerfile>`_
* `Intro Guide to Dockerfile Best Practices <https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/>`_

Once you have a simple docker container up and running, the user can use the scripts/batch files present in the compiler directory to build and run the docker container for the executor.

The dockerfile defines an image that loads the necessary files into the container to run the executor. The script enables the user to use that dockerfile.

For the docker script to run successfully, the models and the data need to be present in one single directory so that the script can load the user work directory as a volume into the container.

Loading the user work directory as a volume enables the executor to output the files in the user work directory itself.

The docker script along with the dockerfile must be built from the compiler directory. The dockerfile only has access to files within its build context. 

The build context is the files and folders the dockerfile can access and that is the current working directory of the dockerfile.

The Dockerfile
-----------------------
The dockerfile imports the base python image with the specified version and builds the docker container.

It takes in four arguments which can be set by the user if they want to build the docker container by themselves.

The dockerfile context is the project root directory which is the compiler folder. The dockerfile must be executed from that location as it needs to access the compiler root directory for the compiler_types_hdr file and also look for the compiler shared object.

It copies all the files needed by the executor from the compiler directory and places them in the docker container to be executed using the COPY command in the dockerfile.
It runs the following command to call the executor with the input yaml file.
::
    python3 ./scripts/axons_ml_nn_compiler_executor.py <yaml_file_fullpath.yaml>

e.g.
::
    python3 ./scripts/axons_ml_nn_compiler_executor.py C:\Users\zaan\Desktop\windows_docker_test\input.yaml

Building the Dockerfile
-----------------------
The user can build their own docker container using the command –
::
    docker build -t <container_image_name> ./ \
    --build-arg compiler_root=<compiler_root_dir> \
    --build-arg yaml_file=<input_yaml_file_name> \
    --build-arg root_dir=<executor_root_dir> \
    --build-arg work_dir=<executor_work_dir>**

The compiler root directory is the value which is copied to the environment variable COMPILER_ROOT_FOLDER for the executor to look for the shared object.

The executor root and work directories are the location in the docker container where the user wants to place the files of the executor.

The input yaml file name is simply the file name of the input yaml file. (May not be needed)

Running the docker container
-----------------------

The following command can be used to run the docker container.
::
    docker run -v <user_workspace>:<workspace_dir> <container_image_name> "./<executor_work_dir>/<input_yaml_file_name>"

The user workspace is where the user work directory is located. This location is mounted to the container as a volume at the workspace_dir location in the docker container. 

Mounting the location of the user work directory allows the container to output the files directly in the user work directory.

The workspace directory is the full path to the executor_work_dir.

The container image name is the same name used when building the container.

The executor work directory is the location inside the container where the input yaml file and other files needed by the executor are referenced into.

Running the docker script
-----------------------
The docker script builds and also runs the docker container end to end without the need of entering multiple inputs while building and running the container. 

It only needs the full file path of the input yaml file. It extracts the user work directory and runs the docker container.

Start the docker container application in windows before running the docker script.

The user can run the script to build and run the docker container using the run_docker.bat / .sh  file present in the compiler directory.

The following command can be used to build and run the container.

  ``run_docker.bat <docker_image_name> <user_work_directory\input_yaml_file_name.yaml>``

e.g 
 ``run_docker.bat windows_docker_image C:\Users\zaan\Desktop\windows_docker_test\input.yaml``

or for wsl/linux/MacOS

  ``./run_docker.sh <docker_image_name> <user_work_directory/input_yaml_file_name.yaml>``

e.g 
  ``./run_docker.sh wsl_docker_image /mnt/c/Users/zaan/Desktop/wsl_docker_test/input.yaml``

Where,

*docker_image_name* unique name to distinguish the multiple docker builds, can be reused to run different yaml file as inputs

*user_work_directory* the user work directory as explained above, is the directory where the input yaml file is located. It is necessary that the files referenced in the input yaml file are also placed in the user work directory and the paths to those files are updated accordingly in the yaml file for the docker container to work without any errors.

*input_yaml_file_name* the full name of the input yaml file present inside the user work directory along with the correct extension i.e. yml/yaml.

Alternative to Docker : Using Podman
-------------------------------------------

Podman is a daemonless alternative to Docker. You can install podman by following the instructions on the `link <https://podman.io/docs/installation?>`_ .

After installation is done, you can setup a simple container by following the `link <https://github.com/containers/podman/blob/main/docs/tutorials/podman_tutorial.md#running-a-sample-container>`_ . 

Podman uses the same Dockerfile syntax so no changes required except replacing docker with podman.

When using podman as an alternative you can simply switch all the above commands to use *podman* instead of *docker*.
::
    podman build -t <container_image_name> ./ \
    --build-arg compiler_root=<compiler_root_dir> \
    --build-arg yaml_file=<input_yaml_file_name> \
    --build-arg root_dir=<executor_root_dir> \
    --build-arg work_dir=<executor_work_dir>**

    podman run -v <user_workspace>:<workspace_dir> <container_image_name> "./<executor_work_dir>/<input_yaml_file_name>"

The script *run_podman.bat* can be used to get the container up and running using podman on windows. Users can replace *docker* with *podman* for linux and macos in the script files once they have a podman machine running. 

e.g.
  ``run_podman.bat <docker_image_name> <user_work_directory\input_yaml_file_name.yaml>``


Error Codes
******************
The following table describes the error codes returned from the executor.
When an error occurs, the logs should be inspected for further details.

.. list-table:: Error Codes
   :widths: 25 170
   :header-rows: 1

   * - Error/Info Code
     - Description
   * - -900
     - generic error code
   * - -901
     - warning : operator supported but skipped
   * - -902
     - operator before softmax has activation function which is not yet supported, try skipping softmax
   * - -903
     - default error code for exceptions when calling the compiler library 
   * - -904
     - tflite file is None or empty
   * - -905
     - invalid test labels format 
   * - -906
     - compiler library is 'None' 
   * - -907
     - error when pre-processing input data/models from the yaml file
   * - -908
     - exception occured when generating the bin file 
   * - -909
     - operator has a fused activation function followed by a LeakyReLU Operator, cannot set leaky_relu as activation function
   * - -910
     - operator is supported as an Activation Function and not an operator
   * - -912
     - cannot set custom activation function to None
   * - -913
     - operator is combined to be a persistent variable
   * - -914
     - operator converted to be a passthrough operation
   * - -915
     - error when loading the custom user handle for handling test labels  
   * - -916
     - generic assertion error
   * - -917     
     - operator is a passthrough operation
   * - -918
     - model is not supported due to unsupported operation or constraints
   * - -919
     - error when creating TfliteAxonGraph object
   * - -920
     - error when handling operator attributes before CPU Extension Operation
   * - -921
     - error when setting custom activation function before CPU Extension Operation
   * - -922
     - CPU Extension Operation is 'None'!
   * - -923
     - CPU Extension Operation Handle threw an error!


Using the Scanner Script
*****************************

An additional utility script is provided to quickly **scan a TensorFlow Lite (TFLite) model**
and determine whether it is supported on **Axon**.

The `script <axons_tflite_model_scan.py>`_ can be run directly from the command line by
providing the full path to your TFLite model file.

**Example:**

.. code-block:: bash

    python axons_tflite_model_scan.py C:/user/fullpath/ei_fomo_face_detection_q.lite

After execution, the script will print the following information to the console:

* **PASS** : if the model is fully supported on Axons
* **FAIL** : if the model is not supported, along with detailed reasons  

Any constraints or compatibility issues found during the scan will be displayed as warnings,
prefixed with *WARN*.

Additionally, the script will suggest if the *transposed* model can be executed successfully using the executor.
