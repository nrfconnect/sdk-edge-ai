.. _axon_npu_tflite_compiler:

Axon NPU TFLITE compiler
########################

.. contents::
   :local:
   :depth: 2

The following guide provides an overview of the Axon NPU TFLite compiler, including how it works, its input parameters, and instructions for setting up and running the compiler.

Overview
********

The Axon NPU TFLite compiler converts a TensorFlow Lite (``.tflite`` or ``.lite``) model into object code that can run on the Axon NPU.
Optionally, the compiler can run inference on a provided test dataset to validate that the compiled model produces results that match TensorFlow Lite accuracy.
When test data is provided, the compiler can also generate test vectors for on‑target verification.

Workflow
========

The compilation flow consists of two components:

* Executor script (:file:`axons_ml_nn_compiler_executor.py`) - A Python script that parses the input configuration, prepares the model, and invokes the compiler library.

* Compiler library - The compiler shared library performs the actual compilation and generates Axon object code.
  It also writes the output headers that are included in the application build.

  The compiler library is delivered as platform-specific binaries:

  * Windows: :file:`bin/Windows/axons_ml_nn_compiler_lib_amd64.dll`
  * Linux: :file:`bin/Linux/libaxons_ml_nn_compiler_lib_amd64.so`
  * macOS: :file:`bin/Darwin/libaxons_ml_nn_compiler_lib_arm64.dylib`

All compiler inputs are provided through a YAML file.
You can find its template in the :file:`compiler_input_yaml_template.yaml` file.

The compiler writes its output to an ``output`` directory located under the YAML file’s workspace directory.
The output includes:

* A C header file containing the compiled Axon object and referenced constants
* An optional test vectors header file when test data is provided, used for verifying inference behavior on the target device

Directory layout
****************

This section describes the directory structure used by the Axon NPU TFLite compiler.

Workspace directory
===================

The workspace directory is the directory containing the input YAML file.

* All paths in the YAML file are relative to this directory (unless you are providing absolute paths).
* The compiler output directory is created directly under this location.

Tools directory
===============

You should not modify the tools directory structure, as the executor expects the following:

* Header files in an :file:`include` directory parallel to :file:`scripts`.
* The compiler shared library in a directory parallel to :file:`scripts`.

You must activate the Python environment from this directory to run the executor.

.. _axon_npu_tflite_compiler_setup_executor:

Setting up the executor
***********************

Before you can run the compiler, you need to :ref:`set up a Python environment with the required dependencies <axon_setup_compiler>`.

Configuring input parameters
****************************

The executor input is a YAML file specified as a command‑line argument.
The file defines one or more models and the parameters used to compile each model.

Use the :file:`compiler_input_yaml_template.yaml` file as a starting point.
Use forward slashes (``/``) in file paths to ensure cross‑platform compatibility.

Primary Parameters
==================

These parameters control the basic behavior of the compiler.
Most parameters are mandatory. Others are optional, but enable core behavior such as
running test vectors and estimating accuracy.

.. list-table:: Primary Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - Name
     - Type
     - Description

   * - MODEL_NAME
     - STR
     - Short‑hand name of the model.
       This name is incorporated into output file names and C symbols, so it must not
       contain characters that are invalid for file systems or C identifiers.

       C symbols are more restrictive: the name must contain only alphanumeric characters
       and underscores, and must not start with a number.

       Always mandatory.

   * - TFLITE_MODEL
     - STR
     - Path and name of the int8 quantized ``.tflite`` or ``.lite`` file to compile for Axon.

       When ``TFLITE_MODEL`` is provided instead of ``FLOAT_MODEL`` + ``TRAIN_DATA`` +
       ``TEST_DATA``, floating‑point model accuracy is not calculated and quantization
       loss is not reported.

       Mandatory unless ``FLOAT_MODEL`` and ``TRAIN_DATA`` are provided.

   * - FLOAT_MODEL
     - STR
     - Path and name of the floating‑point model file to compile for Axon.
       Used to calculate floating‑point model accuracy when ``TEST_DATA`` is provided.

       ``TRAIN_DATA`` must be provided if ``TFLITE_MODEL`` is not provided, as it is required
       to calculate quantization parameters.

       Must be a ``.h5`` Keras model file.

       Mandatory (along with ``TRAIN_DATA``) if ``TFLITE_MODEL`` is not provided.

   * - TRAIN_DATA
     - STR
     - Path and name of the training dataset in floating‑point format.
       Must be a NumPy file in a format supported by the corresponding model.

       Used only when converting a floating‑point model into a TFLite model if
       ``TFLITE_MODEL`` is not provided.

       Mandatory if ``TFLITE_MODEL`` is not provided.

   * - TEST_DATA
     - STR
     - Path and name of the test dataset in floating‑point format.
       Must be a NumPy file in a format supported by the model.

       Optional. Required if accuracy results and a test vectors header file are desired.

   * - TEST_LABELS
     - STR
     - Path and name of the test dataset labels file.
       Textual label translation is specified separately using ``CLASSIFICATION_LABELS``.

       Must be provided in NumPy format.

       Optional unless ``TEST_DATA`` is provided.
       Required for calculating accuracy results from test data.

   * - TEST_LABELS_FORMAT
     - STR
     - Specifies how the ``TEST_LABELS`` file is interpreted.
       Supported options are:

       ``just_labels``
         Each label is a single integer representing the classification index (zero‑based).

       ``last_layer_vector``
         Labels are full output vectors from the last model layer.
         The classification index is the index of the maximum value in the vector.

       ``edge_impulse_labels``
         Label format used by Edge Impulse models.
         Labels are stored in the first column followed by three additional values.
         The executor reads the first column and subtracts 1, as labels are one‑based.

       If the labels are not in one of the supported formats, a custom format can be used
       together with a user‑defined handler function specified by
       ``USER_HANDLE_TEST_LABELS``.

   * - CLASSIFICATION_LABELS
     - LIST
     - Text representation of each classification index, ordered by index.
       This list is conveyed to the target and allows inference code to translate numeric
       classification results into meaningful labels.

       Example (KWS 12 classifications):

       ``["silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]``

       Optional.

   * - TEST_VECTORS
     - LIST or ``all``
     - List of indices and/or index ranges from ``TEST_DATA`` on which inference is run
       to calculate accuracy.

       Example:

       ``[0, 2-6, 10, 34]``

       This runs inference on vectors 0, 2, 3, 4, 5, 6, 10, and 34.

       Useful for reducing execution time or focusing on specific test cases.

       The literal value ``all`` specifies the entire test dataset.

   * - HEADER_FILE_TEST_VECTOR_CNT
     - INT
     - Number of test vectors generated in the test vectors header file.
       Default value is zero.

       The first ``HEADER_FILE_TEST_VECTOR_CNT`` vectors from ``TEST_VECTORS`` are written
       to the output header file and can be used for sanity‑checking inference on the
       target device.

       Larger values increase file size and may exceed available device memory.

       Optional. Default is ``0``.

Variant parameters
==================

Variant parameters affect the output generated by the compiler for a given model.
Each parameter can be assigned either a single value or a set of values.

When a set of values is provided, the compiler builds all permutations across the specified variant parameters.
Each permutation generates distinct output files and symbols.

A comparison table is produced showing memory footprint, accuracy, and estimated performance for each variant, allowing designers to select the most suitable configuration.

.. list-table:: Variant Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - Name
     - Type
     - Description

   * - TRANSPOSE_KERNEL
     - BOOL
     - If ``True``, transposes kernel and filter dimensions so that output height and width
       are swapped.

       This can improve performance on Axon hardware when the model has a smaller width
       dimension than height.

       Default: ``False``.

Advanced parameters
===================

Advanced parameters control memory thresholds, precision handling, logging, and user‑defined
hooks for accuracy calculation and label processing.

.. list-table:: Advanced Parameters
   :widths: 25 5 170
   :header-rows: 1

   * - Name
     - Type
     - Description

   * - INTERLAYER_BUFFER_SIZE
     - INT
     - Size threshold for the interlayer buffer used to store intermediate layer results.
       This buffer is provisioned on the device through the build system and must be large
       enough to accommodate the largest layer of any model included in the build.

       The compiler reports the required interlayer buffer size for all configured models
       and variants. If the required size exceeds this threshold, an error message is
       generated.

       This threshold does not affect compilation or inference within the compiler.

       Default: ``125000``.

   * - PSUM_BUFFER_SIZE
     - INT
     - Size threshold for the partial‑sum (psum) buffer.
       This buffer is used only for 2D convolutions when ``conv2d_settings`` is not
       ``local_psum`` and ``psum_buffer_placement`` is ``dedicated_buffer``.

       As with ``INTERLAYER_BUFFER_SIZE``, exceeding this threshold generates an error
       message but does not prevent compilation.

   * - USER_HANDLE_ACCURACY_RESULTS
     - STR
     - Custom handler for calculating model accuracy.

       The executor can calculate accuracy automatically for classification models.
       Models that determine accuracy using distances or other techniques require a
       user‑provided handler.

       The value maps to a function name in ``user_handler_functions.py``, for example:

       ``user_handler_functions.<user_handler_function_name>``

       The specified function is loaded and executed by the executor.

   * - USER_HANDLE_TEST_LABELS
     - STR
     - Custom handler for processing test labels when ``TEST_LABELS_FORMAT`` is set to
       ``custom``.

       Test labels are expected to contain the true class indices ranging from 0 to
       (number of classifications − 1).

       For example, for a 10‑class image classification model with labels:

       ``["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]``

       Valid label values range from 0 (airplane) to 9 (truck).

       The labels must be provided in NumPy format.

   * - LOG_LEVEL
     - STR
     - Log verbosity level for files written to the workspace ``logs`` directory.

       Supported values: ``debug``, ``info``, ``warn``, ``error``, ``critical``.

       Default: ``info``.

   * - PRECISION_THRESHOLD
     - FLOAT
     - Confidence threshold that a classification must meet to be considered valid.

       Classifications below this threshold are marked as *inconclusive*, representing
       a none‑of‑the‑above result.

       Increasing this value improves precision (fewer false positives) at the cost of
       reduced accuracy.

       Only applicable when softmax is the final model operation.

       Must be between 0 and 1. A value of 0 disables this feature.

   * - PRECISION_MARGIN
     - FLOAT
     - Minimum required margin between the highest and second‑highest classification
       scores.

       If the margin is not met, the classification is marked as inconclusive and is not
       counted against precision.

       Only applicable when softmax is enabled.

   * - RESHAPE_INPUT
     - BOOL
     - When ``True``, reshapes test data inputs to match the model input shape if the only
       required transformation is a simple reshape.

       The executor checks the total input length and reshapes accordingly.

       Default: ``False``.

       NOTE:
       This only addresses shape mismatches.
       Any other required preprocessing (for example normalization or feature extraction)
       must be handled by the user.

       Example:
       If the model expects input shape ``1x96x96x3`` and the test data is flattened to
       ``1x27648``, enabling ``RESHAPE_INPUT`` reshapes the data to match the model input.

Sample TinyML models
********************

TinyML Commons sample models are provided as examples that you can use to run the executor.
For each sample model, a corresponding compiler input YAML file named :file:`compiler_sample_<model_name>_input.yml` is provided.
You can use these files directly with the executor.

For example, the image classification model provides the :file:`models/tinyml/image_classification/compiler_sample_ic_input.yaml` file.

To run the executor successfully, you must obtain all required model artifacts by following the instructions in the corresponding :file:`models/README` file. These instructions explain how to download or generate the model files and how to place them in the expected directory structure.
They also include information on obtaining the models and the associated training, and testing datasets.

.. note::

  You can only use the :file:`compiler_sample_<model_name>_input.yml` files if you exactly follow the instructions for obtaining the dataset and model files.

You must place model artifacts, such as the TFLite model file, Keras model file, training and test NumPy files, in the exact locations expected by the YAML file. If the files are missing or are located elsewhere, the executor will fail to run.

You can also write your own scripts to obtain data and train models.
These scripts can use the Axon feature extractor together with the executor to run custom models on Axon.

Running the Compiler
********************

Run the executor from the command line:

.. code-block:: console

   python scripts/axons_ml_nn_compiler_executor.py <path/to/input.yaml>

Paths may be absolute or relative to the current directory.

.. _axon_npu_tflite_compiler_docker:

Using Docker (Optional)
***********************

Docker provides a fully isolated way to run the compiler without installing dependencies locally.
Set it up using instructions in :ref:`axon_setup_compiler`.

Once you have installed and verified Docker, you can use the scripts and batch files provided in the compiler directory to build and run a Docker container for the executor.
The Dockerfile defines an image that loads all required files into the container so that the executor can be run inside the container.

For the Docker workflow to function correctly, all models and datasets must be placed in a single directory.
This directory is mounted into the container as a volume and serves as your workspace.

Mounting your workspace as a volume, allows the executor to write output files directly into your working directory on the host system.

You must execute the Docker script and Dockerfile from the compiler directory.
The Docker build context is the compiler directory itself, and the Dockerfile can only access files within this context.

The Dockerfile
==============

The Dockerfile uses a base Python image with the required Python version and builds a Docker image capable of running the executor.

The Dockerfile accepts four build arguments, which you can customize if you are building the container manually.
The Dockerfile build context is the compiler root directory.
It must be executed from this directory because it needs access to the compiler root folder for the :file:`compiler_types_hdr` file and the compiler shared library.

During the build process, the Dockerfile copies all files required by the executor from the compiler directory into the container using the ``COPY`` command.

The Docker container runs the executor using the following command:

.. code-block:: console

    python3 ./scripts/axons_ml_nn_compiler_executor.py <yaml_file_fullpath.yaml>

For example:

.. code-block:: console

    python3 ./scripts/axons_ml_nn_compiler_executor.py C:\Users\zaan\Desktop\windows_docker_test\input.yaml

.. _axon_npu_tflite_compiler_podman:

Alternative to Docker: Podman
*****************************

Podman is a daemonless alternative to Docker.

#. Install Podman by following the `Podman installation guide`_.

#. Set up and run a `simple container with Podman <Setting up Podman container_>`_.

   Podman uses the same Dockerfile syntax as Docker, so no changes to the Dockerfile are required.
   When using Podman, simply replace ``docker`` with ``podman`` in all commands.

   .. code-block:: console

       podman build -t <container_image_name> ./ \
       --build-arg compiler_root=<compiler_root_dir> \
       --build-arg yaml_file=<input_yaml_file_name> \
       --build-arg root_dir=<executor_root_dir> \
       --build-arg work_dir=<executor_work_dir>

     podman run -v <user_workspace>:<workspace_dir> <container_image_name> "./<executor_work_dir>/<input_yaml_file_name>"

#. You can use the :file:`run_podman.bat` to build and run the container using Podman on Windows.
   For Linux and macOS, replace ``docker`` with ``podman`` in the existing script files once a Podman machine is running.
   For example:

   .. code-block:: console

       run_podman.bat <docker_image_name> <user_work_directory\input_yaml_file_name.yaml>

Troubleshooting
***************

When the executor fails, inspect the logs in the workspace ``logs`` directory.

The following table lists error codes returned by the executor. When an error occurs,
the logs should be inspected for additional details.

.. list-table:: Error Codes
   :widths: 25 170
   :header-rows: 1

   * - Error/Info Code
     - Description
   * - -900
     - generic error code
   * - -901
     - warning: operator supported but skipped
   * - -902
     - operator before softmax has an activation function that is not yet supported; try skipping softmax
   * - -903
     - default error code for exceptions when calling the compiler library
   * - -904
     - tflite file is None or empty
   * - -905
     - invalid test labels format
   * - -906
     - compiler library is None
   * - -907
     - error during preprocessing of input data or models from the YAML file
   * - -908
     - exception occurred when generating the binary file
   * - -909
     - operator has a fused activation function followed by a LeakyReLU operator
   * - -910
     - operator is supported as an activation function and not as an operator
   * - -912
     - cannot set custom activation function to None
   * - -913
     - operator combined into a persistent variable
   * - -914
     - operator converted into a passthrough operation
   * - -915
     - error when loading the custom user handler for test labels
   * - -916
     - generic assertion error
   * - -917
     - operator is a passthrough operation
   * - -918
     - model not supported due to unsupported operation or constraints
   * - -919
     - error when creating TfliteAxonGraph object
   * - -920
     - error when handling operator attributes before CPU extension operation
   * - -921
     - error when setting custom activation function before CPU extension operation
   * - -922
     - CPU extension operation is None
   * - -923
     - CPU extension operation handle threw an error

Verifying model support (scanner script)
****************************************

Use an additional utility script to scan a TensorFlow Lite (TFLite) model and determine whether it is supported on Axon.
You can run the :file:`axons_tflite_model_scan.py`  script directly from the command line by providing the full path to the TFLite model file:

.. code-block:: bash

    python axons_tflite_model_scan.py C:/user/fullpath/ei_fomo_face_detection_q.lite

After execution, the script prints the following information to the console:

* ``PASS`` if the model is fully supported on Axon
* ``FAIL`` if the model is not supported, along with detailed reasons

Any constraints or compatibility issues are displayed as warnings prefixed with ``WARN``.
The script also indicates whether a transposed version of the model can be executed successfully using the executor.
