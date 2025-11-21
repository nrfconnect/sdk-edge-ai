.. _nrf_edgeai_lib:

NRF Edge AI Library
###################

.. contents::
   :local:
   :depth: 2

The NRF Edge AI library provides a lightweight C-based runtime for deploying machine learning models on Nordic Semiconductor devices.
Models are generated using the Nordic Edge AI Lab web tooling and distributed as C source files that integrate seamlessly with the nRF Connect SDK build system.
For more information about integrating NRF Edge AI models, see :ref:`ug_nrf_edgeai`.

Overview
********

The library consists of several core modules:

* **Runtime** — Core inference engine with model initialization, input feeding, and inference execution.
* **DSP (Digital Signal Processing)** — Signal processing primitives for feature extraction (FFT, spectral analysis, statistics, etc.).
* **NN (Neural Network)** — Support for Neuton neural network models and inference.
* **Types** — Type definitions and data structures for model I/O and processing pipelines.

Key characteristics:

* **C99 standard** — Written in portable C with no external dependencies beyond libc.
* **No dynamic memory** — Memory footprint is fixed and predictable; ideal for embedded systems.
* **Minimal stack usage** — Optimized for constrained microcontroller environments.

Core API
********

The runtime API consists of a small number of essential functions for model inference:

**Initialization and Inference**

* :c:func:`nrf_edgeai_init` — Initialize the runtime (call once at startup).
* :c:func:`nrf_edgeai_feed_inputs` — Feed raw input data (sensor readings, signal samples) to the model.
* :c:func:`nrf_edgeai_run_inference` — Execute the model and compute predictions.

**Input Information**

* :c:func:`nrf_edgeai_input_type` — Get the data type of input features (int16, float32, etc.).
* :c:func:`nrf_edgeai_uniq_inputs_num` — Get the number of unique input features the model expects.
* :c:func:`nrf_edgeai_input_window_size` — Get the size of the input window (for time-series models).
* :c:func:`nrf_edgeai_input_subwindows_num` — Get the number of subwindows in the input.

**Model Information**

* :c:func:`nrf_edgeai_model_neurons_num` — Get the number of neurons in the model.
* :c:func:`nrf_edgeai_model_weights_num` — Get the number of weights in the model.
* :c:func:`nrf_edgeai_model_outputs_num` — Get the number of model outputs.
* :c:func:`nrf_edgeai_model_task` — Get the model task type (classification, regression, anomaly detection).

**Solution and Version Information**

* :c:func:`nrf_edgeai_solution_id_str` — Get the solution ID string.
* :c:func:`nrf_edgeai_solution_runtime_version` — Get the version of the solution runtime.
* :c:func:`nrf_edgeai_runtime_version` — Get the version of the Edge AI library runtime.
* :c:func:`nrf_edgeai_is_runtime_compatible` — Check version compatibility between library and solution.

**Output Access**

After a successful :c:func:`nrf_edgeai_run_inference`, results are available via the ``decoded_output`` member of the context:

* ``p_edgeai->decoded_output.classif`` — Classification results (predicted class, probabilities, number of classes).
* ``p_edgeai->decoded_output.regression`` — Regression results (predicted values array, number of outputs).
* ``p_edgeai->decoded_output.anomaly`` — Anomaly detection results (anomaly score).

Configuration and Kconfig
**************************

The library is configured through Kconfig in the nRF Connect SDK build system.
Enable support by setting:

* :kconfig:option:`CONFIG_EDGEAI` — Enables the NRF Edge AI library and integration.

Related options include:

* :kconfig:option:`CONFIG_NEWLIB_LIBC` — Required by most Edge AI models; enables Newlib C library.
* :kconfig:option:`CONFIG_NEWLIB_LIBC_FLOAT_PRINTF` — Enables floating-point support in printf (useful for logging results).
* :kconfig:option:`CONFIG_FPU` — Enables floating-point unit (FPU) if available on the hardware; speeds up inference.

For detailed configuration information, refer to the integration guide (:ref:`ug_nrf_edgeai`).

Module Structure
****************

**Runtime Module** (:file:`include/nrf_edgeai/rt/`)

- :file:`nrf_edgeai_runtime.h` — Core runtime API (init, feed inputs, run inference).
- :file:`nrf_edgeai_types.h` — Type definitions (context, interfaces, metadata).
- :file:`nrf_edgeai_input_types.h` — Input handling types and structures.
- :file:`nrf_edgeai_model_types.h` — Model definitions and output structures.
- :file:`nrf_edgeai_output_types.h` — Classification, regression, and anomaly output types.
- :file:`nrf_edgeai_dsp_pipeline_types.h` — DSP pipeline configuration.

**DSP Module** (:file:`include/nrf_edgeai/dsp/`)

Provides signal processing functions organized by category:

- **Fast Math** — Basic mathematical operations.
- **Spectral Analysis** — FFT, frequency-domain processing, spectral features.
- **Statistic** — Statistical measures (mean, variance, RMS, entropy, etc.).
- **Transform** — Signal transformations (FFT, RFHT, Mel-spectrogram, etc.).
- **Support** — Utility functions (windowing, quantization, scaling, clipping).

**NN Module** (:file:`include/nrf_edgeai/nn/`)

- :file:`nrf_nn.h` — Neural network abstraction interface.
- :file:`nrf_nn_neuton.h` — Neuton-specific neural network implementation.

Usage Pattern
*************

A typical inference workflow:

.. code-block:: c

   #include <nrf_edgeai/nrf_edgeai.h>
   #include <nrf_edgeai_generated/nrf_edgeai_user_model.h>

   static nrf_edgeai_t* p_edgeai = NULL;

   /* Initialize at startup */
   void init_model(void)
   {
       p_edgeai = nrf_edgeai_user_model();
       nrf_edgeai_init(p_edgeai);
   }

   /* Feed data and run inference */
   void process_sensor_data(int16_t* sensor_readings, uint32_t count)
   {
       nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, sensor_readings, count);
       if (res == NRF_EDGEAI_ERR_SUCCESS) {
           res = nrf_edgeai_run_inference(p_edgeai);
           if (res == NRF_EDGEAI_ERR_SUCCESS) {
               /* Access results via p_edgeai->decoded_output */
           }
       }
   }

Data Types
**********

**Error Codes**

The library uses :c:type:`nrf_edgeai_err_t` for operation status:

- ``NRF_EDGEAI_ERR_SUCCESS`` — Operation completed successfully.
- ``NRF_EDGEAI_ERR_NOT_CONFIGURED`` — Runtime not initialized.
- ``NRF_EDGEAI_ERR_INVALID_ARG`` — Invalid argument provided.
- ``NRF_EDGEAI_ERR_INVALID_SIZE`` — Invalid data size.

**Model Task Types**

The :c:type:`nrf_edgeai_model_task_t` enum specifies the model's task:

- ``NRF_EDGEAI_TASK_MULT_CLASS`` — Multi-class classification.
- ``NRF_EDGEAI_TASK_BIN_CLASS`` — Binary classification.
- ``NRF_EDGEAI_TASK_REGRESSION`` — Regression (numeric prediction).
- ``NRF_EDGEAI_TASK_ANOMALY`` — Anomaly detection.

**Input Types**

Models accept input data of type :c:type:`nrf_edgeai_input_type_t`:

- ``NRF_EDGEAI_INPUT_TYPE_INT8`` — 8-bit signed integer.
- ``NRF_EDGEAI_INPUT_TYPE_INT16`` — 16-bit signed integer.
- ``NRF_EDGEAI_INPUT_TYPE_FLOAT32`` — 32-bit floating-point.
- Other types as supported by the model.

Building and Linking
********************

The library is automatically built and linked when :kconfig:option:`CONFIG_EDGEAI` is enabled.
The build system:

1. Locates the precompiled static library for your target architecture (e.g., ``libnrf_edgeai_cortex-m4.a``).
2. Includes the header files from :file:`include/nrf_edgeai/`.
3. Compiles and liks your Nordic Edge AI Lab-generated model sources with the your application. (e.g., :file:`nrf_edgeai_user_model.c`)
4. Compiles and links your firmware application with the library.

API documentation
*****************

| Header files: :file:`include/nrf_edgeai/`

For detailed API reference including function signatures, parameters, and return values, refer to the header files listed above.
Doxygen comments in the headers provide full documentation of each function and data structure.

See Also
********

* :ref:`ug_nrf_edgeai` — Integration guide with examples.
* :ref:`samples_nrf_edgeai` — Sample applications demonstrating the library.
