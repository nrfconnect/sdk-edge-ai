.. _ug_nrf_edgeai_integration:

nRF Edge AI Library integration
###############################

.. contents::
   :local:
   :depth: 2

The |EAI| provides a lightweight library that enables deploying Edge AI applications on Nordic Semiconductor devices.

Integration prerequisites
*************************

Before you start using the nRF Edge AI integration, make sure you have completed the following prerequisites:

* `Installation of the nRF Connect SDK <nRF Connect SDK installation guide_>`_
* Setup of a :ref:`supported development kit <nrf_edgeai_requirements>`

.. _ug_nrf_edgeai_integration_enabling_kconfig:

Enabling the integration in Kconfig
===================================

To enable Nordic Edge AI Lab model integration in your application, ensure you have enabled the ``CONFIG_NRF_EDGEAI`` and ``CONFIG_NEWLIB_LIBC`` Kconfig options.
If your application prints model outputs (and they are floating-point), also enable ``CONFIG_NEWLIB_LIBC_FLOAT_PRINTF``.
For better inference speed on supported hardware, enable the ``CONFIG_FPU`` Kconfig option.

These options should be set in your application's :file:`prj.conf` file.

Solution architecture
*********************

The integration is implemented as a C library (:file:`lib/nrf_edgeai`) that adapts signal processing premitives, models and runtime helpers from the `Nordic Edge AI Lab`_ to the |NCS| build system.
Models are typically distributed as generated C sources (header + C files), which are consumed by the sample applications.
The library exposes a small API that abstracts model initialization, input feeding, and inference execution so that applications can remain model-agnostic.

Runtime and Nordic Edge AI Lab models integration
*************************************************

This section explains how to integrate models generated with the Nordic Edge AI Lab into your |NCS| application.
The instructions are divided into general workflow steps and detailed CMake integration steps.

Integration workflow
====================

The Edge AI Lab models integration in the |NCS| consists of the following steps:

1. Generate a model using the `Nordic Edge AI Lab`_ web tooling.
#. Place the generated files into your workspace, for example, :file:`samples/nrf_edgeai/<sample>/src/nrf_edgeai_generated`.
#. Replace samples user data, feeding, and result handling code with application-specific logic as required by your generated solution.
#. Enable :ref:`the integration in the application configuration<ug_nrf_edgeai_integration_enabling_kconfig>` and, if required, select a full libc implementation.
#. Build the application.

CMake integration steps
=======================

After configuring the application and placing the generated model files in your workspace, you need to update your build system to include and link the Edge AI library and its associated headers.
The following steps explain how to integrate the required static library and header files into your project’s CMake-based build system:

1. Generate a model using the `Nordic Edge AI Lab`_ web tooling and place the generated folders in your workspace.
2. Add the static library file (for example, :file:`nrf_edgeai/lib/libnrf_edgeai_cortex-m33.a`) to your application's build.
   Open your application's :file:`CMakeLists.txt` file and add the following line to link the library:

   .. code-block:: cmake

      zephyr_link_libraries(${CMAKE_CURRENT_LIST_DIR}/src/nrf_edgeai/lib/libnrf_edgeai_cortex-m33.a)

3. Include the :file:`nrf_edgeai/include/` path to your application include path.
   Open the :file:`CMakeLists.txt` file and add the following line to include the header files:

   .. code-block:: cmake

      target_include_directories(app PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/nrf_edgeai/include)

4. Include header files from :file:`nrf_edgeai/include/` and from :file:`nrf_edgeai/nrf_edgeai_generated` in your application's source files:

   .. code-block:: c

      #include <nrf_edgeai/nrf_edgeai.h>
      #include "nrf_edgeai_generated/nrf_edgeai_user_model.h"

Building samples
****************

The repository contains :ref:`reference samples <samples_nrf_edgeai_overview>` that demonstrate different types of models and runtimes.
These samples illustrate how to add generated model sources to an application and how to call the wrapper API.

Examples
========

See the following code snippets, that show the most common integration patterns: model initialization, feeding inputs (single vector or sliding window), running
inference and reading results for classification, regression and anomaly use cases.
Adapt them to your application and model specifics.

* :ref:`Classification <runtime_classification_sample_inference>`
* :ref:`Regression <runtime_regression_sample_inference>`
* :ref:`Anomaly detection <runtime_anomaly_sample_inference>` - In anomaly detection mode, the model's inference yields an anomaly score, indicating the similarity of input data to the "normal" data used for training.
  A higher anomaly score signifies greater deviation from the normal data, while a score close to zero indicates normal data.
  Because the model learns only from normal data, it cannot predict the presence of anomalies, only deviation from the normal data.
  For this reason, you must set a threshold based on the anomaly score to identify anomalies.

  The strategy for setting an anomaly threshold depends entirely on the intended use case.
  In some highly sensitive scenarios, such as analyzing bearing or gear vibrations, normal data may produce anomaly scores as low as ``0.00005``, while anomalous data may reach around ``0.00025``.
  In other applications, anomaly scores may span a much broader range (for example, ``0`` to ``1000``).

  Therefore, you must evaluate the anomaly scores generated by your model within the context of your own application.
  This may involve simulating anomalies after deployment or using the `EdgeAI Inference Runner`_ tool with collected abnormal data to determine an appropriate and reliable threshold for anomaly detection.

Memory and footprint
********************

The total memory footprint consists of the runtime engine and context, the user-generated model, and any configured signal-processing pipeline.
The runtime engine and its associated context typically require about 2 kB of FLASH and 0.5–1 kB of SRAM.
The size of the user-generated model, which is trained for a specific dataset, depends on your selected settings and overall model complexity.
In most cases, the requirement is between 1 and 10 kB.
Memory usage for the signal-processing pipeline is determined by the configuration parameters specified in the `Signal Processing Block`_ of `Nordic Edge AI Lab`_.

This variability results from the library including only those algorithms and processing blocks that you explicitly select.
Unused modules are not included in the deployed solution, which helps ensure that memory is used efficiently and without unnecessary overhead.

In practice, the overall memory footprint is typically around 2–5 kB of SRAM and 5–10 kB of FLASH.

Library support
***************

The following library supports the nRF Edge AI integration:

* :file:`lib/nrf_edgeai` — The wrapper and runtime glue.
