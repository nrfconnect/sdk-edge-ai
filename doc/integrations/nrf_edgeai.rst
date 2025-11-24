.. _ug_nrf_edgeai:

nRF Edge AI integration
#######################

.. contents::
   :local:
   :depth: 2

The nRF Edge AI add-on provides a lightweight wrapper that enables deploying
machine learning models on Nordic devices using the Nordic Edge AI Lab runtime. 
In this documentation the underlying platform is referred to as `Nordic Edge AI Lab`_.

Integration prerequisites
*************************

Before you start using the nRF Edge AI integration, ensure the following are in place:

* Installation of the nRF Connect SDK and the corresponding Zephyr toolchain.
* A supported development board. The reference samples are exercised for the
   following Zephyr board targets (use the board ID with `west build -b`):

   - ``nrf52dk/nrf52832``
   - ``nrf52840dk/nrf52840``
   - ``nrf5340dk/nrf5340/cpuapp``
   - ``nrf5340dk/nrf5340/cpuapp/ns``
   - ``nrf54l15dk/nrf54l05/cpuapp``
   - ``nrf54l15dk/nrf54l10/cpuapp``
   - ``nrf54l15dk/nrf54l15/cpuapp``
   - ``nrf54h20dk/nrf54h20/cpuapp``
   - ``thingy53/nrf5340/cpuapp``

   See the individual samples' ``sample.yaml`` files under
   ``samples/nrf_edgeai/<sample>/sample.yaml`` for exact supported platforms
   and CI test coverage.

Solution architecture
*********************

The integration is implemented as a wrapper library (`lib/nrf_edgeai`) that
adapts models and runtime helpers from the `Nordic Edge AI Lab`_ to the nRF
Connect SDK build system. Models are typically distributed as generated C
sources (header + C files) which are consumed by the sample applications.

The wrapper exposes a small API that abstracts model initialization, input
feeding and inference execution so that applications can remain model-agnostic.

Kconfig and common options
**************************

Common configuration options used by the integration are described in an
include file. It lists the most important Kconfig options and their purpose:

.. include:: ../includes/include_kconfig_edgeai.txt

Integration overview
*********************

Typical integration steps:

1. Prepare or generate a model using the `Nordic Edge AI Lab`_ web tooling.
2. Place the generated source files into the application (for example
   ``samples/nrf_edgeai/<sample>/src/nrf_edgeai_generated``).
3. Replace samples user data, feeding and result handling code with application-specific
   logic as required by your generated solution.
4. Enable the integration in the application configuration (see the
   Kconfig options) and, if required, select a full libc implementation.
5. Build the application using the standard `west build` flow.

Building and running
********************

Project-level instructions and common build/run notes are available in the
following include, which also shows example `west build` commands:

.. include:: ../includes/include_building_and_running_edgeai.txt

Samples and examples
********************

The repository contains a few reference samples that demonstrate different
types of models and runtimes. These samples illustrate how to add
generated model sources to an application and how to call the wrapper API:

* ``samples/nrf_edgeai/regression`` — a regression example that shows how to
  run a numeric prediction model on-device.
* ``samples/nrf_edgeai/classification`` — a classification example demonstrating
  multi-class inference and result reporting.
* ``samples/nrf_edgeai/anomaly`` — an anomaly-detection example.

Memory and footprint
********************

Memory footprint depends strongly on the chosen model and runtime options
(for example, full libc vs minimal libc). See the memory include for a
summary and guidance:

.. include:: ../includes/memory_requirement_edgeai.txt

Library support
***************

The following libraries in this repository support the nRF Edge AI integration:

* :file:`lib/nrf_edgeai` — the wrapper and runtime glue.

Examples
********

The following code snippets show the most common integration patterns: model
initialization, feeding inputs (single vector or sliding window), running
inference and reading results for classification, regression and anomaly use
cases. These examples are provided as a convenience; adapt them to your
application and model specifics.

Classification:

.. include:: ../includes/classification_integration_examples_edgeai.txt

Regression:

.. include:: ../includes/regression_integration_examples_edgeai.txt

Anomaly detection:
In anomaly detection mode, the model's inference yields an Anomaly Score, 
indicating the similarity of input data to the "normal" data used for training.
A higher Anomaly Score signifies greater deviation from the normal data, 
while a score close to zero indicates normal data. Because the model learns only from normal data,
it cannot predict the presence of anomalies, only deviation from the normal data,
so user must set a threshold based on the Anomaly Score to identify anomalies.

The strategy for setting an anomaly threshold depends entirely on the intended use case.
In some highly sensitive scenarios - such as analyzing bearing or gear vibrations - normal data may produce anomaly scores as low as 0.00005, 
while anomalous data may reach around 0.00025.
In other applications, anomaly scores may span a much broader range (e.g., 0 to 1000).

Therefore, users must evaluate the anomaly scores generated by their model within the context of their own application.
This may involve simulating anomalies after deployment or using the `EdgeAI Inference Runner`_ tool with collected abnormal data to determine an appropriate and reliable threshold for anomaly detection.


.. include:: ../includes/anomaly_integration_examples_edgeai.txt

