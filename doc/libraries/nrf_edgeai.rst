.. _nrf_edgeai_lib:

nRF Edge AI Library
###################

.. contents::
   :local:
   :depth: 2

The nRF Edge AI library is an optimized compute library for deploying Edge AI applications on Nordic Semiconductor devices.

Overview
********

The library consists of several core modules:

.. image:: ./images/nrf_edgeai_diagram.png
    :alt: nRF Edge AI Library Modules
    :align: center

* **Runtime** — Compute engine for running `Nordic Edge AI Lab`_ generated models.
* **DSP (Digital Signal Processing)** — Signal processing primitives for feature extraction (FFT, spectral analysis, statistics, etc.).
* **NN (Neural Network)** — Inference engine for neural network models.

.. toctree::
   :maxdepth: 1

   nrf_edgeai/runtime
   nrf_edgeai/dsp
   nrf_edgeai/nn

Key characteristics:

* **C99 standard** — Written in portable C with no external dependencies beyond libc.
* **No dynamic memory** — Memory footprint is fixed and predictable; ideal for embedded systems.
* **Minimal stack usage** — Optimized for constrained microcontroller environments.
* **Precompiled libraries** — Provided as precompiled static libraries for common Cortex-M architectures.

Configuration and Kconfig
**************************

The library is configured through Kconfig in the nRF Connect SDK build system.
Enable support by setting:

* :kconfig:option:`CONFIG_NRF_EDGEAI` — Enables the nRF Edge AI library and integration.

Related options include:

* :kconfig:option:`CONFIG_NEWLIB_LIBC` — Required by most Edge AI models; enables Newlib C library.
* :kconfig:option:`CONFIG_NEWLIB_LIBC_FLOAT_PRINTF` — Enables floating-point support in printf (useful for logging results).
* :kconfig:option:`CONFIG_FPU` — Enables floating-point unit (FPU) if available on the hardware; speeds up inference.

For detailed configuration information, refer to the integration guide (:ref:`ug_nrf_edgeai`).

Building and Linking
********************

The library is automatically built and linked when :kconfig:option:`CONFIG_NRF_EDGEAI` is enabled.
The build system:

1. Locates the precompiled static library for your target architecture (e.g., ``libnrf_edgeai_cortex-m4.a``).
2. Includes the header files from :file:`include/nrf_edgeai/`.
3. Compiles and liks your `Nordic Edge AI Lab`_-generated model sources with the your application. (e.g., :file:`nrf_edgeai_user_model.c`)
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
