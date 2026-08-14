.. _quick_start_axon_dsp_intrinsics:
.. _setup_axon:

Axon DSP intrinsics
###################

.. contents::
   :local:
   :depth: 2

The following guide outlines the requirements that you need to fulfill before you start working with the Axon NPU using Axon DSP intrinsics.
This is a sub-set of the steps needed to use direct Axon NPU API for model inference. The :ref:`axon_npu_tflite_compiler` is not required to use Axon DSP intrinsics.

To follow this guide, you should be familiar with embedded systems development and C‑based APIs.

.. _axon_requirements_hardware:

Hardware requirements
*********************

Axon NPU library is included as part of the :ref:`lib_axon` and provided as compiled binaries for Cortex-M33F architectures.
Axon NPU is currently available on the `nRF54LM20B`_ device.

.. _axon_requirements_software:

Software requirements
*********************

To start working with the Axon NPU, complete the setup based on your use case:

* If you want to deploy algorithms on the device, you just need to complete :ref:`setup_sdk` to install |NCS| and toolchain.
* If you want to develop and evaluate algorithms off-target, you only need enable simulator application development in VS Code.


.. _quick_start_axon_driver_algorithm_development:

Algorithm Development
*********************

The goal is to develop the algorithm in a manner that allows it to be built for both zephyr and Axon Simulator host (linux, Wndows, MacOS) targets.

To that end, it is developed in C code, and utilizes the Axon NPU host platform abstractions for functions with platform system dependencies that need to executed in both places (ie, ``printf => nrf_axon_platform_printf``),
and bracket simulator-only code with macros (ie ``#ifdef AXON_SIMULATOR``). :ref:`test_dsp_intrinsics` can be used as a template.

During development in the simulator, you will have full access to all the tools of the VS Code environment at your disposal. Single-step debugging, memory inspection, log messages, etc.

Algorithm Device Verification
*****************************

At any time during the algorithm development stage, the application can be targeted for the zephyr device to verify execution an measure performance.
It is recommended that you follow the model of :ref:`test_dsp_intrinsics` and provide test vectors with expected results to verify the zephyr device is bit-exact with the simulator.


Whether you're using a pre-trained model or one you have trained yourself, you will need to run it through this compilation process.

Algorithm Device Deployment
***************************

It is recommended that the algorithm be stored as library code, and that the development application with test vectors be retained for regression testing and reference purposes.


