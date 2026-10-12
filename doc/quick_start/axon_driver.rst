.. _quick_start_axon_driver:
.. _setup_axon:

Axon driver
###########

The Axon driver gives you direct access to the Axon NPU without the abstraction layer of the |EAILib| API.
Both workflows below use the same driver initialization and hardware, but they target different workloads.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Workflow
     - Training toolchain
     - Best for
   * - :ref:`quick_start_axon_driver_inference`
     - :ref:`Axon NPU TFLite compiler <axon_npu_tflite_compiler>`
     - Running compiled TensorFlow Lite models with custom inference pipelines and direct NPU control
   * - :ref:`quick_start_axon_dsp_intrinsics`
     - None
     - Hardware-accelerated signal processing with pre-defined DSP functions, without a compiled model

Axon NPU library is included as part of the :ref:`lib_axon` and is currently available on the `nRF54LM20B`_ device.

.. toctree::
   :maxdepth: 2
   :caption: Subpages:

   axon_driver_inference
   axon_driver_intrinsics
