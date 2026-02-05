.. _axon_samples:

Axon NPU samples and tests
##########################

This section lists the available samples and tests that demonstrate the use of the |EAI| with the Axon NPU driver.

The samples cover neural network inference, DSP‑style algorithms for feature extraction, and combined use cases that integrate both approaches.

Each sample can be built either for a Zephyr‑based target or for the Axon simulator. 
The simulator is not a standalone tool or dedicated hardware. 
Instead, it is provided as a static library that is linked into a host‑based console application. 
While it is not cycle‑accurate compared to hardware, it is computationally bit‑exact and can be used to estimate performance.


.. toctree::
   :maxdepth: 1
   :caption: Subpages

   /../samples/axon/hello_axon/README.rst
   /../samples/axon/test_nn_inference/README.rst
