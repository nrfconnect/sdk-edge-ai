.. _samples_nrf_edge_ai:
.. _tests:

Samples and tests
#################

The |EAI| repository provides several samples showcasing its functionality.
You can build the samples for a variety of board targets and configure them for different usage scenarios.

In the |EAI| repository, all samples are placed in the :file:`samples` directory.

The samples are grouped by workflow, including deploying models from `Nordic Edge AI Lab`_, running inference on the Axon NPU, integrating `Edge Impulse`_, and collecting sensor data for model training.

* :ref:`samples_nrf_edgeai_overview` - Deploy `Nordic Edge AI Lab`_ models (regression, classification, anomaly) with the nRF Edge AI runtime.
* :ref:`edge_impulse_samples` - Integrate `Edge Impulse`_ models and run inference with the |EI| SDK.
* :ref:`axon_samples` - Run models on the Axon NPU, including low-power operation.
* :ref:`samples_other_overview` - Stream sensor data to :ref:`data_forwarder_host_tool` or `Edge Impulse's data forwarder`_ CLI.

.. toctree::
   :maxdepth: 2
   :caption: Subpages:

   samples/edge_ai.rst
   samples/edge_impulse.rst
   samples/axon.rst
   samples/other.rst
