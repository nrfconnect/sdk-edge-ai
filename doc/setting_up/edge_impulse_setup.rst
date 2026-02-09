.. _setup_edge_impulse:

Setting up Edge Impulse
#######################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with |EI| in |EAI|.

.. _edge_impulse_requirements_hardware:

Hardware requirements
*********************

The support of specific hardware platforms depends on the sample application you wish to run.
For details, see the :ref:`edge_impulse_samples` page, or refer to individual samples' :file:`sample.yaml` files in the :file:`samples/edge_impulse/<sample>/` directory.

|EI| SDK is provided in form of source code and can be built during the application build process for a hardware architecture of the target device.
This means SDK and models can be built for any Nordic Semiconductor's device with ARM Cortex-M4F and Cortex-M33F architectures.

There is also a possibility to deploy |EI| models utilizing Axon NPU, which is designed to accelerate machine learning inference on selected Nordic Semiconductor's devices.
Currently, these models can be run only on the `nRF54LM20B`_ device.

.. _edge_impulse_requirements_software:

Software requirements
*********************

To start working with the |EI| SDK, you must:

1. Complete :ref:`setup_sdk` (includes |NCS|, toolchain, and |EI| SDK).
#. Create an `Edge Impulse studio account <Edge Impulse studio signup_>`_ if you want to train and deploy your own machine learning models.
#. Follow the `Edge Impulse CLI installation guide`_ to install Edge Impulse command line tools.
   They include, for example, ``edge-impulse-data-forwarder`` which can be used to forward data from a board to |EIS| for training machine learning models.
