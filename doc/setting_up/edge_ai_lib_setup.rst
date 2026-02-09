.. _setup_nrf_edgeai_lib:

Setting up nRF Edge AI Library
##############################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with the |EAILib|.

.. _nrf_edgeai_requirements_hardware:

Hardware requirements
*********************

The support of specific hardware platforms depends on the sample application you wish to run.
For details, see the :ref:`samples_nrf_edgeai_overview` page, or refer to individual samples' :file:`sample.yaml` files in the :file:`samples/nrf_edgeai/<sample>/sample.yaml` or :file:`applications/<application>/sample.yaml` directory.

Some additional hardware, like extension boards, may be required to run specific samples and applications.
Refer to their documentation for details.

The |EAILib| is provided as compiled binaries for ARM Cortex-M4F and Cortex-M33F architectures.

.. _nrf_edgeai_requirements_software:

Software requirements
*********************

Complete :ref:`setup_sdk` to install |NCS|, toolchain, and |EAILib|.

Optionally, create a `Nordic Edge AI Lab`_ account if you want to prepare and deploy your own machine learning models.
The samples and applications provided in the |EAI| use pre-deployed models, so an account is not required to run them.
