.. _setup_nrf_edgeai_lib:

Setting nRF Edge AI Library
###########################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with the |EAILib|.

.. _nrf_edgeai_requirements_software:

Software requirements
*********************

Complete :ref:`setup_sdk` to install |NCS|, toolchain, and |EAILib|.

Optionally, create a `Nordic Edge AI Lab account <Nordic Edge AI Lab signup>`_ if you want to prepare and deploy your own machine learning models.
The samples and applications provided in the |EAI| use pre-deployed models, so an account is not required to run them.

.. _nrf_edgeai_requirements_hardware:

Hardware requirements
*********************

The support of specific hardware platforms depends on the sample application you wish to run.
For details, see the :ref:`samples_nrf_edgeai_overview` page, or refer to individual samples' :file:`sample.yaml` files in the :file:`samples/nrf_edgeai/<sample>/sample.yaml` or :file:`applications/<application>/sample.yaml` directory.

Some additional hardware, like extension boards, may be required to run specific samples and applications.
Refer to their documentation for details.

The |EAILib| is provided as compiled binaries for ARM Cortex-M4F and Cortex-M33F architectures, so it is suitable for use on various Nordic Semiconductor's devices:

* `nRF52 Series`_ (Cortex-M4F) - :file:`libnrf_edgeai-cortex-m4.a`
* `nRF53 Series`_ (Cortex-M33F) - :file:`libnrf_edgeai-cortex-m33.a`
* `nRF54 Series`_ (Cortex-M33F) - :file:`libnrf_edgeai-cortex-m33.a`

It is required that the target device has an FPU to run the library, as the compiled binaries are built with floating point hard ABI flag.

.. _requirements_memory:

RAM and flash memory requirements
*********************************

|EAILib| is highly optimized for minimal memory footprint.
Ensure you have:

* RAM: 1-5 KB
* Flash: 5-10 KB

RAM and flash memory requirement values differ depending on user model and programmed sample.
