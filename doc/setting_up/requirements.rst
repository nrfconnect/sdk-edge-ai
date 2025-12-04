.. _nrf_edgeai_requirements:

nRF Edge AI Library requirements
################################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with the nRF Edge AI library.

.. _nrf_edgeai_requirements_software:

Software requirements
*********************

To start working with the nRF Edge AI, you must install the `nRF Connect SDK`_, including all its prerequisites and the |NCS| toolchain.

.. _nrf_edgeai_requirements_hardware:

Hardware requirements
*********************

The support of specific hardware platforms depends on the sample application you wish to run.
For details, see the :ref:`samples_nrf_edgeai_overview` page, or refer to individual samples' :file:`sample.yaml` files in the :file:`samples/nrf_edgeai/<sample>/sample.yaml` directory.

The nRF Edge AI library is provided as compiled binaries for ARM Cortex-M4F and Cortex-M33F architectures.
Additionally, it is suitable for use on various Nordic Semiconductor's devices:

* `nRF52 Series`_ (Cortex-M4F) - :file:`libnrf_edgeai-cortex-m4.a`
* `nRF53 Series`_ (Cortex-M33F) - :file:`libnrf_edgeai-cortex-m33.a`
* `nRF54 Series`_ (Cortex-M33F) - :file:`libnrf_edgeai-cortex-m33.a`

.. _requirements_memory:

RAM and flash memory requirements
*********************************

nRF Edge AI library is highly optimized for minimal memory footprint.
Ensure you have:

* RAM: 1-5 KB
* Flash: 5-10 KB

RAM and flash memory requirement values differ depending on user model and programmed sample.
