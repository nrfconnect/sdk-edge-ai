.. _ug_edge_impulse:

Edge Impulse integration
########################

.. contents::
   :local:
   :depth: 2

`Edge Impulse`_ is a development platform that can be used to enable `embedded machine learning`_ on |NCS| devices.
You can use this platform to collect data from sensors, train machine learning model, and then deploy it to your Nordic Semiconductor's device.

Integration prerequisites
*************************

Before you start the |NCS| integration with Edge Impulse, make sure that the following prerequisites are completed:

* :ref:`Installation of the nRF Connect SDK <installation>`.
* Setup of the required :term:`Development Kit (DK)`.
* Creation of an `Edge Impulse studio account <Edge Impulse studio signup_>`_ and an Edge Impulse project.

Solution architecture
*********************

The usage of `Edge Impulse`_ SDK is demonstrated by the :ref:`hello_ei_sample` sample

.. _ug_edge_impulse_adding:

Integration overview
********************

Before integrating the Edge Impulse machine learning model into an |NCS| application, you must prepare and deploy the machine learning model for your embedded device.
This model is prepared using the `Edge Impulse studio`_ external web tool.
It relies on sensor data that can be provided by different sources, for example data forwarder.
Check the :ref:`ei_data_forwarder_sample` sample for a demonstration of how you can send sensor data to Edge Impulse studio using `Edge Impulse's data forwarder`_.

The machine learning model is distributed as a single :file:`zip` archive that includes C++ library sources.
This file is used by the |NCS| build system to build the Edge Impulse library.

Integration steps
*****************

Complete the following steps to generate the archive and add it to the build system:

1. :ref:`ug_edge_impulse_adding_preparing`
#. :ref:`ug_edge_impulse_adding_building`

.. _ug_edge_impulse_adding_preparing:

.. rst-class:: numbered-step

Preparing the machine learning model
====================================

To prepare the machine learning model, use `Edge Impulse studio`_ and follow one of the tutorials described in `Edge Impulse getting started guide`_.
For example, you can try the `Continuous motion recognition tutorial`_.
This tutorial will guide you through the following steps:

1. Collecting data from sensors and uploading the data to Edge Impulse studio.

   .. note::
     You can use one of the development boards supported directly by Edge Impulse or your mobile phone to collect the data.
     You can also modify the :ref:`ei_data_forwarder_sample` sample or :ref:`nrf_machine_learning_app` application and use it to forward data from a sensor that is connected to any board available in the |NCS|.

#. Designing your machine learning model (an *impulse*).
#. Deploying the machine learning model to use it on an embedded device.
   As part of this step, you must select the :guilabel:`C++ library` to generate the required :file:`zip` file that contains the source files for building the Edge Impulse library in |NCS|.

.. _ug_edge_impulse_adding_building:

.. rst-class:: numbered-step

Building the machine learning model in |NCS|
============================================

After preparing the :file:`zip` archive, you can use the |NCS| build system to build the C++ library with the machine learning model.
Complete the following steps to configure the building process:

1. Make sure that the following Kconfig options are enabled:

   * :kconfig:option:`CONFIG_CPP`
   * :kconfig:option:`CONFIG_STD_CPP11`
   * :kconfig:option:`CONFIG_REQUIRES_FULL_LIBCPP`

   .. note::
      The :kconfig:option:`CONFIG_FPU` Kconfig option is implied by default if floating point unit (FPU) is supported by the hardware.
      Using FPU speeds up calculations.

#. Make sure that the :kconfig:option:`CONFIG_FP16` Kconfig option is disabled.
   The Edge Impulse library is not compatible with half-precision floating point support introduced in Zephyr.

Applications and samples
************************

The following samples demonstrate the Edge Impulse integration in the |NCS|:

* :ref:`hello_ei_sample` sample - demonstrates the usage of the wrapper.
* :ref:`ei_data_forwarder_sample` sample - demonstrates how you can send sensor data to Edge Impulse studio using `Edge Impulse's data forwarder`_.
