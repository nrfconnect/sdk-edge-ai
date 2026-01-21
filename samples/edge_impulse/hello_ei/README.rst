.. _hello_ei_sample:

Edge Impulse: Hello Edge Impulse
#####################

.. contents::
   :local:
   :depth: 2

The Hello Edge Impulse sample demonstrates usage of `Edge Impulse`_ SDK and custom machine learning model when :ref:`integrating Edge Impulse with the nRF Connect SDK <ug_edge_impulse>`.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample:
1. Provides input data to the `Edge Impulse`_ model.
#. Starts predictions using the machine learning model.
#. Displays the prediction results and time measurements to the user.

By default, the sample uses a pre-trained machine learning model and 2 input data series representing a sine wave and a triangle wave.

Configuration
*************

|config|

The sample can be configured using the following Kconfig options:

.. options-from-kconfig::
   :show-type:

Using your own machine learning model
=====================================

To run the sample using a custom machine learning model, you must complete the following setup:

1. Prepare your own machine learning model.

   To prepare the machine learning model, use `Edge Impulse studio`_ and follow one of the tutorials described in `Edge Impulse getting started guide`_.
   For example, you can try the `Continuous motion recognition tutorial`_.
   This tutorial will guide you through the following steps:

   * Collecting data from sensors and uploading the data to Edge Impulse studio.

      .. note::
      You can use one of the development boards supported directly by Edge Impulse or your mobile phone to collect the data.
      You can also modify the :ref:`ei_data_forwarder_sample` sample and use it to forward data from a sensor that is connected to any board available in the |NCS|.

   * Designing your machine learning model (an *impulse*).
   * Deploying the machine learning model to use it on an embedded device.
      As part of this step, you must select the :guilabel:`C++ library` to generate the required :file:`zip` file that contains the source files for building the Edge Impulse library in |NCS|.

#. Select the Edge Impulse model by completing the following steps:

   a. Set the :kconfig:option:`CONFIG_EDGE_IMPULSE_URI` to URI of your machine learning model.
      You can set it to one of the following values:

      * An absolute or relative path to a file in the local file system.
        For this variant, you must download the :file:`zip` file manually and place it under path defined by the Kconfig option.
        The relative path is tracked from the application source directory (``APPLICATION_SOURCE_DIR``).
        CMake variables that are part of the path are expanded.
      * Any downloadable URI supported by CMake's ``file(DOWNLOAD)`` command.
        For this variant, the |NCS| build system will download the :file:`zip` file automatically during build.
        The :file:`zip` file is downloaded into your application's :file:`build` directory.

        If the URI requires providing an additional API key in the HTTP header, you can provide it using the :c:macro:`EI_API_KEY_HEADER` CMake definition.
        The API key is provided using a format in which *key_name* is followed by *key_value*.
        For example, if the URI uses ``x-api_key`` for authentication, the :c:macro:`EI_API_KEY_HEADER` can be defined as follows: ``x-api-key:aaaabbbbccccdddd``.
        The ``aaaabbbbccccdddd`` is a sample *key_value*.
        See :ref:`cmake_options` for more information about defining CMake options for command line builds and |nRFVSC|.
        See `Downloading model directly from Edge Impulse studio`_ for details about downloading model directly from the Edge Impulse studio.

#. Define the input data for the machine learning model in :file:`samples/edge_impulse/hello_ei/src/include/input_data.h`.
#. Check the example input data in your Edge Impulse studio project:

   a. Go to the :guilabel:`Live classification` tab.
   #. In the **Classifying existing test sample** panel, select one of the test samples.
   #. Click :guilabel:`Load sample` to display the raw data preview.

      .. figure:: ./doc/images/ei_loading_test_sample.png
         :scale: 50 %
         :alt: Loading test sample input data in Edge Impulse studio

         Loading test sample input data in Edge Impulse studio

      The classification results will be displayed, with raw data preview.

      .. figure:: ./doc/images/ei_raw_features.png
         :scale: 50 %
         :alt: Raw data preview in Edge Impulse studio

         Raw data preview in Edge Impulse studio

#. Copy information from the **Raw features** list into an array defined in the :file:`input_data.h` file.

.. note::
    If you provide more input data than a single input window can hold, the prediction will be triggered multiple times.
    The input window will be shifted by one input frame between subsequent predictions.
    The prediction will be retriggered until there is no more input data.

Downloading model directly from Edge Impulse studio
---------------------------------------------------

As an example of downloadable URI, you can configure the |NCS| build system to download your model directly from the Edge Impulse studio.
You can download a model from either a private or a public project.

.. tabs::

   .. group-tab:: Private project

      When downloading from a private project, you must provide an API key for authentication.

      1. Set :kconfig:option:`CONFIG_EDGE_IMPULSE_URI` to the URI from Edge Impulse studio:

         .. parsed-literal::
            :class: highlight

            CONFIG_EDGE_IMPULSE_URI="https:\ //studio.edgeimpulse.com/v1/api/*XYZ*/deployment/download?type=zip"

         Set *XYZ* to the project ID of your Edge Impulse project.
         You can check the project ID of your project in the **Project info** panel under :guilabel:`Dashboard`.

         .. figure:: ./doc/images/ei_project_id.png
            :scale: 50 %
            :alt: Project ID in Edge Impulse studio dashboard

            Project ID in Edge Impulse studio dashboard

      #. Define the :c:macro:`EI_API_KEY_HEADER` CMake option (see :ref:`cmake_options`) as ``x-api-key:[ei api key]`` to provide the x-api-key associated with your Edge Impulse project.
         To check what to provide as the *[ei api key]* value, check your API keys under the :guilabel:`Keys` tab in the Edge Impulse project dashboard.

         .. figure:: ./doc/images/ei_api_key.png
            :scale: 50 %
            :alt: API key under the Keys tab in Edge Impulse studio

            API key under the Keys tab in Edge Impulse studio

   .. group-tab:: Public project

      When downloading from a public project, no authentication is required.

      1. Check the ID of the public project:

         a. Check the project ID of your project in the **Project info** panel under :guilabel:`Dashboard`.
         #. Provide this project ID in the *XYZ* field in the following URL:

            .. parsed-literal::
               :class: highlight

               https:\ //studio.edgeimpulse.com/v1/api/*XYZ*/versions/public

         #. Paste the URL into your browser.
            The ID of the public project is returned as the value of the ``publicProjectId`` field.
            For example:

            .. parsed-literal::
               :class: highlight

               {"success":true,"versions":[{"version":1,"publicProjectId":66469,"publicProjectUrl":"https://studio.edgeimpulse.com/public/66468/latest"}]}

            In this example, the *XYZ* project ID is ``66468``, while the ``publicProjectId`` equals ``66469``.

      #. Set :kconfig:option:`CONFIG_EDGE_IMPULSE_URI` to the following URI from Edge Impulse studio:

         .. parsed-literal::
            :class: highlight

            CONFIG_EDGE_IMPULSE_URI="https:\ //studio.edgeimpulse.com/v1/api/*XYZ*/deployment/download?type=zip&modelType=int8"

         Set the *XYZ* to the public project ID from previous step.
         Using the example above, this would be ``66469``.

         .. note::
            This URI includes the ``modelType=int8`` parameter because from public Edge Impulse projects you can only download quantized models created with Edge Impulse's EON Compiler.

Building and running
********************

.. |sample path| replace:: :file:`samples/edge_impulse/hello_ei`

Testing
=======

After programming the sample to your development kit, test it by performing the following steps:

1. |connect_terminal|
#. Reset the kit.
#. Observe that output similar to the following is logged on UART:

   .. parsed-literal::
      :class: highlight

      *** Booting nRF Connect SDK v3.2.0-5dcc6bd39b0f ***
      *** Using Zephyr OS v4.2.99-a57ad913cf4e ***
      I: === Model info ===
      I: Input frame size: 3
      I: Input window size: 312
      I: Input frequency: 52
      I: Label count: 3
      I: Labels assigned:
      I: - idle
      I: - sine
      I: - triangle
      I: Has anomaly: yes
      I: Running inference on sine wave input data
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.99219
      I: triangle => 0.00781
      I: anomaly: -0.12298
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 1 ms
      I: DSP operations completed in 5 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.99219
      I: triangle => 0.00781
      I: anomaly: -0.12898
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 0 ms
      I: DSP operations completed in 6 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.99219
      I: triangle => 0.00781
      I: anomaly: -0.12708
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 1 ms
      I: DSP operations completed in 5 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: End of input data reached
      I:
      I: Running inference on triangle wave input data
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.00000
      I: triangle => 0.99609
      I: anomaly: -0.26885
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 1 ms
      I: DSP operations completed in 5 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.00000
      I: triangle => 0.99609
      I: anomaly: -0.27599
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 0 ms
      I: DSP operations completed in 6 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: === Inference result ===
      I: idle => 0.00000
      I: sine => 0.00000
      I: triangle => 0.99609
      I: anomaly: -0.28299
      I: === Inference time profiling ===
      I: Full inference completed in 6 ms
      I: Classification completed in 1 ms
      I: DSP operations completed in 5 ms
      I: Anomaly detection completed in 0 ms
      I:
      I: End of input data reached

The observed classification results depend on machine learning model and input data.

Dependencies
************

This sample uses the following Zephyr libraries:

* Logging

This sample uses the following external components:

* `Edge Impulse`_ SDK
