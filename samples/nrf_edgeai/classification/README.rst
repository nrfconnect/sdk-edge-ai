.. _runtime_classification_sample:

Classification sample
#####################

.. contents::
   :local:
   :depth: 2

This sample demonstrates a multi-class classifier that identifies parcel delivery states (Idle, Shaking, Impact, Free Fall, Carrying, In Car, Placed) from a
stream of acceleration magnitude samples.

By default, the model is compiled directly into the application image, same as any other nRF Edge AI sample.
On the nRF54LM20 DK boards, you can opt into loading the model from a dedicated ``model_storage`` flash partition at runtime instead, so that flashing a new model package - independently of the application binary, and without mcuboot - is enough to change what the device predicts.
See `Model-only OTA update`_ below.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample evaluates activity classification over consecutive 50-sample windows, printing both the predicted activity class and the associated confidence probabilities for each window.
The model processes a single input feature—acceleration magnitude—collected over 50 consecutive samples for each prediction.

It classifies each window into one of seven possible activity classes:

* IDLE
* SHAKING
* IMPACT
* FREE_FALL
* CARRYING
* IN_CAR
* PLACED

Each time a complete 50-sample window is received, the model outputs its prediction and corresponding probabilities for each possible class.
The model does not require additional context beyond each window to make its predictions.
The predicted class for each window is the one with the highest confidence probability.

Configuration
*************

|config|

The project configuration for this sample is provided in :file:`samples/nrf_edgeai/classification/prj.conf`.

Model backend (Neuton and Axon)
===============================

The sample can use either of two model backends, selected in Kconfig:

* Neuton (CPU) — Runs on the application core.
  It is supported on all nRF Edge AI boards.
* Axon (NPU) — Runs on the Axon neural processing unit.
  It is available only on SoCs with Axon NPU.

To select the model backend, set the ``CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_NEUTON`` or ``CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_AXON`` Kconfig option in your :file:`prj.conf` file.
See board-specific configuration and overlays in the :file:`samples/nrf_edgeai/classification/boards/` folder.
When using the Axon backend, the generated model saves its buffer requirements in the :file:`prj_example.conf` file as the ``CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE`` and ``CONFIG_NRF_AXON_PSUM_BUFFER_SIZE`` Kconfig options.
You must manually include these values in your :file:`prj.conf` file before building.

Configuration options
=====================

In your :file:`prj.conf` file, the following settings are applied to ensure the sample builds and runs correctly:

.. code-block:: ini

   CONFIG_NRF_EDGEAI=y
   CONFIG_FPU=y
   CONFIG_CONSOLE=y
   CONFIG_UART_CONSOLE=y
   CONFIG_RTT_CONSOLE=n
   CONFIG_PICOLIBC_IO_FLOAT=y

:kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA` (see `Model-only OTA update`_) selects ``FLASH``/``FLASH_MAP``/``CRC`` and the matching ``MODEL_OTA``/``MODEL_OTA_NEUTON``/``MODEL_OTA_AXON`` options automatically, so they do not need to be listed here.

.. include:: /includes/include_kconfig_edgeai.txt

Building and running
********************

.. include:: /includes/include_building_and_running_edgeai.txt

Testing
*******

The application automatically evaluates a set of validation cases using windows of 50 samples each.
For each case, the sample prints the predicted activity class, its probability, and the expected ground truth label:

.. code-block:: console

    In 7 classes, predicted 1 with probability 0.945678
    Expected class SHAKING - predicted SHAKING

#. Observe the output for each window:

   * The line reports the total number of supported classes (7 in this case).
   * ``predicted`` shows the model's chosen class index, along with its probability for this window.
   * The following line compares the expected (ground truth) class to the predicted class.

#. Check that the predicted class matches the expected class for each validation sample.

#. Review the class probability for additional insight into the model's confidence in its predictions.

.. _runtime_classification_model_ota:

Model-only OTA update
======================

On the nRF54LM20 DK boards, this sample does not use mcuboot, so its second application slot (``slot1_partition``) is unused.
The board overlays in :file:`samples/nrf_edgeai/classification/boards/` repurpose that space as a dedicated ``model_storage`` partition instead, sized to comfortably fit larger models too.
Build with :kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA` set to ``y`` to load the model from ``model_storage`` at runtime instead of compiling it in: at boot (and every 5 seconds thereafter), the sample reads and validates a small header-plus-payload "model package" from that partition and wires it up for inference - see :ref:`lib_model_ota` for how the package format, host-side packaging tools, and on-device loading work.
Flashing a new package to ``model_storage`` is enough to change what the device predicts, without rebuilding or reflashing the application.

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp samples/nrf_edgeai/classification \
       -- -DCONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_AXON=y -DCONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA=y

On an unprovisioned (or invalid) ``model_storage`` partition, the sample logs the following and skips inference every 5 seconds until a valid package is flashed:

.. code-block:: console

  No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

A classification model has no output scale to speak of (unlike the regression sample's Neuton model), so the OTA loaders here need nothing beyond the model's own weights/topology.

Packaging a Neuton model
-------------------------

Neuton packages only need the model's raw arrays (weights, topology), with no embedded addresses.
With both :kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_NEUTON` and :kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA` enabled, this sample's package is built automatically as part of a normal build - :file:`CMakeLists.txt`'s ``nrf_neuton_model_package()`` call runs :file:`package_model_neuton.py` against :file:`src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c` (this model's own generated source, standing in for a real training run's output), with no separate build or manual packaging step needed:

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/classification \
       -- -DCONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA=y

This produces ``build/classification/classification_model_pkg.bin``/``.hex``.

To package a different (for example freshly retrained) model instead, point :file:`package_model_neuton.py` at its own generated source directly:

.. code-block:: console

   python3 tools/model_ota/package_model_neuton.py \
     path/to/nrf_edgeai_user_model.c --name parcel_classification --version 1.0.0 -o model_v1 \
     --dts build/classification/zephyr/zephyr.dts

``--dts`` reads the ``model_storage`` partition's actual address and size from a build's generated :file:`zephyr.dts` and preflight-checks the package fits, instead of trusting the tool's nRF54LM20 DK defaults to still match your build; point it at any existing OTA-enabled build of this sample (Neuton or Axon - the partition layout is the same either way).

Packaging an Axon model
-------------------------

Axon packages are built automatically as part of a normal application build once :kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_AXON` and :kconfig:option:`CONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA` are both enabled - no separate build or manual packaging step is needed. See :ref:`lib_model_ota` ("Build-time model packaging") for how :file:`CMakeLists.txt`'s ``nrf_axon_model_stub()`` call does this, and for how Axon packages are put together.

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/nrf_edgeai/classification \
       -- -DCONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_AXON=y -DCONFIG_NRF_EDGEAI_CLASSIFICATION_MODEL_OTA=y

This produces ``build/classification/classification_model_pkg.bin``/``.hex``.

Flashing a package
-------------------

Build and flash the application as usual, then flash the model package it produced to the ``model_storage`` partition:

.. code-block:: console

   nrfutil device program --firmware build/classification/classification_model_pkg.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` ensures the board resumes execution automatically; without it, ``nrfutil`` leaves the CPU halted after flashing.

.. _runtime_classification_sample_inference:

Manual inference using the API
==============================

You can also perform manual inference in your own application code by providing sample data and inspecting the model output.
This example uses the compiled-in model retrieval pattern (``nrf_edgeai_user_model()``); if you build this sample with `Model-only OTA update`_ enabled, the rest of the inference API (``nrf_edgeai_feed_inputs()``, ``nrf_edgeai_run_inference()``, ``decoded_output.classif``) is identical either way.

The following example demonstrates how to initialize the model, feed your own 50-sample window, and print out the predicted class and its probabilities:

.. code-block:: c

    #include <nrf_edgeai/nrf_edgeai.h>
    #include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
    #include <assert.h>
    #include <stdio.h>
    // In this example, our raw features is a window of N elements with 3 accelerometer axis values
    // The number of raw features and their order should be the same as in the training dataset file
    int16_t raw_features[] =
    {
        Accelerometer_X0,
        Accelerometer_Y0,
        Accelerometer_Z0,
        /* ... */
        Accelerometer_Xn,
        Accelerometer_Yn,
        Accelerometer_Zn,
    };
    // Pointer to user model
    static nrf_edgeai_t* p_edgeai = NULL;

    void user_init_edegeai_model(void)
    {
        // Get user model pointer
        p_edgeai = nrf_edgeai_user_model();
        // Init EdgeAI library based on user solution, should be called once!
        nrf_edgeai_err_t res = nrf_edgeai_init(p_edgeai);
        // Optional check for success, #include <assert.h> required
        assert(res == NRF_EDGEAI_ERR_SUCCESS);
    }
    //
    // ....
    //
    void user_feed_data_to_model(void)
    {
        // Feed and prepare raw inputs for the model inference
        nrf_edgeai_err_t res = nrf_edgeai_feed_inputs(p_edgeai, raw_features,
                                                nrf_edgeai_uniq_inputs_num(p_edgeai) *
                                                nrf_edgeai_input_window_size(p_edgeai));

        // Check if input data is prepared and ready for model inference
        if (res == NRF_EDGEAI_ERR_SUCCESS)
        {
            // Run model inference
            res = nrf_edgeai_run_inference(p_edgeai);
            // Check if model inference is ready and successful
            if (res == NRF_EDGEAI_ERR_SUCCESS)
            {
                uint16_t predicted_class = p_edgeai->decoded_output.classif.predicted_class;
                size_t num_classes = p_edgeai->decoded_output.classif.num_classes;
                // Get probability depending on model quantization: f32, q16, q8. Here is an example for f32 model
                const flt32_t* p_probabilities = p_edgeai->decoded_output.classif.probabilities.p_f32;

                printf("Predicted class %u with probability %f, in %u classes\r\n", predicted_class,
                                                                                    p_probabilities[predicted_class],
                                                                                    num_classes);
            }
        }

    }


This example prints the predicted activity class, the associated probability, and the number of supported classes:

.. code-block:: console

   Predicted class 1 with probability 0.945678, out of 7 classes

If you want to compare to a known expected class or print human-readable class labels, you can add such logic based on the application's requirements.

Dependencies
************

* Model-only OTA update PoC library (:file:`lib/model_ota`, see :ref:`lib_model_ota`), only when `Model-only OTA update`_ is enabled
* Header file: :file:`include/zephyr/kernel.h`
