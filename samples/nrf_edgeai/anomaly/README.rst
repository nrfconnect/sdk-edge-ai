.. _runtime_anomaly_sample:

Anomaly detection sample
########################

.. contents::
   :local:
   :depth: 2

This sample demonstrates an anomaly-detection model that monitors dual-axis vibration
data to detect gear faults.

By default, the model is compiled directly into the application image, same as any other nRF Edge AI sample.
On the nRF54LM20 DK boards, you can opt into loading the model from a dedicated ``model_storage`` flash partition at runtime instead, so that flashing a new model package - independently of the application binary, and without mcuboot - is enough to change what the device predicts.
See `Model-only OTA update`_ below.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample evaluates anomaly scores over a series of 128-sample windows, printing the computed score and comparing it to the configured threshold for each window.
Scores that exceed the threshold indicate a potential fault in the observed signal.

The model uses 2 input features for each prediction, corresponding to the X and Y vibration axes.
These input features are interleaved in the input data sequence.
The model processes one window of 128 samples at a time, requiring the full window to generate a score.
It does not analyze individual samples in isolation.
Predictions are only made after a complete window is received.
Each prediction from the model is a single floating-point number representing the computed anomaly score.

Configuration
*************

|config|

The project configuration for this sample is provided in :file:`samples/nrf_edgeai/anomaly/prj.conf`.

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

:kconfig:option:`CONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA` (see `Model-only OTA update`_) selects ``FLASH``/``FLASH_MAP``/``CRC`` and the matching ``MODEL_OTA``/``MODEL_OTA_NEUTON`` options automatically, so they do not need to be listed here.

.. include:: /includes/include_kconfig_edgeai.txt

Building and running
********************

.. include:: /includes/include_building_and_running_edgeai.txt

Testing
=======

The application automatically processes windows of 128 samples and computes an anomaly score for each window upon startup.
For each input case, the sample prints output similar to the following, providing the computed score and a human-readable verdict:

.. code-block:: console

    Anomaly score for GOOD gear data: 0.000010
    Verdict: NORMAL (score < threshold)

.. code-block:: console

    Anomaly score for ANOMALOUS gear data: 0.000120
    Verdict: ANOMALY DETECTED (score >= threshold)

#. Observe the printed results for each test window:

   * Anomaly score shows the model's computed score for the current window of vibration data.
   * Verdict indicates whether the model classifies the input as normal or anomalous based on the configured threshold.

#. Confirm that you see output lines for each window processed.
#. Check that windows representing normal (good) data have scores below the threshold, and that anomalous data yields scores at or above the threshold.
#. Adjust the anomaly score threshold as needed for your specific application and use case.

.. _runtime_anomaly_model_ota:

Model-only OTA update
======================

On the nRF54LM20 DK boards, this sample does not use mcuboot, so its second application slot (``slot1_partition``) is unused.
The board overlays in :file:`samples/nrf_edgeai/anomaly/boards/` repurpose that space as a dedicated ``model_storage`` partition instead, sized to comfortably fit larger models too.
Build with :kconfig:option:`CONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA` set to ``y`` to load the model from ``model_storage`` at runtime instead of compiling it in: at boot (and every 5 seconds thereafter), the sample reads and validates a small header-plus-payload "model package" from that partition and wires it up for inference - see :ref:`lib_model_ota` for how the package format, host-side packaging tools, and on-device loading work.
Flashing a new package to ``model_storage`` is enough to change what the device predicts, without rebuilding or reflashing the application.

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp samples/nrf_edgeai/anomaly \
       -- -DCONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA=y

On an unprovisioned (or invalid) ``model_storage`` partition, the sample logs the following and skips inference every 5 seconds until a valid package is flashed:

.. code-block:: console

  No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

Unlike the regression/classification samples, this model is Neuton-only (there is no Axon variant) and quantized (q16), and it is an anomaly-detection task: its package carries not only ``MODEL_OUTPUT_SCALE_MIN``/``MAX`` but also ``MODEL_AVERAGE_EMBEDDING``, all three of which the OTA loader patches into ``decoded_output.anomaly.meta`` on a successful load.

Packaging the model
---------------------

Neuton packages only need the model's raw arrays (weights, topology), with no embedded addresses.
With :kconfig:option:`CONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA` enabled, this sample's package is built automatically as part of a normal build - :file:`CMakeLists.txt`'s ``nrf_neuton_model_package()`` call runs :file:`package_model_neuton.py` against :file:`src/nrf_edgeai_generated/nrf_edgeai_user_model.c` (this model's own generated source, standing in for a real training run's output), with no separate build or manual packaging step needed:

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/anomaly \
       -- -DCONFIG_NRF_EDGEAI_ANOMALY_MODEL_OTA=y

This produces ``build/anomaly/anomaly_model_pkg.bin``/``.hex``.

To package a different (for example freshly retrained) model instead, point :file:`package_model_neuton.py` at its own generated source directly:

.. code-block:: console

   python3 tools/model_ota/package_model_neuton.py \
     path/to/nrf_edgeai_user_model.c --name gear_anomaly --version 1.0.0 -o model_v1 \
     --dts build/anomaly/zephyr/zephyr.dts

``--dts`` reads the ``model_storage`` partition's actual address and size from a build's generated :file:`zephyr.dts` and preflight-checks the package fits, instead of trusting the tool's nRF54LM20 DK defaults to still match your build.

Flashing a package
-------------------

Build and flash the application as usual, then flash the model package it produced to the ``model_storage`` partition:

.. code-block:: console

   nrfutil device program --firmware build/anomaly/anomaly_model_pkg.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` ensures the board resumes execution automatically; without it, ``nrfutil`` leaves the CPU halted after flashing.

.. _runtime_anomaly_sample_inference:

Manual inference using the API
==============================

You can also run inference manually in your own application code.
This example uses the compiled-in model retrieval pattern (``nrf_edgeai_user_model()``); if you build this sample with `Model-only OTA update`_ enabled, the rest of the inference API (``nrf_edgeai_feed_inputs()``, ``nrf_edgeai_run_inference()``, ``decoded_output.anomaly``) is identical either way.

The following example demonstrates how to initialize the model, feed your own window of sensor data, and print out the computed anomaly score:

.. code-block:: c

    #include <nrf_edgeai/nrf_edgeai.h>
    #include <nrf_edgeai_generated/nrf_edgeai_user_model.h>
    #include <assert.h>
    #include <stdio.h>
    // User should define Anomaly Score Threshold to identify anomalies by himself,
    // specific to user application
    #define USER_DEFINED_ANOMALY_SCORE_THRESHOLD 0.6f
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
                flt32_t anomaly_score = p_edgeai->decoded_output.anomaly.score;

                printf("Predicted Anomaly score: %f\r\n", anomaly_score);

                if (anomaly_score > USER_DEFINED_ANOMALY_SCORE_THRESHOLD)
                {
                    printf("Anomaly detected!\n");
                }
            }
        }
    }

This example prints the predicted anomaly score, and outputs a verdict based on whether the score exceeds the configured threshold:

.. code-block:: console

    Predicted anomaly score: 0.000120
    Anomaly detected! (score >= threshold)

You may modify this logic to match your own application requirements, and experiment with different values for the anomaly score threshold as needed.

Dependencies
************

* Model-only OTA update PoC library (:file:`lib/model_ota`, see :ref:`lib_model_ota`), only when `Model-only OTA update`_ is enabled
* Header file: :file:`include/zephyr/kernel.h`
