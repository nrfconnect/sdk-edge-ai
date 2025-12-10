.. _runtime_anomaly_sample:

Anomaly detection sample
########################

.. contents::
   :local:
   :depth: 2

This sample demonstrates an anomaly-detection model that monitors dual-axis vibration
data to detect gear faults.

Requirements
************

The sample supports the following development kits:

.. list-table::
   :header-rows: 1

   * - Hardware platforms
     - PCA
     - Board name
     - Board target
   * - `nRF54H20 DK`_
     - PCA10175
     - `nrf54h20dk`_
     - ``nrf54h20dk/nrf54h20/cpuapp``
   * - `nRF54L15 DK`_
     - PCA10156
     - `nrf54l15dk`_
     - ``nrf54l15dk/nrf54l15/cpuapp``
   * - `nRF54L15 DK (emulating nRF54L10) <nRF54L15 DK_>`_
     - PCA10156
     - `nrf54l15dk <nRF54L15 emulation_>`_
     - ``nrf54l15dk/nrf54l10/cpuapp``
   * - `nRF54L15 DK (emulating nRF54L05) <nRF54L15 DK_>`_
     - PCA10156
     - `nrf54l15dk <nRF54L15 emulation_>`_
     - ``nrf54l15dk/nrf54l05/cpuapp``
   * - `nRF5340 DK`_
     - PCA10095
     - `nrf5340dk`_
     - | ``nrf5340dk/nrf5340/cpuapp``
       | ``nrf5340dk/nrf5340/cpuapp/ns``
   * - `Thingy:53`_
     - PCA20053
     - `thingy53`_
     - ``thingy53/nrf5340/cpuapp``
   * - `nRF52 DK`_
     - PCA10040
     - `nrf52dk`_
     - ``nrf52dk/nrf52832``
   * - `nRF52840 DK <nRF52 DK_>`_
     - PCA10059
     - `nrf52840dk`_
     - ``nrf52840dk/nrf52840``

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
   CONFIG_NEWLIB_LIBC=y
   CONFIG_FPU=y
   CONFIG_CONSOLE=y
   CONFIG_UART_CONSOLE=y
   CONFIG_RTT_CONSOLE=n
   CONFIG_NEWLIB_LIBC_FLOAT_PRINTF=y

.. include:: ../../includes/include_kconfig_edgeai.txt

Building and running
********************

.. include:: ../../includes/include_building_and_running_edgeai.txt

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

.. _runtime_anomaly_sample_inference:

Manual inference using the API
==============================

You can also run inference manually in your own application code.

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

* Header file: :file:`include/zephyr/kernel.h`
