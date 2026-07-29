.. _runtime_regression_sample:

Regression sample
#################

.. contents::
   :local:
   :depth: 2

The following sample demonstrates running a generated regression model to predict a continuous air quality value based on gas sensor and environmental data.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample validates model predictions over a set of 29 test cases, printing the predicted value, expected value, and absolute error for each sample.
The model takes 9 input values for each prediction.
These inputs are:

* Carbon monoxide (CO) concentration
* 5 readings from different PT08S sensors
* Temperature
* Relative humidity (RH)
* Absolute humidity (AH)

The model makes a prediction every time it receives a single new set of input data (that is, after each individual sample).
It does not need multiple samples collected over time to make a prediction.
Each prediction from the model is a single floating-point number representing the estimated air quality value.

Configuration
*************

|config|

The project configuration for this sample is provided in :file:`samples/nrf_edgeai/regression/prj.conf`.

Model backend (Neuton and Axon)
===============================

The sample can use either of two model backends, selected in Kconfig:

* Neuton (CPU) — Runs on the application core.
  It is supported on all nRF Edge AI boards.
* Axon (NPU) — Runs on the Axon neural processing unit.
  It is available only on SoCs with Axon NPU.

To select the model backend, set the ``CONFIG_NRF_EDGEAI_REGRESSION_MODEL_NEUTON`` or ``CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON`` Kconfig option in your :file:`prj.conf` file.
See board-specific configuration and overlays in the :file:`samples/nrf_edgeai/regression/boards/` folder.
When using the Axon backend, the generated model saves its buffer requirements in the :file:`prj_example.conf` file as the ``CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE`` and ``CONFIG_NRF_AXON_PSUM_BUFFER_SIZE`` Kconfig options.
You must manually include these values in your :file:`prj.conf` file before building.

Build types
===========

The sample supports the following build types:

.. list-table:: Regression sample build types
   :widths: auto
   :header-rows: 1

   * - Build type
     - File name
     - Description
   * - Default
     - :file:`prj.conf`
     - Compiled-in model.
   * - Model OTA
     - :file:`sysbuild_model_ota.conf`
     - Partition-resident model with MCUboot and model SMP DFU on nRF54LM20 DK (Neuton on lm20a, Axon on lm20b).

See `Custom build types`_ and `Providing CMake options`_ for more information.

.. _runtime_regression_sample_model_ota:

Model-only OTA update
=====================

On nRF54LM20 DK, this sample can use MCUboot with two updateable images:

* **Image 0 (firmware):** dual-slot swap-using-move over ``slot0_partition`` / ``slot1_partition``.
* **Image 1 (regression model):** single-slot layout in ``model_storage`` (``slot2_partition`` / ``slot3_partition`` alias the same region).

The devicetree fragments live under :file:`dts/` (included from the application overlay and :file:`sysbuild/mcuboot/boards/`).

At **build time**, the sample builds a partition-resident model image and a wired loader:

* Neuton (``nrf54lm20a``): ``model_ota_neuton_wire()`` / ``nrf_edgeai_load_user_model_90508()``
* Axon (``nrf54lm20b``): ``model_ota_axon_edgeai_wire()`` / ``nrf_edgeai_load_user_model_36025()``

At **boot**, the sample calls the matching loader, validates the partition image, and runs inference.
Inference is paused during SMP uploads to the model image and a **reset** is required before running against a newly uploaded model.

:file:`prj.conf` enables the compiled-in model.
Partition-resident models and MCUboot/SMP DFU are optional via ``SB_CONFIG_APP_MODEL_OTA`` (set in :file:`sysbuild_model_ota.conf`).

Build combinations:

* **Compiled-in model** (default):

  .. code-block:: console

     west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp samples/nrf_edgeai/regression
     west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp samples/nrf_edgeai/regression

* **Partition-resident model + model SMP DFU:**

  .. code-block:: console

     west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/regression \
       -- -DSB_EXTRA_CONF_FILE=sysbuild_model_ota.conf
     west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/nrf_edgeai/regression \
       -- -DSB_EXTRA_CONF_FILE=sysbuild_model_ota.conf

This produces:

* ``build/regression/regression_model_mcuboot.signed.{bin,hex}`` - regression model (MCUboot image 1)
* ``build/regression/zephyr/zephyr.signed.{bin,hex}`` - application firmware (MCUboot image 0)
* ``build/regression_provision.hex`` - merged bootloader, application, and signed model

First-time provisioning must flash the **full sysbuild image chain**, not the application ``zephyr.hex`` alone:

.. code-block:: console

   west flash -d build --recover --no-rebuild

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

.. include:: /includes/include_kconfig_edgeai.txt

Building and running
********************

.. include:: /includes/include_building_and_running_edgeai.txt

Testing
=======

The application runs 29 validation test cases automatically upon startup.
For each case, the sample prints a line similar to the following:

.. code-block:: console

  Air quality - Predicted: 12.345678, Expected: 14.300000, absolute error 1.954322

#. Observe the results printed:

   * ``Predicted value`` corresponds to the air quality value predicted by the model for the given input.
   * ``Expected value`` corresponds to the reference value the model should ideally predict for this input.
   * ``Absolute error`` corresponds to the difference between the predicted and expected values.

#. Confirm that a total of 29 lines are printed, each corresponding to one validation sample.
#. Inspect the absolute error value for each line to verify that the model's predictions are close to the expected values.
   Acceptable error margins depend on your use case or specified requirements in your project.

.. _runtime_regression_sample_inference:

Manual inference using the API
==============================

You can also run inference manually in your own application code.

The following example demonstrates how to initialize the model, feed your own test data, and print out predicted values:

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
                const flt32_t* p_predicted_values = p_edgeai->decoded_output.regression.p_outputs;
                size_t values_num = p_edgeai->decoded_output.regression.outputs_num;

                printf("Predicted target values:\r\n");
                for (size_t i = 0; i < values_num; i++)
                {
                    printf("%f,", p_predicted_values[i]);
                }
                printf("\r\n");
            }
        }
    }

This example prints only the predicted model value(s):

.. code-block:: console

  Predicted target values:
  12.345678,

If you wish to validate predictions (as done in the automated validation), you add code to compare the prediction to a known expected value, and print the absolute error.

Dependencies
************

* Header file: :file:`include/zephyr/kernel.h`
