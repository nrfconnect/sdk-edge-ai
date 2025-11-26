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

The application runs 29 validation test cases automatically upon startup.
For each case, the sample prints a line similar to the following:

.. code-block:: console

  Air quality - Predicted value: 12.345678, Expected value: 14.300000, absolute error 1.954322

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
