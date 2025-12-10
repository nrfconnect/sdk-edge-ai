.. _runtime_classification_sample:

Classification sample
#####################

.. contents::
   :local:
   :depth: 2

This sample demonstrates a multi-class classifier that identifies parcel delivery states (Idle, Shaking, Impact, Free Fall, Carrying, In Car, Placed) from a
stream of acceleration magnitude samples.

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

.. _runtime_classification_sample_inference:

Manual inference using the API
==============================

You can also perform manual inference in your own application code by providing sample data and inspecting the model output.

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

* Header file: :file:`include/zephyr/kernel.h`
