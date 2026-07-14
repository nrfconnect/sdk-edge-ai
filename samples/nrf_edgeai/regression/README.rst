.. _runtime_regression_sample:

Regression sample
#################

.. contents::
   :local:
   :depth: 2

The following sample demonstrates running a generated regression model to predict a continuous air quality value based on gas sensor and environmental data.

By default, the model itself is not compiled into the application image: at boot (and periodically thereafter) the sample loads and validates a "model package" from a dedicated ``model_storage`` flash partition, and only then runs inference against it.
Flashing a new model package to ``model_storage`` — independently of the application binary, and without mcuboot — is enough to change what the device predicts.
See `Model-only OTA update`_ below, including how to opt out of it and restore the compiled-in-model behavior instead.

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

Selecting a backend only determines which model package type the sample expects to find in the ``model_storage`` flash partition at runtime — it does not compile in a model of either type. See `Model-only OTA update`_.

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

:kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA` (see `Model-only OTA update`_) selects ``FLASH``/``FLASH_MAP``/``CRC`` and the matching ``MODEL_OTA``/``MODEL_OTA_NEUTON``/``MODEL_OTA_AXON`` options automatically, so they do not need to be listed here.

.. include:: /includes/include_kconfig_edgeai.txt

Building and running
********************

.. include:: /includes/include_building_and_running_edgeai.txt

Testing
=======

On a device with an unprovisioned (or invalid) ``model_storage`` partition, the sample logs the following and skips inference every 5 seconds until a valid package is flashed:

.. code-block:: console

  No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

Once ``model_storage`` holds a valid package matching the selected backend (see `Model-only OTA update`_), the sample runs 29 validation test cases every 5 seconds.
For each case, it prints a line similar to the following:

.. code-block:: console

  Air quality - Predicted: 12.345678, Expected: 14.300000, absolute error 1.954322

#. Observe the results printed:

   * ``Predicted value`` corresponds to the air quality value predicted by the model for the given input.
   * ``Expected value`` corresponds to the reference value the model should ideally predict for this input.
   * ``Absolute error`` corresponds to the difference between the predicted and expected values.

#. Confirm that a total of 29 lines are printed, each corresponding to one validation sample.
#. Inspect the absolute error value for each line to verify that the model's predictions are close to the expected values.
   Acceptable error margins depend on your use case or specified requirements in your project.

.. _runtime_regression_model_ota:

Model-only OTA update
======================

This sample does not use mcuboot, so its second application slot (``slot1_partition``) is unused on the boards it supports.
The board overlays in :file:`samples/nrf_edgeai/regression/boards/` repurpose that space as a dedicated ``model_storage`` partition instead, sized to comfortably fit larger models too.
At boot (and every 5 seconds thereafter), the sample reads and validates a small header-plus-payload "model package" from ``model_storage`` and wires it up for inference — see :ref:`lib_model_ota` for how the package format, host-side packaging tools, and on-device loading work.
Flashing a new package to ``model_storage`` is enough to change what the device predicts, without rebuilding or reflashing the application.

Making model OTA optional
--------------------------

Model-only OTA is enabled by default (:kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA` defaults to ``y``).
Build with it disabled to restore this sample's original, pre-model-OTA behavior instead: the selected backend's model (Neuton or Axon) is compiled directly into the application image, no ``model_storage`` partition or flash package is involved, and the 29 test cases are validated once at boot — asserting on the expected accuracy — rather than being reloaded and re-validated every 5 seconds.

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp samples/nrf_edgeai/regression \
       -- -DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON=y -DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA=n

Packaging a Neuton model
-------------------------

Neuton packages only need the model's raw arrays (weights, topology, output scaling), with no embedded addresses.
Like Axon (see below), this sample's ``model_v1``-equivalent package is now built automatically as part of a normal build (:kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_NEUTON` + :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA`, both on by default on boards other than ``nrf54lm20b``) - :file:`CMakeLists.txt`'s ``nrf_neuton_model_package()`` call runs :file:`package_model_neuton.py` against :file:`tools/model_ota/models/regression_v1_generated.c` (a restored copy of this model's original generated source, standing in for a real training run's output) with no separate build or manual packaging step needed:

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/regression

This produces ``build/regression/regression_model_pkg.bin``/``.hex``.

To package a different (for example freshly retrained) model instead, point :file:`package_model_neuton.py` at its own generated source directly:

.. code-block:: console

   python3 tools/model_ota/package_model_neuton.py \
     path/to/nrf_edgeai_user_model.c --name aq_regression --version 1.0.0 -o model_v1 \
     --dts build/regression/zephyr/zephyr.dts

:file:`tools/model_ota/models/regression_v2.json` is a hand-tweaked variant with no corresponding generated source, useful for observing a change in predictions after an update; package it with :file:`package_model.py` instead (see :ref:`lib_model_ota`, "Host-side packaging tools").

``--dts`` reads the ``model_storage`` partition's actual address and size from a build's generated :file:`zephyr.dts` and preflight-checks the package fits, instead of trusting the tool's nRF54LM20 DK defaults to still match your build; point it at any existing build of this sample (Neuton or Axon - the partition layout is the same either way).

Packaging an Axon model
------------------------

Axon packages are built automatically as part of a normal application build (:kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON` + :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA`, both on by default on ``nrf54lm20b``) - no separate build or manual packaging step is needed. See :ref:`lib_model_ota` ("Build-time model packaging") for how :file:`CMakeLists.txt`'s ``nrf_axon_model_stub()`` call does this, and for how Axon packages are put together.

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/nrf_edgeai/regression

This produces ``build/regression/regression_model_pkg.bin``/``.hex``.

Flashing a package
-------------------

Build and flash the application as usual, then flash the model package it produced to the ``model_storage`` partition:

.. code-block:: console

   nrfutil device program --firmware build/regression/regression_model_pkg.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` ensures the board resumes execution automatically; without it, ``nrfutil`` leaves the CPU halted after flashing.

Repeat with a package built from :file:`regression_v2.json` (Neuton), or a hand-tweaked generated Axon model header, rebuilt, to observe predictions change after the update — no application rebuild or reflash required.

.. _runtime_regression_sample_inference:

Manual inference using the API
==============================

You can also run inference manually in your own application code.

The following example demonstrates how to initialize the model, feed your own test data, and print out predicted values, using the compiled-in model retrieval pattern (``nrf_edgeai_user_model()``) typical of nRF Edge AI samples.
This sample itself instead loads its model from flash at runtime — see `Model-only OTA update`_ — but the rest of the inference API (``nrf_edgeai_feed_inputs()``, ``nrf_edgeai_run_inference()``, ``decoded_output.regression``) is identical either way.

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

* Model-only OTA update PoC library (:file:`lib/model_ota`, see :ref:`lib_model_ota`)
* Header file: :file:`include/zephyr/kernel.h`
