.. _runtime_regression_sample:

Regression sample
#################

.. contents::
   :local:
   :depth: 2

The following sample demonstrates running a generated regression model to predict a continuous air quality value based on gas sensor and environmental data.

By default, the model itself is not compiled into the application image: at boot (and periodically thereafter) the sample loads and validates a "model package" from a dedicated ``model_storage`` flash partition, and only then runs inference against it.
Flashing a new model package to ``model_storage`` — independently of the application binary — is enough to change what the device predicts.
The sample also runs MCUboot for signed dual-slot firmware and model updates. See `Model-only OTA update`_ and `MCUboot and DFU`_.

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

Selecting a backend only determines which model package type the sample expects to find in the ``model_storage`` flash partition at runtime - it does not compile in a model of either type. See `Model-only OTA update`_.

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
     - Model OTA over SMP (UART) and MCUboot dual-image boot.
   * - BLE Memfault gateway
     - :file:`prj_ble_memfault.conf`
     - Adds Bluetooth LE SMP and the Memfault MCUmgr command group so a phone or gateway can read device identity, fetch a model release from Memfault, and upload it over BLE.

See `Custom build types`_ and `Providing CMake options`_ for more information.

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

On nRF54LM20 DK, this sample uses MCUboot with two updateable images:

* **Image 0 (firmware):** dual-slot swap-using-move over ``slot0_partition`` / ``slot1_partition``.
* **Image 1 (model):** layout selected at **sysbuild** time (see `Model slot layout`_ below).

The devicetree fragments live under :file:`dts/` (included from the application overlay and :file:`sysbuild/mcuboot/boards/`).

At boot the sample reads and validates a header-plus-payload "model package" from ``model_storage`` (MCUboot slot2 primary) and wires it up for inference — see :ref:`lib_model_ota` for the package format, host-side packaging tools, and on-device loading work.
With the default single-slot model layout, inference runs against that loaded model until an SMP upload to image 1 starts; the application then pauses inference until you reset after the upload completes.
With the dual-slot model layout, the sample reloads from ``model_storage`` every 5 seconds and can keep running inference while a new model is uploaded to the staging slot.
The loader skips the 32-byte MCUboot header automatically when present.

Model slot layout
-----------------

Choose the model profile in sysbuild Kconfig (fixed at build time; changing it requires a full reflash with ``--recover``).

Single-slot model (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``SB_CONFIG_NRF_EDGEAI_REGRESSION_MODEL_SLOT_SINGLE=y`` (default)

* One 340 kB ``model_storage`` region; devicetree labels it as both ``slot2_partition`` and ``slot3_partition``.
* Larger application slots (684 kB each).
* SMP model uploads overwrite ``model_storage`` in place. **No MCUboot revert** to a previous model in another slot.
* The application loads the model once at boot, pauses inference during image-1 SMP uploads, and requires a **reset** after upload before running against the new model.
* Build produces ``regression_model_mcuboot.signed.bin`` and ``regression_model_mcuboot.signed.hex`` (same image; use either for provision or SMP).

Dual-slot model
^^^^^^^^^^^^^^^

``SB_CONFIG_NRF_EDGEAI_REGRESSION_MODEL_SLOT_DUAL=y``

* Separate equal 340 kB slots: ``slot2_partition`` / ``model_storage`` (live) and ``slot3_partition`` (staging).
* Smaller application slots (460 kB each) to fit both model slots.
* SMP uploads target slot3; MCUboot swaps on reboot. If the application fails to load the new model, image 1 stays unconfirmed and MCUboot **reverts** on the next reset.
* Build produces ``regression_model_mcuboot.signed.hex`` (first flash to ``model_storage``) and ``regression_model_mcuboot.signed.bin`` (SMP OTA). Both are unconfirmed at build time; after a dual-slot SMP swap the application confirms image 1 once the new model loads successfully.

Example dual-slot sysbuild invocation:

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20a/cpuapp samples/nrf_edgeai/regression -d build_dual \
       -- -DSB_CONFIG_NRF_EDGEAI_REGRESSION_MODEL_SLOT_DUAL=y

MCUboot and DFU
----------------

Both assets follow the normal MCUboot + MCUmgr path: upload a signed image, then test/reset so MCUboot applies the update (swap for dual-slot model, in-place for single-slot).

| Asset | Image index | Upload target | After reboot |
|-------|-------------|---------------|--------------|
| Firmware | 0 | ``slot1`` (secondary) | MCUboot swaps image 0 |
| Model (dual-slot) | 1 | ``slot3`` (secondary) | MCUboot swaps image 1; live model in ``model_storage`` |
| Model (single-slot) | 1 | same ``model_storage`` region | In-place overwrite of ``model_storage`` |

First-time provisioning must flash the **full sysbuild image chain**, not the application ``zephyr.hex`` alone.
A normal ``west flash`` also programs the MCUboot-signed model image to ``model_storage`` (see :file:`sysbuild.cmake`).

.. code-block:: console

   west flash -d build --recover --no-rebuild

Or with ``nrfutil`` using the build-generated merged image that includes bootloader, application, and model:

.. code-block:: console

   nrfutil device program --firmware build/regression_provision.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

If MCUboot reports ``magic=unset`` and ``Unable to find bootable image``, the bootloader or signed application slot was not programmed. Reflash with ``--recover`` (or full chip erase) using either command above, or reflash the merged bootloader/application image and model separately:

.. code-block:: console

   west flash -d build --recover --no-rebuild
   west flash --hex-file build/regression/regression_model_mcuboot.signed.hex --no-rebuild

Do **not** flash the raw ``regression_model_pkg.hex`` alone when MCUboot image 1 is enabled: ``model_storage`` must contain a valid MCUboot header at boot. The build produces ``regression_model_mcuboot.signed.hex`` for provisioning; ``regression_model_pkg.hex`` remains useful for direct payload inspection and matches the bytes inside the signed image body.

Model OTA over SMP (UART)
^^^^^^^^^^^^^^^^^^^^^^^^^

Close any serial monitor on the application UART port, then:

.. code-block:: console

   mcumgr -c acm1 image upload -e -n 1 build/regression/regression_model_mcuboot.signed.bin
   mcumgr -c acm1 image list
   mcumgr -c acm1 image test <model_hash>
   mcumgr -c acm1 reset

**Dual-slot only:** after reset, if the application loads the swapped model successfully, it calls ``boot_write_img_confirmed_multi(1)`` so the update survives the next reboot. If load fails, the image stays unconfirmed and MCUboot reverts on the following reset.

**Single-slot:** there is no separate staging slot or revert. Inference is paused while image 1 is uploaded because SMP writes to the same ``model_storage`` region the model executes from. After upload completes, **reset the device** before validating the new model; the sample does not hot-reload an in-place SMP update.

Firmware-only OTA uses image index 0 (omit ``-n 1``) and ``build/regression/zephyr/zephyr.signed.bin``. Each image can be updated independently.

Model OTA over BLE (Memfault gateway)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The device does not download model images from Memfault by itself in this flow.
A BLE-connected gateway (phone, PC, or custom tool) reads Memfault metadata from the device, fetches the signed model from Memfault, and uploads it with SMP image management (same MCUboot image 1 path as UART above).

Build and flash with the BLE overlay (dual-slot model layout is recommended so a bad model can revert):

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20a/cpuapp samples/nrf_edgeai/regression -d build_ble \
       -- -DEXTRA_CONF_FILE=prj_ble_memfault.conf \
          -DSB_CONFIG_NRF_EDGEAI_REGRESSION_MODEL_SLOT_DUAL=y \
          -DCONFIG_MEMFAULT_NCS_PROJECT_KEY=<your-memfault-project-key>
   west flash -d build_ble --recover --no-rebuild

Set ``CONFIG_MEMFAULT_NCS_DEVICE_ID`` in :file:`prj_ble_memfault.conf` (or pass ``-DCONFIG_MEMFAULT_NCS_DEVICE_ID=...``) so each board has a unique serial in Memfault.

In Memfault, create model releases using software type ``regression-model`` (separate from the application type ``regression-app`` reported by the device).
Upload ``build_ble/regression/regression_model_mcuboot.signed.bin`` as the OTA payload for that software type.
Use a separate Memfault project for model-only releases, or the same project with a distinct software type, depending on how you want to manage cohorts.

Gateway workflow:

#. Connect to the device over Bluetooth LE (advertised as ``EdgeAI Regression``).
#. If you previously connected with an older firmware that required SMP pairing, remove the bond in the phone OS Bluetooth settings or in Device Manager before retrying.
#. Read Memfault MCUmgr group 128, command 0 (device info) and command 1 (project key).
   The gateway needs ``device_serial``, ``hardware_version``, and ``project_key`` to query Memfault.
   Use software type ``regression-model`` (not ``regression-app`` from device info) when calling the Memfault releases API for model binaries.
#. Compare the model version on the device (``mcumgr image list``, image index 1) with the latest Memfault release for ``regression-model``.
#. Download ``regression_model_mcuboot.signed.bin`` from Memfault when an update is available.
#. Upload to the device over BLE SMP, then test and reset:

   .. code-block:: console

      mcumgr --conntype ble --connstring peer_name='EdgeAI Regression' \
          image upload -e -n 1 build_ble/regression/regression_model_mcuboot.signed.bin
      mcumgr --conntype ble --connstring peer_name='EdgeAI Regression' image list
      mcumgr --conntype ble --connstring peer_name='EdgeAI Regression' image test <model_hash>
      mcumgr --conntype ble --connstring peer_name='EdgeAI Regression' reset

   On Linux you may need ``peer_id=<BLE address>`` instead of ``peer_name`` if name-based lookup fails.

#. After reset, confirm the sample loads the new model and (dual-slot only) calls ``boot_write_img_confirmed_multi(1)`` when validation succeeds.

Ready-made gateways that speak SMP over BLE include `nRF Connect Device Manager`_ (application firmware, image 0) and custom tools built on the Memfault MCUmgr command group (`Memfault in nRF Connect SDK`_).
For image 1 (model-only) uploads, use ``mcumgr`` over BLE as shown above or extend your gateway to pass ``-n 1`` / image index 1 to SMP image management.

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
Like Axon (see below), this sample's ``model_v1``-equivalent package is now built automatically as part of a normal build (:kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_NEUTON` + :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA`, both on by default on boards other than ``nrf54lm20b``) - :file:`CMakeLists.txt`'s ``nrf_neuton_model_package()`` call runs :file:`package_model_neuton.py` against :file:`src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c` (this model's own generated source, standing in for a real training run's output) with no separate build or manual packaging step needed:

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/regression

This produces ``build/regression/regression_model_pkg.bin``/``.hex``.

To package a different (for example freshly retrained) model instead, point :file:`package_model_neuton.py` at its own generated source directly:

.. code-block:: console

   python3 tools/model_ota/package_model_neuton.py \
     path/to/nrf_edgeai_user_model.c --name aq_regression --version 1.0.0 -o model_v1 \
     --dts build/regression/zephyr/zephyr.dts

:file:`src/nrf_edgeai_generated/Neuton/regression_v2.json` is a hand-tweaked variant with no corresponding generated source, useful for observing a change in predictions after an update; package it with :file:`package_model.py` instead (see :ref:`lib_model_ota`, "Host-side packaging tools").

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
