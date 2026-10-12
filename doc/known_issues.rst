.. _edge_ai_known_issues:

Known issues
############

.. only:: release or pr

   See `known issues page on the main branch`_ for the complete list of issues for all releases.

.. only:: latest or pr

   .. contents::
      :local:
      :depth: 2


   Known issues listed on this page and tagged with the selected release are valid for that release.
   Use the release filter to view known issues for a specific release.
   A known issue can list one or both of the following entries:

   * **Affected platforms:**

     If a known issue does not have any specific platforms listed, it is valid for all hardware platforms.

   * **Workaround:**

     Some known issues have a workaround.
     Sometimes, they are discovered later and added over time.

   .. known-issues-filter::

   .. rst-class:: wontfix v2-3-0 v2-2-0 v2-1-0 v2-0-0

   Axon NPU asynchronous inference affects ongoing inference
      The asynchronous inference prematurely copies the input vector into the :term:`Axon interlayer buffer` before the inference is enqueued.
      This might overwrite the data of any inference that is already running.
      The affected Axon driver versions are from 0.7.0 to 1.5.0.

      **Workaround:** Migrate to nRF Edge AI release v3.0.0 or later release.
      If you must stay on an affected release, use synchronous inferences, or ensure that no inference will be executed during an :c:func:`nrf_axon_nn_model_infer_async` call.

   .. rst-class:: v2-3-0

   NCSDK-40932: DFU fails on Thingy:53 in the Gesture Recognition application
      When using the Device Firmware Update (DFU) feature in the :ref:`Gesture Recognition application <app_gesture_recognition>` on Thingy:53, the DFU process fails.

      **Affected platforms:** Thingy:53

   .. rst-class:: v2-2-0

   NCSDK-40250: Bootloader Serial Recovery mode is disabled in release configurations on Thingy:53
      In the :ref:`Gesture Recognition application <app_gesture_recognition>`, bootloader Serial Recovery mode is disabled in release configurations.

      **Workaround:** Enable the following Kconfig options in the MCUboot configuration file :file:`configuration/thingy53_nrf5340_cpuapp/images/mcuboot/prj_release.conf`:

      .. code-block:: ini

         CONFIG_MCUBOOT_SERIAL=y
         CONFIG_GPIO=y

      This restores Serial Recovery mode at the cost of increased current consumption.

      **Affected platforms:** Thingy:53

   .. rst-class:: v2-0-0

   DRGN-27788: Bluetooth LE disables RRAM low-latency mode when using AXON NPU and Bluetooth LE simultaneously on the nRF54LM20B SoC
      When running AXON and Bluetooth LE together on an nRF54LM20B device, Bluetooth LE might disable the RRAM low-latency mode during radio activity, which may slow down or corrupt an ongoing inference.
      MPSL sets STANDBY mode in ``NRF_RRAMC->POWER.LOWPOWERCONFIG`` at the start of each radio slot and restores the application init value at the end.
      In a power-optimized application, if the radio slot ends while an inference is running on Axon, the low-power (``NRF_RRAMC_LP_POWER_OFF``) value will be forced by MPSL, slowing down the rest of the inference.

      **Workaround:** Use the ``CONFIG_MPSL_FORCE_RRAM_ON_ALL_THE_TIME`` Kconfig option to keep RRAM permanently in STANDBY mode.
      This setting increases power consumption but ensures reliable performance.

      **Affected platforms:** nRF54LM20B SoC
