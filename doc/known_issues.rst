.. _edge_ai_known_issues:

Known issues
############

.. contents::
   :local:
   :depth: 2

Known issues listed on this page are valid for the current state of development.
For the list of potential issues found regarding Axon NPU, see the :ref:`axon_npu_changelog` page.
A known issue can list one or both of the following entries:

* **Affected platforms:**

  If a known issue does not have any specific platforms listed, it is valid for all hardware platforms.

* **Workaround:**

  Some known issues have a workaround.
  Sometimes, they are discovered later and added over time.

List of known issues for v2.0.0 release
***************************************

DRGN-27788: Bluetooth LE disables RRAM low-latency mode when using AXON NPU and Bluetooth LE simultaneously on the nRF54LM20B SoC
  When running AXON and Bluetooth LE together on nRF54LM20B, Bluetooth LE might disable RRAM low-latency mode during radio activity, which may slow or corrupt an ongoing inference. 
  MPSL sets STANDBY mode in ``NRF_RRAMC->POWER.LOWPOWERCONFIG`` at the start of each radio slot and restores the application init value at the end.
  In a power-optimized application, if the radio slot ends while an inference is running on Axon, the low-power (``NRF_RRAMC_LP_POWER_OFF``) value will be forced by MPSL, slowing down the rest of the inference.

  **Workaround:** Use ``CONFIG_MPSL_FORCE_RRAM_ON_ALL_THE_TIME`` to keep RRAM permanently in STANDBY mode.
  This setting increases power consumption but ensures reliable performance.
  
  **Affected platforms:** nRF54LM20B SoC
