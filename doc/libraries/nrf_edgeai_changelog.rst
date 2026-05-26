.. _nrf_edgeai_changelog:

nRF Edge AI Library changelog
#############################

.. contents::
   :local:
   :depth: 2

See the list of changes for a specific release of the |EAILib|.

Release v2.2.1 (09 April 2026)
******************************

This release is tagged as ``NRF-EDGEAI-RELEASE-2.2.1`` (internal release commit ``2e7626e4b9d0bd84c5a07251da0d91927ab83cff``).

* Added support for Axon driver version 1.1.0.

* Fixed:

  * Argument handling in positive and negative sigma crossing rate feature extraction functions (:c:func:`nrf_edgeai_feature_pscr`, :c:func:`nrf_edgeai_feature_nscr`) across all supported integer and float types.
  * Type casting for ``sigma_factor`` and ``lag`` parameters in time-domain DSP feature extraction functions to correctly use the declared ``nrf_dsp_sigma_factor_t`` and ``uint8_t`` types.
  * Refactored DSP clipping functions for Q63-to-Q31 and Q63-to-Q15 fixed-point conversions to ensure correct saturation behavior.

Known issues
============

There are no critical known issues identified for this release.

Compatibility
=============

* Nordic Edge AI Lab solutions version: 2.2.0 - 2.2.1
* Axon driver version: 1.1.0

Release v2.2.0 (06 March 2026)
******************************

This release is tagged as ``NRF-EDGEAI-RELEASE-2.2.0`` (internal release commit ``7a38f672a9e71949ba5f0ff43acbc7e865fb89f5``).

* Added:

  * Axon model support across the nRF Edge AI library, including model type handling and runtime integration alongside existing Neuton support.
  * Inference lifecycle APIs for initialization and deinitialization.
  * Custom-domain feature extraction and improved audio mel processing and wake-word decoding flow.

* Fixed Doxygen warnings and documentation issues.

Known issues
============

There are no critical known issues identified for this release.

Compatibility
=============

* Nordic Edge AI Lab solutions version: 2.2.0
* Axon driver version: 0.7.0 - 1.0.1

Release v1.0.0 (05 December 2025)
*********************************

This release is tagged as ``RELEASE-NRF-EDGEAI-1.0.0`` (internal release commit ``9501cebe17ec8d404298f8657fa3549c3ce5c453``).

* Added initial project release based on Neuton libc project.

Known issues
============

There are no critical known issues identified for this release.

Compatibility
=============

* Nordic Edge AI Lab solutions version: 1.0.0
