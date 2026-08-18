.. _nrf_edgeai_changelog:

nRF Edge AI Library changelog
#############################

.. contents::
   :local:
   :depth: 2

See the list of changes for a specific release of the |EAILib|.

Release v3.0.0 (17 August 2026)
*******************************

This release is tagged as ``NRF-EDGEAI-RELEASE-3.0.0`` (internal release commit ``6dfb365bca9400f463b8c16773887b6d6fa39550``).

This is a major release.
The runtime major version was raised from 2 to 3, which makes it incompatible with solutions generated for the 2.x runtime.

* Breaking changes:

  * Runtime version raised to 3.0.0. :c:func:`nrf_edgeai_is_runtime_compatible` compares the major version of the runtime library against the solution, so a model generated for a 2.x runtime now causes :c:func:`nrf_edgeai_init` to fail with ``NRF_EDGEAI_ERR_INCOMPATIBLE``. Regenerate the solution with Nordic Edge AI Lab 3.0.0.
  * Feature processing was split into two separate pipeline stages: DSP feature extraction and feature scaling. The runtime interface structure ``nrf_edgeai_interfaces_t`` gained a new mandatory ``scale_features`` member (``nrf_edgeai_iface_scale_features_t``), which every solution context must populate.
  * The combined ``nrf_edgeai_process_features_<mode>_<input>_<output>`` interface family was removed and replaced:

    * ``process_features`` now provides DSP extraction only — ``nrf_edgeai_process_features_dsp_i8()``, ``nrf_edgeai_process_features_dsp_i16()``, ``nrf_edgeai_process_features_dsp_f32()``, and ``nrf_edgeai_process_features_empty()``.
    * Scaling moved to the new ``nrf_edgeai_scale_features_*`` family declared in ``nrf_edgeai_scale_features.h``.
    * Scaling mode prefixes were renamed: ``scale_vector_*`` is now ``input_vector_*``, and ``scale_window_*`` is now ``input_window_*``.
  * The ``nrf_edgeai_t`` runtime context gained a ``state`` member, changing the structure layout. Applications and generated solutions must be recompiled against the new headers.

* Added:

  * The public :c:func:`nrf_edgeai_process_features` API, allowing DSP feature extraction to be triggered independently of model inference.
  * Runtime state tracking via ``nrf_edgeai_state_t``, with state bit definitions (``NRF_EDGEAI_STATE_RT_INITIALIZED``, ``NRF_EDGEAI_STATE_RT_INPUTS_COLLECTED``, ``NRF_EDGEAI_STATE_RT_FEATURES_PROCESSED``, ``NRF_EDGEAI_STATE_RT_INFERENCE_COMPLETED``) and composite readiness masks. The runtime now validates its state on entry to the input, feature processing, and inference stages.
  * Two error codes for state violations: ``NRF_EDGEAI_ERR_UNINITIALIZED`` (-8) and ``NRF_EDGEAI_ERR_WRONG_STATE`` (-9).
  * :c:func:`nrf_edgeai_dsp_features_ctx` to obtain read-only access to the DSP feature extraction context, so computed features can be used from the application.

* :c:func:`nrf_edgeai_run_inference` keeps the previous single-call workflow working. If features have not been processed yet, it invokes :c:func:`nrf_edgeai_process_features` internally and propagates its status code unchanged.


* Fixed:

  * :c:func:`nrf_edgeai_feed_inputs` now rejects calls on a context that was never successfully initialized, returning ``NRF_EDGEAI_ERR_UNINITIALIZED`` instead of forwarding the data to an uninitialized input window context.

Known issues
============

There are no critical known issues identified for this release.

Compatibility
=============

* Nordic Edge AI Lab solutions version: 3.0.0
* Axon driver version: 1.3.0 - 1.5.0

Release v2.2.2 (03 July 2026)
******************************

This release is tagged as ``NRF-EDGEAI-RELEASE-2.2.2`` (internal release commit ``d33025f8dcb96859626bdabc671b108983f1c0d8``).

* Added support for Axon driver version 1.3.0.

* Fixed:

  * Compatibility of the library with PicoLib in the |NCS|. ``_impure_ptr`` references are now removed from the audio microfrontend utils.

Known issues
============

There are no critical known issues identified for this release.

Compatibility
=============

* Nordic Edge AI Lab solutions version: 2.2.0 - 2.2.2
* Axon driver version: 1.1.0 - 1.3.0

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
