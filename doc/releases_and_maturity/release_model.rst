.. _release_model:

Release model
#############

.. contents::
   :local:
   :depth: 2

This page describes version identifiers, release artifacts, and compatibility rules for the |EAI|, |NCS|, Axon NPU software, and the |EAILib|.

Add-on release identifiers
**************************

The version string is defined in the :file:`VERSION` file at the repository root.
Git tags use a leading ``v`` prefix, for example ``v|release_version|``.

Use the tagged add-on version together with the |NCS| revision listed in :file:`west.yml`.
Do not mix an add-on tag with a different |NCS| version unless a release note or changelog explicitly states that combination is supported.

Component version streams
*************************

The following table summarizes the version identifiers:

.. list-table:: Version streams
   :header-rows: 1
   :widths: 22 28 50

   * - Component
     - Version source
     - Where to find it
   * - |EAI| add-on
     - ``MAJOR.MINOR.PATCH`` from :file:`VERSION`
     - Git tag ``v|release_version|``, :ref:`release_notes`
   * - |NCS|
     - ``MAJOR.MINOR.PATCH`` revision pinned in :file:`west.yml`
     - :file:`west.yml` project ``nrf`` revision (for example ``|ncs_version|``)
   * - Axon NPU software
     - Independent Axon release, for example ``2.0.1``
     - :ref:`axon_npu_changelog`, driver and compiler binaries in the add-on tree
   * - |EAILib|
     - Independent library release, for example ``3.0.0``
     - :ref:`nrf_edgeai_changelog`, :c:func:`nrf_edgeai_runtime_version`
   * - Nordic Edge AI Lab solutions
     - Solution export version embedded in generated model sources
     - :c:func:`nrf_edgeai_solution_runtime_version`, compatibility notes in :ref:`nrf_edgeai_changelog`

Axon NPU compatibility
**********************

Axon NPU software consists of the on-device driver, the host-side TFLite compiler, and the compiled model artifacts they produce.

Backward compatibility (older models, newer driver)
===================================================

Compiled Axon models built with an older compiler and driver generation generally run on a newer Axon driver in the same major feature set.
When a compiler or driver fix changes numerical behavior, recompile affected models with the compiler version shipped in your target add-on release.

Since Axon release 1.2.0, compiled model headers can declare a minimum supported Axon version.
If a model requires driver features that are not present, build or initialization fails instead of running with undefined behavior.

Forward compatibility (newer models, older driver)
==================================================

Newer compiled models may run on an older Axon driver only when they do not depend on features introduced after that driver version.
When in doubt, compile models with the Axon compiler version bundled in your selected |EAI| tag and run them with the matching driver from the same tag.

|EAILib| compatibility
**********************

The |EAILib| integrates Neuton CPU models and Axon NPU models through a shared runtime API.
Each library release lists the Nordic Edge AI Lab solution versions and Axon driver versions it supports in :ref:`nrf_edgeai_changelog`.

Runtime compatibility between library and model
===============================================

Models exported from Nordic Edge AI Lab embed a solution runtime version.
At startup, applications should verify that the on-device library matches that export using :c:func:`nrf_edgeai_is_runtime_compatible`.
If the check fails, regenerate or redeploy the model for the |EAILib| version in your add-on release instead of forcing a mismatch at runtime.

Backward compatibility (older models, newer library)
======================================================

Within the supported ranges published in :ref:`nrf_edgeai_changelog`, a model generated for an older Nordic Edge AI Lab solution version is expected to work with a newer |EAILib| from a later add-on release.
Patch and minor library releases focus on fixes and additive capability, such as support for additional Axon driver versions.

When release notes or the library changelog call out API or pipeline changes, regenerate model sources from Nordic Edge AI Lab before upgrading the add-on in a product firmware line.

Forward compatibility (newer models, older library)
=====================================================

A model generated for a newer Nordic Edge AI Lab solution version is not guaranteed to run on an older |EAILib|.
The :c:func:`nrf_edgeai_is_runtime_compatible` check is the supported way to detect this condition.

Axon-backed |EAILib| releases also document which Axon driver versions they support.
