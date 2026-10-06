.. _releases_and_maturity:

Releases and maturity
#####################

The |EAI| is released as tagged versions of its Git repository and west manifest.
Each release bundles libraries, tools, samples, and documentation at fixed component revisions and pins a compatible |NCS| revision in :file:`west.yml`.

The versioning scheme follows `Semantic versioning`_.
Every tagged release is identified with a version string in ``MAJOR.MINOR.PATCH`` format.
Axon NPU software and the |EAILib| use separate version streams inside each add-on release.
For versioning details and compatibility rules, see :ref:`release_model`.

Each release is documented in :ref:`release_notes`, with component-level changes in :ref:`axon_npu_changelog` and :ref:`nrf_edgeai_changelog`.
:ref:`software_maturity` summarizes which features are supported or experimental in the current release.
If an issue is found after a release, it is listed on :ref:`known_issues`.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Subpages:

   releases_and_maturity/release_model
   releases_and_maturity/release_notes
   releases_and_maturity/software_maturity
   releases_and_maturity/known_issues
