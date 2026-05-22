.. _nrf_edgeai_obsv_memfault_mds:

nRF Edge AI: Observability over Memfault MDS
########################################

.. contents::
   :local:
   :depth: 2

The nRF Edge AI Observability over Memfault MDS sample demonstrates how to push
model-inference observability to a Memfault project using the
:ref:`mds_readme` (Memfault Diagnostic Service) BLE GATT service as the
transport.

The sample uses **synthetic** 4-class probability data - no real model is
loaded. It exercises the full on-device observability pipeline so you can verify
Memfault ingestion end to end before wiring in a real inference loop.

Overview
********

Every second the sample generates a 4-element probability vector where one
"dominant" class carries roughly 0.70 of the mass. The dominant class rotates
through ``0 -> 1 -> 2 -> 3`` every 30 s (configurable via Kconfig). These
vectors are fed into the Edge AI observability library, which maintains a
probability-distribution histogram and a class-to-class transition matrix.

The :kconfig:option:`CONFIG_NRF_EDGEAI_OBSV_MEMFAULT` glue refreshes a
serialized snapshot every 60 s (via
:kconfig:option:`CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC`)
and stages it as a Memfault Custom Data Recording (CDR). The Memfault
packetizer drains the CDR when a phone or gateway subscribes to MDS.

Data flow::

   synth -> obsv_update() -> [library counters]
                                      |
                   (every 60 s)       v
                          collect() -> CDR snapshot
                                      |
                     phone connects + subscribes to MDS
                                      |
                                      v
                       Memfault packetizer streams chunks -> Memfault cloud

Requirements
************

The sample is currently built for nRF54LM20DK:

* ``nrf54lm20dk/nrf54lm20a/cpuapp``
* ``nrf54lm20dk/nrf54lm20b/cpuapp``

You also need:

* A Memfault account and project (`Memfault quickstart`_).
* The **Memfault** mobile app or a Memfault-aware gateway to drain data over
  BLE.

User interface
**************

There is no user interaction - the sample advertises, runs the synthetic
generator, and periodically snapshots the observability. Progress is visible via
UART log output and on the MDS phone app connection status.

Configuration
*************

Before building, set your Memfault project key in :file:`prj.conf` (or via
a ``-D`` option):

.. code-block:: kconfig

   CONFIG_MEMFAULT_NCS_PROJECT_KEY="<your-memfault-project-key>"

Optional sample knobs (see :file:`Kconfig`):

* :kconfig:option:`CONFIG_NRF_EDGEAI_OBSV_SAMPLE_INFERENCE_PERIOD_MS` - how
  often to feed a synthetic probability vector (default 1000 ms).
* :kconfig:option:`CONFIG_NRF_EDGEAI_OBSV_SAMPLE_ROTATION_PERIOD_SEC` - how
  often the dominant class rotates (default 30 s).
* :kconfig:option:`CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC` -
  how often the observability library stages a fresh CDR payload (default 60 s
  in this sample).

Building and running
********************

Build for the nRF54LM20DK:

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20a/cpuapp samples/nrf_edgeai/obsv_memfault_mds
   west flash

Open a serial terminal at 115200-8-N-1. You should see:

.. code-block:: console

   Starting nRF Edge AI Observability over Memfault MDS sample
   Advertising successfully started
   Bluetooth initialized

Connect with the **Memfault** mobile app (or any MDS-capable gateway):

1. Scan for the device advertising as ``nRF Edge AI Observability``.
2. Pair and bond (MDS access requires an encrypted link).
3. Subscribe to the MDS characteristic.
4. The packetizer streams any pending CDR chunks up to the Memfault cloud.

Within the Memfault UI the payload shows up as a Custom Data Recording with
reason ``edgeai_obsv`` and MIME type ``application/octet-stream``.
Decode it on the server side using the observability packet format described in
:file:`sdk-edge-ai/include/nrf_edgeai_obsv/nrf_edgeai_obsv.h`.

Testing
*******

This sample is intended for manual verification against a real Memfault
project. The in-tree CI variant is build-only and simply validates that all
dependencies compile together on the supported nRF54LM20DK targets.

Dependencies
************

This sample uses the following |NCS| components:

* :ref:`mds_readme`
* ``memfault-firmware-sdk`` (CDR source API)
* ``sdk-edge-ai`` observability library (:file:`lib/nrf_edgeai_obsv/`)
* ``sdk-edge-ai`` Memfault CDR glue
  (:file:`lib/nrf_edgeai_obsv_memfault/`)

.. _Memfault quickstart: https://docs.memfault.com/docs/mcu/introduction
