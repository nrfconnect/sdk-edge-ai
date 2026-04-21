.. _app_matter_switch:

Matter Axon-based switch application
####################################

.. contents::
   :local:
   :depth: 2

TBD

Pin mapping
===========

The application was tested with an `Adafruit PDM MEMS Microphone`_ module.
It can be powered from 1.8V ``VDD:IO`` supply.
The following table show how to connect this module to the DK:

.. list-table::
   :header-rows: 1

   * - Adafurit DMIC
     - nRF54LM20 DK
   * - ``3V``
     - ``VDD:IO``
   * - ``GND``
     - ``GND``
   * - ``SEL``
     - ``GND``
   * - ``CLK``
     - ``P1.4``
   * - ``DAT``
     - ``P1.5``

The ``SEL`` pin is responsible for selecting audio channel and connecting it to ground selects left channel.

To use other microphones, adapt configuration parameters of PDM in the :file:`src/dmic.c` file.

User interface
***************

LED1:
   Blinks for one second when the wakeword is detected.

UART30:
   Prints wakeword state messages:

   * ``Waiting for wakeword...``
   * ``Wakeword detected``

Configuration
*************

TBD

Configuration options
=====================

TBD

Building and running
********************

TBD

Testing
=======

TBD

Dependencies
************

This application uses the following |EAI| library:

* :ref:`nrf_edgeai_lib`

This application uses the following Zephyr libraries:

* `Logging`_
* Audio (DMIC)
* UART Driver

API documentation
*****************

.. doxygengroup:: ww_kws
