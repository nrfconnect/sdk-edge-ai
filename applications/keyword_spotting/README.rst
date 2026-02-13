.. _app_keyword_spotting:

Keyword Spotting
################

.. contents::
   :local:
   :depth: 2

.. The Keyword Spotting application demonstrates using |EAILib| for wakeword-gated keyword spotting.
It utilize Axon-based models for both wakeword detection and keyword spotting.

Application overview
********************

The application is using single channel audio collected via digital microphone to detect spoken keywords using using machine learning models.
To remove false positives from keyword spotting, the application is also using wakeword detection.
Initially application is wakeword detection mode.
When wakeword is detected application switch to keyword spotting for (TODO) 3 seconds.
When keyword is spotted, it is signaled to the user and keyword spotting window is extended to 3 seconds.
After this window application switch back to wakeword detection.

Requirements
************

The application supports the following development kit:

..
   table-from-sample-yaml::

The application also requires PDM digital microphone connected to pins specified in :file:`boards/nrf54lm20dk_nrf54lm20a_cpuapp.overlay`.
The application was tested with `Adafruit PDM MEMS Microphone`_.
Using other microphones may require adapting configuration parameter of PDM in :file:`src/dmic.c`.

User interface
***************

LED0:
   On when the application is in keyword spotting mode.

LED1:
   Blinks when keyword was spotted.

UART30:
   Prints information about switching mode and spotted keywords.

Configuration*
**************

|config|

Some title*
===========

.. note::
   If required, add subsections for additional configuration scenarios that require a different building procedure.

Configuration options*
======================

TODO

|application_kconfig|

..
   options-from-kconfig::
   :show-type:

..
   note::
   * Use the following syntax to list all the sample-specific configuration options in the :file:`Kconfig` file.

   .. code-block::

      .. options-from-kconfig::
         :show-type:

   Make sure all other configuration options are listed in the section at the bottom of the page.
   * The syntax allows application configuration options to link to the option descriptions in the same way as the library configuration options link to the corresponding Kconfig descriptions (``:kconfig:option:`CONFIG_APPLICATION```, which results in :kconfig:option:`CONFIG_APPLICATION`).
   * For the |nRFVSC| instructions, list the configuration options as they are stated on the **Generate Configuration** screen.
   * Use ``:option:`SAMPLE_CONFIG``` to link to sample specific option.
   * The tech writer team needs to review the :file:`Kconfig` file, where the sample-specific Kconfig options are defined.

Additional configuration*
=========================

..
   note::
   * Add this section to describe and link to any library configuration options that might be important to run this application.
     You can link to options with ``:kconfig:option:`CONFIG_FOO```.
   * You need not list all possible configuration options, but only the ones that are relevant.

   Check and configure the following library options that are used by the application:

   * :kconfig:option:`CONFIG_PDN_DEFAULT_APN` - Used for manual configuration of the Access Point Name (APN).
   * :kconfig:option:`CONFIG_MODEM_ANTENNA_GNSS_EXTERNAL` - Selects an external GNSS antenna.

Configuration files*
====================

.. note::
   Add this section if the application provides predefined configuration files.

The application provides predefined configuration files for typical use cases.
You can find the configuration files in the :file:`XXX` directory.

The following files are available:

* :file:`filename.conf` - Specific scenario.
  This configuration file ...

Building and running
********************

..
   note::
   * Include the standard text for building - either ``.. include:: /includes/application_build_and_run.txt`` or ``.. include:: /includes/application_build_and_run_ns.txt`` for the board targets that use :ref:`Cortex-M Security Extensions <app_boards_spe_nspe>`.
   * The main supported IDE for |NCS| is |VSC|, with the |nRFVSC| installed.
     Therefore, build instructions for the |nRFVSC| are required.
     Build instructions for the command line are optional.
   * See the link to the `nRF Connect for Visual Studio Code`_ documentation site for basic instructions on building with the extension.
   * If the application uses a non-standard setup, point it out and link to more information, if possible.

.. |application path| replace:: :file:`applications/XXX`

..
   include:: /includes/application_build_and_run.txt

Some title*
===========

.. note::
   If required, add subsections for additional build instructions.
   Use these subsections sparingly and only if the content does not fit into other sections (mainly `Configuration*`_).
   If the additional build instructions are valid for other applications as well, consider adding them to the :ref:`configuration_and_build` section instead and linking to them.

Testing
=======

|test_application|

#. |connect_kit|
#. |connect_terminal_kit| The application is using both serial ports.
#. On the terminal connected to second serial port you can observe the application logs.
#. Say (TODO) "OK Nordic" followed by one of the keywords.
#. On the terminal connected to first serial port the application shows similar output:

.. code-block:: console

   TODO
   Waiting for wakeword
   Wakeword detected
   Waiting for keywords
   Keyword detected: Up, 97%
   Keyword detection timeout
   Waiting for wakeword

Application output*
===================

.. note::
   Add the full output of the application in this section or include parts of the output in the testing steps in the previous section.

The application shows the following output:

.. code-block:: console

   [00:00:02.029,174] <inf> zigbee_app_utils: Zigbee stack initialized

References*
***********

.. note::
   Provide a link to other relevant documentation for the user to get more information.

.. tip::
   Do not include links to documents that are common to all or many of our applications. For example, :ref:`installation` section.

Related projects and applications*
==================================

.. note::
   Add links to projects and/or applications that demonstrate or implement some or all of the features of this application.
   For example, the :ref:`matter_weather_station_app` application is part of the `Matter`_ project.

Related samples*
================
.. note::
   Add links to the samples that show aspects of the application.
   A sample showcases one feature or component.
   An application uses several features or components.
   Include links to samples that showcase features included in the application.

Dependencies*
*************

.. note::
   * List all relevant dependencies, for example, libraries, other tool or service (third-party) references, certification requirements (if applicable).
   * Standard libraries (for example, :file:`types.h`, :file:`errno.h`, or :file:`printk.h`) need not be listed.
   * Delete the parts that are not relevant.
   * Drivers can be listed under libraries.
   * If possible, link to the respective dependency.
     If there is no documentation for the dependency, include the path.
   * Add the appropriate secure firmware component that the application supports.

This application uses the following |NCS| libraries:

* :ref:`app_event_manager`
* :ref:`lib_aws_iot`

It uses the following `sdk-nrfxlib`_ library:

* :ref:`nrfxlib:nrf_modem`

It uses the following Zephyr libraries:

* :ref:`zephyr:logging_api`
* :ref:`zephyr:kernel_api`:

  * :file:`include/kernel.h`

In addition, it uses the following secure firmware component:

* :ref:`Trusted Firmware-M <ug_tfm>`

The application also uses drivers from `nrfx`_.

Internal modules*
*****************

.. note::
   Add this section if there are internal modules that must be documented.
   If there are complex modules that cannot fit on one page, add them on separate pages.

API documentation*
******************

.. note::
   Add the following section if the application uses API documentation.
   Add subsections if the application uses different components with their own APIs.

.. code-block::

   | Header file: :file:`*provide_the_path*`
   | Source files: :file:`*provide_the_path*`

   .. doxygengroup:: *doxygen_group_name*
