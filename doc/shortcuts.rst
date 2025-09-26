.. |NCS| replace:: nRF Connect SDK
.. |EAI| replace:: Edge AI add-on
.. |nRFVSC| replace:: nRF Connect for VS Code extension
.. |config| replace:: See :ref:`configure_application` for information about how to permanently or temporarily change the configuration.
.. |connect_terminal| replace:: Connect to the kit with a terminal emulator (for example, the `Serial Terminal app`_).
   See :ref:`test_and_optimize` for the required settings and steps.
.. |sysbuild_autoenabled_ncs| replace:: When building :ref:`repository applications <create_application_types_repository>` in the :ref:`SDK repositories <dm_repo_types>`, building with sysbuild is :ref:`enabled by default <sysbuild_enabled_ncs>`.
   If you work with out-of-tree :ref:`freestanding applications <create_application_types_freestanding>`, you need to manually pass the ``--sysbuild`` parameter to every build command or :ref:`configure west to always use it <sysbuild_enabled_ncs_configuring>`.
.. |thingy53_sample_note| replace:: If you build this application for Thingy:53, it enables additional features. See :ref:`thingy53_app_guide` for details.
.. |54H_engb_2_8| replace:: The nRF54H20 DK Engineering A and B (up to version 0.8.2) are no longer supported starting with |NCS| v2.9.0.
.. |test_sample| replace:: After programming the sample to your development kit, complete the following steps to test it:
.. |connect_kit| replace:: Connect the kit to the computer using a USB cable.
   The kit is assigned a serial port.
   Serial ports are referred to as COM ports on Windows, /dev/ttyACM devices on Linux, and /dev/tty devices on macOS.
   To list Nordic Semiconductor devices connected to your computer together with their serial ports, open a terminal and run the ``nrfutil device list`` command.
   Alternatively, check your operating system's device manager or its equivalent.
