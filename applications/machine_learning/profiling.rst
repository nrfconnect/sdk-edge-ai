.. _nrf_machine_learning_app_profiling:

nRF Machine Learning: Profiling
###############################

The nRF Machine Learning application provides tools for comparing the resource usage by different models, frameworks and acceleration.
The current configurations support measuring of the following factors:

* Power consumption
* RAM and NVM usage
* Max heap usage
* Max stack usage of the model inference thread

Configurations
**************

To receive data about power consumption, RAM and NVM usage, build the application using the :file:`prj_profile_*.conf` configuration.
It disables the functionalities that highly affect power consumption like UART serial communication.
Connect the `Power Profiler Kit II (PPK2)`_ to your computer and the device.
Flash the device and run sampling from the `Power Profiler app`_ for required time to get sensible average of current consumption.
RAM and NVM usage can be acquired from build logs.

To get more information about power consumption, especially during model inference, use the logic ports feature of the PPK2.
Connect two pins specified in the ``inference-position-gpios`` phandle under the ``/zephyr,user`` devicetree node to any logic ports of PPK2.
These two pins toggle their output in response to the ``sensor_event`` and ``ml_result_event`` respectively.
The Power Profiler app monitors these changes as digital channels.
You can use these digital channels to identify the model inference period.
Mark this period in Power Profiler app to get current consumption value during the inference and rough, external estimate of the inference time.

.. figure:: /applications/images/ml_app_profiling.png
   :alt: screenshot of Power Profiler app showing inference identified by digital channels

   Using the Power Profiler app to get average current consumption during model inference.
   Digital channel `0` changes on every ``sensor_event`` and digital channel `1` changes on every ``ml_results_event``.
   These channels allow to position where the inference starts and ends.

To receive data about max usage of heap and inference thread stack, build the application using the :file:`prj_profile_*.conf` configuration and :file:`memory_profile.conf` as extra configuration.
This extra configuration re-enables logging over UART and heap and stack profiling options.
Flash the device and connect to it using a terminal emulator.
Observe the logged information about heap usage:

.. code-block:: shell

   max system bytes =        520
   system bytes     =        520
   in use bytes     =        520
   zephyr system heap: allocated 48, free 3964, max allocated 220, heap size 4096

The first three lines correspond to heap area in free RAM used by libc.
This area is used by the common ``malloc``, for example by Edge Impulse.
The fourth line correspond to Zephyr system heap area.
This area is is used by Zephyr's ``k_malloc``.
Its size is configurable with a Kconfig option and taken into account in the RAM usage report of the build log.

To acquire max stack usage of the inference thread, use the `Thread Viewer of nRF Connect for Visual Studio Code <nRF Connect for Visual Studio Code: Thread Viewer_>`_.

The wildcard ``*`` in :file:`prj_profile_*.conf` represents one of the available combinations of model, framework and compute acceleration.
These configuration files are present in board-specific directory inside :file:`applications/machine_learning/configuration` directory.

Additional configurations
*************************

An additional :file:`enable_bt.conf` configuration file is provided as many business applications use a Bluetooth connection.
This allows to compare applications using different models in context of using the Bluetooth stack.
This configuration enables the Nordic Status Message Service on Bluetooth Low Energy.

Stub configuration
******************

The :file:`prj_profile_stub.conf` configuration file is provided as a baseline for comparing different combinations of the model, framework, and compute acceleration.
It uses the stub model runner.
