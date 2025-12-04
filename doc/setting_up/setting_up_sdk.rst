.. _nrf_edgeai_setting_up_sdk:

Setting up the SDK
##################

.. contents::
   :local:
   :depth: 2

Once you have installed the |NCS| environment, complete the following steps.

Get the nRF Edge AI Add-on code
*******************************

The nRF Edge AI Add-on is distributed as a Git repository, and is managed through its own west manifest.
The compatible nRF Connect SDK version is specified in the :file:`west.yml` file.

.. tabs::

   .. group-tab:: nRF Connect for Visual Studio Code

      Clone the nRF Edge AI Add-on code, together with the compatible |NCS|:

      1. Open the nRF Connect extension in Visual Studio Code by clicking its icon in the :guilabel:`Activity Bar`.
      #. In the extension's :guilabel:`Welcome View`, click on :guilabel:`Create a new application`.
         The list of actions appears in the Visual Studio Code's quick pick.
      #. Click :guilabel:`Browse nRF Connect SDK Add-on Index`.
         The list of available nRF Connect SDK Add-ons appears in the Visual Studio Code's quick pick.
      #. Select :guilabel:`nRF Edge AI Add-on`.
      #. Select the Add-on version to install.

      The Add-on and compatible |NCS| installation starts and it can take several minutes.

   .. group-tab:: Command line

      1. Initialize the nRF Edge AI repository, using one of the following methods:

         .. tabs::

            .. tab:: Direct initialization (Recommended)

               a. Initialize west with the remote manifest.

                  .. code-block:: console

                     west init -m https://github.com/nrfconnect/sdk-edge-ai

            .. tab:: Manual cloning and initialization

               a. Clone the nRF Edge AI repository into the :file:`sdk-edge-ai` directory.

                  .. code-block:: console

                     git clone https://github.com/nrfconnect/sdk-edge-ai.git sdk-edge-ai

               #. Initialize west with local manifest.

                  .. code-block:: console

                     west init -l sdk-edge-ai

      #. Update all repositories, by running the following command:

         .. code-block:: console

            west update

         Depending on your connection, the update might take some time.
