.. _setup_sdk:

Setting up the SDK
##################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with |EAI|.

Get the nRF Edge AI Add-on code
*******************************

The nRF Edge AI Add-on is distributed as a Git repository, and is managed through its own west manifest.
The compatible nRF Connect SDK version is specified in the :file:`west.yml` file.
To get the nRF Edge AI Add-on code, you can either:

* Use the `nRF Connect for Visual Studio Code`_ extension, which provides a convenient way to clone the Add-on and compatible |NCS| version.
* Clone the Add-on repository and initialize west with the local manifest.

.. tabs::

   .. group-tab:: nRF Connect for Visual Studio Code

      Clone the nRF Edge AI Add-on code, together with the compatible |NCS|:

      1. Ensure you have installed `Visual Studio Code`_ and the `nRF Connect for Visual Studio Code`_ extension.
      #. Follow the `nRF Connect SDK installation guide`_ to install |NCS| prerequisites and toolchain v3.2.0.

         .. note::

            The compatible version of the |NCS| will be cloned with the |EAI| repository in the following steps.
            The version of |NCS| is fixed to the version of |EAI| and is hard-coded in the :file:`west.yml` file of the |EAI|.

      #. Open the nRF Connect extension in Visual Studio Code by clicking its icon in the :guilabel:`Activity Bar`.
      #. In the extension's :guilabel:`Welcome View`, click on :guilabel:`Create a new application`.
         The list of actions appears in the Visual Studio Code's quick pick.
      #. Click :guilabel:`Browse nRF Connect SDK Add-on Index`.
         The list of available nRF Connect SDK Add-ons appears in the Visual Studio Code's quick pick.
      #. Select :guilabel:`nRF Edge AI Add-on`.
      #. Select the Add-on version to install.
         Depending on the speed of your internet connection, the update might take some time.

   .. group-tab:: Command line

      1. Follow the `nRF Connect SDK installation guide`_ to install |NCS| prerequisites and toolchain v3.2.0.

         .. note::

            The compatible version of the |NCS| will be cloned with the |EAI| repository in the following steps.
            The version of |NCS| is fixed to the version of |EAI| and is hard-coded in the :file:`west.yml` file of the |EAI|.

      #. Launch installed toolchain:

         .. tabs::

            .. group-tab:: Windows

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version v3.2.0 --terminal

            .. group-tab:: Linux

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version v3.2.0 --shell

            .. group-tab:: MacOS

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version v3.2.0 --shell

      #. Initialize the nRF Edge AI repository, using one of the following methods:

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

         Depending on the speed of your Internet connection, the update might take some time.
