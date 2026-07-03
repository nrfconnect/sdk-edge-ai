.. _setup_sdk:

Setting up the SDK
##################

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with |EAI|.

Get the nRF Edge AI Add-on code
*******************************

The |EAI| is distributed as a Git repository, and is managed through its own west manifest.
The compatible |NCS| version is specified in the :file:`west.yml` file.
To get the |EAI| code, you can either:

* Use the `nRF Connect for Visual Studio Code`_ extension, which provides a convenient way to clone the Add-on and compatible |NCS| version.
* Clone the Add-on repository and initialize west with the local manifest.
* Extend your west manifest with Add-on as a project.
* Clone the Add-on repository and add the Add-on as a Zephyr module through CMake or environment variable.

.. tabs::

   .. group-tab:: nRF Connect for Visual Studio Code

      .. note::

         Use this method when you wish to specifically evaluate Edge AI capabilities, but do not have the |NCS| setup yet.

      Clone the |EAI| code, together with the compatible |NCS|:

      1. Ensure you have installed `Visual Studio Code`_ and the `nRF Connect for Visual Studio Code`_ extension.
      #. Follow the `nRF Connect SDK installation guide`_ to install |NCS| prerequisites and toolchain |toolchain_ncs_id|.

         .. note::

            The compatible version of the |NCS| will be cloned with the |EAI| repository in the following steps.
            The version of |NCS| is fixed to the version of |EAI| and is hard-coded in the :file:`west.yml` file of the |EAI|.

      #. Open the nRF Connect extension in Visual Studio Code by clicking its icon in the :guilabel:`Activity Bar`.
      #. In the extension's :guilabel:`Welcome View`, click on :guilabel:`Create a new application`.
         The list of actions appears in the Visual Studio Code's quick pick.
      #. Click :guilabel:`Browse nRF Connect SDK Add-on Index`.
         The list of available nRF Connect SDK Add-ons appears in the Visual Studio Code's quick pick.
      #. Select :guilabel:`Edge AI Add-on`.
      #. Select the Add-on version to install.
         Depending on the speed of your internet connection, the update might take some time.

   .. group-tab:: Command line for local manifest

      .. note::

         Use this method when you wish to specifically evaluate Edge AI capabilities, but do not have the |NCS| setup yet.

      1. Follow the `nRF Connect SDK installation guide`_ to install |NCS| prerequisites and toolchain |toolchain_ncs_id|.

         .. note::

            The compatible version of the |NCS| will be cloned with the |EAI| repository in the following steps.
            The version of |NCS| is fixed to the version of |EAI| and is hard-coded in the :file:`west.yml` file of the |EAI|.

      #. Launch installed toolchain:

         .. tabs::

            .. group-tab:: Windows

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version |toolchain_ncs_id| --terminal

            .. group-tab:: Linux

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version |toolchain_ncs_id| --shell

            .. group-tab:: MacOS

               .. code-block:: console

                  nrfutil sdk-manager toolchain launch --ncs-version |toolchain_ncs_id| --shell

      #. Initialize the nRF Edge AI repository, using one of the following methods:

         .. tabs::

            .. tab:: Direct initialization (Recommended)

               a. Initialize west with the remote manifest.

                  .. code-block:: console

                     west init --manifest-rev v|release_version| --manifest-url https://github.com/nrfconnect/sdk-edge-ai

            .. tab:: Manual cloning and initialization

               a. Clone the nRF Edge AI repository into the :file:`sdk-edge-ai` directory.

                  .. code-block:: console

                     git clone --branch v|release_version| https://github.com/nrfconnect/sdk-edge-ai.git sdk-edge-ai

               #. Initialize west with local manifest.

                  .. code-block:: console

                     west init --local sdk-edge-ai

      #. Update all repositories, by running the following command:

         .. code-block:: console

            west update

         Depending on the speed of your Internet connection, the update might take some time.

   .. group-tab:: Add-on as a manifest project

      .. note::

         Use this method when running a `Workspace application`_ or if you prefer to use a modification of the |NCS| manifest.

      1. Add Add-on repository as a project in west manifest by including the following lines in your :file:`west.yml` under the ``projects`` key:

         .. code-block:: yaml

            - name: edge-ai
              url: https://github.com/nrfconnect/sdk-edge-ai
              revision: v|release_version|
              import: true

         If you have already included the |NCS| in your west manifest, remove it or replace the above ``import`` key with the mapping:

         .. code-block:: yaml

            import:
               name-blocklist:
               - nrf

         Your west manifest must specify compatible versions of |NCS| and Add-on.

      #. Perform ``west update`` to pull the Add-on repository.

   .. group-tab:: Add-on as an extra Zephyr module

      .. note::

         Use this method if you have a prior installation of the |NCS| and would like to evaluate or use the |EAI| with that installation.
         This method will allow you to keep the |NCS| manifest unmodified.

      Before using this approach, ensure that a compatible version of the |NCS| is installed.
      To identify the compatible |NCS| version, check the :file:`west.yml` file of the |EAI|.
      Since west does not manage the Add-on in this setup, you are responsible for keeping the versions synchronized.

      1. Clone the Add-on repository:

         .. code-block:: console

            git clone --branch v|release_version| https://github.com/nrfconnect/sdk-edge-ai

      #. Set the CMake or environment variable ``EXTRA_ZEPHYR_MODULES`` to the Add-on code path.
         Use absolute path to ensure proper path resolution.
         Check the `Environment Variables`_ documentation for different ways of setting environment variables in Zephyr.

         .. note::

            If you wish to:

            * Use |EI| in Zephyr library deployment.
            * Use |EI| samples from Add-on.

            Repeat the steps above for `edge-impulse-sdk-zephyr`_.
            The ``EXTRA_ZEPHYR_MODULES`` variable should be set to ``<path to Add-on>;<path to Edge Impulse SDK>``.
