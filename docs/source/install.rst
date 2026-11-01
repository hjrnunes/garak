..  headings: = - ^ "

Installation
============

.. _installation:

garak is a command-line tool.
It is developed in Linux, MacOS, and Windows.

Installing with pip
---------------------

.. code-block:: console

   $ python3 -m pip install garak


Installing the Development Version
----------------------------------

The version of garak is updated periodically.
To install from source in GitHub, run the following command:

.. code-block:: console

   $ python3 -m pip install -U git+https://github.com/NVIDIA/garak.git@main


Optional MiDojo Support
-----------------------

garak supports Python 3.10 or later.
The MiDojo injection-spec probe (``probes.midojo.InjectionSpecProbe``) also needs MiDojo, which needs Python 3.12 or later.
To use it, run garak on Python 3.12 or later and install the ``midojo`` package and ``midojo_sdk`` with its ``mcp`` extra.
The probe also needs the injection-spec modules MiDojo's rename introduced: ``midojo.injection_spec``, ``midojo.evaluation_handle`` and ``midojo.injection_spec_document``, whose reader accepts ``midojo-injection-spec-v1`` documents. MiDojo commit ``2a794f3`` or later provides them, including the ``OpenAIResponsesAgentClient`` that takes ``allowed_tools``, and the probe refuses to load with an older MiDojo:

.. code-block:: console

   $ python3 -m pip install "<path to MiDojo>" "<path to MiDojo>/interception/python-sdk[mcp]"

See :doc:`garak.probes.midojo.InjectionSpecProbe <probes/midojo>` for details.
``probes.ipi_episode.IPIEpisode`` is the deprecated alias of the same probe.
