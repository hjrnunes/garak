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
The MiDojo episode generator (``generators.midojo``) also needs MiDojo, which needs Python 3.12 or later.
To use it, run garak on Python 3.12 or later and install the ``midojo`` package and ``midojo_sdk`` with its ``mcp`` extra.
The generator needs a MiDojo whose ``OpenAIResponsesAgentClient`` takes ``allowed_tools`` (MiDojo commit b9319ac or later), and refuses to load with an older one:

.. code-block:: console

   $ python3 -m pip install "<path to MiDojo>" "<path to MiDojo>/interception/python-sdk[mcp]"

See :doc:`garak.generators.midojo <generators/midojo>` for details.
