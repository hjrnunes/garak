Injectors
=========

An injector arms the interceptor after the evidence collector opens and before
the environment sets up the target. Install ``garak[midojo-mim]`` for the
``mim`` implementation. Start a run, arm each chain, pass its tool block to
your target setup, then disarm each chain before closing the run. Disarm and
close even after a failed chain; a lost binding cannot be released. Never
share a live run or binding with another worker.

.. toctree::
   :maxdepth: 1

   injectors/base
   injectors/mim
