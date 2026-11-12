Environments
============

An environment owns chain isolation and observations, not target setup or
tool bindings. The consumer supplies bindings and runs any target-configured
hooks. A reset, if supported, precedes setup; nothing resets afterwards.
Use ``check_attempt(repeated=True)`` before a repeated attempt.

.. toctree::
   :maxdepth: 1

   environments/base
   environments/disposable
   environments/unmanaged
