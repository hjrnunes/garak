Run coordinator
===============

.. automodule:: garak.coordinator
   :members:

The harness owns one ``RunCoordinator`` and enters its run scope before
opening chains. A chain starts its evidence collector, arms the injector, and
opens the environment in that order. It retains the same environment for all
turns. Call ``seal(agent_output)`` to release resources and fix the evidence
snapshot. Leaving a chain without sealing still releases its resources.
Pass ``incomplete=`` a tuple of evidence names (``function_calls``, ``turns``,
``agent_output``) to mark evidence you know a failure cut short; the names join
the ones the coordinator marks, and a repeat call returns the first snapshot and
ignores a different ``agent_output`` or ``incomplete``.
Entering the same chain again is refused. ``max_live`` caps open chains;
opening a second chain requires the environment to declare target isolation.
An unmanaged target cannot run concurrent chains even when the cap exceeds one.

``turn_timeout`` applies **per turn**, not to the whole chain. A watchdog
closes a timed-out chain even if the caller remains blocked. To bound a whole
chain, schedule ``chain.cancel("deadline")`` with a caller-owned timer.
``drain_timeout`` bounds unfinished call admission after closure.
``close_timeout`` bounds each blocking release and observation operation;
``start_timeout``, ``bind_timeout``, and ``setup_timeout`` bound run acquisition,
binding, and environment setup. The setup bound defaults to 60 seconds because
target setup hooks may perform slow external work. A cancellation during chain
entry waits no longer than the remaining start, bind, and setup bounds plus
bounded cleanup; a scope returned after setup times out is closed once.
The defaults are 60 seconds per turn, 2 seconds for drain, 10 seconds for
each close, and 15 seconds each for start and bind. A stalled external
operation reports a timeout after its bound. A late release is best-effort
on a daemon thread; its failure is logged by type name only and never holds
the timed-out caller. An external close that never returns cannot be forcibly
stopped. The injector's own bind and release timeouts also apply independently.

If closure fails, every later closer receives the same failure. When the
collector has reached its collecting stage, ``seal`` stores a snapshot with
``function_calls`` and ``environment`` marked incomplete, then raises the
closure failure; it never returns the snapshot as clean. The conservative
markers reflect an unconfirmed release or an incomplete environment teardown.
If closure did not reach collecting, sealing remains unavailable and every
attempt raises the stored failure. When a body already raised, its original
exception propagates; a note naming only cleanup failure types is best-effort
and may occur once per exit path. ``chain.result["cleanup_failures"]`` is the
authoritative type-name record even if the original exception refuses notes.

The coordinator stays in the parent process. Its run, chain, and turn scopes
refuse pickling. The returned tool block contains a private credential;
do not print or persist it. The run keeps only weak references to closed
chains for run-level failure reporting.
