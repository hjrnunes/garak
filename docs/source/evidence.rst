evidence
========

The evidence collector records tool calls made during one evaluation. Install the
``midojo`` extra to encode call bodies; the modules remain importable without it.
Records freeze JSON values when they arrive, so later changes to a caller's objects
cannot change evidence. A channel sets provenance, never the caller's body.

Call IDs identify recorded calls. Repeating the same ID and the same body inserts
nothing and returns the first record. A different body under the same ID is rejected.
Comparison is type for type: ``True``, ``1`` and ``1.0`` differ, as do ``0.0`` and
``-0.0``. Object key order does not matter. Values must be JSON-only: exact strings,
integers, booleans, finite floats, null, lists and objects with string keys.

A disposition says what the interception layer did with a call. A replay may leave it
out, because an omitted disposition says nothing and repeats the first record. A
disposition the replay names must be the first record's, so a replay cannot add one to
a record that has none.

A call starts as an unfinished record, and its finish completes that record in place,
keeping the start's position, provenance, timestamp and turn. A start needs a call ID,
because the ID is what pairs it with its finish. A call that starts and
never finishes stays visible: a lost report hides whatever the call did. A replay of a
start compares only the fields a start carries, so a finish's result never conflicts
with the start it completes.

Per-call cost
-------------

Intake cost depends on the size of the call being recorded, not on the number of
calls already held: records are immutable and shared, the ledger appends to a list
and a dictionary index, and no event copies the ledger.

Copying the whole evaluation on every event is the obvious way to isolate state, and
it makes each call slower than the last. A ledger that deep-copied the evaluation three
times per event took about 0.5 ms per whole-call record for the first calls and about
5 ms by call 500, growing in a straight line. This ledger takes about 0.021 ms per call
with a 1 KB result for calls 20 to 70 and about 0.024 ms for calls 470 to 520, measured
on a development laptop.

``tests/evidence/test_ledger_growth.py`` guards the bound three ways, for whole-call
records and for starts and finishes alike:

* Over 500 calls, per-call time and retained memory must stay flat. This catches heavy
  per-record work such as re-exporting or deep-copying every earlier call.
* A copy of 500 references is too quick for a timer, so a second test logs every
  whole-container method (iteration, slicing, copying, ``index``, ``keys``, ``items``)
  that intake, replay, lookup or refusal calls on the ledger's list and indexes, and
  fails on any. The start and completion paths run under the same guard.
* Code that calls a built-in method directly skips those logs, so a third test times
  12,000 calls and fails when the last thousand cost more than 1.5 times the first. A
  start-and-finish run at the same scale fails the same way.

The tests catch regressions made by accident. They cannot stop code written to hide
work from them, which review has to catch.

.. toctree::
   :maxdepth: 1

   evidence/codec
   evidence/records
   evidence/ledger
