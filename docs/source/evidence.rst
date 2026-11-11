evidence
========

The evidence collector records tool calls made during one evaluation. Install the
``midojo-spec`` extra to encode call bodies; the modules remain importable without it.
Records freeze JSON values when they arrive, so later changes to a caller's objects
cannot change evidence. A channel sets provenance, never the caller's body.

Call IDs identify recorded calls. Repeating the same ID and the same body inserts
nothing and returns the first record. A different body under the same ID is rejected.
Comparison is type for type: ``True``, ``1`` and ``1.0`` differ, as do ``0.0`` and
``-0.0``. Object key order does not matter. Values must be JSON-only: exact strings,
integers, booleans, finite floats, null, lists and objects with string keys.

Per-call cost
-------------

Intake cost depends on the size of the call being recorded, not on the number of
calls already held: records are immutable and shared, the ledger appends to a list
and a dictionary index, and no event copies the ledger.

.. toctree::
   :maxdepth: 1

   evidence/codec
   evidence/records
