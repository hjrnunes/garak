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

Recording is at most once, not exactly once. A call ID makes a retry safe, because a
repeat never adds a second record. It does not recover a record that never reached the
collector. The sections below describe how the evidence marks a loss it can see; a
loss it cannot see leaves no trace.

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

Late facts and capture failures
-------------------------------

A write a stage refuses when it has moved on is kept as a late fact, once per distinct
write, with the evidence keys it marks incomplete and its payload as frozen JSON. A
capture failure records the other loss: a call record the interception layer could not
deliver, identified by its call, function and phase alone, so a resent report replays
whatever error it carries.

Capture links
-------------

A capture link names the seam a channel records through. Its state is read from the
binding itself: the beat clock, the loss flag and the peer's answer. A lost or stalled
link is a loss the records cannot show; a released link is clean. A link's first end
stands, so a release that follows a loss does not undo it. Links are
diagnostics, never evidence: a lost or stalled one marks the call capture incomplete,
and the collector's summary of them never enters a digest.

The collector
-------------

A :class:`~garak.evidence.collector.Collector` stages one evaluation's evidence:
``open`` admits every write, ``closing`` admits no new call but lets calls already
started finish until none is unfinished or the closure deadline passes, ``collecting``
takes the final observations, and ``sealed`` fixes the immutable snapshot. Each event
takes its arrival time from an injectable monotonic clock, so an event arriving past
the deadline finds the collector collecting even when nobody drained it. A closure wait
must be finite and not negative, and a link name opens once, so that a second link
cannot hide the loss of the first.

The disposition rule on completion is narrower than a symmetric wildcard: the start's
and the finish's dispositions disagree only when both are named and differ; a start
that named none lets the finish name one. Late facts never change a sealed snapshot or
its digest; they appear in the collector's overlay status, and the ones kept before
the seal are folded into the snapshot's evidence status.

The seal reads every capture link's state from the binding itself, never waiting for a
dispatcher to notice a loss, so a stalled dispatcher with an expired binding cannot
hold the seal open or let it claim a complete capture. An interception layer in the
agent may fail open, so an agent-mode evaluation sealed with no record at all has an
incomplete capture, and so does any call such a layer started whose finish never
arrived; a host-started call does not mark the capture, because the host sends the
seal and the finish alike.

The sealed snapshot
-------------------

The seal fixes an immutable snapshot named by the shared digest of its content: no
caller chooses the ID, and equal content has an equal one. The exported JSON is fresh
and detached on every read, so mutating it, or the objects the snapshot was built from,
changes neither the digest nor a later read. A snapshot may be pickled and copied,
because it is a pure frozen value; an unfinished call seals with ``complete`` false and
``result`` null, and no result is ever invented for it.

Each sealed call record has ``pre_environment`` and ``post_environment`` fields, and
both are null: nothing in v1 produces an environment for a single call. The
snapshot's own ``pre_environment`` and ``environment`` are the only observation of the
world.

Delivery facts and exposure
---------------------------

A sealed call keeps the delivery facts its interceptor reported: for each
injection instruction, whether its selector matched, how the payload was
transformed and which parts of the result carry it. :mod:`garak.evidence.delivery`
flattens the facts and derives, from them and the plan, an exposure per
instruction and per boundary (``emitted``, ``agent_tool_result`` and
``model_input``), each ``exposed``, ``not_exposed`` or ``unknown``. Nothing past
the interceptor is observed, so an emitted payload's ``agent_tool_result`` is
unknown and every payload's ``model_input`` is. A missing fact is a gap, never
an absence: only a complete capture whose every call to the tool ran to its end
shows a payload was not emitted. Delivery and exposure are report only and never
change a grade.

The injection spec document and the trace
-----------------------------------------

An injection spec document says what to run: the turns, the plan that places
payloads, the security and utility checks, the environment, the runtime, the
interception settings and the tool policy.
:class:`garak.evidence.injection_spec.InjectionSpec` reads a
``midojo-injection-spec-v1`` document the way the MiDojo runtime does, with the
runtime's refusals in the same order, and holds it as one read-only value. It
checks every member, including the ones a grade never reads, so a driver never
runs a half-checked document. A version other than v1 raises
``UnsupportedDocumentVersion``; a check that needs an OpenShell observation on
an ``unmanaged`` runtime carries the gap
``check:openshell-predicate-on-unmanaged-runtime``; and a document that declares
the ``openshell`` runtime is refused last, as ``UnsupportedRuntime``, after every
other member passed.

The reader holds no environment model, so it differs from the runtime in six
ways. It reads an environment the runtime's model inference refuses (a key such
as ``model_config``). It refuses a value outside JSON (bytes, a date) that the
runtime keeps; it holds NaN and infinity, as the runtime does. It checks only
the ``type`` of an ``openshell`` runtime. It reads checks with the strict reader
the grade uses, which refuses a few malformed checks the runtime accepts
(``output_contains: 1``). It refuses a check on which the runtime fails with a
bare ``KeyError`` (an ``env_field_equals`` without a ``value``) as a document
error. And a ``reply_judge`` check is refused until the reply judge is read.
Where both refuse, the class and the ``gap`` agree and the wording of a check's
refusal may not. An OpenShell predicate is checked as the runtime's constructors
check it, so a malformed one has no ``gap`` and a well-formed one has the
OpenShell gap.

:mod:`garak.evidence.trace` states the two values a consumer reads from a trace:
the sealed snapshot as the runtime serves it, and the verdict. Serialised with
sorted keys they are the runtime's bytes. A snapshot sealed with a null
``pre_environment`` or ``environment`` digests differently from one sealed with
``{}``, and the runtime never serves it, so the trace refuses it: seal an
``unmanaged`` runtime's evaluation with ``pre_environment={}``.

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
   evidence/facts
   evidence/links
   evidence/intake
   evidence/seal
   evidence/delivery
   evidence/injection_spec
   evidence/trace
   evidence/sealing
   evidence/world
   evidence/collector
