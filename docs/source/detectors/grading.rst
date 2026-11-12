garak.detectors.grading
=======================

Grade a sealed snapshot against the checks of an injection spec document,
without the MiDojo runtime. Install the optional ``midojo`` extra to use it.

``grade_sealed(document, snapshot)`` returns a ``Verdict`` whose
``to_document()`` is the ``midojo-verdict-v1`` document the runtime writes for
the same document and snapshot. The document is a plain mapping with its
``security`` and ``utility`` checks, its ``plan`` and its ``runtime``; the
snapshot is a ``garak.evidence.seal.Snapshot``, whose digest is verified first.

A grade is True, False or None. None means the evidence could not decide the
check: the grade's validity is then ``invalid`` and lists the unknown leaf
checks, and the grade is never a decided result. A check that is not authored
grades False and ``not_applicable``. The combinators ``all_of``, ``any_of`` and
``not`` follow Kleene's three-valued logic and assess every branch. Evidence
the snapshot marks incomplete leaves a check that needs a call's absence
unknown, so an incomplete capture never shows that an attack failed.

Beside ``security`` the verdict reports whether the agent ``attempted`` what the
security check looks for, whether a refusal ``blocked`` the attempt, whether the
effect was observed on the calls that ran to their end, and whether the plan's
payload was delivered. Each is True, False or None.

The grader refuses, and writes no verdict, when it cannot grade what it is
given. ``DocumentRefused`` covers a check it cannot read, an OpenShell
predicate, a runtime other than ``unmanaged``, a ``reply_judge`` check and a plan
that does not decode. ``EvidenceRefused`` covers a snapshot whose digest no
longer matches, an environment field a check reads that the snapshot lacks, and
a malformed delivery fact.

The grader's checks are held to byte-identical verdicts on a corpus the
runtime graded, kept under ``tests/_assets/detectors/grading`` with the
script that regenerates it.

.. automodule:: garak.detectors.grading
   :members:
   :undoc-members:
   :show-inheritance:
