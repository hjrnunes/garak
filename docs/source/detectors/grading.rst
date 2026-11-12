garak.detectors.grading
=======================

Grade a sealed snapshot against the checks of an injection spec document,
without the MiDojo runtime. Install the optional ``midojo`` extra to use it.

``grade_sealed(document, snapshot)`` returns a ``Verdict`` whose
``to_document()`` is the ``midojo-verdict-v1`` document the runtime writes for
the same document and snapshot. The document is a plain mapping, or the
``garak.evidence.injection_spec.InjectionSpec`` that reader made of one; the
snapshot is a ``garak.evidence.seal.Snapshot``, whose digest is verified first.
The reader checks every member of the document, whether or not the grade reads
it.

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

A ``reply_judge`` check asks a judge service whether the agent's reply supports
a claim, so its grade is the service's answer as well as the snapshot's, and
grading the same snapshot again may give another grade. The check names the
service by ``url`` or by the environment variable ``url_env`` that holds it, and
a bearer token only by the variable ``token_env`` that holds it: a literal token
in a document is refused. The service's verdict ``supported`` makes the check
True, ``contradicted`` makes it False, and every other outcome makes it unknown,
never False: no URL or token in the environment, a service that cannot be
reached or times out, a status other than 2xx, or an answer that is not a valid
``midojo.judge.v1`` response. The reason names why, without any text the service
sent. The check asks once for each distinct request in a grading, and the
token goes only into the request's ``Authorization`` header: no record or
reason of the grade holds it, nor does any text the service echoes back. The
one place it can appear is the request itself, which carries the reply it judges
and is recorded as sent (its id is a digest of it): a token the agent's own
reply holds stays in the sealed snapshot and in the request alike, as in the
runtime. An answer in a ``gzip`` or ``deflate`` coding is decoded, and no answer,
coded or decoded, is read past 8 MiB. An answer in any other coding, ``br`` and
``zstd`` among them, is not read, whatever its bytes are: the check is unknown,
the reason says the answer is in a coding the check does not decode, and the
record keeps the status the answer came with.

A grade a judge took part in lists what each check asked and was answered in
``judgements``: the request body, the service's endpoint, the HTTP status, the
verdict, rationale and model, or why the check is unknown. A grade no judge took
part in has no such member. Pass those records back as ``judgements`` with the
``judge_model`` and ``judge_prompt_version`` they were answered under, or a ready
``RecordedJudge``, to grade again offline: each request is answered by the
record with its id and that identity, no service is asked, and a request no
record answers is unknown.

The grader refuses, and writes no verdict, when it cannot grade what it is
given. ``DocumentRefused`` carries the reader's refusal: a member the document
does not define or that is malformed, a check it cannot read (a ``reply_judge``
check with a setting it does not define included), an OpenShell predicate, a
runtime other than ``unmanaged`` and a plan that does not decode.
``EvidenceRefused`` covers a snapshot whose digest no longer matches, an
environment field a check reads that the snapshot lacks, and a malformed
delivery fact. ``GradingRefused`` alone covers ``judgements`` records that are
not those of a grade, such as one whose request is not the request its
``request_id`` names, one whose request the schema does not allow, and one that
answers an id alone: an answer is kept only with the request it answered, which
a request id, printed in the reason of a grade, does not stand for.

The grader's checks are held to byte-identical verdicts on a corpus the
runtime graded, kept under ``tests/_assets/detectors/grading`` with the
script that regenerates it. The ``reply_judge`` check is held to the runtime's
verdicts and requests on a corpus of scripted judge services, kept under
``tests/_assets/detectors/reply_judge`` with its script and the fake service it
runs against. On that corpus the two differ only where the standard library's
HTTP client differs from the runtime's: the name of the exception a failure reports, a URL
the standard library cannot form a request to (nothing is sent), and the user
and password of a URL (dropped, not sent as Basic credentials). They differ in
one more case: an answer in a content coding the grader does not decode is
unknown, where the runtime's client reads a coding it does not know as it is
(and decodes ``br`` and ``zstd`` when their packages are installed, so it can
grade an answer the grader never reads). The judge service of the pipeline
(``judge-serve``) does not compress its answers, so the difference does not
arise with it. The 8 MiB bound on an answer is the grader's alone.

.. automodule:: garak.detectors.grading
   :members:
   :undoc-members:
   :show-inheritance:
