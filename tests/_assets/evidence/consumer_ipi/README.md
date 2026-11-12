# Consumer injected-agent fixtures

Byte copies of six of the consumer's Garak-route fixtures, which a MiDojo run sealed
and graded: `delivered`, `undelivered`, `incomplete`, `blocked`, `sequential` and
`reply-judged`. The consumer's `tampered` case is left out because it edits a snapshot
after the seal. `MANIFEST.json` names the source commit and the SHA-256 of each file.

Each case holds:

| File | What it holds |
| --- | --- |
| `ipi-episodes.json` | the compiled source entry; its first entry's `injection_spec` is the document that was sealed and graded |
| `snapshot.json` | the snapshot the runtime served for the seal, with its `snapshot_id` |
| `verdict.json` | the `midojo-verdict-v1` document the runtime graded it with |

`tests/evidence/test_trace.py` reads each document and snapshot through
`garak.evidence.trace` and compares the two values it states with these files byte for
byte. The `reply-judged` verdict is the oracle for the reply judge: the document is
refused until the reply judge is part of the grader, and the test names that.

Refresh the files from the consumer, never by hand, and update `MANIFEST.json` with them.
