# Replay probe assets

These files back `tests/probes/test_probe_injection_golden.py`.

- `scn-002.attempt.jsonl` is one attempt line, as the run of the earlier route wrote it: the final attempt (status 2) of a single-turn conversation graded by a native `RubricJudge`, with three tool calls made by the target. It is one line of a report file, kept as written apart from two replaced values: the address of the tool server (`http://192.0.2.10:9000/sse`, a documentation address) and the model name (`test-model`). The test replays the target's reply and the judge's reply from the line to a scripted endpoint, runs the probe and the detector, and requires the attempt the run writes to equal this line (except `uuid`, which differs on every run).
- `cases.json` holds the other cases: the conversation source entries, the target's replies in order, and the judge's reply. `replay_endpoint.py` answers them, as the test does in process, and serves them over HTTP. In the two failed cases the endpoint answers the first request with a 503. The attempt keeps the error note (type, status code and body) and has no verdict, because the probe removes the tool-call condition and the judge rubric from it. The sequential case also ends its conversation there.

To serve a case over HTTP, for a run of the command line in another process:

```sh
cd tests/_assets/probes/injection
python replay_endpoint.py cases.json sequential-command 9000
```

Then point the target and the judge at `http://127.0.0.1:9000/v1/` with `max_retries: 0`.
