#!/usr/bin/env python3

"""
analyze a garak report.jsonl log file

print out summary stats, and which prompts led to failures

usage:

./analyze_log.py <report.jsonl filename>

"""

import sys
import json

import garak
import garak.resources.metrics
from garak.exception import ReportIncompatibleError


def _read_pairs(record: dict) -> dict:
    """The pair counts of a utility_eval record, with their rates computed again"""
    try:
        return garak.resources.metrics.pool_pairs(None, record["pairs"])
    except ValueError as e:
        raise ReportIncompatibleError(
            "Report utility_eval entry has malformed pairs; "
            "the report was likely generated with a different garak version"
        ) from e


def analyze_log(report_path: str) -> None:
    started_attempt_uuids = set([])
    completed_attempt_uuids = set([])

    current_probe_classname = None
    utility = garak.resources.metrics.UtilityScope(report_path)

    with open(report_path, "r", encoding="utf-8") as reportfile:
        for _, line in enumerate(reportfile):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if not isinstance(record, dict):
                continue

            if "entry_type" not in record.keys():
                continue

            utility.observe(record)

            if record["entry_type"] in ("config", "init"):
                print("##", line)
                continue

            elif record["entry_type"] == "attempt":
                if record["status"] == 1:  # started
                    started_attempt_uuids.add(record["uuid"])
                    continue

                if record["status"] == 2:
                    completed_attempt_uuids.add(record["uuid"])

                if record["probe_classname"] != current_probe_classname:
                    print("## PROBE:", record["probe_classname"])
                    current_probe_classname = record["probe_classname"]

                # for each detector:
                for detector_name, detector_scores in record[
                    "detector_results"
                ].items():
                    if utility.covers(current_probe_classname, detector_name):
                        continue  # a completed task is not a hit
                    normalised_scores = [
                        1 if i >= 0.5 else 0 for i in detector_scores if i is not None
                    ]
                    # are there any hits?
                    if sum(normalised_scores):
                        # if so print prompt & hit rate
                        # i guess make this sortable so give probe, detector, hit rate, prompt
                        hit_rate = sum(normalised_scores) / len(normalised_scores)
                        print(
                            "\t".join(
                                [
                                    current_probe_classname,
                                    detector_name,
                                    f"{hit_rate:0.2%}",
                                    repr(record["prompt"]),
                                ]
                            )
                        )
            elif record["entry_type"] == "eval":
                print(
                    "\t".join(
                        map(
                            str,
                            [
                                record["probe"],
                                record["detector"],
                                "%0.4f"
                                % (
                                    record["passed"] / record["total_evaluated"]
                                    if record["total_evaluated"]
                                    else 0
                                ),
                                record["total_processed"],
                            ],
                        )
                    )
                )
            elif record["entry_type"] == "utility_eval":
                pairs = _read_pairs(record) if "pairs" in record else None
                scope = (
                    f" (utility under attack: {garak.resources.metrics.PLAIN_COVERS})"
                    if pairs
                    else ""
                )
                print(
                    f"## utility, not an attack: {record['probe']} {record['detector']} "
                    f"completed {record['completed']}/{record['total_evaluated']}{scope}"
                )
                if pairs:
                    drop = (
                        f"{pairs['drop']:.4f} over"
                        if pairs["valid"]
                        else "no rate with"
                    )
                    print(
                        f"## paired drop: {garak.resources.metrics.PAIRS_COVERS}: "
                        f"{drop} {pairs['valid']}/{pairs['pairs']} "
                        f"valid pairs, {pairs['invalid']} invalid"
                    )

    if not started_attempt_uuids:
        print("## no attempts in log")
    else:
        completion_rate = len(completed_attempt_uuids) / len(started_attempt_uuids)
        print("##", len(started_attempt_uuids), "attempts started")
        print("##", len(completed_attempt_uuids), "attempts completed")
        print(f"## attempt completion rate {completion_rate:.0%}")


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    import argparse

    garak._config.load_config()
    print(
        f"garak {garak.__description__} v{garak._config.version} ( https://github.com/NVIDIA/garak )"
    )

    parser = argparse.ArgumentParser(
        prog="python -m garak.analyze.analyze_log",
        description="Analyze a garak JSONL report and emit summary lines",
        epilog="See https://github.com/NVIDIA/garak",
        allow_abbrev=False,
    )
    # Support both positional and -r/--report_path for backward compatibility
    parser.add_argument("report_path", nargs="?", help="Path to the garak JSONL report")
    parser.add_argument(
        "-r",
        "--report_path",
        dest="report_path_opt",
        help="Path to the garak JSONL report",
    )
    args = parser.parse_args(argv)
    report_path = args.report_path_opt or args.report_path
    if not report_path:
        parser.error("a report path is required (positional or -r/--report_path)")

    sys.stdout.reconfigure(encoding="utf-8")
    analyze_log(report_path)


if __name__ == "__main__":
    main()
