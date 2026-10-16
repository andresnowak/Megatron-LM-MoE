"""Send existing Alps results to a report-only GitHub Actions workflow."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[3]


def gh_json(*arguments):
    completed = subprocess.run(
        ["gh", *arguments], text=True, capture_output=True, check=True, cwd=REPO_ROOT, timeout=60
    )
    return json.loads(completed.stdout)


def tested_sha(directory, override):
    if override:
        sha = override
    elif (directory / "launch.txt").is_file():
        lines = (directory / "launch.txt").read_text().splitlines()
        sha = next(
            (line.removeprefix("Commit: ") for line in lines if line.startswith("Commit: ")), ""
        )
    else:
        raise ValueError("No launch.txt: supply --sha with the recorded tested commit")
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
        raise ValueError("Invalid tested commit SHA")
    return sha


def junit_summary(directory):
    counts = defaultdict(Counter)
    failures = set()
    for path in directory.rglob("*.xml"):
        root = ElementTree.parse(path).getroot()
        match = re.search(r"\.rank-(\d+)\.xml$", path.name)
        rank = match.group(1) if match else "0"
        for suite in root.iter("testsuite"):
            counts[rank]["tests"] += int(suite.get("tests", 0))
            counts[rank]["failed"] += int(suite.get("failures", 0))
            counts[rank]["failed"] += int(suite.get("errors", 0))
            counts[rank]["skipped"] += int(suite.get("skipped", 0))
        for case in root.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failures.add(f'{case.get("classname", "")}::{case.get("name", "")}')
    columns = []
    for field in ("tests", "failed", "skipped"):
        values = [count[field] for count in counts.values()]
        if not values:
            columns.append("—")
        elif min(values) == max(values):
            columns.append(str(values[0]))
        else:
            columns.append(f"{min(values)}–{max(values)}")
    return {
        "tests_per_rank": columns[0],
        "failures_errors_per_rank": columns[1],
        "skipped_per_rank": columns[2],
        "reporting_ranks": len(counts),
        "failing_tests": sorted(failures)[:5],
        "has_failures": bool(failures),
    }


def build_report(directory, results, suite, sha):
    selections = []
    with results.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("Results ledger is empty")
    for row in rows:
        name = row.get("group") or row.get("selection") or row.get("case")
        if not name or "result" not in row:
            raise ValueError("Expected group/selection/case and result columns")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError(f"Unsupported selection name: {name}")
        group_dir = directory / name
        if not group_dir.is_dir():
            group_dir = directory / "run" / suite / name
        ranks = row.get("ranks")
        inner_ledger = group_dir / "results.tsv"
        if not ranks and inner_ledger.is_file():
            with inner_ledger.open() as stream:
                inner_rows = csv.DictReader(stream, delimiter="\t")
                ranks = next(
                    (item.get("ranks") for item in inner_rows if item.get("group") == name), None
                )
        expected_ranks = int(ranks) if ranks else None
        if suite == "functional":
            expected_ranks = 1
        selections.append(
            {
                "name": name,
                "outcome": row["result"],
                "expected_reporting_ranks": expected_ranks,
                "junit": junit_summary(group_dir),
            }
        )
    return {"version": 1, "suite": suite, "sha": sha, "selections": selections}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing the results ledger")
    parser.add_argument("--suite", choices=("unit", "functional"), required=True)
    parser.add_argument(
        "--sha", help="Recorded tested commit; defaults to launch.txt, never current HEAD"
    )
    parser.add_argument(
        "--results", default="results.tsv", help="Ledger filename relative to directory"
    )
    parser.add_argument("--repo", help="OWNER/REPO; defaults to the current GitHub repository")
    parser.add_argument(
        "--ref", help="Branch containing the workflow; defaults to the default branch"
    )
    parser.add_argument("--publish", action="store_true", help="Dispatch the report-only workflow")
    args = parser.parse_args()
    directory = args.directory.resolve()
    sha = tested_sha(directory, args.sha)
    report = build_report(directory, directory / args.results, args.suite, sha)
    if not args.publish:
        print(json.dumps(report, indent=2))
        print("Preview only. No GitHub writes; use --publish to dispatch the reporting workflow.")
        return
    repo_info = gh_json(
        "repo",
        "view",
        *([args.repo] if args.repo else []),
        "--json",
        "nameWithOwner,defaultBranchRef",
    )
    repo = repo_info["nameWithOwner"]
    report["sha"] = gh_json("api", f"repos/{repo}/commits/{sha}")["sha"]
    payload = json.dumps({"report": json.dumps(report, separators=(",", ":"))})
    if len(payload.encode()) > 60000:
        raise ValueError("Report exceeds workflow input size; publish smaller selections")
    subprocess.run(
        [
            "gh",
            "workflow",
            "run",
            "alps-test-results.yml",
            "--repo",
            repo,
            "--ref",
            args.ref or repo_info["defaultBranchRef"]["name"],
            "--json",
        ],
        input=payload,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    print(f"Report submitted: https://github.com/{repo}/actions/workflows/alps-test-results.yml")


if __name__ == "__main__":
    main()
