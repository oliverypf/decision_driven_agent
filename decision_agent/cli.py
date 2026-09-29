"""JSON-in/JSON-out CLI for hook integration."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .controller import DecisionController
from .decision.failure_router import FailureInput
from .decision.metrics import MetricsAggregator


def _load_json(source: str) -> dict[str, Any]:
    if source == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Input JSON must be an object")
    return value


def _dump(value: Any) -> None:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="decision-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("sufficiency", "stop", "failure-route", "tool-risk"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("input", help="JSON file path, or - for stdin")
        command_parser.add_argument(
            "--store-dir",
            default=None,
            help=(
                "Evidence store directory used when input omits evidence; "
                "omit it to use an isolated temporary audit store"
            ),
        )

    development_parser = subparsers.add_parser(
        "development-plan",
        help="JEV selects the next action, files, direction or verification plan",
    )
    development_parser.add_argument("input", help="JSON file path, or - for stdin")
    development_parser.add_argument(
        "--store-dir", default=None, help="Evidence store directory for the decision record"
    )

    verify_parser = subparsers.add_parser(
        "development-verify",
        help="Verify executed files and validation evidence against the selected scope",
    )
    verify_parser.add_argument("input", help="JSON file path, or - for stdin")
    verify_parser.add_argument(
        "--store-dir", default=None, help="Evidence store directory for the verification record"
    )

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.add_argument(
        "path",
        nargs="?",
        default=".decision/evidence",
        help="Evidence file, evidence store directory, or evidence root",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "metrics":
        _dump(MetricsAggregator().aggregate_path(args.path))
        return 0
    payload = _load_json(args.input)
    # Standalone CLI calls must not append to the project-level hook evidence
    # root by default. Callers that need durable, task-scoped evidence can
    # opt in explicitly with --store-dir.
    store_dir = args.store_dir or tempfile.mkdtemp(prefix="decision-agent-cli-")
    controller = DecisionController.from_directory(store_dir)

    if args.command == "development-plan":
        result = controller.plan_development(
            str(payload.get("stage", "")),
            problem=payload.get("problem") or {},
            candidates=payload.get("candidates") or [],
            evidence=payload.get("evidence") or [],
            timeout=payload.get("timeout"),
        )
        _dump(result)
        return 0

    if args.command == "development-verify":
        result = controller.verify_development_execution(
            problem=payload.get("problem") or {},
            selected_files=payload.get("selected_files") or [],
            changed_files=payload.get("changed_files") or [],
            validation_evidence=payload.get("validation_evidence") or [],
        )
        _dump(result)
        return 0

    if args.command == "failure-route":
        _dump(controller.route_failure(FailureInput(**payload)))
        return 0

    if args.command == "tool-risk":
        _dump(
            controller.judge_tool_use(
                tool_name=str(payload.get("tool_name", "")),
                tool_input=payload.get("tool_input"),
                requirement=str(payload.get("requirement", "")),
                domain=payload.get("domain") or payload.get("task_domain"),
            )
        )
        return 0

    evidence = payload.get("evidence")
    if evidence is None:
        evidence = [record.to_dict() for record in controller.evidence()]
    common = {
        "requirement": str(payload.get("requirement", "")),
        "acceptance_criteria": payload.get("acceptance_criteria", []),
        "evidence": evidence,
        "domain": payload.get("domain") or payload.get("task_domain"),
    }
    if args.command == "sufficiency":
        _dump(controller.judge_sufficiency(**common))
        return 0

    _dump(
        controller.judge_stop(
            **common,
            iteration=int(payload.get("iteration", 0)),
            agent_requested_stop=bool(payload.get("agent_requested_stop", True)),
        )
    )
    return 0
