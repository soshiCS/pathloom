"""Command-line entrypoint.

    python3 -m src.cua discover --goal ... --url ... --name ... --param NAME=VALUE   # LLM discovery -> artifact
    python3 -m src.cua replay --artifact artifacts/<name>.vN.json --param NAME=VALUE  # deterministic replay
    python3 -m src.cua discover-campaign --spec scenarios/<name>.json           # one discovery per declared
                                                                               # scenario -> one graph artifact
    python3 -m src.cua stability --artifact ... --runs 3 --param NAME=VALUE    # N fresh unattended replays -> report
    python3 -m src.cua approve --artifact ... --report <report.json> --reviewer NAME  # draft -> approved version

Replay runs a schema 1.0 artifact as a linear flow and a schema 2.0 capability graph as a graph.
Lifecycle: draft -> supervised replay or stability runs -> approve -> approved -> unattended replay.
An unattended replay (--operator none) of a draft stops with artifact_not_approved before any
browser exists; approval is a local review record backed by stability reports, not a signature.

The production flow is: discovery -> artifact -> deterministic replay -> structured result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from urllib.parse import urlparse

from . import artifact as artifact_module
from . import graph as graph_module
from .agent import DEFAULT_MAX_VISION_ATTEMPTS, DiscoveryFailed, discover
from .approval import ApprovalError, approve
from .campaign import CampaignError, CampaignFailed, load_spec, run_campaign
from .escalation import ConsoleOperator, Escalator, NoOperator, SessionControl
from .evidence import RunLog
from .graph_replay import DEFAULT_IRREVERSIBLE_POLICY, IRREVERSIBLE_POLICIES
from .lifecycle import (SUPERVISED, UNATTENDED, is_approved, load_any_version, not_approved_result, policy_for,
                        replay_any, write_bundle)
from .planner import (DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL, ClaudePlanner,
                      OpenAIPlanner)
from .policy import Policy
from .stability import run_stability
from .surface import PlaywrightSurface


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "discover":
        return cmd_discover(args)
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "discover-campaign":
        return cmd_discover_campaign(args)
    if args.command == "stability":
        return cmd_stability(args)
    if args.command == "approve":
        return cmd_approve(args)
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m src.cua", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")

    disc = commands.add_parser("discover", help="LLM-guided discovery; writes a capability artifact")
    disc.add_argument("--goal", required=True, help="natural-language goal")
    disc.add_argument("--url", required=True, help="entry URL of the target site")
    disc.add_argument("--name", required=True, help="capability name (snake_case)")
    disc.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                      help="LLM provider used during discovery (default: anthropic)")
    disc.add_argument("--model", default=None,
                      help="provider model id (defaults to ANTHROPIC_MODEL or OPENAI_MODEL)")
    disc.add_argument("--max-steps", type=int, default=15)
    disc.add_argument("--outcome", action="append", default=[], metavar="CODE=TEXT",
                      help="extra business outcome identified by on-screen text, e.g. invalid_credentials='do not match'")
    disc.add_argument("--missing-outcome", action="append", default=[], metavar="CODE=TEXT",
                      help="extra business outcome identified by absent text, e.g. product_not_found='{{product_name}}'")
    add_vision_options(disc)
    add_shared_run_options(disc)

    camp = commands.add_parser("discover-campaign",
                               help="LLM-guided discovery of every scenario declared in a JSON spec, merged into "
                                    "one capability graph")
    camp.add_argument("--spec", required=True,
                      help="path to the campaign JSON (name, goal, url, selectors, scenarios)")
    camp.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                      help="LLM provider used during discovery (default: anthropic)")
    camp.add_argument("--model", default=None,
                      help="provider model id (defaults to ANTHROPIC_MODEL or OPENAI_MODEL)")
    camp.add_argument("--max-steps", type=int, default=15)
    add_vision_options(camp)
    add_shared_run_options(camp, params=False)

    rep = commands.add_parser("replay", help="deterministic replay of a saved artifact (no LLM)")
    rep.add_argument("--artifact", required=True, help="path to artifacts/<name>.vN.json")
    rep.add_argument("--irreversible-policy", choices=IRREVERSIBLE_POLICIES, default=DEFAULT_IRREVERSIBLE_POLICY,
                     help="schema 2.0 graphs: what to do at an action whose effect is irreversible "
                          "(default: confirm with a human; unknown effects always need a human)")
    add_shared_run_options(rep)

    stab = commands.add_parser("stability", help="replay one invocation N times, unattended and on a fresh session "
                                                 "each time, and write a report that approval can use")
    stab.add_argument("--artifact", required=True, help="path to artifacts/<name>.vN.json (either schema)")
    stab.add_argument("--runs", type=int, default=3, help="number of replays (default: 3)")
    add_shared_run_options(stab, operator=False)

    appr = commands.add_parser("approve", help="turn a draft into the next approved version, given eligible "
                                               "stability reports (a local review record, not a signature)")
    appr.add_argument("--artifact", required=True, help="path to the draft artifact")
    appr.add_argument("--report", action="append", required=True, metavar="REPORT_JSON",
                      help="stability report; repeat once per selector assignment for a campaign graph")
    appr.add_argument("--reviewer", required=True, help="who reviewed the evidence")
    return parser


def add_vision_options(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--vision-fallback", action="store_true",
                     help="opt in to the bounded screenshot fallback when the planner is stuck (discovery only; "
                          "masked viewport screenshots are sent to the provider and cost money)")
    sub.add_argument("--max-vision-attempts", type=int, default=DEFAULT_MAX_VISION_ATTEMPTS,
                     help=f"total visual attempts per discovery run (default: {DEFAULT_MAX_VISION_ATTEMPTS})")


def add_shared_run_options(sub: argparse.ArgumentParser, params: bool = True, operator: bool = True) -> None:
    if params:   # a campaign takes its parameters and sensitivity from the spec
        sub.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="input parameter")
        sub.add_argument("--sensitive", action="append", default=[], metavar="NAME",
                         help="mark a parameter as sensitive (never shown to the model or written to disk)")
    sub.add_argument("--allow-host", action="append", default=[], help="allowlisted host (default: the entry host)")
    if operator:  # stability runs are unattended by definition
        sub.add_argument("--operator", choices=["console", "none"], default="console",
                         help="console = a human can take over via the terminal; none = unattended (approved "
                              "artifacts only)")
    sub.add_argument("--headed", action="store_true", help="show the browser window")
    sub.add_argument("--quiet", action="store_true", help="do not echo log events to stderr")


# ---------- commands ----------

def cmd_discover(args: argparse.Namespace) -> int:
    params = parse_params(args.param)
    sensitive = set(args.sensitive)
    secrets = tuple(str(params[name]) for name in sensitive if name in params)
    log = RunLog("discovery", secrets=secrets, echo=not args.quiet)
    policy = Policy(allowed_hosts=args.allow_host or [urlparse(args.url).hostname or ""])
    extra_outcomes = [{"code": code, "kind": "business", "source": "reviewer", "detect": {"text_contains": text}}
                      for code, text in (item.split("=", 1) for item in args.outcome)]
    extra_outcomes += [{"code": code, "kind": "business", "source": "reviewer", "detect": {"text_missing": text}}
                       for code, text in (item.split("=", 1) for item in args.missing_outcome)]

    surface = PlaywrightSurface(headless=not args.headed, secrets=secrets)
    escalator = Escalator(make_operator(args), SessionControl(), log)
    try:
        planner = make_planner(args)
        built = discover(goal=args.goal, name=args.name, params=params, surface=surface,
                         planner=planner, policy=policy, escalator=escalator, log=log,
                         entry_url=args.url, sensitive=sensitive, max_steps=args.max_steps,
                         extra_outcomes=extra_outcomes, vision=planner if args.vision_fallback else None,
                         max_vision_attempts=args.max_vision_attempts)
        path = artifact_module.save(built, secrets)
        log.event("artifact_saved", path=str(path))
        evidence_dir = log.copy_to_evidence()
        shutil.copy2(path, evidence_dir / path.name)
        print(f"\nDiscovery succeeded. Artifact: {path}\nEvidence: {evidence_dir}")
        return 0
    except DiscoveryFailed as error:
        log.event("discovery_failed", error=str(error))
        log.screenshot(surface, "failed")
        evidence_dir = log.copy_to_evidence()
        print(f"\nDiscovery failed: {error}\nEvidence: {evidence_dir}")
        return 2
    finally:
        surface.close()


def cmd_discover_campaign(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
    except CampaignError as error:
        print(f"Cannot load campaign spec: {error}")
        return 2
    allowed_hosts = args.allow_host or [urlparse(spec.url).hostname or ""]
    try:
        result = run_campaign(
            spec, surface_factory=lambda secrets: PlaywrightSurface(headless=not args.headed, secrets=secrets),
            planner_factory=lambda scenario: make_planner(args), operator=make_operator(args),
            allowed_hosts=allowed_hosts, max_steps=args.max_steps, echo=not args.quiet, spec_path=args.spec,
            vision_fallback=args.vision_fallback, max_vision_attempts=args.max_vision_attempts)
    except CampaignFailed as error:
        print(f"\nCampaign failed: {error}\nNo artifact was saved. Summary: {error.summary_path}")
        return 2
    print(f"\nCampaign succeeded: {len(result.scenarios)} scenario(s) merged into {result.artifact_path}"
          f"\nSummary: {result.summary_path}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    params = parse_params(args.param)
    secrets = tuple(str(params[name]) for name in args.sensitive if name in params)
    log = RunLog("replay", secrets=secrets, echo=not args.quiet)
    try:
        loaded = load_any_version(args.artifact)
    except artifact_module.ArtifactError as error:
        print(f"Cannot load artifact: {error}")
        return 2
    if not is_approved(loaded):
        if args.operator == "none":
            # The gate: no browser, no action; still a structured result and an evidence bundle.
            result = not_approved_result(loaded)
            log.event("replay_blocked", capability=loaded.name, version=loaded.version, status=loaded.status,
                      outcome_code=result.outcome_code, reason=result.observed)
            print(f"Refusing unattended replay: {result.observed}")
            return finish_replay(args, log, loaded, result, params)
        warning = (f"WARNING: {loaded.name} v{loaded.version} has status {loaded.status!r}; this is a supervised "
                   f"test run. Run `stability` and `approve` before replaying it unattended.")
        log.event("draft_warning", capability=loaded.name, version=loaded.version, status=loaded.status)
        print(warning)

    surface = PlaywrightSurface(headless=not args.headed, secrets=secrets)
    escalator = Escalator(make_operator(args), SessionControl(), log)
    try:
        result = replay_any(loaded, params, surface, policy_for(loaded, args.allow_host), escalator, log,
                            purpose=UNATTENDED if args.operator == "none" else SUPERVISED,
                            irreversible_policy=args.irreversible_policy)
    finally:
        surface.close()
    return finish_replay(args, log, loaded, result, params)


def finish_replay(args: argparse.Namespace, log: RunLog, loaded, result, params: dict) -> int:
    evidence_dir = write_bundle(log, loaded, args.artifact, result, params, args.sensitive)
    print("\nReplay result:")
    print(json.dumps(asdict(result), indent=2))
    print(f"Evidence: {evidence_dir}")
    return 0 if result.status in ("success", "business_outcome") else 2


def cmd_stability(args: argparse.Namespace) -> int:
    params = parse_params(args.param)
    try:
        report_path = run_stability(
            args.artifact, params, list(args.sensitive), args.runs,
            surface_factory=lambda secrets: PlaywrightSurface(headless=not args.headed, secrets=secrets),
            allowed_hosts=args.allow_host or None, echo=not args.quiet)
    except artifact_module.ArtifactError as error:
        print(f"Cannot load artifact: {error}")
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"\nStability: {report['runs_completed']}/{report['runs_requested']} runs completed, "
          f"success rate {report['success_rate']:.0%}, clean-run rate {report['clean_run_rate']:.0%}, "
          f"recoveries {report['total_recoveries']}, interventions {report['total_interventions']}, "
          f"drift signals {report['total_drift_signals']}")
    print(f"Eligible for approval: {report['eligible_for_approval']}"
          + (f" ({'; '.join(report['ineligible_reasons'])})" if report["ineligible_reasons"] else ""))
    print(f"Report: {report_path}")
    return 0 if report["eligible_for_approval"] else 2


def cmd_approve(args: argparse.Namespace) -> int:
    try:
        path = approve(args.artifact, args.report, args.reviewer)
    except (ApprovalError, artifact_module.ArtifactError) as error:
        print(f"Approval refused: {error}")
        return 2
    print(f"Approved: {path} (from {args.artifact}, reviewed by {args.reviewer})")
    return 0


# ---------- helpers ----------

def parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--param expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        params[name.strip()] = value
    return params


def make_operator(args: argparse.Namespace):
    return ConsoleOperator() if args.operator == "console" else NoOperator()


def make_planner(args: argparse.Namespace):
    """Construct only the provider selected for this discovery run."""
    if args.provider == "openai":
        return OpenAIPlanner(args.model or DEFAULT_OPENAI_MODEL)
    return ClaudePlanner(args.model or DEFAULT_ANTHROPIC_MODEL)


if __name__ == "__main__":
    sys.exit(main())
