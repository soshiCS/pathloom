"""Build, validate, version, save, and load capability artifacts.

The artifact is the contract between discovery and replay. It contains the recorded
flow (steps + locator ladders), the typed inputs a caller must supply, the typed
outputs it gets back, the declared non-success outcomes, and the success checkpoint.
It never contains the model transcript, model reasoning, or any input value.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import Artifact, Locator, Step
from .policy import redact

SCHEMA_VERSION = "1.0"
ARTIFACTS_DIR = Path("artifacts")
STEP_ACTIONS = {"navigate", "click", "type", "extract"}
OUTCOME_KINDS = {"business", "recoverable"}
PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
PLACEHOLDER_TOKEN = re.compile(r"(\{\{\w+\}\})")   # split-friendly: keeps the placeholders as segments


class ArtifactError(ValueError):
    """The artifact is malformed, unsupported, or unsafe to run."""


# ---------- parameters ----------

def parameterize(value: str | None, params: dict) -> str | None:
    """Replace discovery-time literals with {{name}} placeholders.

    Only whole tokens are replaced ("User" never matches inside "standard_user"), longer
    values go first so one value cannot clobber a longer one containing it, and values
    shorter than two characters are skipped: they would match almost anything. Text that
    is already a placeholder is left alone, so a value that happens to equal another
    input's name (lookup_method = "member_id" next to {{member_id}}) cannot corrupt it.
    """
    if value is None:
        return None
    segments = PLACEHOLDER_TOKEN.split(value)          # [text, {{placeholder}}, text, ...]
    for index in range(0, len(segments), 2):
        segments[index] = parameterize_text(segments[index], params)
    return "".join(segments)


def parameterize_text(text: str, params: dict) -> str:
    for name, literal in sorted(params.items(), key=lambda item: -len(str(item[1]))):
        literal = str(literal)
        if len(literal) < 2:
            continue
        bounded = r"(?<!\w)" + re.escape(literal) + r"(?!\w)"
        text = re.sub(bounded, "{{" + name + "}}", text)
    return text


LOCATOR_TEXT_FIELDS = ("name", "text", "context")


def parameterize_locator(locator: Locator, params: dict) -> Locator:
    """Locators can depend on inputs too: "the Add to cart button in the {{product_name}} card".

    When a rung depends on an input, the rungs that do not (structural path, coordinates)
    are dropped: they point at whatever happened to be there during discovery, and falling
    back to them would silently act on the wrong item for a different input.
    """
    strategies = []
    for strategy in locator.strategies:
        strategies.append({key: (parameterize(value, params) if key in LOCATOR_TEXT_FIELDS else value)
                           for key, value in strategy.items()})
    parameterized = [s for s in strategies if locator_placeholders(s)]
    return Locator(strategies=parameterized or strategies)


def locator_placeholders(strategy: dict) -> set[str]:
    names: set[str] = set()
    for key in LOCATOR_TEXT_FIELDS:
        names |= placeholders_in(str(strategy.get(key, "")))
    return names


def substitute_locator(locator: Locator, params: dict) -> Locator:
    strategies = []
    for strategy in locator.strategies:
        strategies.append({key: (substitute(value, params) if key in LOCATOR_TEXT_FIELDS else value)
                           for key, value in strategy.items()})
    return Locator(strategies=strategies)


def substitute(value: str | None, params: dict) -> str | None:
    """Fill {{name}} placeholders with concrete values (replay direction)."""
    if value is None:
        return None
    return PLACEHOLDER.sub(lambda match: str(params[match.group(1)]), value)


def placeholders_in(value: str | None) -> set[str]:
    return set(PLACEHOLDER.findall(value or ""))


# ---------- build ----------

def next_version(name: str) -> int:
    """Artifacts are immutable once saved; a re-recording gets the next version number."""
    existing = [int(m.group(1)) for path in ARTIFACTS_DIR.glob(f"{name}.v*.json")
                if (m := re.search(r"\.v(\d+)\.json$", path.name))]
    return max(existing, default=0) + 1


def build(
    name: str,
    goal: str,
    surface_meta: dict,
    params: dict,
    steps: list[Step],
    outputs: dict,
    outcomes: list[dict],
    success: dict,
    run_id: str,
    sensitive: set[str] | None = None,
    planner_name: str = "",
) -> Artifact:
    """Build a typed, versioned artifact from a successful discovery run."""
    sensitive = sensitive or set()
    inputs = {
        param: {"type": "string", "required": True, "sensitive": param in sensitive,
                "description": f"Value typed where '{param}' was used during discovery"}
        for param in params
    }
    artifact = Artifact(
        schema_version=SCHEMA_VERSION,
        name=name,
        version=next_version(name),
        status="draft",
        description=goal,
        surface=surface_meta,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        outcomes=outcomes,
        success=success,
        provenance={
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "planner": planner_name,
            "step_count": len(steps),
        },
    )
    validate(artifact)
    return artifact


# ---------- validate ----------

def validate(artifact: Artifact) -> None:
    """Reject artifacts replay cannot run safely. Raises ArtifactError with a precise reason."""
    if artifact.schema_version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {artifact.schema_version!r}, expected {SCHEMA_VERSION!r}")
    validate_contract(artifact)
    if not artifact.steps:
        raise ArtifactError("artifact has no steps")
    declared_inputs = set(artifact.inputs)
    seen_ids: set[str] = set()
    for step in artifact.steps:
        validate_step(step, seen_ids, declared_inputs, artifact.outputs)


def validate_contract(artifact) -> None:
    """The checks every schema version shares: identity, surface, inputs/outputs/outcomes, success.

    Takes anything with the version-1 metadata fields (Artifact or a capability graph).
    """
    if not artifact.name or not re.fullmatch(r"[a-z0-9_]+", artifact.name):
        raise ArtifactError("name must be a non-empty snake_case identifier")
    if artifact.status not in ("draft", "approved"):
        raise ArtifactError(f"unknown status {artifact.status!r}")
    if not artifact.surface.get("entry_url"):
        raise ArtifactError("surface.entry_url is required")
    if not artifact.success:
        raise ArtifactError("success checkpoint is required")
    validate_placeholders(artifact.success, set(artifact.inputs), "success checkpoint")
    for outcome in artifact.outcomes:
        _validate_outcome(outcome)
    for name, spec in artifact.outputs.items():
        if "type" not in spec:
            raise ArtifactError(f"output {name!r} has no type")


def validate_step(step: Step, seen_ids: set[str], declared_inputs: set[str], outputs: dict) -> None:
    if not step.id or step.id in seen_ids:
        raise ArtifactError(f"step id {step.id!r} is missing or duplicated")
    seen_ids.add(step.id)
    if step.risk not in ("safe", "risky"):
        raise ArtifactError(f"step {step.id}: risk must be safe or risky")
    validate_action(step, declared_inputs, outputs, f"step {step.id}")


def validate_action(step, declared_inputs: set[str], outputs: dict, where: str) -> None:
    """The rules for what an action does: takes a Step or a graph node's action payload."""
    if step.action not in STEP_ACTIONS:
        raise ArtifactError(f"{where}: unknown action {step.action!r}")
    if step.action == "navigate" and not step.value:
        raise ArtifactError(f"{where}: navigate needs a url value")
    if step.action in ("click", "type", "extract") and not (step.target and step.target.strategies):
        raise ArtifactError(f"{where}: {step.action} needs a target locator")
    if step.action == "type" and step.value is None:
        raise ArtifactError(f"{where}: type needs a value")
    if step.action == "extract" and step.value not in outputs:
        raise ArtifactError(f"{where}: extract writes undeclared output {step.value!r}")
    unknown = placeholders_in(step.value) - declared_inputs
    if unknown:
        raise ArtifactError(f"{where}: placeholders {sorted(unknown)} are not declared inputs")
    if step.target:
        for strategy in step.target.strategies:
            for key in LOCATOR_TEXT_FIELDS:
                unknown = placeholders_in(str(strategy.get(key, ""))) - declared_inputs
                if unknown:
                    raise ArtifactError(f"{where}: locator placeholders {sorted(unknown)} are not declared inputs")
    if step.checkpoint:
        validate_placeholders(step.checkpoint, declared_inputs, f"{where} checkpoint")


def validate_placeholders(checkpoint: dict, declared_inputs: set[str], where: str) -> None:
    for value in checkpoint.values():
        unknown = placeholders_in(str(value)) - declared_inputs
        if unknown:
            raise ArtifactError(f"{where}: placeholders {sorted(unknown)} are not declared inputs")


def _validate_outcome(outcome: dict) -> None:
    if not outcome.get("code") or outcome.get("kind") not in OUTCOME_KINDS:
        raise ArtifactError(f"outcome {outcome!r} needs a code and a kind in {sorted(OUTCOME_KINDS)}")
    detect = outcome.get("detect") or {}
    if not any(detect.get(key) for key in ("text_contains", "text_missing", "dialog_contains")):
        raise ArtifactError(f"outcome {outcome['code']}: detect needs text_contains, text_missing, or dialog_contains")
    if outcome["kind"] == "recoverable" and not outcome.get("recover"):
        raise ArtifactError(f"outcome {outcome['code']}: recoverable outcomes need a recover action")


# ---------- save / load ----------

def save(artifact: Artifact, secrets: tuple[str, ...] = ()) -> Path:
    """Persist an artifact as JSON. Refuses to write if a known secret value is present."""
    validate(artifact)
    raw_text = json.dumps(asdict(artifact), ensure_ascii=False)
    for secret in secrets:
        # A secret inside the artifact means parameterization failed; masking it would only
        # produce a flow that types "[REDACTED]" on replay, so refuse instead.
        if secret and secret in raw_text:
            raise ArtifactError("refusing to save: a sensitive input value was not parameterized")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f"{artifact.name}.v{artifact.version}.json"
    path.write_text(dumps(artifact, secrets), encoding="utf-8")
    return path


def dumps(artifact: Artifact, secrets: tuple[str, ...] = ()) -> str:
    """Reviewable JSON with sensitive keys and any registered secret value redacted."""
    return json.dumps(redact(asdict(artifact), secrets), indent=2, ensure_ascii=False)


def load(path: str | Path) -> Artifact:
    """Load and validate a version-1 artifact (the version replay runs).

    A version-2 capability graph is loaded through graph.load_graph, which also accepts
    version-1 files and converts them in memory.
    """
    data = read_json(path)
    version = data.get("schema_version") if isinstance(data, dict) else None
    if version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {version!r}, expected {SCHEMA_VERSION!r} "
                            f"(a 2.0 capability graph loads through graph.load_graph)")
    artifact = from_dict(data)
    validate(artifact)
    return artifact


def read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot read artifact {path}: {error}") from error


def from_dict(data: dict) -> Artifact:
    """Rebuild the dataclasses from plain JSON, checking that required keys exist."""
    required = {"schema_version", "name", "version", "status", "description", "surface",
                "inputs", "outputs", "steps", "outcomes", "success", "provenance"}
    missing = required - set(data)
    if missing:
        raise ArtifactError(f"artifact is missing keys: {sorted(missing)}")
    steps = [step_from_dict(raw) for raw in data["steps"]]
    return Artifact(
        schema_version=data["schema_version"], name=data["name"], version=int(data["version"]),
        status=data["status"], description=data["description"], surface=data["surface"],
        inputs=data["inputs"], outputs=data["outputs"], steps=steps, outcomes=data["outcomes"],
        success=data["success"], provenance=data["provenance"],
    )


def step_from_dict(raw: dict) -> Step:
    target = Locator(strategies=raw["target"]["strategies"]) if raw.get("target") else None
    return Step(id=raw.get("id", ""), action=raw.get("action", ""), target=target,
                value=raw.get("value"), checkpoint=raw.get("checkpoint"), risk=raw.get("risk", "safe"))
