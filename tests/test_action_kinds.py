"""The runtime's action kinds, the planner tool's kinds, the artifact's actions and the policy's allowlist agree."""
from typing import get_args

from src.cua.artifact import ACTIONS
from src.cua.models import ActionKind
from src.cua.planner import CHOOSE_ACTION_TOOL
from src.cua.policy import SURFACE_ACTIONS
from tests.scripted_planner import ScriptedPlanner


def test_every_planner_kind_is_a_runtime_action_kind():
    runtime = set(get_args(ActionKind))
    tool_kinds = set(CHOOSE_ACTION_TOOL["input_schema"]["properties"]["kind"]["enum"])
    assert tool_kinds <= runtime, tool_kinds - runtime
    assert {"navigate", "back", "click", "type", "select", "extract", "extract_many", "reuse_candidate", "done",
            "stuck"} == runtime


def test_artifact_and_policy_subsets_are_exactly_the_surface_actions():
    runtime = set(get_args(ActionKind))
    assert ACTIONS == SURFACE_ACTIONS == runtime - {"done", "stuck", "reuse_candidate"}


def test_the_scripted_planner_produces_only_runtime_kinds():
    import inspect
    import re
    source = inspect.getsource(ScriptedPlanner.decide)
    produced = set(re.findall(r'Action\(kind="([a-z_]+)"', source)) | {"click", "type", "extract", "navigate", "done", "stuck"}
    assert produced <= set(get_args(ActionKind))
