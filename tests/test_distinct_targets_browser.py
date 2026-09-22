"""Same role, accessible name and enclosing item, different structural paths: replay reads three distinct cells."""
import pytest

from src.cua import agent as agent_module
from src.cua.agent import discover
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

pytestmark = pytest.mark.browser

ROWS = """<!doctype html><html><head><title>Rows</title></head><body><h1>Rows</h1>
<table><tr><th>Kind</th><th>Value</th></tr>%s</table></body></html>"""
THREE = ROWS % "".join(f"<tr><td>Item</td><td>{v}</td></tr>" for v in (11, 22, 33))
ONE = ROWS % "<tr><td>Item</td><td>11</td></tr>"


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    root = tmp_path_factory.mktemp("rows")
    (root / "three.html").write_text(THREE)
    (root / "one.html").write_text(ONE)
    live.three, live.one = (root / "three.html").as_uri(), (root / "one.html").as_uri()
    yield live
    live.close()


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 1.0)


def test_same_named_cells_are_read_as_three_distinct_values_and_ambiguity_emits_nothing(surface):
    policy = Policy(allowed_hosts=[""])
    contract = {"values": {"type": "list", "required": True, "min_items": 3, "max_items": 3,
                           "items": {"type": "integer", "pattern": r"(\d+)"}}}
    log = RunLog("discovery")
    artifact = discover(goal="read the three values", name="row_values", params={}, surface=surface,
                        planner=ScriptedPlanner([ScriptedStep("extract_many", "cell", texts=["11", "22", "33"],
                                                              output_name="values", pattern=r"(\d+)"),
                                                 ScriptedStep("done", expect="Rows")]),
                        policy=policy, escalator=Escalator(NoOperator(), SessionControl(), log), log=log,
                        entry_url=surface.three, output_contract=contract)
    validate(artifact)
    node = linear_path(artifact)[-1]
    names = {(t.strategies[0]["role"], t.strategies[0]["name"]) for t in node.action.targets}
    assert names == {("cell", "Item Value")}                                   # identical first rungs
    assert len({t.strategies[1]["selector"] for t in node.action.targets}) == 3   # distinct structural paths
    replay_log = RunLog("replay")
    result = replay(artifact, {}, surface, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "success" and result.outputs == {"values": [11, 22, 33]}

    artifact.surface["entry_url"] = surface.one
    linear_path(artifact)[0].action.value = surface.one
    replay_log = RunLog("replay")
    result = replay(artifact, {}, surface, policy, Escalator(NoOperator(), SessionControl(), replay_log), replay_log)
    assert result.status == "failure" and result.outcome_code == "ambiguous_target" and result.outputs == {}
