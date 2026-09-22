"""A role or text rung resolves only when it is unique; several same-named controls fall through to their
structural rungs, and a ladder with nothing unique is `ambiguous_target` with no action performed."""
import json

import pytest

from src.cua.artifact import build_linear
from src.cua.escalation import NoOperator
from src.cua.models import ActionError, Element, GraphAction, Locator, Observation
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RunLog, SessionControl, linear_node

FILINGS = "https://filings.test/"
DOCUMENTS = {1: "report-q1.htm", 2: "report-q2.htm", 3: "report-q3.htm"}


def el(role, name="", text="", ref="", context=""):
    return Element(role=role, name=name, text=text or name, box=(10, 10, 50, 20), ref=ref, context=context)


class FilingsSurface:
    """A results table whose three rows each carry a link named Documents; each opens its own detail page."""

    def __init__(self, rows=(1, 2, 3)):
        self.rows = list(rows)
        self.url, self.screen = "about:blank", "blank"
        self.history: list[tuple[str, str]] = []
        self.actions: list[tuple] = []

    def elements(self):
        if self.screen == "results":
            return [el("heading", text="Filings"), el("link", "Home", ref="a:home")] + [
                el("link", "Documents", ref=f"tr:{row} > a", context="Interactive Data") for row in self.rows]
        if self.screen.startswith("detail:"):
            row = int(self.screen.split(":")[1])
            return [el("heading", text="Filing Detail"), el("text", text=f"Document: {DOCUMENTS[row]}", ref=f"p:doc:{row}")]
        return []

    def observe(self):
        return Observation(url=self.url, elements=self.elements())

    def matches(self, strategy):
        found = []
        for element in self.elements():
            if strategy["kind"] == "role" and (element.role, element.name) == (strategy["role"], strategy["name"]) \
                    and (not strategy.get("context") or element.context == strategy["context"]):
                found.append(element)
            elif strategy["kind"] == "text" and strategy["text"] in element.text:
                found.append(element)
        return found

    def resolve(self, locator, exclude=frozenset()):
        for strategy in locator.strategies:
            if strategy["kind"] in ("role", "text"):
                found = self.matches(strategy)
                element = found[0] if len(found) == 1 else None
            else:
                element = next((e for e in self.elements() if strategy["kind"] == "css" and e.ref == strategy["selector"]), None)
            if element is None or element.ref in exclude:
                continue
            return element
        return None

    def navigate(self, url):
        self.actions.append(("navigate", url))
        self.url, self.screen = url, "results"

    def back(self):
        self.actions.append(("back",))
        self.screen, self.url = self.history.pop()

    def click(self, target):
        self.actions.append(("click", target.ref))
        if target.ref.startswith("tr:"):
            self.history.append((self.screen, self.url))
            row = int(target.ref.split(":")[1].split()[0])
            self.screen, self.url = f"detail:{row}", FILINGS + f"filing/{row}"
        return None

    def type(self, target, value):
        raise ActionError("nothing to type", performed="no")

    def screenshot(self, path):
        open(path, "wb").write(b"\x89PNG\r\n\x1a\n")
        return path


def documents_ladder(row: int, with_css=True) -> Locator:
    rungs = [{"kind": "role", "role": "link", "name": "Documents", "context": "Interactive Data"}]
    if with_css:
        rungs.append({"kind": "css", "selector": f"tr:{row} > a"})
    return Locator(strategies=rungs)


def filings_artifact(with_css=True, parameterized=False):
    nodes = [linear_node("s1", GraphAction(action="navigate", target=None, value=FILINGS,
                                           checkpoint={"text_contains": "Filings"}))]
    step = 2
    for row in (1, 2, 3):
        target = documents_ladder(row, with_css)
        if parameterized:
            target.strategies[0]["name"] = "{{link_name}}"
        nodes.append(linear_node(f"s{step}", GraphAction(action="click", target=target, value=None,
                                                         checkpoint={"text_contains": "Filing Detail"})))
        nodes.append(linear_node(f"s{step + 1}", GraphAction(action="extract", value=f"document_{row}",
                                                             target=Locator(strategies=[{"kind": "text", "text": "Document:"}]))))
        nodes.append(linear_node(f"s{step + 2}", GraphAction(action="back", target=None, value=None,
                                                             checkpoint={"text_contains": "Filings"})))
        step += 3
    outputs = {f"document_{row}": {"type": "string", "required": True, "pattern": r"Document: (.+)", "description": "",
                                   "example": DOCUMENTS[row]} for row in (1, 2, 3)}
    return build_linear(name="filing_documents", goal="open each filing and read its document", 
                        surface_meta={"kind": "web", "app": "filings", "entry_url": FILINGS, "allowed_hosts": ["filings.test"]},
                        params={"link_name": "Documents"} if parameterized else {}, nodes=nodes, outputs=outputs,
                        outcomes=[], success={"text_contains": "Filings"},
                        run_id="discovery-test", sensitive=set(), planner_name="scripted")


def run(artifact, surface=None):
    surface = surface or FilingsSurface()
    log = RunLog("replay")
    params = {"link_name": "Documents"} if "link_name" in artifact.inputs else {}
    result = replay(artifact, params, surface, Policy(allowed_hosts=["filings.test"]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    return result, surface, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


def test_identical_role_rungs_fall_through_to_the_structural_rungs_and_open_each_row():
    result, surface, log = run(filings_artifact())
    assert result.status == "success"
    assert result.outputs == {"document_1": "report-q1.htm", "document_2": "report-q2.htm", "document_3": "report-q3.htm"}
    assert [a for a in surface.actions if a[0] == "click"] == [("click", "tr:1 > a"), ("click", "tr:2 > a"), ("click", "tr:3 > a")]
    clicks = [e for e in events(log, "target_resolved") if e["step_id"] in ("s2", "s5", "s8")]
    assert [(e["rung"], e["strategy"]) for e in clicks] == [(1, "css")] * 3
    assert [e["matches"] for e in events(log, "rung_ambiguous")] == [3, 3, 3]


def test_parameterized_single_targets_use_css_only_to_disambiguate_semantic_matches():
    result, surface, log = run(filings_artifact(parameterized=True))
    assert result.status == "success"
    assert [a for a in surface.actions if a[0] == "click"] == [
        ("click", "tr:1 > a"), ("click", "tr:2 > a"), ("click", "tr:3 > a")]
    clicks = [e for e in events(log, "target_resolved") if e["step_id"] in ("s2", "s5", "s8")]
    assert [(e["rung"], e["strategy"]) for e in clicks] == [(1, "css")] * 3


def test_parameterized_structural_rung_cannot_select_outside_the_semantic_matches():
    artifact = filings_artifact(parameterized=True)
    artifact.nodes[1].action.target.strategies[1]["selector"] = "a:home"
    result, surface, log = run(artifact)
    assert result.status == "failure" and result.outcome_code == "ambiguous_target" and result.step_id == "s2"
    assert [a for a in surface.actions if a[0] == "click"] == []
    assert events(log, "rung_skipped")[0]["reason"] == \
        "structural rung did not disambiguate a parameterized semantic match"


def test_without_structural_rungs_the_first_click_is_ambiguous_and_nothing_is_clicked():
    result, surface, log = run(filings_artifact(with_css=False))
    assert result.status == "failure" and result.outcome_code == "ambiguous_target" and result.step_id == "s2"
    assert result.expected.startswith("exactly one control matching") and result.outputs == {}
    assert [a for a in surface.actions if a[0] == "click"] == [] and surface.screen == "results"
    assert events(log, "ambiguous_target")[0]["action"] == "click"


def test_a_genuinely_unique_role_locator_still_resolves_at_rung_zero():
    artifact = filings_artifact()
    home = linear_node("s11", GraphAction(action="click", target=Locator(strategies=[
        {"kind": "role", "role": "link", "name": "Home"}, {"kind": "css", "selector": "a:home"}]), value=None))
    artifact.nodes.insert(-1, home)
    from src.cua.models import GraphEdge, Guard
    artifact.edges = [e for e in artifact.edges if e.source != "s10"] + [
        GraphEdge(source="s10", target="s11", guards=[Guard(kind="always")], priority=0),
        GraphEdge(source="s11", target="success", guards=[Guard(kind="always")], priority=0)]
    result, surface, log = run(artifact)
    assert result.status == "success"
    [resolved] = [e for e in events(log, "target_resolved") if e["step_id"] == "s11"]
    assert (resolved["rung"], resolved["strategy"]) == (0, "role")


def test_a_ladder_with_no_match_at_all_is_still_a_missing_target():
    result, surface, log = run(filings_artifact(), FilingsSurface(rows=()))
    assert result.status == "failure" and result.outcome_code == "target_not_found" and result.step_id == "s2"
    assert events(log, "target_unresolved")[0]["ambiguous"] is False
