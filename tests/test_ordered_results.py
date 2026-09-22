"""Ordered multi-result work: the run fixes which results it is reporting before it leaves the list page, and
later detail pages follow those identities even when the list reorders underneath."""
import json

import pytest

from src.cua import agent as agent_module
from src.cua.agent import DiscoveryFailed, Recorder, discover, snapshot_instability, stable_result_list
from src.cua.artifact import linear_path, validate
from src.cua.escalation import NoOperator
from src.cua.models import Action, ActionError, Element, Locator, Observation
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RecordingOperator, RunLog, SessionControl
from tests.scripted_planner import ScriptedPlanner, ScriptedStep

LIST_URL = "https://results.test/"
PROJECTS = [("Alpha Project", "P-1", "2026-07-22"), ("Beta Project", "P-2", "2026-06-29"),
            ("Gamma Project", "P-3", "2026-05-14")]
def list_spec(pattern="(.+)"):
    return {"type": "list", "required": True, "min_items": 3, "max_items": 3,
            "items": {"type": "string", "pattern": pattern}}


CONTRACT = {"names": list_spec(), "ids": list_spec(r"Id: (.+)"), "dates": list_spec(r"Date: (.+)")}


def el(role, name="", text="", ref=""):
    return Element(role=role, name=name, text=text or name, box=(10, 10, 80, 20), ref=ref)


class ResultsSurface:
    """A result list whose order can change while a detail page is open, as a live list does."""

    def __init__(self, reorder_after=None, unstable=False, identities=True):
        self.order = list(range(len(PROJECTS)))
        self.reorder_after = reorder_after          # rotate the list once, after this many detail visits
        self.unstable = unstable                    # reshuffle on every observation
        self.identities = identities                # whether result titles are perceivable at all
        self.visits = 0
        self.screen, self.url = "list", LIST_URL
        self.actions: list[tuple] = []
        self.history: list[tuple] = []

    def rows(self):
        return [PROJECTS[index] for index in self.order]

    def elements(self):
        if self.screen == "list":
            out = [el("heading", text="Results"), el("text", text="Sorted newest first")]
            for position, (title, ident, date) in enumerate(self.rows()):
                label = title if self.identities else ""
                out.append(el("link", label, text=title if self.identities else "Open result",
                              ref=f"row:{position}"))
            return out
        title, ident, date = PROJECTS[int(self.screen.split(":")[1])]
        return [el("heading", text="Detail"), el("text", text=f"Name: {title}", ref="d:name"),
                el("text", text=f"Id: {ident}", ref="d:id"), el("text", text=f"Date: {date}", ref="d:date")]

    def observe(self):
        if self.unstable and self.screen == "list":
            # a live list whose rows keep being replaced: every look shows a different set, so no order holds
            self.shuffles = getattr(self, "shuffles", 0) + 1
            self.order = [(index + self.shuffles) % len(PROJECTS) for index in self.order][:-1]
        return Observation(url=self.url, elements=self.elements())

    def matches(self, strategy):
        found = []
        for element in self.elements():
            if strategy["kind"] == "role" and (element.role, element.name) == (strategy["role"], strategy["name"]):
                found.append(element)
            elif strategy["kind"] == "text" and strategy["text"] in element.text:
                found.append(element)
        return found

    def resolve(self, locator, exclude=frozenset()):
        for strategy in locator.strategies:
            if strategy["kind"] in ("role", "text"):
                hits = self.matches(strategy)
                element = hits[0] if len(hits) == 1 else None
            else:
                element = next((e for e in self.elements() if e.ref == strategy.get("selector")), None)
            if element is None or element.ref in exclude:
                continue
            return element
        return None

    def navigate(self, url):
        self.actions.append(("navigate", url))
        self.url, self.screen = url, "list"

    def back(self):
        self.actions.append(("back",))
        self.screen, self.url = self.history.pop()
        self.visits += 1
        if self.reorder_after is not None and self.visits == self.reorder_after:
            self.order = self.order[1:] + self.order[:1]      # the live list refreshes: yesterday's first is now second

    def click(self, target):
        self.actions.append(("click", target.name or target.text))
        if target.ref.startswith("row:"):
            self.history.append((self.screen, self.url))
            index = self.order[int(target.ref.split(":")[1])]
            self.screen, self.url = f"detail:{index}", LIST_URL + f"item/{index}"
        return None

    def type(self, target, value):
        raise ActionError("nothing to type", performed="no")

    def screenshot(self, path):
        open(path, "wb").write(b"\x89PNG\r\n\x1a\n")
        return path


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(agent_module, "EXPECT_TIMEOUT_S", 0.0)
    monkeypatch.setattr(agent_module, "SNAPSHOT_RECHECK_MS", 0)


def snapshot_first(titles=("Alpha Project", "Beta Project", "Gamma Project")):
    """Read the identity column, then open each result by name and append its remaining values."""
    script = [ScriptedStep("extract_many", "link", texts=list(titles), output_name="names", pattern="(.+)")]
    for title in titles:
        script += [ScriptedStep("click", "link", title, expect="Detail"),
                   ScriptedStep("extract", "text", text="Id:", output_name="ids", pattern=r"Id: (.+)",
                                output_mode="append"),
                   ScriptedStep("extract", "text", text="Date:", output_name="dates", pattern=r"Date: (.+)",
                                output_mode="append"),
                   ScriptedStep("back", expect="Results")]
    return script + [ScriptedStep("done", expect="Results")]


def run(script, surface=None, operator=None, contract=CONTRACT, max_steps=40):
    surface = surface or ResultsSurface()
    log = RunLog("discovery")
    artifact = discover(goal="report the first three results", name="ordered_results", params={}, surface=surface,
                        planner=ScriptedPlanner(script), policy=Policy(allowed_hosts=["results.test"]),
                        escalator=Escalator(operator or NoOperator(), SessionControl(), log), log=log,
                        entry_url=LIST_URL, output_contract=contract, max_steps=max_steps)
    return artifact, surface, log


def events(log, name):
    return [json.loads(line) for line in log.path.read_text().splitlines() if f'"event": "{name}"' in line]


# ---------- the live failure: the list reorders while a detail page is open ----------

def test_a_list_that_reorders_after_the_first_detail_page_keeps_the_snapshot_order():
    surface = ResultsSurface(reorder_after=1)
    artifact, surface, log = run(snapshot_first(), surface=surface)
    validate(artifact)
    assert [e["output_name"] for e in events(log, "result_snapshot_taken")] == ["names"]
    # the list rotated after the first visit, and every output still reads in the order first committed to
    assert artifact.outputs["names"]["example"] == ["Alpha Project", "Beta Project", "Gamma Project"]
    assert artifact.outputs["ids"]["example"] == ["P-1", "P-2", "P-3"]
    assert artifact.outputs["dates"]["example"] == ["2026-07-22", "2026-06-29", "2026-05-14"]


def test_the_identities_are_captured_before_the_first_detail_page_is_opened():
    artifact, surface, log = run(snapshot_first())
    kinds = [a[0] for a in surface.actions]
    assert kinds.index("click") > 0                       # a navigation first, then the column read, then clicks
    steps = [n.action.action for n in linear_path(artifact)]
    assert steps[:2] == ["navigate", "extract_many"]      # the snapshot precedes every detail visit
    assert steps[2] == "click"


def test_each_later_result_is_opened_by_its_saved_identity_not_its_position():
    surface = ResultsSurface(reorder_after=1)
    artifact, surface, log = run(snapshot_first(), surface=surface)
    clicked = [a[1] for a in surface.actions if a[0] == "click"]
    assert clicked == ["Alpha Project", "Beta Project", "Gamma Project"]
    ladders = [n.action.target.strategies[0] for n in linear_path(artifact) if n.action.action == "click"]
    assert [rung["name"] for rung in ladders] == ["Alpha Project", "Beta Project", "Gamma Project"]
    assert all(rung["kind"] == "role" for rung in ladders)     # semantic identity, never a row index


def test_five_aligned_lists_stay_aligned():
    contract = {"names": list_spec(), "ids": list_spec(r"Id: (.+)"), "dates": list_spec(r"Date: (.+)"),
                "names_again": list_spec(r"Name: (.+)"), "ids_again": list_spec(r"Id: (.+)")}
    script = [ScriptedStep("extract_many", "link", texts=[p[0] for p in PROJECTS], output_name="names", pattern="(.+)")]
    for title in (p[0] for p in PROJECTS):
        script += [ScriptedStep("click", "link", title, expect="Detail"),
                   ScriptedStep("extract", "text", text="Id:", output_name="ids", pattern=r"Id: (.+)", output_mode="append"),
                   ScriptedStep("extract", "text", text="Date:", output_name="dates", pattern=r"Date: (.+)", output_mode="append"),
                   ScriptedStep("extract", "text", text="Name:", output_name="names_again", pattern=r"Name: (.+)", output_mode="append"),
                   ScriptedStep("extract", "text", text="Id:", output_name="ids_again", pattern=r"Id: (.+)", output_mode="append"),
                   ScriptedStep("back", expect="Results")]
    artifact, surface, log = run(script + [ScriptedStep("done", expect="Results")],
                                 surface=ResultsSurface(reorder_after=2), contract=contract)
    examples = {name: artifact.outputs[name]["example"] for name in contract}
    assert examples["names"] == examples["names_again"] == [p[0] for p in PROJECTS]
    assert examples["ids"] == examples["ids_again"] == [p[1] for p in PROJECTS]
    assert examples["dates"] == [p[2] for p in PROJECTS]


# ---------- the gate itself ----------

def test_appending_before_a_snapshot_exists_is_refused_with_the_reason():
    surface = ResultsSurface()
    script = [ScriptedStep("click", "link", "Alpha Project", expect="Detail"),
              ScriptedStep("extract", "text", text="Id:", output_name="ids", pattern=r"Id: (.+)", output_mode="append"),
              ScriptedStep("back", expect="Results")] + snapshot_first()
    artifact, surface, log = run(script)
    [refused] = events(log, "action_failed")
    assert refused["performed"] == "no" and "nothing fixes which results they are yet" in refused["reason"]
    assert artifact.outputs["ids"]["example"] == ["P-1", "P-2", "P-3"]      # the run recovered and stayed in order


def test_an_unstable_list_is_retried_within_a_bound_then_refuses_the_snapshot():
    surface = ResultsSurface(unstable=True)
    rows = [e for e in ResultsSurface().elements() if e.role == "link"]
    action = Action(kind="extract_many", targets=rows, output_name="names")
    log = RunLog("discovery")
    problem = stable_result_list(action, surface, log)
    assert problem is not None and "kept changing while its order was being read" in problem
    assert len(events(log, "result_list_unstable")) == agent_module.MAX_SNAPSHOT_ATTEMPTS
    assert events(log, "result_list_stable") == []
    assert surface.actions == []                              # nothing was clicked while the order was in doubt
    assert surface.shuffles >= agent_module.MAX_SNAPSHOT_ATTEMPTS
    # a list that holds still is accepted on the first look
    steady = RunLog("discovery")
    assert stable_result_list(action, ResultsSurface(), steady) is None
    assert events(steady, "result_list_stable")[0]["results"] == 3


def test_a_list_with_no_capturable_identity_refuses_before_any_output_is_recorded():
    """Rows with no distinguishable identity cannot be snapshotted, so the first append is refused outright."""
    recorder = Recorder({}, CONTRACT)
    append = Action(kind="extract", target=el("text", text="Id: P-1"), output_name="ids", output_mode="append")
    problem = recorder.output_conflict(append)
    assert problem is not None and "nothing fixes which results they are yet" in problem
    assert recorder.outputs == {} and recorder.nodes == []          # nothing partial was recorded

    # and in a run: the anonymous list offers no identity column, so the append is refused and a person decides
    surface = ResultsSurface(identities=False)
    operator = RecordingOperator("abort")
    with pytest.raises(DiscoveryFailed, match="human aborted"):
        run([ScriptedStep("click", "link", text="Open result", expect="Detail"),
             ScriptedStep("extract", "text", text="Id:", output_name="ids", pattern=r"Id: (.+)", output_mode="append"),
             ScriptedStep("extract", "text", text="Id:", output_name="ids", pattern=r"Id: (.+)", output_mode="append")],
            surface=surface, operator=operator)
    assert "nothing fixes which results they are yet" in operator.requests[0].reason


# ---------- ordinary work is untouched ----------

def test_a_single_result_task_is_unchanged():
    contract = {"name": {"type": "string", "required": True, "pattern": r"Name: (.+)"}}
    artifact, surface, log = run([ScriptedStep("click", "link", "Alpha Project", expect="Detail"),
                                  ScriptedStep("extract", "text", text="Name:", output_name="name", pattern=r"Name: (.+)"),
                                  ScriptedStep("done", expect="Detail")], contract=contract)
    assert artifact.outputs["name"]["example"] == "Alpha Project" and events(log, "action_failed") == []


def test_lists_read_entirely_from_the_list_page_are_unchanged():
    contract = {"names": list_spec(), "ids": list_spec()}
    artifact, surface, log = run([
        ScriptedStep("extract_many", "link", texts=[p[0] for p in PROJECTS], output_name="names", pattern="(.+)"),
        ScriptedStep("extract_many", "link", texts=[p[0] for p in PROJECTS], output_name="ids", pattern="(.+)"),
        ScriptedStep("done", expect="Results")], contract=contract)
    assert artifact.outputs["names"]["example"] == [p[0] for p in PROJECTS]
    assert events(log, "action_failed") == []


# ---------- the artifact stands on its own ----------

def test_the_artifact_replays_with_no_planner_and_no_snapshot_state():
    artifact, _, _ = run(snapshot_first(), surface=ResultsSurface(reorder_after=1))
    fresh = ResultsSurface()
    log = RunLog("replay")
    result = replay(artifact, {}, fresh, Policy(allowed_hosts=["results.test"]),
                    Escalator(NoOperator(), SessionControl(), log), log)
    assert result.status == "success"
    assert result.outputs["names"] == [p[0] for p in PROJECTS]
    assert result.outputs["ids"] == [p[1] for p in PROJECTS]
    assert result.outputs["dates"] == [p[2] for p in PROJECTS]
    assert "snapshot" not in json.dumps(artifact.provenance)


def test_the_gate_only_governs_several_equally_sized_lists():
    recorder = Recorder({}, CONTRACT)
    assert recorder.aligned_lists() == ["dates", "ids", "names"]
    assert Recorder({}, {"one": CONTRACT["ids"]}).aligned_lists() == []
    mixed = {"a": CONTRACT["ids"], "b": {**CONTRACT["ids"], "max_items": 5}}
    assert Recorder({}, mixed).aligned_lists() == []
    unsized = {"a": CONTRACT["ids"], "b": {"type": "list", "required": True, "items": {"type": "string"}}}
    assert Recorder({}, unsized).aligned_lists() == []
    assert Recorder({}, {}).aligned_lists() == []


# ---------- stability is judged on semantic identity, never on structure ----------

def identity_row(text, ref="", box=(0, 0, 0, 0), context=""):
    return Element(role="link", name="", text=text, ref=ref, box=box, context=context)


def wanted_identities(*texts):
    from src.cua.agent import result_identity
    return [result_identity(identity_row(text)) for text in texts]


TITLES = ("Alpha Project", "Beta Project", "Gamma Project")


def test_re_rendered_rows_with_new_refs_paths_geometry_and_context_stay_stable():
    from src.cua.agent import snapshot_instability

    wanted = wanted_identities(*TITLES)
    rerendered = [identity_row("Alpha Project", ref="ng-9 > a:nth-of-type(4)", box=(7, 88, 120, 24), context="Row 9"),
                  identity_row("Beta Project", ref="ng-3 > a:nth-of-type(1)", box=(7, 140, 118, 24), context="Row 3"),
                  identity_row("Gamma Project", ref="ng-7 > a:nth-of-type(2)", box=(7, 190, 121, 24), context="Row 7")]
    assert snapshot_instability(wanted, rerendered) is None


def test_whitespace_and_case_differences_are_normalized():
    from src.cua.agent import snapshot_instability

    wanted = wanted_identities(*TITLES)
    noisy = [identity_row("  Alpha   Project "), identity_row("BETA PROJECT"), identity_row("gamma\tproject")]
    assert snapshot_instability(wanted, noisy) is None


def test_unrelated_content_appearing_does_not_make_the_list_unstable():
    from src.cua.agent import snapshot_instability

    wanted = wanted_identities(*TITLES)
    with_extras = [identity_row("Cookie notice"), identity_row("Alpha Project"), identity_row("Advertisement"),
                   identity_row("Beta Project"), identity_row("Live chat"), identity_row("Gamma Project"),
                   identity_row("Footer link")]
    assert snapshot_instability(wanted, with_extras) is None


@pytest.mark.parametrize("rows, problem", [
    ((TITLES[1], TITLES[0], TITLES[2]), "reordered"),
    ((TITLES[0], "Delta Project", TITLES[2]), "missing"),
    ((TITLES[0], TITLES[1]), "missing"),
    ((TITLES[0], TITLES[0], TITLES[1], TITLES[2]), "ambiguous"),
])
def test_reordered_replaced_missing_and_duplicate_identities_all_fail(rows, problem):
    from src.cua.agent import snapshot_instability

    assert snapshot_instability(wanted_identities(*TITLES), [identity_row(text) for text in rows]) == problem


def test_a_read_that_named_one_result_twice_is_ambiguous():
    from src.cua.agent import snapshot_instability

    wanted = wanted_identities(TITLES[0], TITLES[0], TITLES[1])
    assert snapshot_instability(wanted, [identity_row(text) for text in TITLES]) == "ambiguous"


def test_the_snapshot_survives_a_list_that_re_renders_between_looks():
    """The live shape: the same three results, re-rendered with fresh references on every observation."""

    class ReRendering(ResultsSurface):
        """The same three results, re-rendered with fresh structure, geometry and enclosing item each look."""

        def elements(self):
            rows = super().elements()
            if self.screen != "list":
                return rows
            self.renders = getattr(self, "renders", 0) + 1
            for position, row in enumerate(rows):
                if row.role == "link":
                    row.box = (7, 40 * position + self.renders, 120, 24)
                    row.context = f"Render {self.renders}"
            return rows

    surface = ReRendering()
    artifact, surface, log = run(snapshot_first(), surface=surface)
    validate(artifact)
    assert [e["output_name"] for e in events(log, "result_snapshot_taken")] == ["names"]
    assert events(log, "result_list_unstable") == []
    assert artifact.outputs["names"]["example"] == list(TITLES)
    assert artifact.outputs["ids"]["example"] == ["P-1", "P-2", "P-3"]


def test_the_stability_log_never_carries_an_identity_or_a_value():
    surface = ResultsSurface(unstable=True)
    rows = [e for e in ResultsSurface().elements() if e.role == "link"]
    log = RunLog("discovery")
    stable_result_list(Action(kind="extract_many", targets=rows, output_name="names"), surface, log)
    text = log.path.read_text()
    for title, ident, _ in PROJECTS:
        assert title not in text and ident not in text
    assert '"problem"' in text and '"results": 3' in text
