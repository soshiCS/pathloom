"""Three identically named links in real Chromium: replay opens rows 1, 2 and 3 through their structural rungs."""
import json

import pytest

from src.cua.artifact import build_linear, linear_path
from src.cua.escalation import NoOperator
from src.cua.models import GraphAction, Locator
from src.cua.policy import Policy
from src.cua.replay import replay
from tests.context import Escalator, RunLog, SessionControl, linear_node

pytestmark = pytest.mark.browser

DOCS = {1: "report-q1.htm", 2: "report-q2.htm", 3: "report-q3.htm"}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("filings")
    rows = "".join(f"<tr><td>Interactive Data</td><td><a href='detail{n}.html'>Documents</a></td></tr>" for n in DOCS)
    (root / "index.html").write_text(f"<!doctype html><html><head><title>Filings</title></head><body><h1>Filings</h1>"
                                     f"<table><tr><th>Kind</th><th>Links</th></tr>{rows}</table></body></html>")
    for n, name in DOCS.items():
        (root / f"detail{n}.html").write_text(f"<!doctype html><html><head><title>Detail</title></head><body>"
                                              f"<h1>Filing Detail</h1><p>Document: {name}</p></body></html>")
    return root


@pytest.fixture(scope="module")
def surface(site):
    pytest.importorskip("playwright")
    from src.cua.surface import PlaywrightSurface
    try:
        live = PlaywrightSurface(headless=True, timeout_ms=2500)
    except Exception as error:
        pytest.skip(f"Chromium is not available: {error}")
    live.entry = (site / "index.html").as_uri()
    yield live
    live.close()


def artifact_for(surface, with_css=True):
    surface.navigate(surface.entry)
    links = [e for e in surface.observe().elements if e.role == "link" and e.name == "Documents"]
    assert len(links) == 3 and len({e.ref for e in links}) == 3
    rung = {"kind": "role", "role": "link", "name": "Documents", "context": links[0].context}
    nodes = [linear_node("s1", GraphAction(action="navigate", target=None, value=surface.entry, checkpoint={"text_contains": "Filings"}))]
    step = 2
    for index, link in enumerate(links, start=1):
        rungs = [dict(rung)] + ([{"kind": "css", "selector": link.ref}] if with_css else [])
        nodes += [linear_node(f"s{step}", GraphAction(action="click", target=Locator(strategies=rungs), value=None,
                                                      checkpoint={"text_contains": "Filing Detail"})),
                  linear_node(f"s{step + 1}", GraphAction(action="extract", value=f"document_{index}",
                                                          target=Locator(strategies=[{"kind": "text", "text": "Document:"}]))),
                  linear_node(f"s{step + 2}", GraphAction(action="back", target=None, value=None, checkpoint={"text_contains": "Filings"}))]
        step += 3
    outputs = {f"document_{n}": {"type": "string", "required": True, "pattern": r"Document: (.+)", "description": "", "example": DOCS[n]}
               for n in DOCS}
    return build_linear(name="filing_documents", goal="read each filing's document", params={}, nodes=nodes, outputs=outputs,
                        surface_meta={"kind": "web", "app": "filings", "entry_url": surface.entry, "allowed_hosts": [""]},
                        outcomes=[], success={"text_contains": "Filings"}, run_id="discovery-test", sensitive=set(),
                        planner_name="scripted")


def run(surface, artifact):
    log = RunLog("replay")
    result = replay(artifact, {}, surface, Policy(allowed_hosts=[""]), Escalator(NoOperator(), SessionControl(), log), log)
    return result, log


def test_same_named_links_open_their_own_rows_and_read_three_different_documents(surface):
    result, log = run(surface, artifact_for(surface))
    assert result.status == "success"
    assert [result.outputs[f"document_{n}"] for n in (1, 2, 3)] == list(DOCS.values())
    clicks = [json.loads(l) for l in log.path.read_text().splitlines() if '"target_resolved"' in l]
    assert [(e["rung"], e["strategy"]) for e in clicks if e["step_id"] in ("s2", "s5", "s8")] == [(1, "css")] * 3


def test_without_structural_rungs_replay_is_ambiguous_and_never_leaves_the_results_page(surface):
    result, log = run(surface, artifact_for(surface, with_css=False))
    assert result.status == "failure" and result.outcome_code == "ambiguous_target" and result.step_id == "s2"
    assert surface.observe().url == surface.entry and result.outputs == {}
    assert '"event": "ambiguous_target"' in log.path.read_text() and '"event": "acted", "step_id": "s2"' not in log.path.read_text()
