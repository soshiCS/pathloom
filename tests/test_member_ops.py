"""MemberOps Sandbox: the fictional back-office app behaves deterministically in every runtime mode."""
import http.cookiejar
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

from examples.member_ops.app import MODES, MemberOpsApp, serve_in_thread


class Browser:
    """A cookie-keeping HTTP client that reports 4xx/5xx bodies instead of raising."""

    def __init__(self, base: str):
        self.base = base
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def get(self, path: str):
        return self._open(urllib.request.Request(self.base + path))

    def post(self, path: str, **form):
        return self._open(urllib.request.Request(self.base + path, urllib.parse.urlencode(form).encode()))

    def _open(self, request):
        try:
            with self.opener.open(request, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    def login(self):
        return self.post("/login", username="operator", password="training_only")

    def to_review(self, member="1001", kind="savings", deposit="500"):
        self.login()
        self.post("/members/search", by="member_id", member_id=member)
        return self.post(f"/members/{member}/accounts/review", type=kind, opening_deposit=deposit)


@pytest.fixture
def sandbox(request):
    mode = getattr(request, "param", "normal")
    server, base = serve_in_thread(mode)
    yield server.app, base
    server.shutdown()
    server.server_close()


def browser(base: str) -> Browser:
    return Browser(base)


def state(base: str) -> dict:
    return json.loads(Browser(base).get("/__state")[1])


# ---------- startup, reset, login ----------

def test_server_starts_in_every_mode_and_resets():
    for mode in MODES:
        server, base = serve_in_thread(mode)
        try:
            assert state(base)["mode"] == mode
            client = browser(base)
            client.to_review()
            client.to_review()                 # session_expired_once bounces the first review; the second works
            client.post("/members/1001/accounts/open")
            assert state(base)["created"] and state(base)["sessions"] >= 1
            assert Browser(base).post("/__reset") == (200, "reset\n")
            assert state(base)["created"] == [] and state(base)["sessions"] == 0
            assert [a for a in state(base)["accounts"]["1001"]] == ["CHK-4321", "SAV-8765"]
        finally:
            server.shutdown()
            server.server_close()
    with pytest.raises(ValueError, match="mode must be one of"):
        MemberOpsApp(mode="chaos")


def test_login_success_and_failure(sandbox):
    app, base = sandbox
    client = browser(base)
    assert client.get("/")[0] == 200 and "Sign in" in client.get("/")[1]
    status, body = client.post("/login", username="operator", password="wrong")
    assert status == 200 and "Invalid training credentials" in body and "Member lookup" not in body
    status, body = client.login()
    assert status == 200 and "Member lookup" in body and "operator" in body
    assert "training_only" not in body                                   # the password is never echoed
    assert client.get("/logout")[1].count("Sign in") >= 1
    assert "Please sign in to continue" in client.get("/members")[1]


# ---------- lookup, profile, outcomes ----------

def test_both_lookup_methods_reach_the_same_profile(sandbox):
    _, base = sandbox
    client = browser(base)
    client.login()
    by_id = client.post("/members/search", by="member_id", member_id="1001")[1]
    by_phone = client.post("/members/search", by="phone", phone="5550101")[1]
    for body in (by_id, by_phone):
        assert "Member profile" in body and "Alex Morgan" in body and "New sub-account" in body
        assert body.count("Details") == 2 and "Account type" in body     # repeated labels per account row
    assert 'href="/members?by=phone"' in client.get("/members")[1]
    assert "Phone number" in client.get("/members?by=phone")[1]


def test_member_not_found_and_permission_denied(sandbox):
    _, base = sandbox
    client = browser(base)
    client.login()
    status, body = client.post("/members/search", by="member_id", member_id="4040")
    assert status == 200 and "No member found for the supplied lookup value" in body and "member ID 4040" in body
    status, body = client.post("/members/search", by="phone", phone="")
    assert "No member found" in body and "blank" in body
    status, body = client.post("/members/search", by="phone", phone="5550102")
    assert status == 403 and "This member record is restricted" in body and "Jordan Lee" not in body
    assert client.get("/members/1002")[0] == 403 and client.get("/members/9999")[0] == 404


# ---------- preparing a sub-account ----------

@pytest.mark.parametrize("kind, label, deposit, shown", [("savings", "Savings", "500", "500.00"),
                                                         ("checking", "Checking", "250", "250.00")])
def test_savings_and_checking_preparation_reach_a_review_page(sandbox, kind, label, deposit, shown):
    _, base = sandbox
    client = browser(base)
    client.login()
    choose = client.get("/members/1001/accounts/new")[1]
    assert "Choose the account type" in choose and f"?type={kind}" in choose
    form = client.get(f"/members/1001/accounts/new?type={kind}")[1]
    assert f"Account type: <strong>{label}</strong>" in form and "Opening deposit (USD)" in form
    status, review = client.to_review(kind=kind, deposit=deposit)
    assert status == 200 and "Review new sub-account" in review
    assert f"Member name: Alex Morgan" in review and f"Account type: {label}" in review
    assert f"Opening deposit: $ {shown}" in review and "Confirm and Open Account" in review
    assert "Nothing has been created yet" in review and state(base)["created"] == []
    assert "Review new sub-account" in client.get("/members/1001/accounts/review")[1]   # the draft survives a reload


def test_review_outputs_match_the_campaign_regexes(sandbox):
    import re
    from src.cua.campaign import load_spec
    _, base = sandbox
    review = browser(base).to_review(kind="checking", deposit="250")[1]
    lines = re.findall(r"<li>(.*?)</li>", review)
    outputs = load_spec("scenarios/member_account_prepare.json").outputs
    found = {}
    for name, spec in outputs.items():
        for line in lines:
            match = re.search(spec["pattern"], line)
            if match:
                found[name] = match.group(1)
    assert found == {"member_name": "Alex Morgan", "account_type": "Checking", "opening_deposit": "250.00"}


def test_validation_errors_keep_the_form_and_create_nothing(sandbox):
    _, base = sandbox
    client = browser(base)
    for bad in ("", "abc", "10", "24.99"):
        status, body = client.to_review(deposit=bad)
        assert status == 200 and "Opening deposit must be a number of at least 25.00" in body
        assert "Review new sub-account" not in body and "Opening deposit (USD)" in body
    assert "Choose an account type first" in client.post("/members/1001/accounts/review", type="bond",
                                                         opening_deposit="500")[1]
    assert state(base)["created"] == []


# ---------- the irreversible action ----------

def test_final_action_changes_state_only_when_actually_submitted(sandbox):
    _, base = sandbox
    client = browser(base)
    client.to_review(kind="savings", deposit="500")
    assert state(base)["created"] == [] and state(base)["accounts"]["1001"] == ["CHK-4321", "SAV-8765"]
    status, body = client.post("/members/1001/accounts/open")
    assert status == 200 and "Sub-account opened" in body and "SAV-9001" in body
    assert state(base)["created"] == [{"member": "1001", "number": "SAV-9001", "type": "Savings",
                                       "balance": "500.00", "opened": "2026-09-15"}]
    assert state(base)["accounts"]["1001"][-1] == "SAV-9001"
    assert "SAV-9001" in client.get("/members/1001")[1]
    # Submitting again without a fresh review does nothing.
    status, body = client.post("/members/1001/accounts/open")
    assert "Sub-account opened" not in body and len(state(base)["created"]) == 1


# ---------- runtime modes ----------

@pytest.mark.parametrize("sandbox", ["slow_once"], indirect=True)
def test_slow_once_answers_503_exactly_once_per_session(sandbox):
    _, base = sandbox
    client = browser(base)
    client.login()
    started = time.time()
    status, body = client.get("/members/1001")
    assert status == 503 and "warming up" in body and time.time() - started >= 1.0
    assert client.get("/members/1001")[0] == 200
    assert client.get("/members/1001")[0] == 200
    other = browser(base)
    other.login()
    assert other.get("/members/1001")[0] == 503                        # a new browser session gets its own one


@pytest.mark.parametrize("sandbox", ["known_interstitial"], indirect=True)
def test_known_interstitial_shows_until_dismissed(sandbox):
    _, base = sandbox
    client = browser(base)
    assert 'role="dialog"' not in client.get("/login")[1]              # never on the login page
    body = client.login()[1]
    assert 'role="dialog"' in body and "Scheduled maintenance" in body and ">Dismiss<" in body
    assert 'role="dialog"' in client.get("/members/1001")[1]           # every page, until dismissed
    client.post("/notice/dismiss", back="/members")
    assert 'role="dialog"' not in client.get("/members")[1] and 'role="dialog"' not in client.get("/members/1001")[1]


@pytest.mark.parametrize("sandbox", ["unexpected_dialog"], indirect=True)
def test_unexpected_dialog_persists_until_acknowledged(sandbox):
    app, base = sandbox
    client = browser(base)
    client.login()
    assert 'role="dialog"' not in client.get("/members")[1]
    first = client.get("/members/1001")[1]                                 # first profile GET: visible
    assert 'role="dialog"' in first and "Compliance attestation required" in first and ">Acknowledge<" in first
    assert 'role="dialog"' in client.get("/members/1001")[1]              # second GET: still visible
    client.get("/members")
    assert 'role="dialog"' in client.get("/members/1001")[1]              # revisit: still visible
    assert 'role="dialog"' in client.get("/members/1003")[1]              # another profile: still visible
    status, body = client.post("/attest", back="/members/1001")          # only this acknowledges it
    assert status == 200 and "Member profile" in body and 'role="dialog"' not in body
    assert 'role="dialog"' not in client.get("/members/1001")[1]          # and it stays gone for the session
    other = browser(base)
    other.login()
    assert 'role="dialog"' in other.get("/members/1001")[1]               # a new session must acknowledge it too
    Browser(base).post("/__reset")
    fresh = browser(base)
    fresh.login()
    assert 'role="dialog"' in fresh.get("/members/1001")[1]               # reset clears acknowledgments


@pytest.mark.parametrize("sandbox", ["slow_after_open_once"], indirect=True)
def test_slow_after_open_once_opens_the_account_but_fails_the_confirmation_page_once(sandbox):
    _, base = sandbox
    client = browser(base)
    client.to_review()
    status, body = client.post("/members/1001/accounts/open")
    assert status == 503 and "may or may not have completed" in body
    assert len(state(base)["created"]) == 1                              # the side effect did happen
    client.to_review()
    status, body = client.post("/members/1001/accounts/open")
    assert status == 200 and "Sub-account opened" in body and len(state(base)["created"]) == 2


def test_concurrent_openings_allocate_distinct_accounts(sandbox):
    import threading
    _, base = sandbox
    numbers, errors = [], []

    def one():
        try:
            client = browser(base)
            client.to_review()
            status, body = client.post("/members/1001/accounts/open")
            assert status == 200 and "Sub-account opened" in body
            numbers.append(body.split("Account ")[1].split(" ")[0])
        except Exception as error:      # collected, not raised inside the thread
            errors.append(error)

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == [] and len(set(numbers)) == 8
    snapshot = state(base)
    assert len(snapshot["created"]) == 8 and sorted(c["number"] for c in snapshot["created"]) == sorted(numbers)
    assert snapshot["accounts"]["1001"][2:] == [c["number"] for c in snapshot["created"]]


@pytest.mark.parametrize("sandbox", ["session_expired_once"], indirect=True)
def test_session_expires_once_on_the_first_review(sandbox):
    _, base = sandbox
    client = browser(base)
    status, body = client.to_review()
    assert status == 200 and "Your session has expired" in body and "Sign in" in body
    assert "Please sign in to continue" in client.get("/members")[1]   # really signed out
    status, body = client.to_review()                                  # sign in again: it works now
    assert "Review new sub-account" in body and state(base)["created"] == []


def test_icon_only_control_and_legacy_markup_are_present(sandbox):
    _, base = sandbox
    client = browser(base)
    body = client.login()[1]
    assert 'role="button" data-test="alerts-toggle"' in body           # icon-only control with a test id
    assert '<td class="label">Username</td>' in client.get("/logout")[1]   # legacy table form
    assert "No real members, accounts or credentials" in body


# ---------- concurrency ----------

def run_threads(target, count: int, timeout: float = 30.0) -> None:
    """Run `target` on `count` threads and fail loudly on a hang instead of blocking the suite."""
    threads = [threading.Thread(target=target, daemon=True) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    assert not any(t.is_alive() for t in threads), "a request thread hung"


import threading  # noqa: E402  (kept next to the helper that needs it)


@pytest.mark.parametrize("sandbox", ["slow_once"], indirect=True)
def test_two_simultaneous_requests_on_one_session_consume_a_once_condition_once(sandbox):
    _, base = sandbox
    client = browser(base)                     # one cookie jar: one session shared by both threads
    client.login()
    statuses, errors = [], []

    def one():
        try:
            statuses.append(client.get("/members/1001")[0])
        except Exception as error:
            errors.append(error)

    run_threads(one, 2)
    assert errors == [] and sorted(statuses) == [200, 503]


def test_reset_racing_with_an_opening_leaves_a_consistent_snapshot(sandbox):
    app, base = sandbox
    paused, release = threading.Event(), threading.Event()
    app._test_hooks = {"before_open": lambda: (paused.set(), release.wait(timeout=10))}
    client = browser(base)
    client.to_review()
    outcome = {}

    def open_it():
        outcome["response"] = client.post("/members/1001/accounts/open")

    worker = threading.Thread(target=open_it, daemon=True)
    worker.start()
    assert paused.wait(timeout=10)             # the request is inside open_account, before the locked block
    Browser(base).post("/__reset")             # the reset wins the race
    release.set()
    worker.join(timeout=15)
    assert not worker.is_alive()
    app._test_hooks = {}
    snapshot = state(base)
    created = [c["number"] for c in snapshot["created"]]
    assert created == [] or created == ["SAV-9001"]               # either outcome is acceptable ...
    assert all(number in snapshot["accounts"]["1001"] for number in created)   # ... but never inconsistent
    assert snapshot["accounts"]["1001"][:2] == ["CHK-4321", "SAV-8765"]
    assert len(snapshot["accounts"]["1001"]) == 2 + len(created)
