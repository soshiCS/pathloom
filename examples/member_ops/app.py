"""The MemberOps Sandbox application: routes, in-memory state, runtime modes, HTML.

Screens: login -> member lookup -> profile (or not found / access denied) -> new sub-account
(choose type, enter opening deposit) -> review -> "Confirm and Open Account" (the irreversible
action) -> opened. Pathloom drives it through the browser only; the `/__reset` and `/__state`
endpoints exist for tests and demonstrations, not for the automation.

Runtime modes (`--mode`), each deterministic and, where "once", once per browser session:
  normal                nothing special
  slow_once             the first profile load per session answers 503 after a short delay
  known_interstitial    a dismissible maintenance notice dialog after login, until dismissed
  unexpected_dialog     a compliance dialog on the profile page that no artifact knows about,
                        shown until it is acknowledged: reloading does not clear it; the human
                        must click Acknowledge on the same session and resume
  session_expired_once  the first review submission per session bounces to the login page
  slow_after_open_once  the account is opened, but the first confirmation page per session
                        answers 503: the result of the irreversible action is uncertain

Locking: one `threading.Lock` guards every read and write of shared state (the session table,
each Session's fields, the member records, the created list). State transitions are small
locked blocks (`get`, `set`, `claim_once`, `snapshot_member`, the allocation inside
`open_account`); rendering, the slow_once sleep and socket I/O happen outside the lock, on
copies. `claim_once` makes every "once per session" condition an atomic check-and-set, so two
simultaneous requests on one session cannot both consume it. `open_account` re-resolves the
member under the lock, so a reset racing with it can never leave `created` and the member's
account list disagreeing.
"""
from __future__ import annotations

import html
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .data import ACCOUNT_TYPES, MEMBERS, MINIMUM_DEPOSIT, TRAINING_PASSWORD, TRAINING_USER

MODES = ("normal", "slow_once", "known_interstitial", "unexpected_dialog", "session_expired_once",
         "slow_after_open_once")
COOKIE = "mo_session"
NOTICE_TEXT = "Scheduled maintenance: MemberOps will be unavailable tonight from 22:00 to 23:00."
COMPLIANCE_TEXT = "Compliance attestation required. Confirm with your supervisor before continuing with this member."
SLOW_DELAY_S = 1.5


@dataclass
class Session:
    user: str | None = None
    notice_dismissed: bool = False
    compliance_acknowledged: bool = False   # only POST /attest sets this; rendering the dialog does not
    slow_served: bool = False
    expiry_served: bool = False
    slow_after_open_served: bool = False
    draft: dict | None = None      # the sub-account being prepared: member id, type, deposit


@dataclass
class MemberOpsApp:
    mode: str = "normal"
    sessions: dict = field(default_factory=dict)
    members: dict = field(default_factory=lambda: {k: {**v, "accounts": [dict(a) for a in v["accounts"]]}
                                                   for k, v in MEMBERS.items()})
    created: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    def reset(self) -> None:
        with self.lock:
            self.sessions.clear()
            self.created.clear()
            self.members = {k: {**v, "accounts": [dict(a) for a in v["accounts"]]} for k, v in MEMBERS.items()}

    def state(self) -> dict:
        with self.lock:
            return {"mode": self.mode, "sessions": len(self.sessions), "created": [dict(c) for c in self.created],
                    "accounts": {k: [a["number"] for a in v["accounts"]] for k, v in self.members.items()}}

    # ---------- small locked state transitions ----------

    def get(self, session: Session, field: str):
        with self.lock:
            return getattr(session, field)

    def set(self, session: Session, **fields) -> None:
        with self.lock:
            for field_name, value in fields.items():
                setattr(session, field_name, value)

    def claim_once(self, session: Session, flag: str) -> bool:
        """Atomically consume a once-per-session condition; True for exactly one caller."""
        with self.lock:
            if getattr(session, flag):
                return False
            setattr(session, flag, True)
            return True

    def snapshot_member(self, member_id: str) -> dict | None:
        """A copy of a member record for rendering; the live record is only touched under the lock."""
        with self.lock:
            member = self.members.get(member_id)
            return None if member is None else {**member, "accounts": [dict(a) for a in member["accounts"]]}

    def _hook(self, name: str) -> None:
        """Test-only pause points for deterministic race tests (see tests/test_member_ops.py)."""
        hook = getattr(self, "_test_hooks", {}).get(name)
        if hook is not None:
            hook()

    # ---------- request handling ----------

    def handle(self, method: str, raw_path: str, cookies: dict, form: dict) -> tuple[int, dict, bytes]:
        """Dispatch one request; returns (status, headers, body)."""
        parts = urlsplit(raw_path)
        path, query = parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}
        if path == "/__reset":                     # test and demo endpoints: no session, no HTML
            self.reset()
            return 200, {"Content-Type": "text/plain"}, b"reset\n"
        if path == "/__state":
            return 200, {"Content-Type": "application/json"}, json.dumps(self.state()).encode()
        session_id = cookies.get(COOKIE)
        headers = {"Content-Type": "text/html; charset=utf-8"}
        with self.lock:
            if session_id not in self.sessions:
                session_id = secrets.token_hex(8)
                self.sessions[session_id] = Session()
                headers["Set-Cookie"] = f"{COOKIE}={session_id}; Path=/; HttpOnly"
            session = self.sessions[session_id]
        request = Request(method, path, query, form, session)
        status, body = self.route(request)
        if isinstance(body, Redirect):
            headers["Location"] = body.location
            return HTTPStatus.SEE_OTHER, headers, b""
        return status, headers, body.encode("utf-8")

    def route(self, r: Request) -> tuple[int, str | Redirect]:
        if r.path == "/":
            return 200, Redirect("/members" if self.get(r.session, "user") else "/login")
        if r.path == "/login":
            return self.login(r)
        if r.path == "/logout":
            self.set(r.session, user=None)
            return 200, Redirect("/login")
        if self.get(r.session, "user") is None:
            return 200, self.login_page(r, message="Please sign in to continue.")
        if r.path == "/notice/dismiss" and r.method == "POST":
            self.set(r.session, notice_dismissed=True)
            return 200, Redirect(r.form.get("back", "/members"))
        if r.path == "/attest" and r.method == "POST":
            self.set(r.session, compliance_acknowledged=True)
            return 200, Redirect(r.form.get("back", "/members"))
        if r.path == "/members":
            return 200, self.lookup_page(r)
        if r.path == "/members/search":
            return self.search(r)
        if r.path == "/reports":
            return 200, self.page(r, "Reports", "<p class='muted'>Reporting is not available in the sandbox.</p>")
        if r.path == "/alerts":
            return 200, self.page(r, "Alerts", "<p class='muted'>No alerts.</p>")
        segments = r.path.strip("/").split("/")
        if len(segments) >= 2 and segments[0] == "members":
            member = self.snapshot_member(segments[1])
            if member is None:
                return 404, self.page(r, "Member not found",
                                      "<p class='error'>No member found for the supplied lookup value.</p>")
            if member["restricted"]:
                return 403, self.page(r, "Access denied", "<p class='error'>This member record is restricted. "
                                                          "Ask a supervisor for access.</p>")
            rest = segments[2:]
            if not rest:
                return self.profile(r, member)
            if rest == ["accounts", "new"]:
                return 200, self.new_account_page(r, member)
            if rest == ["accounts", "review"] and r.method == "POST":
                return self.review(r, member)
            if rest == ["accounts", "review"]:
                if self.get(r.session, "draft"):
                    return 200, self.review_page(r, member)
                return 200, Redirect(f"/members/{member['id']}/accounts/new")
            if rest == ["accounts", "open"] and r.method == "POST":
                return self.open_account(r, member)
        return 404, self.page(r, "Not found", "<p class='error'>That page does not exist.</p>")

    # ---------- screens ----------

    def login(self, r: Request) -> tuple[int, str | Redirect]:
        if r.method == "POST":
            if r.form.get("username") == TRAINING_USER and r.form.get("password") == TRAINING_PASSWORD:
                self.set(r.session, user=TRAINING_USER)
                return 200, Redirect("/members")
            return 200, self.login_page(r, error="Invalid training credentials. Check the username and password.")
        return 200, self.login_page(r)

    def login_page(self, r: Request, error: str = "", message: str = "") -> str:
        body = f"""
<h1>Sign in</h1>
<p class="muted">Training environment. Use the training credentials supplied for this session.</p>
{'<p class="error">' + html.escape(error) + '</p>' if error else ''}
{'<p class="notice">' + html.escape(message) + '</p>' if message else ''}
<form method="post" action="/login" class="legacy-form">
  <table class="form-table">
    <tr><td class="label">Username</td><td><input type="text" name="username" size="24"></td></tr>
    <tr><td class="label">Password</td><td><input type="password" name="password" size="24"></td></tr>
    <tr><td></td><td><input type="submit" value="Sign in"></td></tr>
  </table>
</form>"""
        return self.page(r, "Sign in", body, chrome=False)

    def lookup_page(self, r: Request, error: str = "") -> str:
        by = r.query.get("by", "member_id")
        tab = lambda key, text: (f'<a class="tab active" aria-current="page" href="/members?by={key}">{text}</a>'
                                 if by == key else f'<a class="tab" href="/members?by={key}">{text}</a>')
        if by == "phone":
            field_html = ('<label for="phone">Phone number</label> '
                          '<input type="text" id="phone" name="phone" size="16" placeholder="digits only">')
        else:
            field_html = ('<label for="member_id">Member ID</label> '
                          '<input type="text" id="member_id" name="member_id" size="12">')
        body = f"""
<h1>Member lookup</h1>
<div class="tabs">{tab("member_id", "By member ID")} {tab("phone", "By phone")}</div>
{'<p class="error">' + html.escape(error) + '</p>' if error else ''}
<form method="post" action="/members/search" class="inline-form">
  <input type="hidden" name="by" value="{html.escape(by)}">
  {field_html}
  <button type="submit">Search</button>
</form>
<p class="muted">Recent members</p>
<table class="grid">
  <tr><th>Member ID</th><th>Name</th><th>Branch</th><th></th></tr>
  {''.join(f'<tr><td>{m["id"]}</td><td>{html.escape(m["name"])}</td><td>{m["branch"]}</td>'
           f'<td><a href="/members/{m["id"]}">Open</a></td></tr>' for m in self.member_list() if not m["restricted"])}
</table>"""
        return self.page(r, "Member lookup", body)

    def member_list(self) -> list[dict]:
        with self.lock:
            return [dict(m) for m in self.members.values()]

    def search(self, r: Request) -> tuple[int, str | Redirect]:
        by = r.form.get("by", "member_id")
        value = (r.form.get("phone") if by == "phone" else r.form.get("member_id")) or ""
        value = value.strip()
        matches = [m for m in self.member_list()
                   if (m["phone"] == value.replace("-", "").replace(" ", "") if by == "phone" else m["id"] == value)]
        if not value or not matches:
            what = "phone number" if by == "phone" else "member ID"
            r.query["by"] = by
            return 200, self.lookup_page(
                r, error=f"No member found for the supplied lookup value ({what} {value or 'blank'}).")
        return 200, Redirect(f"/members/{matches[0]['id']}")

    def profile(self, r: Request, member: dict) -> tuple[int, str]:
        if self.mode == "slow_once" and self.claim_once(r.session, "slow_served"):
            time.sleep(SLOW_DELAY_S)                                   # outside the lock
            return 503, self.page(r, "Service warming up", "<p class='error'>Member directory is warming up. "
                                                             "Retry in a moment.</p>", chrome=False)
        dialog = ""
        if self.mode == "unexpected_dialog" and not self.get(r.session, "compliance_acknowledged"):
            dialog = self.dialog(COMPLIANCE_TEXT, "/attest", "Acknowledge", back=f"/members/{member['id']}")
        rows = "".join(f"<tr><td>{a['number']}</td><td>{a['type']}</td><td class='num'>$ {a['balance']}</td>"
                       f"<td>{a['opened']}</td><td><a href='/members/{member['id']}#{a['number']}'>Details</a></td></tr>"
                       for a in member["accounts"])
        body = f"""
<h1>Member profile</h1>
<table class="kv">
  <tr><th>Member ID</th><td>{member['id']}</td></tr>
  <tr><th>Name</th><td>{html.escape(member['name'])}</td></tr>
  <tr><th>Phone</th><td>{member['phone']}</td></tr>
  <tr><th>Status</th><td><span class="badge">{member['status']}</span></td></tr>
  <tr><th>Member since</th><td>{member['since']}</td></tr>
  <tr><th>Branch</th><td>{member['branch']}</td></tr>
</table>
<div class="section-head"><h2>Accounts</h2>
  <a class="btn primary" href="/members/{member['id']}/accounts/new">New sub-account</a></div>
<table class="grid">
  <tr><th>Account</th><th>Account type</th><th>Balance</th><th>Opened</th><th></th></tr>
  {rows}
</table>
<p class="muted">Adjustments and closures are handled by the servicing desk.</p>
{dialog}"""
        return 200, self.page(r, "Member profile", body)

    def new_account_page(self, r: Request, member: dict, error: str = "", deposit: str = "") -> str:
        kind = r.query.get("type") or (self.get(r.session, "draft") or {}).get("type") or r.form.get("type", "")
        if kind not in ACCOUNT_TYPES:
            choose = "".join(f'<a class="btn" href="/members/{member["id"]}/accounts/new?type={key}">{label}</a> '
                             for key, label in ACCOUNT_TYPES.items())
            body = f"""
<h1>New sub-account for {html.escape(member['name'])}</h1>
{'<p class="error">' + html.escape(error) + '</p>' if error else ''}
<p>Choose the account type to prepare.</p>
<div class="choices">{choose}</div>
<p class="muted">Member ID {member['id']}. Existing accounts: {len(member['accounts'])}.</p>"""
            return self.page(r, "New sub-account", body)
        label = ACCOUNT_TYPES[kind]
        body = f"""
<h1>New sub-account for {html.escape(member['name'])}</h1>
<p>Account type: <strong>{label}</strong> <a class="small" href="/members/{member['id']}/accounts/new">change</a></p>
{'<p class="error">' + html.escape(error) + '</p>' if error else ''}
<form method="post" action="/members/{member['id']}/accounts/review" class="legacy-form">
  <input type="hidden" name="type" value="{kind}">
  <table class="form-table">
    <tr><td class="label">Opening deposit (USD)</td>
        <td><input type="text" name="opening_deposit" size="12" value="{html.escape(deposit)}"> <span class="muted">minimum {MINIMUM_DEPOSIT:.2f}</span></td></tr>
    <tr><td class="label">Funding source</td><td>Branch cash desk (default)</td></tr>
    <tr><td></td><td><button type="submit">Continue to review</button></td></tr>
  </table>
</form>"""
        return self.page(r, "New sub-account", body)

    def review(self, r: Request, member: dict) -> tuple[int, str | Redirect]:
        if self.mode == "session_expired_once" and self.claim_once(r.session, "expiry_served"):
            self.set(r.session, user=None)
            return 200, self.login_page(r, message="Your session has expired. Please sign in again.")
        kind = r.form.get("type", "")
        raw = (r.form.get("opening_deposit") or "").strip().replace(",", "")
        try:
            amount = float(raw)
        except ValueError:
            amount = -1.0
        if kind not in ACCOUNT_TYPES:
            return 200, self.new_account_page(r, member, error="Choose an account type first.")
        if amount < MINIMUM_DEPOSIT:
            r.query["type"] = kind
            error = f"Opening deposit must be a number of at least {MINIMUM_DEPOSIT:.2f}."
            return 200, self.new_account_page(r, member, deposit=raw, error=error)
        self.set(r.session, draft={"member": member["id"], "type": kind, "deposit": amount})
        return 200, self.review_page(r, member)

    def review_page(self, r: Request, member: dict) -> str:
        draft = dict(self.get(r.session, "draft"))
        label = ACCOUNT_TYPES[draft["type"]]
        body = f"""
<h1>Review new sub-account</h1>
<p class="notice">Nothing has been created yet. Check the details below.</p>
<ul class="summary">
  <li>Member ID: {member['id']}</li>
  <li>Member name: {html.escape(member['name'])}</li>
  <li>Account type: {label}</li>
  <li>Opening deposit: $ {draft['deposit']:.2f}</li>
  <li>Funding source: Branch cash desk</li>
</ul>
<form method="post" action="/members/{member['id']}/accounts/open" class="actions">
  <button type="submit" class="danger">Confirm and Open Account</button>
  <a class="btn" href="/members/{member['id']}/accounts/new?type={draft['type']}">Back</a>
</form>"""
        return self.page(r, "Review new sub-account", body)

    def open_account(self, r: Request, member: dict) -> tuple[int, str | Redirect]:
        """The irreversible action. The draft check, the number allocation and both appends are one locked
        transition against the *current* member record, so a reset in flight cannot split them."""
        self._hook("before_open")
        with self.lock:
            draft = r.session.draft
            live = self.members.get(member["id"])
            if not draft or draft["member"] != member["id"] or live is None:
                return 200, Redirect(f"/members/{member['id']}/accounts/new")
            number = f"{'SAV' if draft['type'] == 'savings' else 'CHK'}-{9000 + len(self.created) + 1}"
            account = {"number": number, "type": ACCOUNT_TYPES[draft["type"]],
                       "balance": f"{draft['deposit']:,.2f}", "opened": "2026-09-15"}
            live["accounts"].append(account)
            self.created.append({"member": member["id"], **account})
            reference = f"{number}-{len(self.created):04d}"
            r.session.draft = None
        if self.mode == "slow_after_open_once" and self.claim_once(r.session, "slow_after_open_served"):
            return 503, self.page(r, "Service warming up", "<p class='error'>The confirmation page could not be "
                                                             "loaded. The request may or may not have completed.</p>",
                                  chrome=False)
        body = f"""
<h1>Sub-account opened</h1>
<p class="notice">Account {number} ({account['type']}) was opened for {html.escape(member['name'])} with an opening
deposit of $ {account['balance']}. Reference {reference}.</p>
<a class="btn" href="/members/{member['id']}">Back to member</a>"""
        return 200, self.page(r, "Sub-account opened", body)

    # ---------- layout ----------

    def dialog(self, text: str, action: str, button: str, back: str) -> str:
        return (f'<div class="modal-backdrop"><div role="dialog" aria-modal="true" class="modal">'
                f'<p>{html.escape(text)}</p><form method="post" action="{action}">'
                f'<input type="hidden" name="back" value="{html.escape(back)}"><button type="submit">{button}</button>'
                f'</form></div></div>')

    def page(self, r: Request, title: str, body: str, chrome: bool = True) -> str:
        notice = ""
        if chrome and self.mode == "known_interstitial" and not self.get(r.session, "notice_dismissed"):
            notice = self.dialog(NOTICE_TEXT, "/notice/dismiss", "Dismiss", back=r.path)
        header = f"""
<div class="topbar">
  <span class="brand">MemberOps <span class="env">SANDBOX</span></span>
  <nav class="nav"><a href="/members">Members</a><a href="/reports">Reports</a></nav>
  <span class="spacer"></span>
  <a role="button" data-test="alerts-toggle" class="icon-button" href="/alerts">
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1a4 4 0 0 0-4 4v3l-1.5 2.5h11L12 8V5a4 4 0 0 0-4-4zm-1.5 11a1.5 1.5 0 0 0 3 0z" fill="currentColor"/></svg>
  </a>
  <span class="user">{html.escape(self.get(r.session, "user") or '')}</span>
  <a class="small" href="/logout">Sign out</a>
</div>"""
        if not chrome:
            header = '<div class="topbar"><span class="brand">MemberOps <span class="env">SANDBOX</span></span></div>'
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)} - MemberOps Sandbox</title>
<style>{CSS}</style></head><body>{header}<main class="content">{body}</main>{notice}
<div class="footer">MemberOps Sandbox is a fictional training application. No real members, accounts or credentials.</div>
</body></html>"""


class Redirect:
    def __init__(self, location: str):
        self.location = location


class Request:
    def __init__(self, method: str, path: str, query: dict, form: dict, session: Session):
        self.method, self.path, self.query, self.form, self.session = method, path, query, form, session


CSS = """
body{margin:0;font:13px/1.4 Verdana,Arial,sans-serif;background:#eef1f5;color:#1d2733}
.topbar{display:flex;align-items:center;gap:14px;background:#1f3a5f;color:#fff;padding:6px 14px}
.brand{font-weight:bold;letter-spacing:.3px}.env{font-size:10px;background:#c9a227;color:#1d2733;padding:1px 5px;border-radius:2px;margin-left:4px}
.nav a{color:#dbe6f3;margin-right:12px;text-decoration:none}.nav a:hover{text-decoration:underline}
.spacer{flex:1}.user{font-size:12px;color:#dbe6f3}.icon-button{display:inline-flex;width:24px;height:24px;align-items:center;justify-content:center;color:#fff;border:1px solid #4b6a8f;border-radius:3px}
.content{max-width:880px;margin:16px auto;background:#fff;border:1px solid #cfd6df;padding:14px 18px}
h1{font-size:18px;margin:0 0 8px}h2{font-size:14px;margin:0}
.muted{color:#6b7684}.error{color:#a3231b;background:#fbe9e7;border:1px solid #e6b3ae;padding:6px 8px}
.notice{color:#1f3a5f;background:#e8f0fa;border:1px solid #b9cdea;padding:6px 8px}
table.form-table td{padding:4px 6px}td.label{text-align:right;color:#4b5563}
table.kv th{text-align:left;padding:3px 10px 3px 0;color:#4b5563;font-weight:normal}table.kv td{padding:3px 0}
table.grid{border-collapse:collapse;width:100%;margin-top:6px}table.grid th,table.grid td{border:1px solid #d9dfe7;padding:4px 6px;text-align:left}
table.grid th{background:#f3f5f8}td.num{text-align:right}
.tabs{margin:8px 0}.tab{display:inline-block;padding:4px 10px;border:1px solid #cfd6df;border-bottom:none;background:#f3f5f8;color:#1f3a5f;text-decoration:none}
.tab.active{background:#fff;font-weight:bold}
.inline-form{margin:0 0 10px;padding:8px;border:1px solid #cfd6df}.inline-form label{margin-right:6px}
.section-head{display:flex;align-items:center;justify-content:space-between;margin-top:14px}
.btn{display:inline-block;padding:4px 10px;border:1px solid #7a8798;background:#f7f8fa;color:#1d2733;text-decoration:none;border-radius:2px}
.btn.primary{background:#1f3a5f;color:#fff;border-color:#1f3a5f}.choices .btn{margin-right:8px}
button{padding:4px 10px}button.danger{background:#a3231b;color:#fff;border:1px solid #7a1912}
.actions .btn{margin-left:8px}.summary li{margin:2px 0}.badge{background:#d9f2e3;color:#1c6b3a;padding:1px 6px;border-radius:2px}
.small{font-size:11px}
.modal-backdrop{position:fixed;inset:0;background:rgba(20,30,45,.45);display:flex;align-items:center;justify-content:center}
.modal{background:#fff;border:1px solid #7a8798;padding:14px 18px;max-width:420px;box-shadow:0 4px 18px rgba(0,0,0,.3)}
.footer{max-width:880px;margin:8px auto 20px;color:#6b7684;font-size:11px}
"""


# ---------- HTTP plumbing ----------

def make_handler(app: MemberOpsApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MemberOpsSandbox/1.0"

        def do_GET(self):
            self.dispatch("GET", {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else ""
            form = {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}
            self.dispatch("POST", form)

        def dispatch(self, method: str, form: dict):
            cookies = {}
            if self.headers.get("Cookie"):
                jar = SimpleCookie(self.headers["Cookie"])
                cookies = {key: morsel.value for key, morsel in jar.items()}
            status, headers, body = app.handle(method, self.path, cookies, form)
            self.send_response(int(status))
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):   # keep the demo terminal quiet
            pass

    return Handler


def serve(port: int = 8765, mode: str = "normal", host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Create the server (not yet serving); the caller runs serve_forever or a thread."""
    app = MemberOpsApp(mode=mode)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.app = app
    return server


def serve_in_thread(mode: str = "normal") -> tuple[ThreadingHTTPServer, str]:
    """For tests and demonstrations: an ephemeral port on localhost, served on a daemon thread."""
    server = serve(port=0, mode=mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"
