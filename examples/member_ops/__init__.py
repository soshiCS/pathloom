"""MemberOps Sandbox: a fictional banking back-office application used as a controlled Pathloom target.

Everything is in memory and deterministic: fictional members, training credentials, no network
beyond the local socket, no persistence. Run it with `python -m examples.member_ops --port 8765`.
"""
from .app import MemberOpsApp, MODES, serve

__all__ = ["MemberOpsApp", "MODES", "serve"]
