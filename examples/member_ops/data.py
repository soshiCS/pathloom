"""Fictional, deterministic data for the MemberOps Sandbox. Nothing here is real."""
from __future__ import annotations

TRAINING_USER = "operator"
TRAINING_PASSWORD = "training_only"   # a training credential for a fictional app; treated as sensitive by Pathloom

MEMBERS = {
    "1001": {"id": "1001", "name": "Alex Morgan", "phone": "5550101", "status": "Active", "restricted": False,
             "since": "2019-03-12", "branch": "Riverside",
             "accounts": [{"number": "CHK-4321", "type": "Checking", "balance": "2,450.00", "opened": "2019-03-12"},
                          {"number": "SAV-8765", "type": "Savings", "balance": "12,000.00", "opened": "2020-07-01"}]},
    "1002": {"id": "1002", "name": "Jordan Lee", "phone": "5550102", "status": "Active", "restricted": True,
             "since": "2016-11-30", "branch": "Harbor",
             "accounts": [{"number": "CHK-1188", "type": "Checking", "balance": "780.15", "opened": "2016-11-30"}]},
    "1003": {"id": "1003", "name": "Sam Patel", "phone": "5550103", "status": "Active", "restricted": False,
             "since": "2022-01-18", "branch": "Riverside",
             "accounts": [{"number": "SAV-2210", "type": "Savings", "balance": "3,300.00", "opened": "2022-01-18"}]},
}

ACCOUNT_TYPES = {"savings": "Savings", "checking": "Checking"}
MINIMUM_DEPOSIT = 25.0
