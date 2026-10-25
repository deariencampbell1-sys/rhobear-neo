"""W1-02 metering — the owner workspace never meters.

install_id = 'deariencampbell1-sys' (the owner workspace) and any install
whose `plan` is empty skip BOTH the credit gate and the debit. Customers with
a plan are gated on balance <= 0 (and told so on the PR) and debited — that
customer gate is proven here without a database and in tests/test_jobs_pg.py
(TestAlreadyEscalatedPG.test_out_of_credits_records_parked_not_escalated)
against real Postgres.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.jev_review import JevAssessment
from src.neo_worker import OWNER_INSTALL_ID, meters_credits, run_neo

SHA = "a" * 40


def _cfg() -> Config:
    cfg = Config()
    cfg.database_url = ""
    cfg.deepseek_model = "deepseek-v4-flash"
    cfg.max_builder_rounds = 2
    cfg.auto_merge_default = False
    cfg.deepseek_timeout = 60
    return cfg


def _evidence(sha: str) -> dict:
    return {"ok": True, "head_sha": sha, "decided_sha": sha, "review_sha": sha,
            "review_bound_ok": True, "trusted_states": ["success"],
            "trusted_green": True, "failing_checks": 0, "rollup_green": True,
            "trusted_producer_ok": True, "review_sha_match": True,
            "unresolved_critical_high": 0, "coverage_complete": True}


def _inst(install_id: str, plan: str, balance: int) -> dict:
    return {"install_id": install_id, "org": install_id, "auto_merge": None,
            "plan": plan, "credits_balance": balance}


# ---------------------------------------------------------------------------
# meters_credits — who pays
# ---------------------------------------------------------------------------

class TestMetersCredits:
    def test_owner_workspace_never_meters(self):
        assert meters_credits(_inst(OWNER_INSTALL_ID, "pro", 0)) is False

    def test_owner_workspace_never_meters_even_with_balance(self):
        assert meters_credits(_inst(OWNER_INSTALL_ID, "enterprise", 999)) is False

    def test_empty_plan_never_meters(self):
        assert meters_credits(_inst("acme", "", 0)) is False

    def test_missing_plan_never_meters(self):
        assert meters_credits({"install_id": "acme"}) is False

    def test_customer_with_plan_meters(self):
        assert meters_credits(_inst("acme", "starter", 5)) is True


# ---------------------------------------------------------------------------
# run_neo — gate and debit behaviour, no database needed
# ---------------------------------------------------------------------------

def _run(inst: dict, verdict: str = "FIX-FORWARD"):
    state = MagicMock()
    state.install.return_value = inst
    state.already_acted.return_value = False
    state.builder_rounds.return_value = 0
    jobs = MagicMock()
    jobs.claim.return_value = True
    wake = {"repo": f"{inst['install_id']}/repo", "sha": SHA, "green": False,
            "context": "rhobear-reviews", "pr_number": 5, "job_id": 1}
    with patch("src.neo_worker.repo_token", return_value="tok"), \
         patch("src.neo_worker.resolve_pr", return_value=5), \
         patch("src.neo_worker._review_evidence", return_value=_evidence(SHA)), \
         patch("src.neo_worker._jev_for_pr",
               return_value=JevAssessment(False, reason="no key")), \
         patch("src.neo_worker._run_agent",
               return_value=({"input_tokens": 10, "output_tokens": 5},
                             verdict, None, None)) as agent, \
         patch("src.neo_worker._gh", return_value=(0, "ok")) as gh, \
         patch("src.neo_worker.ClaudeAgent"):
        run_neo(_cfg(), state, jobs, wake)
    return state, jobs, agent, gh


class TestOwnerWorkspaceNeverMeters:
    def test_owner_workspace_is_not_gated(self):
        state, jobs, agent, gh = _run(_inst(OWNER_INSTALL_ID, "pro", 0))
        jobs.park.assert_not_called()
        agent.assert_called_once()
        assert not any("out of credits" in str(c) for c in gh.call_args_list)

    def test_owner_workspace_is_not_debited(self):
        state, jobs, agent, gh = _run(_inst(OWNER_INSTALL_ID, "pro", 0))
        state.debit.assert_not_called()

    def test_empty_plan_is_not_gated_or_debited(self):
        state, jobs, agent, gh = _run(_inst("acme", "", 0))
        jobs.park.assert_not_called()
        agent.assert_called_once()
        state.debit.assert_not_called()


class TestCustomerStillMeters:
    def test_customer_with_plan_and_no_balance_is_gated_and_told(self):
        state, jobs, agent, gh = _run(_inst("acme", "starter", 0))
        jobs.park.assert_called_once()
        agent.assert_not_called()
        state.debit.assert_not_called()
        bodies = " ".join(str(c) for c in gh.call_args_list)
        assert "out of credits" in bodies, "the customer must be told on the PR"

    def test_customer_with_plan_is_debited(self):
        state, jobs, agent, gh = _run(_inst("acme", "starter", 10))
        state.debit.assert_called_once()
        jobs.park.assert_not_called()
