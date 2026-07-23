"""Tests for the session-security registry (tools/session_security.py).

Pen-test remediation: concurrent-login detection/notification/invalidation, mid-session
IP/User-Agent anomaly detection, idle timeout, and ownership-checked manual termination.
All behaviour is gated by SESSION_SECURITY_ENABLED, which tests enable via monkeypatch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools import session_security as ss


@pytest.fixture(autouse=True)
def _enabled_and_clean_store(monkeypatch):
    """Enable the feature and force a fresh in-memory store for every test."""
    monkeypatch.setattr(ss, "SESSION_SECURITY_ENABLED", True)
    monkeypatch.setattr(ss, "SESSION_SECURITY_MODE", "notify")
    monkeypatch.setattr(ss, "SESSION_SECURITY_BIND_IP", True)
    monkeypatch.setattr(ss, "SESSION_SECURITY_BIND_USER_AGENT", True)
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "notify")
    monkeypatch.setattr(ss, "SESSION_SECURITY_IDLE_TIMEOUT_MINUTES", 0)
    monkeypatch.setattr(ss, "SESSION_SECURITY_COGNITO_GLOBAL_SIGNOUT", False)
    monkeypatch.setattr(ss, "SAVE_LOGS_TO_CSV", False)
    monkeypatch.setattr(ss, "SAVE_LOGS_TO_DYNAMODB", False)
    ss.reset_store_for_tests()
    yield
    ss.reset_store_for_tests()


###
# register_session: disabled / basic behaviour
###


def test_register_session_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_ENABLED", False)
    notice = ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    assert notice is None
    assert ss.list_sessions("alice") == []


def test_register_session_creates_active_record():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    records = ss.list_sessions("alice")
    assert len(records) == 1
    assert records[0].session_hash == "sess-1"
    assert records[0].status == "active"
    assert records[0].ip_address == "1.1.1.1"


def test_register_session_reload_refreshes_same_record_without_duplicate():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    assert len(ss.list_sessions("alice")) == 1


###
# Concurrent-login detection: notify / invalidate / both
###


def test_concurrent_login_notify_mode_flags_sibling_without_terminating(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_MODE", "notify")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    records = {r.session_hash: r for r in ss.list_sessions("alice")}
    assert records["sess-1"].status == "active"
    assert records["sess-2"].status == "active"

    # The older session should have a pending notice, surfaced via heartbeat.
    result = ss.heartbeat("sess-1", "1.1.1.1", "UA1")
    assert result.status == "active"
    assert result.notice is not None
    assert "new sign-in" in result.notice.lower()


def test_concurrent_login_invalidate_mode_terminates_sibling(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_MODE", "invalidate")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    result = ss.heartbeat("sess-1", "1.1.1.1", "UA1")
    assert result.status == "terminated"
    assert result.notice is not None

    # The new session itself remains active.
    result_new = ss.heartbeat("sess-2", "2.2.2.2", "UA2")
    assert result_new.status == "active"


def test_concurrent_login_both_mode_terminates_and_notifies(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_MODE", "both")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    result = ss.heartbeat("sess-1", "1.1.1.1", "UA1")
    assert result.status == "terminated"
    assert result.notice is not None


def test_different_usernames_do_not_trigger_concurrent_login_logic():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("bob", "sess-2", "2.2.2.2", "UA2")

    alice_records = ss.list_sessions("alice")
    bob_records = ss.list_sessions("bob")
    assert len(alice_records) == 1
    assert alice_records[0].status == "active"
    assert len(bob_records) == 1
    assert bob_records[0].status == "active"


def test_concurrent_login_calls_cognito_global_signout_when_configured(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_MODE", "invalidate")
    monkeypatch.setattr(ss, "SESSION_SECURITY_COGNITO_GLOBAL_SIGNOUT", True)
    monkeypatch.setattr(ss, "AWS_USER_POOL_ID", "pool-123")

    called = {}

    def fake_cognito_global_sign_out(username):
        called["username"] = username

    monkeypatch.setattr(ss, "_cognito_global_sign_out", fake_cognito_global_sign_out)

    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    assert called.get("username") == "alice"


###
# Anomaly detection (IP / User-Agent drift)
###


def test_heartbeat_notify_on_ip_change(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "notify")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    result = ss.heartbeat("sess-1", "9.9.9.9", "UA1")
    assert result.status == "active"
    assert result.notice is not None
    assert "connection details" in result.notice.lower()


def test_heartbeat_terminate_on_ip_change(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "terminate")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    result = ss.heartbeat("sess-1", "9.9.9.9", "UA1")
    assert result.status == "terminated"
    assert result.notice is not None


def test_heartbeat_log_only_does_not_change_status(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "log_only")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    result = ss.heartbeat("sess-1", "9.9.9.9", "UA1")
    assert result.status == "active"
    assert result.notice is None


def test_heartbeat_no_anomaly_when_ip_binding_disabled(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_BIND_IP", False)
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "terminate")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    result = ss.heartbeat("sess-1", "9.9.9.9", "UA1")
    assert result.status == "active"


def test_heartbeat_notify_on_user_agent_change(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_ANOMALY_ACTION", "notify")
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA-original")

    result = ss.heartbeat("sess-1", "1.1.1.1", "UA-different")
    assert result.status == "active"
    assert result.notice is not None


###
# Idle timeout
###


def test_heartbeat_idle_timeout_terminates_session(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_IDLE_TIMEOUT_MINUTES", 10)
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    # Simulate the session having been idle for longer than the timeout.
    store = ss._get_store()
    record = store.get("sess-1")
    record.last_seen_time = (
        datetime.now(timezone.utc) - timedelta(minutes=15)
    ).isoformat(timespec="seconds")
    store.put(record)

    result = ss.heartbeat("sess-1", "1.1.1.1", "UA1")
    assert result.status == "terminated"
    assert "inactivity" in result.notice.lower()


def test_heartbeat_within_idle_window_stays_active(monkeypatch):
    monkeypatch.setattr(ss, "SESSION_SECURITY_IDLE_TIMEOUT_MINUTES", 10)
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    result = ss.heartbeat("sess-1", "1.1.1.1", "UA1")
    assert result.status == "active"


def test_heartbeat_unknown_session_hash_is_active_noop():
    result = ss.heartbeat("does-not-exist", "1.1.1.1", "UA1")
    assert result.status == "active"
    assert result.notice is None


###
# Manual, ownership-checked termination
###


def test_terminate_sessions_ends_owned_sibling():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    count = ss.terminate_sessions("alice", ["sess-2"], actor_session_hash="sess-1")
    assert count == 1

    records = {r.session_hash: r for r in ss.list_sessions("alice")}
    assert records["sess-2"].status == "terminated"
    assert records["sess-1"].status == "active"


def test_terminate_sessions_refuses_to_terminate_actors_own_session():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")

    count = ss.terminate_sessions("alice", ["sess-1"], actor_session_hash="sess-1")
    assert count == 0
    assert ss.list_sessions("alice")[0].status == "active"


def test_terminate_sessions_cannot_terminate_another_users_session():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("bob", "sess-2", "2.2.2.2", "UA2")

    # Ownership check: "alice" cannot terminate "bob"'s session even by hash.
    count = ss.terminate_sessions("alice", ["sess-2"], actor_session_hash="sess-1")
    assert count == 0
    assert ss.list_sessions("bob")[0].status == "active"


def test_terminate_sessions_noop_when_disabled(monkeypatch):
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")
    monkeypatch.setattr(ss, "SESSION_SECURITY_ENABLED", False)

    count = ss.terminate_sessions("alice", ["sess-2"], actor_session_hash="sess-1")
    assert count == 0


def test_terminate_sessions_already_terminated_is_noop():
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    ss.register_session("alice", "sess-2", "2.2.2.2", "UA2")

    first = ss.terminate_sessions("alice", ["sess-2"], actor_session_hash="sess-1")
    second = ss.terminate_sessions("alice", ["sess-2"], actor_session_hash="sess-1")
    assert first == 1
    assert second == 0


###
# list_sessions
###


def test_list_sessions_empty_for_unknown_user():
    assert ss.list_sessions("nobody") == []


def test_list_sessions_disabled_returns_empty(monkeypatch):
    ss.register_session("alice", "sess-1", "1.1.1.1", "UA1")
    monkeypatch.setattr(ss, "SESSION_SECURITY_ENABLED", False)
    assert ss.list_sessions("alice") == []


###
# In-memory store directly
###


def test_in_memory_store_list_for_user_isolates_by_username():
    store = ss.InMemorySessionStore()
    store.put(ss.SessionRecord(username="alice", session_hash="a1"))
    store.put(ss.SessionRecord(username="bob", session_hash="b1"))
    assert [r.session_hash for r in store.list_for_user("alice")] == ["a1"]


def test_in_memory_store_delete_removes_record():
    store = ss.InMemorySessionStore()
    store.put(ss.SessionRecord(username="alice", session_hash="a1"))
    store.delete("a1")
    assert store.get("a1") is None


###
# Shared access-log sink
###


def test_log_event_writes_through_shared_access_logger(monkeypatch):
    """Security events go to the same access-log helper (same CSV / DynamoDB table)."""
    calls = []

    def fake_log_platform_access(session_hash, host_name, **kwargs):
        calls.append((session_hash, host_name, kwargs))

    monkeypatch.setattr(ss, "SAVE_LOGS_TO_CSV", True)
    monkeypatch.setattr(ss, "SAVE_LOGS_TO_DYNAMODB", False)
    monkeypatch.setattr(
        "tools.gradio_platform.log_platform_access", fake_log_platform_access
    )

    ss._log_event(
        session_hash="sess-1",
        username="alice",
        event="login",
        ip_address="1.1.1.1",
        user_agent="UA1",
        status="active",
        details="unit-test",
    )

    assert len(calls) == 1
    session_hash, _host, kwargs = calls[0]
    assert session_hash == "sess-1"
    assert kwargs["username"] == "alice"
    assert kwargs["event"] == "login"
    assert kwargs["ip_address"] == "1.1.1.1"


def test_session_security_log_defaults_share_access_log_sink():
    from tools import config

    assert config.SESSION_SECURITY_LOG_FILE_NAME == config.LOG_FILE_NAME
    assert (
        config.SESSION_SECURITY_DYNAMODB_LOG_TABLE_NAME
        == config.ACCESS_LOG_DYNAMODB_TABLE_NAME
    )
    assert "event" in config.ACCESS_LOG_UNIFIED_HEADERS
    assert len(config.ACCESS_LOG_UNIFIED_HEADERS) >= 8
