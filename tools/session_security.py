"""
Session security: concurrent-login detection, active-session management, and mid-session
IP/User-Agent anomaly detection.

Pen-test remediation. Everything here is inert unless SESSION_SECURITY_ENABLED=True
(tools/config.py). The registry is keyed on the app's already-resolved identity
(tools.gradio_platform.resolve_session_identity), so it works the same whether identity
comes from an ALB "authenticate-cognito" header, Gradio-native Cognito login, or (inertly,
since every "session" is then already unique) no auth at all.

What this can and cannot do:
  - Can: track sessions per username with IP/User-Agent/login/last-seen metadata; detect
    concurrent logins for the same user and notify and/or soft-terminate the older session
    at the app layer; keep an account activity logbook; let a user list/terminate their own
    other sessions; detect IP/User-Agent drift mid-session and react.
  - Cannot: directly delete another browser's load-balancer session cookie (e.g. ALB's
    AWSELBAuthSessionCookie) - that store is owned by the load balancer/IdP, not this app.
    SESSION_SECURITY_COGNITO_GLOBAL_SIGNOUT is a best-effort, delayed (next token refresh)
    mitigation at the Cognito layer, not an instant kill.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from tools.config import (
    AWS_ACCESS_KEY,
    AWS_REGION,
    AWS_SECRET_KEY,
    AWS_USER_POOL_ID,
    COGNITO_AUTH,
    HOST_NAME,
    RUN_AWS_FUNCTIONS,
    SAVE_LOGS_TO_CSV,
    SAVE_LOGS_TO_DYNAMODB,
    SESSION_SECURITY_ANOMALY_ACTION,
    SESSION_SECURITY_BIND_IP,
    SESSION_SECURITY_BIND_USER_AGENT,
    SESSION_SECURITY_COGNITO_GLOBAL_SIGNOUT,
    SESSION_SECURITY_DYNAMODB_TABLE_NAME,
    SESSION_SECURITY_ENABLED,
    SESSION_SECURITY_IDLE_TIMEOUT_MINUTES,
    SESSION_SECURITY_MODE,
    SESSION_SECURITY_STORE_BACKEND,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionRecord:
    username: str
    session_hash: str
    ip_address: str = ""
    user_agent: str = ""
    login_time: str = field(default_factory=_now_iso)
    last_seen_time: str = field(default_factory=_now_iso)
    status: str = "active"  # active | terminated
    notice: Optional[str] = None

    def to_item(self) -> Dict[str, Any]:
        return {
            "session_hash": self.session_hash,
            "username": self.username,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "login_time": self.login_time,
            "last_seen_time": self.last_seen_time,
            "status": self.status,
            "notice": self.notice,
        }

    @classmethod
    def from_item(cls, item: Dict[str, Any]) -> "SessionRecord":
        return cls(
            username=item.get("username", ""),
            session_hash=item.get("session_hash", ""),
            ip_address=item.get("ip_address", "") or "",
            user_agent=item.get("user_agent", "") or "",
            login_time=item.get("login_time") or _now_iso(),
            last_seen_time=item.get("last_seen_time") or _now_iso(),
            status=item.get("status") or "active",
            notice=item.get("notice") or None,
        )


@dataclass
class HeartbeatResult:
    status: str
    notice: Optional[str] = None


###
# Storage backends
###


class SessionStore:
    """Minimal storage interface for session records."""

    def get(self, session_hash: str) -> Optional[SessionRecord]:
        raise NotImplementedError

    def put(self, record: SessionRecord) -> None:
        raise NotImplementedError

    def list_for_user(self, username: str) -> List[SessionRecord]:
        raise NotImplementedError

    def delete(self, session_hash: str) -> None:
        raise NotImplementedError


class InMemorySessionStore(SessionStore):
    """Default store. Per-process only: fine for a single instance/task, but concurrent
    logins landing on different worker processes/replicas will not see each other. Use
    the DynamoDB backend for multi-replica deployments."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, SessionRecord] = {}

    def get(self, session_hash: str) -> Optional[SessionRecord]:
        with self._lock:
            return self._records.get(session_hash)

    def put(self, record: SessionRecord) -> None:
        with self._lock:
            self._records[record.session_hash] = record

    def list_for_user(self, username: str) -> List[SessionRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.username == username]

    def delete(self, session_hash: str) -> None:
        with self._lock:
            self._records.pop(session_hash, None)


class DynamoDBSessionStore(SessionStore):
    """Shares session state across multiple app replicas/tasks via DynamoDB."""

    def __init__(self, table_name: str) -> None:
        self._table_name = table_name
        self._dynamodb = self._connect()
        self._table = self._ensure_table()

    @staticmethod
    def _connect():
        try:
            dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
            dynamodb.meta.client.list_tables()
            return dynamodb
        except Exception as exc:
            if RUN_AWS_FUNCTIONS and AWS_ACCESS_KEY and AWS_SECRET_KEY:
                return boto3.resource(
                    "dynamodb",
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=AWS_REGION,
                )
            raise RuntimeError(
                f"AWS credentials for DynamoDB session store not found: {exc}"
            ) from exc

    def _ensure_table(self):
        try:
            table = self._dynamodb.Table(self._table_name)
            table.load()
            return table
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            table = self._dynamodb.create_table(
                TableName=self._table_name,
                KeySchema=[{"AttributeName": "session_hash", "KeyType": "HASH"}],
                AttributeDefinitions=[
                    {"AttributeName": "session_hash", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.meta.client.get_waiter("table_exists").wait(
                TableName=self._table_name
            )
            return table

    def get(self, session_hash: str) -> Optional[SessionRecord]:
        response = self._table.get_item(Key={"session_hash": session_hash})
        item = response.get("Item")
        return SessionRecord.from_item(item) if item else None

    def put(self, record: SessionRecord) -> None:
        self._table.put_item(Item=record.to_item())

    def list_for_user(self, username: str) -> List[SessionRecord]:
        from boto3.dynamodb.conditions import Attr

        items: List[Dict[str, Any]] = []
        scan_kwargs: Dict[str, Any] = {
            "FilterExpression": Attr("username").eq(username)
        }
        while True:
            response = self._table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
        return [SessionRecord.from_item(item) for item in items]

    def delete(self, session_hash: str) -> None:
        self._table.delete_item(Key={"session_hash": session_hash})


_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def _get_store() -> SessionStore:
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is None:
            if (SESSION_SECURITY_STORE_BACKEND or "memory").lower() == "dynamodb":
                try:
                    _store = DynamoDBSessionStore(SESSION_SECURITY_DYNAMODB_TABLE_NAME)
                except Exception as exc:
                    print(
                        "Session security: could not initialise DynamoDB session store, "
                        f"falling back to in-memory ({exc})"
                    )
                    _store = InMemorySessionStore()
            else:
                _store = InMemorySessionStore()
    return _store


def reset_store_for_tests() -> None:
    """Test helper: force a fresh in-memory store on next access."""
    global _store
    _store = None


###
# Account activity logbook (shared access log CSV / DynamoDB table)
###


def _log_event(
    *,
    session_hash: str,
    username: str,
    event: str,
    ip_address: str = "",
    user_agent: str = "",
    status: str = "",
    details: str = "",
) -> None:
    """Append a security event to the existing access log (same file / DynamoDB table)."""
    if not SAVE_LOGS_TO_CSV and not SAVE_LOGS_TO_DYNAMODB:
        return
    try:
        # Local import avoids a circular import at module load
        # (gradio_platform does not import session_security).
        from tools.gradio_platform import log_platform_access

        log_platform_access(
            session_hash,
            HOST_NAME,
            username=username,
            event=event,
            ip_address=ip_address,
            user_agent=user_agent,
            status=status,
            details=details,
        )
    except OSError as exc:
        print(f"Session security: activity log write failed ({exc})")
    except Exception as exc:  # pragma: no cover - defensive, must never break the app
        print(f"Session security: could not write activity log ({exc})")


###
# Cognito / Gradio best-effort session invalidation
###


def _cognito_global_sign_out(username: str) -> None:
    """
    Best-effort, delayed mitigation: invalidates the user's Cognito refresh tokens so that
    an ALB "authenticate-cognito" session (or any other Cognito-token consumer) is forced to
    re-authenticate the next time it tries to refresh. This does NOT immediately revoke an
    already-issued, unexpired access token, so an already-live session can remain usable for
    up to its access-token lifetime. It also cannot be scoped to a single sibling session -
    it affects every token issued for that username, including ones issued moments earlier
    for the very login that triggered this call. For that reason this is opt-in and off by
    default; do not treat it as an instant, targeted session kill.
    """
    if (
        not SESSION_SECURITY_COGNITO_GLOBAL_SIGNOUT
        or not AWS_USER_POOL_ID
        or not username
    ):
        return
    try:
        client = boto3.client("cognito-idp", region_name=AWS_REGION)
        client.admin_user_global_sign_out(
            UserPoolId=AWS_USER_POOL_ID, Username=username
        )
    except (
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        BotoCoreError,
    ) as exc:
        print(
            "Session security: Cognito admin_user_global_sign_out failed "
            f"(check cognito-idp:AdminUserGlobalSignOut permission): {exc}"
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(
            f"Session security: unexpected error during Cognito global sign-out: {exc}"
        )


_gradio_patch_lock = threading.Lock()
_gradio_patch_applied = False
_tracked_blocks: Any = None


def _ensure_gradio_app_capture_patch() -> None:
    """
    Best-effort monkeypatch of gradio.routes.App.configure_app so we can later reach the
    live App instance's ``.tokens`` dict (Gradio's own auth-cookie -> username map), in
    order to purge a user's Gradio-native login tokens immediately on invalidate. Only
    relevant when COGNITO_AUTH=True (Gradio's own login form, as opposed to ALB-header
    auth, where Gradio never owns the session cookie in the first place). Wrapped in
    try/except so that if Gradio's internals change shape in a future version, this
    quietly stops working instead of breaking the app.
    """
    global _gradio_patch_applied
    if _gradio_patch_applied:
        return
    with _gradio_patch_lock:
        if _gradio_patch_applied:
            return
        try:
            import gradio.routes as gr_routes

            original_configure_app = gr_routes.App.configure_app

            def _patched_configure_app(self, blocks):
                original_configure_app(self, blocks)
                try:
                    blocks._session_security_app_instance = self
                except Exception:  # pragma: no cover - defensive
                    pass

            gr_routes.App.configure_app = _patched_configure_app
            _gradio_patch_applied = True
        except Exception as exc:  # pragma: no cover - defensive
            print(
                f"Session security: could not attach Gradio App-capture patch ({exc})"
            )


def register_gradio_blocks(blocks: Any) -> None:
    """Call once with the app's gr.Blocks instance to enable Gradio-native token purge."""
    global _tracked_blocks
    if not SESSION_SECURITY_ENABLED or not COGNITO_AUTH:
        return
    _ensure_gradio_app_capture_patch()
    _tracked_blocks = blocks


def _purge_gradio_tokens(username: str) -> None:
    if not COGNITO_AUTH or _tracked_blocks is None or not username:
        return
    app_instance = getattr(_tracked_blocks, "_session_security_app_instance", None)
    if app_instance is None:
        return
    try:
        tokens = getattr(app_instance, "tokens", None)
        if not isinstance(tokens, dict):
            return
        for token in list(tokens.keys()):
            if tokens.get(token) == username:
                del tokens[token]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Session security: could not purge Gradio auth tokens ({exc})")


###
# Anomaly detection
###


def _detect_anomaly(
    record: SessionRecord, ip_address: str, user_agent: str
) -> Optional[str]:
    changes = []
    if (
        SESSION_SECURITY_BIND_IP
        and record.ip_address
        and ip_address
        and record.ip_address != ip_address
    ):
        changes.append(f"IP address changed from {record.ip_address} to {ip_address}")
    if (
        SESSION_SECURITY_BIND_USER_AGENT
        and record.user_agent
        and user_agent
        and record.user_agent != user_agent
    ):
        changes.append("User-Agent changed")
    return "; ".join(changes) if changes else None


def _pop_notice(store: SessionStore, record: SessionRecord) -> Optional[str]:
    notice = record.notice
    record.notice = None
    store.put(record)
    return notice


###
# Public API
###


def register_session(
    username: str, session_hash: str, ip_address: str = "", user_agent: str = ""
) -> Optional[str]:
    """
    Register (or refresh) an active session for username/session_hash. Detects other
    concurrent active sessions for the same username and applies SESSION_SECURITY_MODE
    to them (notify and/or invalidate). Returns a pending notice for *this* session_hash,
    if one was left by an earlier call (e.g. this tab itself was flagged before reloading).
    """
    if not SESSION_SECURITY_ENABLED or not username or not session_hash:
        return None

    store = _get_store()
    now = _now_iso()

    existing = store.get(session_hash)
    if existing is not None and existing.username == username:
        existing.ip_address = existing.ip_address or ip_address
        existing.user_agent = existing.user_agent or user_agent
        existing.last_seen_time = now
        return _pop_notice(store, existing)

    siblings = [
        r
        for r in store.list_for_user(username)
        if r.session_hash != session_hash and r.status == "active"
    ]

    record = SessionRecord(
        username=username,
        session_hash=session_hash,
        ip_address=ip_address,
        user_agent=user_agent,
        login_time=now,
        last_seen_time=now,
        status="active",
        notice=None,
    )
    store.put(record)
    _log_event(
        session_hash=session_hash,
        username=username,
        event="login",
        ip_address=ip_address,
        user_agent=user_agent,
        status="active",
    )

    if siblings:
        mode = (SESSION_SECURITY_MODE or "notify").lower()
        do_notify = mode in ("notify", "both")
        do_invalidate = mode in ("invalidate", "both")
        details = (
            f"New sign-in detected from {ip_address or 'an unknown location'} at {now}."
        )

        for sibling in siblings:
            if do_invalidate:
                sibling.status = "terminated"
                sibling.notice = (
                    "This session was ended because a new sign-in for your account was "
                    "detected. If this wasn't you, please review your account activity."
                )
                event = "invalidated_by_new_login"
            elif do_notify:
                sibling.notice = (
                    f"A new sign-in for your account was detected from "
                    f"{ip_address or 'a different location'}. If this wasn't you, please "
                    "review your account activity."
                )
                event = "notified_of_new_login"
            else:
                event = None

            if event:
                _log_event(
                    session_hash=sibling.session_hash,
                    username=username,
                    event=event,
                    ip_address=sibling.ip_address,
                    user_agent=sibling.user_agent,
                    status=sibling.status,
                    details=details,
                )
            store.put(sibling)

        if do_invalidate:
            _cognito_global_sign_out(username)
            _purge_gradio_tokens(username)

    return None


def heartbeat(
    session_hash: str, ip_address: str = "", user_agent: str = ""
) -> HeartbeatResult:
    """
    Periodic client poll: updates idle/last-seen tracking, checks for IP/User-Agent drift,
    enforces an optional idle timeout, and surfaces any pending notice (new login elsewhere,
    anomaly, manual termination) to the calling tab.
    """
    if not SESSION_SECURITY_ENABLED or not session_hash:
        return HeartbeatResult(status="active", notice=None)

    store = _get_store()
    record = store.get(session_hash)
    if record is None:
        return HeartbeatResult(status="active", notice=None)

    if record.status == "terminated":
        return HeartbeatResult(status="terminated", notice=_pop_notice(store, record))

    anomaly = _detect_anomaly(record, ip_address, user_agent)
    if anomaly:
        action = (SESSION_SECURITY_ANOMALY_ACTION or "notify").lower()
        details = f"Property change detected: {anomaly}"
        if action == "terminate":
            record.status = "terminated"
            record.notice = (
                "This session was ended because we detected a change in your connection "
                "details (possible session hijacking). Please reload and sign in again."
            )
            _log_event(
                session_hash=session_hash,
                username=record.username,
                event="anomaly_terminated",
                ip_address=ip_address,
                user_agent=user_agent,
                status="terminated",
                details=details,
            )
            return HeartbeatResult(
                status="terminated", notice=_pop_notice(store, record)
            )
        if action == "notify":
            record.notice = (
                "We noticed a change in your connection details during this session. If "
                "this wasn't expected, please check your account activity."
            )
            _log_event(
                session_hash=session_hash,
                username=record.username,
                event="anomaly_notified",
                ip_address=ip_address,
                user_agent=user_agent,
                status=record.status,
                details=details,
            )
        else:
            _log_event(
                session_hash=session_hash,
                username=record.username,
                event="anomaly_logged",
                ip_address=ip_address,
                user_agent=user_agent,
                status=record.status,
                details=details,
            )

    if (
        SESSION_SECURITY_IDLE_TIMEOUT_MINUTES
        and SESSION_SECURITY_IDLE_TIMEOUT_MINUTES > 0
    ):
        try:
            last_seen_dt = datetime.fromisoformat(record.last_seen_time)
        except ValueError:
            last_seen_dt = datetime.now(timezone.utc)
        idle_minutes = (
            datetime.now(timezone.utc) - last_seen_dt
        ).total_seconds() / 60.0
        if idle_minutes >= SESSION_SECURITY_IDLE_TIMEOUT_MINUTES:
            record.status = "terminated"
            record.notice = (
                "This session ended due to inactivity. Please reload and sign in again."
            )
            _log_event(
                session_hash=session_hash,
                username=record.username,
                event="idle_timeout",
                ip_address=ip_address,
                user_agent=user_agent,
                status="terminated",
            )
            return HeartbeatResult(
                status="terminated", notice=_pop_notice(store, record)
            )

    record.last_seen_time = _now_iso()
    record.ip_address = record.ip_address or ip_address
    record.user_agent = record.user_agent or user_agent
    return HeartbeatResult(status="active", notice=_pop_notice(store, record))


def list_sessions(username: str) -> List[SessionRecord]:
    """List all known sessions (active and terminated) for a username, most recent first."""
    if not SESSION_SECURITY_ENABLED or not username:
        return []
    records = _get_store().list_for_user(username)
    records.sort(key=lambda r: r.last_seen_time, reverse=True)
    return records


def terminate_sessions(
    username: str, session_hashes: List[str], actor_session_hash: str = ""
) -> int:
    """
    Manually terminate a user's own other sessions at the app layer. Ownership-checked:
    only sessions belonging to `username` are affected, and the caller's own current
    session_hash (`actor_session_hash`) is always skipped (use the normal logout control
    to end the session you are currently acting from).
    """
    if not SESSION_SECURITY_ENABLED or not username or not session_hashes:
        return 0

    store = _get_store()
    terminated = 0
    for session_hash in session_hashes:
        if not session_hash or session_hash == actor_session_hash:
            continue
        record = store.get(session_hash)
        if (
            record is None
            or record.username != username
            or record.status == "terminated"
        ):
            continue
        record.status = "terminated"
        record.notice = (
            "This session was remotely ended by you from another tab/device."
        )
        store.put(record)
        _log_event(
            session_hash=session_hash,
            username=username,
            event="manual_terminate",
            ip_address=record.ip_address,
            user_agent=record.user_agent,
            status="terminated",
        )
        terminated += 1
    return terminated
