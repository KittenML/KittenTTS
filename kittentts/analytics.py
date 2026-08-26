"""Small, dependency-free analytics client for KittenTTS SDK events."""

import json
import os
import platform as platform_module
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib import error, request

ANALYTICS_ENDPOINT = "https://kittenmlanalytics.com/v1/track"
SDK_TYPE = "python"
DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_PENDING_EVENTS = 1000
DEFAULT_MAX_PENDING_AGE_SECONDS = 30 * 24 * 60 * 60
DEFAULT_MAX_FLUSH_EVENTS = 20
STALE_TEMP_FILE_AGE_SECONDS = 60 * 60
FIRST_RUN_NOTICE = (
    "KittenTTS sends privacy-limited usage analytics (never your text or audio). "
    "Opt out with KITTENTTS_ANALYTICS=0 or KittenTTS(..., analytics=False). "
    "Details: https://github.com/KittenML/KittenTTS/blob/main/docs/analytics.md"
)

_FALSE_VALUES = {"0", "false", "off", "no"}
_TRUE_VALUES = {"1", "true", "on", "yes"}
_MODEL_VERSION_RE = re.compile(r"^(?P<model>.+?)-(?P<version>\d+(?:\.\d+)*(?:-[A-Za-z0-9]+)*)$")


def analytics_enabled(value=True) -> bool:
    if value is False:
        return False
    if _env_truthy("HF_HUB_DISABLE_TELEMETRY") or _env_truthy("DO_NOT_TRACK"):
        return False
    env_value = os.environ.get("KITTENTTS_ANALYTICS")
    if env_value and env_value.strip().lower() in _FALSE_VALUES:
        return False
    return True


def offline_mode() -> bool:
    return _env_truthy("KITTENTTS_OFFLINE") or _env_truthy("HF_HUB_OFFLINE")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    return bool(value and value.strip().lower() in _TRUE_VALUES)


def current_platform() -> str:
    system = platform_module.system().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return "unknown"


def runtime_version() -> str:
    return f"python {sys.version_info.major}.{sys.version_info.minor}"


def parse_model_name(model_name: str) -> Dict[str, str]:
    repo_name = str(model_name).rstrip("/").split("/")[-1] or str(model_name)
    match = _MODEL_VERSION_RE.match(repo_name)
    if not match:
        return {"selected_model": repo_name, "model_version": "unknown"}
    return {
        "selected_model": match.group("model"),
        "model_version": match.group("version"),
    }


def error_code(error_value: BaseException) -> str:
    name = error_value.__class__.__name__
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()
    return words or "UNKNOWN_ERROR"


class AnalyticsTransportError(Exception):
    """Analytics delivery error with retry guidance for the durable queue."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class AnalyticsClient:
    def __init__(
        self,
        sdk_version: str,
        selected_model: str,
        model_version: str,
        asset_source: str,
        enabled: bool = True,
        endpoint: str = ANALYTICS_ENDPOINT,
        anonymous_id_path: Optional[Path] = None,
        pending_dir: Optional[Path] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        post_json: Optional[Callable[[str, Dict[str, str], float], None]] = None,
        async_delivery: bool = True,
        max_pending_events: int = DEFAULT_MAX_PENDING_EVENTS,
        max_pending_age_seconds: float = DEFAULT_MAX_PENDING_AGE_SECONDS,
        max_flush_events: int = DEFAULT_MAX_FLUSH_EVENTS,
    ):
        self.sdk_version = sdk_version
        self.selected_model = selected_model
        self.model_version = model_version
        self.asset_source = asset_source
        self.enabled = analytics_enabled(enabled)
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._post_json = post_json or post_json_request
        self._async_delivery = async_delivery
        self._anonymous_id_path = anonymous_id_path or default_anonymous_id_path()
        self._pending_dir = pending_dir or self._anonymous_id_path.parent / "analytics_pending"
        self._max_pending_events = max(1, int(max_pending_events))
        self._max_pending_age_seconds = max(0.0, float(max_pending_age_seconds))
        self._max_flush_events = max(1, int(max_flush_events))
        self._anonymous_id = None
        self._flush_lock = threading.Lock()
        self._flush_thread = None

        if not self.enabled:
            self._clear_pending()
        else:
            self._trim_pending()
            if not offline_mode() and self._pending_files():
                self._schedule_flush()

    @property
    def anonymous_id(self) -> str:
        if not self._anonymous_id:
            first_run = not self._anonymous_id_path.exists()
            self._anonymous_id = load_or_create_anonymous_id(self._anonymous_id_path)
            if first_run and self.enabled:
                _emit_first_run_notice()
        return self._anonymous_id

    def track_generation(
        self,
        selected_voice: str,
        generation: str,
        sdk_error_code: Optional[str] = None,
    ) -> None:
        try:
            self._track_generation(selected_voice, generation, sdk_error_code=sdk_error_code)
        except Exception:
            return

    def _track_generation(
        self,
        selected_voice: str,
        generation: str,
        sdk_error_code: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "anonymous_id": self.anonymous_id,
            "client_event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sdk_version": self.sdk_version,
            "sdk_type": SDK_TYPE,
            "platform": current_platform(),
            "runtime_version": runtime_version(),
            "selected_model": self.selected_model,
            "model_version": self.model_version,
            "selected_voice": str(selected_voice),
            "generation": generation,
            "asset_source": self.asset_source,
        }
        if sdk_error_code:
            payload["sdk_error_code"] = sdk_error_code

        pending_path = self._persist_pending(payload)
        if offline_mode():
            return
        if pending_path:
            self._schedule_flush()
        else:
            self._schedule_transient(payload)

    def _schedule_flush(self) -> None:
        if self._async_delivery:
            if self._flush_thread and self._flush_thread.is_alive():
                return
            self._flush_thread = threading.Thread(target=self._flush_pending, daemon=True)
            self._flush_thread.start()
        else:
            self._flush_pending()

    def _schedule_transient(self, payload: Dict[str, str]) -> None:
        if self._async_delivery:
            thread = threading.Thread(target=self._deliver, args=(payload,), daemon=True)
            thread.start()
        else:
            self._deliver(payload)

    def _flush_pending(self) -> None:
        if not self._flush_lock.acquire(blocking=False):
            return
        try:
            processed = set()
            while not offline_mode():
                batch = [
                    path for path in self._pending_files() if path not in processed
                ][: self._max_flush_events]
                if not batch:
                    return

                for pending_path in batch:
                    processed.add(pending_path)
                    payload = self._read_pending(pending_path)
                    if payload is None:
                        self._unlink(pending_path)
                        continue

                    result = self._deliver(payload)
                    if result in {"sent", "drop"}:
                        self._unlink(pending_path)
                        continue
                    return
        finally:
            self._flush_lock.release()

    def _deliver(self, payload: Dict[str, str]) -> str:
        try:
            self._post_json(self.endpoint, payload, self.timeout_seconds)
            return "sent"
        except AnalyticsTransportError as exc:
            return "retry" if exc.retryable else "drop"
        except Exception:
            return "retry"

    def _persist_pending(self, payload: Dict[str, str]) -> Optional[Path]:
        temporary = None
        try:
            self._pending_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            event_id = payload["client_event_id"]
            path = self._pending_dir / f"{event_id}.json"
            temporary = self._pending_dir / f".{event_id}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            os.replace(str(temporary), str(path))
            self._trim_pending()
            return path
        except OSError:
            if temporary is not None:
                self._unlink(temporary)
            return None

    def _trim_pending(self) -> None:
        files = self._pending_files(remove_expired=True)
        overflow = len(files) - self._max_pending_events
        for path in files[: max(0, overflow)]:
            self._unlink(path)
        self._remove_stale_temporaries()

    def _remove_stale_temporaries(self) -> None:
        # A crash between temporary-file creation and os.replace leaves
        # ".<event>.<nonce>.tmp" files that the "*.json" queue never touches.
        try:
            candidates = list(self._pending_dir.glob(".*.tmp"))
        except OSError:
            return
        now = time.time()
        for path in candidates:
            try:
                if now - path.stat().st_mtime > STALE_TEMP_FILE_AGE_SECONDS:
                    path.unlink()
            except OSError:
                continue

    def _pending_files(self, remove_expired: bool = False) -> List[Path]:
        try:
            candidates = list(self._pending_dir.glob("*.json"))
        except OSError:
            return []

        now = time.time()
        files = []
        for path in candidates:
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if remove_expired and now - modified > self._max_pending_age_seconds:
                self._unlink(path)
            else:
                files.append((modified, path.name, path))
        files.sort()
        return [item[2] for item in files]

    @staticmethod
    def _read_pending(path: Path) -> Optional[Dict[str, str]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        return value

    def _clear_pending(self) -> None:
        for path in self._pending_files():
            self._unlink(path)

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass


def post_json_request(endpoint: str, payload: Dict[str, str], timeout_seconds: float) -> None:
    body = json.dumps(payload).encode("utf-8")
    sdk_version = str(payload.get("sdk_version") or "unknown").replace("\n", " ").replace("\r", " ")
    req = request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"KittenTTS-Python/{sdk_version}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            response.read()
    except error.HTTPError as exc:
        retryable = exc.code in {408, 425, 429} or exc.code >= 500
        raise AnalyticsTransportError(f"analytics HTTP {exc.code}", retryable=retryable) from exc
    except (error.URLError, TimeoutError, OSError) as exc:
        raise AnalyticsTransportError("analytics delivery failed", retryable=True) from exc


def _emit_first_run_notice() -> None:
    try:
        print(FIRST_RUN_NOTICE, file=sys.stderr)
    except Exception:
        pass


def default_anonymous_id_path() -> Path:
    configured_home = os.environ.get("KITTENTTS_ANALYTICS_HOME")
    if configured_home:
        return Path(configured_home).expanduser() / "anonymous_id"
    try:
        return Path.home() / ".kittentts" / "analytics_id"
    except RuntimeError:
        return Path(os.environ.get("TMPDIR", "/tmp")) / "kittentts" / "analytics_id"


def load_or_create_anonymous_id(path: Path) -> str:
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if is_uuid(existing):
            return existing
    except OSError:
        pass

    anonymous_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(anonymous_id)
    except OSError:
        pass
    return anonymous_id


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (TypeError, ValueError):
        return False
    return True
