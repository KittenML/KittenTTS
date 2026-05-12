"""Small, dependency-free analytics client for KittenTTS SDK events."""

import json
import os
import platform as platform_module
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib import request

ANALYTICS_ENDPOINT = "https://kittentts-analytics.dewana-sl.workers.dev/v1/track"
SDK_TYPE = "python"
DEFAULT_TIMEOUT_SECONDS = 3.0

_MODEL_VERSION_RE = re.compile(r"^(?P<model>.+?)-(?P<version>\d+(?:\.\d+)*(?:-[A-Za-z0-9]+)*)$")


def analytics_enabled(value=True) -> bool:
    if value is False:
        return False
    env_value = os.environ.get("KITTENTTS_ANALYTICS")
    if env_value and env_value.strip().lower() in {"0", "false", "off", "no"}:
        return False
    return True


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


def error_code(error: BaseException) -> str:
    name = error.__class__.__name__
    words = re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()
    return words or "UNKNOWN_ERROR"


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
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        post_json: Optional[Callable[[str, Dict[str, str], float], None]] = None,
        async_delivery: bool = True,
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
        self._anonymous_id = None

    @property
    def anonymous_id(self) -> str:
        if not self._anonymous_id:
            self._anonymous_id = load_or_create_anonymous_id(self._anonymous_id_path)
        return self._anonymous_id

    def track_generation(
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

        if self._async_delivery:
            thread = threading.Thread(target=self._send, args=(payload,), daemon=True)
            thread.start()
        else:
            self._send(payload)

    def _send(self, payload: Dict[str, str]) -> None:
        try:
            self._post_json(self.endpoint, payload, self.timeout_seconds)
        except Exception:
            return


def post_json_request(endpoint: str, payload: Dict[str, str], timeout_seconds: float) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        response.read()


def default_anonymous_id_path() -> Path:
    configured_home = os.environ.get("KITTENTTS_ANALYTICS_HOME")
    if configured_home:
        return Path(configured_home).expanduser() / "anonymous_id"
    return Path.home() / ".kittentts" / "analytics_id"


def load_or_create_anonymous_id(path: Path) -> str:
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if is_uuid(existing):
            return existing
    except OSError:
        pass

    anonymous_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(anonymous_id, encoding="utf-8")
    except OSError:
        pass
    return anonymous_id


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (TypeError, ValueError):
        return False
    return True
