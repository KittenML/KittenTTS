import io
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from kittentts.analytics import (
    FIRST_RUN_NOTICE,
    STALE_TEMP_FILE_AGE_SECONDS,
    AnalyticsClient,
    AnalyticsTransportError,
    _emit_first_run_notice,
    error_code,
    parse_model_name,
    post_json_request,
)
from kittentts.get_model import KittenTTS


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "DO_NOT_TRACK": "",
            "HF_HUB_DISABLE_TELEMETRY": "",
            "HF_HUB_OFFLINE": "",
            "KITTENTTS_ANALYTICS": "",
            "KITTENTTS_OFFLINE": "",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        notice = patch("kittentts.analytics._emit_first_run_notice")
        self.notice = notice.start()
        self.addCleanup(notice.stop)

    def make_client(self, post_json, enabled=True, anonymous_id_path=None, **kwargs):
        if anonymous_id_path is None:
            tempdir = tempfile.TemporaryDirectory()
            self.addCleanup(tempdir.cleanup)
            anonymous_id_path = Path(tempdir.name) / "analytics_id"
        return AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            enabled=enabled,
            anonymous_id_path=anonymous_id_path,
            post_json=post_json,
            async_delivery=False,
            **kwargs,
        )

    def test_disabled_analytics_sends_no_request(self):
        calls = []
        client = self.make_client(lambda *args: calls.append(args), enabled=False)

        client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(calls, [])

    def test_success_event_contains_required_fields(self):
        calls = []
        client = self.make_client(lambda endpoint, payload, timeout: calls.append((endpoint, payload, timeout)))

        client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(len(calls), 1)
        endpoint, payload, timeout = calls[0]
        self.assertEqual(endpoint, "https://kittenmlanalytics.com/v1/track")
        self.assertEqual(timeout, 3.0)
        for key in [
            "anonymous_id",
            "client_event_id",
            "timestamp",
            "sdk_version",
            "sdk_type",
            "platform",
            "runtime_version",
            "selected_model",
            "model_version",
            "selected_voice",
            "generation",
            "asset_source",
        ]:
            self.assertIn(key, payload)
            self.assertTrue(payload[key])
        self.assertNotIn("sdk_error_code", payload)
        self.assertNotIn("ip_address", payload)
        self.assertNotIn("ip_location", payload)

    def test_failure_event_includes_error_code(self):
        calls = []
        client = self.make_client(lambda endpoint, payload, timeout: calls.append(payload))

        client.track_generation(
            selected_voice="Jasper",
            generation="wav",
            sdk_error_code=error_code(ValueError("bad voice")),
        )

        self.assertEqual(calls[0]["sdk_error_code"], "VALUE_ERROR")

    def test_network_error_does_not_raise(self):
        def failing_post(endpoint, payload, timeout):
            raise TimeoutError("timed out")

        client = self.make_client(failing_post)
        client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(len(list(client._pending_dir.glob("*.json"))), 1)

    def test_non_retryable_http_error_is_not_kept(self):
        def rejected_post(endpoint, payload, timeout):
            raise AnalyticsTransportError("invalid event", retryable=False)

        client = self.make_client(rejected_post)
        client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(list(client._pending_dir.glob("*.json")), [])

    def test_payload_error_does_not_raise(self):
        client = self.make_client(lambda *args: None)

        with patch("kittentts.analytics.uuid.uuid4", side_effect=RuntimeError("uuid failed")):
            client.track_generation(selected_voice="Jasper", generation="wav")

    def test_thread_start_error_does_not_raise(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        client = AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            anonymous_id_path=Path(tempdir.name) / "analytics_id",
            post_json=lambda *args: None,
            async_delivery=True,
        )

        with patch("kittentts.analytics.threading.Thread") as thread_class:
            thread_class.return_value.start.side_effect = RuntimeError("thread failed")
            client.track_generation(selected_voice="Jasper", generation="wav")

    def test_async_delivery_uses_daemon_thread(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        client = AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            anonymous_id_path=Path(tempdir.name) / "analytics_id",
            post_json=lambda *args: None,
            async_delivery=True,
        )

        with patch("kittentts.analytics.threading.Thread") as thread_class:
            client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertTrue(thread_class.call_args.kwargs["daemon"])
        thread_class.return_value.start.assert_called_once()

    def test_event_is_persisted_before_background_delivery_starts(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"
        client = AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            anonymous_id_path=path,
            post_json=lambda *args: None,
            async_delivery=True,
        )

        with patch("kittentts.analytics.threading.Thread") as thread_class:
            client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(len(list((path.parent / "analytics_pending").glob("*.json"))), 1)
        thread_class.return_value.start.assert_called_once()

    def test_explicit_offline_mode_queues_then_later_flushes(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"
        calls = []

        with patch.dict(os.environ, {"KITTENTTS_OFFLINE": "1"}):
            offline_client = self.make_client(
                lambda endpoint, payload, timeout: calls.append(payload),
                anonymous_id_path=path,
            )
            offline_client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(calls, [])
        self.assertEqual(len(list((path.parent / "analytics_pending").glob("*.json"))), 1)

        self.make_client(
            lambda endpoint, payload, timeout: calls.append(payload),
            anonymous_id_path=path,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(list((path.parent / "analytics_pending").glob("*.json")), [])

    def test_backlog_larger_than_flush_batch_drains_fully(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        with patch.dict(os.environ, {"KITTENTTS_OFFLINE": "1"}):
            offline_client = self.make_client(lambda *args: None, anonymous_id_path=path)
            for _ in range(5):
                offline_client.track_generation(selected_voice="Jasper", generation="wav")

        calls = []
        self.make_client(
            lambda endpoint, payload, timeout: calls.append(payload),
            anonymous_id_path=path,
            max_flush_events=2,
        )

        self.assertEqual(len(calls), 5)
        self.assertEqual(list((path.parent / "analytics_pending").glob("*.json")), [])

    def test_flush_stops_on_retryable_failure_and_keeps_events(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        with patch.dict(os.environ, {"KITTENTTS_OFFLINE": "1"}):
            offline_client = self.make_client(lambda *args: None, anonymous_id_path=path)
            for _ in range(3):
                offline_client.track_generation(selected_voice="Jasper", generation="wav")

        delivered = []

        def rate_limited_after_first(endpoint, payload, timeout):
            if delivered:
                raise AnalyticsTransportError("analytics HTTP 429", retryable=True)
            delivered.append(payload)

        self.make_client(rate_limited_after_first, anonymous_id_path=path)

        self.assertEqual(len(delivered), 1)
        self.assertEqual(len(list((path.parent / "analytics_pending").glob("*.json"))), 2)

    def test_stale_temporary_files_are_removed(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"
        pending_dir = path.parent / "analytics_pending"
        pending_dir.mkdir(parents=True)
        stale = pending_dir / ".dead-event.abc123.tmp"
        fresh = pending_dir / ".live-event.def456.tmp"
        stale.write_text("{}", encoding="utf-8")
        fresh.write_text("{}", encoding="utf-8")
        old = time.time() - STALE_TEMP_FILE_AGE_SECONDS - 60
        os.utime(stale, (old, old))

        self.make_client(lambda *args: None, anonymous_id_path=path)

        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())

    def test_first_run_notice_is_printed_once(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        first = self.make_client(lambda *args: None, anonymous_id_path=path)
        first.track_generation(selected_voice="Jasper", generation="wav")
        self.assertEqual(self.notice.call_count, 1)

        second = self.make_client(lambda *args: None, anonymous_id_path=path)
        second.track_generation(selected_voice="Jasper", generation="wav")
        self.assertEqual(self.notice.call_count, 1)

    def test_first_run_notice_is_not_printed_when_disabled(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        client = self.make_client(lambda *args: None, enabled=False, anonymous_id_path=path)
        client.track_generation(selected_voice="Jasper", generation="wav")

        self.notice.assert_not_called()
        self.assertFalse(path.exists())

    def test_first_run_notice_writes_opt_out_details_to_stderr(self):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            _emit_first_run_notice()

        self.assertEqual(stderr.getvalue().strip(), FIRST_RUN_NOTICE)
        self.assertIn("KITTENTTS_ANALYTICS=0", FIRST_RUN_NOTICE)

    def test_pending_queue_is_bounded(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        with patch.dict(os.environ, {"KITTENTTS_OFFLINE": "1"}):
            client = self.make_client(
                lambda *args: None,
                anonymous_id_path=path,
                max_pending_events=2,
            )
            for _ in range(3):
                client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertEqual(len(list((path.parent / "analytics_pending").glob("*.json"))), 2)

    def test_opt_out_clears_unsent_events(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "analytics_id"

        with patch.dict(os.environ, {"KITTENTTS_OFFLINE": "1"}):
            client = self.make_client(lambda *args: None, anonymous_id_path=path)
            client.track_generation(selected_voice="Jasper", generation="wav")

        self.make_client(lambda *args: None, enabled=False, anonymous_id_path=path)

        self.assertEqual(list((path.parent / "analytics_pending").glob("*.json")), [])

    def test_ecosystem_telemetry_opt_outs_are_honored(self):
        for variable in ["HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"]:
            with self.subTest(variable=variable), patch.dict(os.environ, {variable: "1"}):
                calls = []
                client = self.make_client(lambda *args: calls.append(args))
                client.track_generation(selected_voice="Jasper", generation="wav")
                self.assertEqual(calls, [])

    def test_post_request_uses_sdk_user_agent(self):
        captured = []

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return b"{}"

        def fake_urlopen(req, timeout):
            captured.append((req, timeout))
            return DummyResponse()

        payload = {"sdk_version": "0.8.1"}

        with patch("kittentts.analytics.request.urlopen", fake_urlopen):
            post_json_request("https://example.com/v1/track", payload, 3.0)

        req, timeout = captured[0]
        self.assertEqual(timeout, 3.0)
        self.assertEqual(req.get_header("User-agent"), "KittenTTS-Python/0.8.1")

    def test_model_metadata_parses_variant_version(self):
        self.assertEqual(
            parse_model_name("KittenML/kitten-tts-nano-0.8-int8"),
            {"selected_model": "kitten-tts-nano", "model_version": "0.8-int8"},
        )

    def test_generate_tracks_success(self):
        model = KittenTTS.__new__(KittenTTS)
        model.model = DummyModel()
        model.analytics = RecordingAnalytics()

        self.assertEqual(model.generate("hello", voice="Jasper"), "audio")
        self.assertEqual(
            model.analytics.events,
            [{"selected_voice": "Jasper", "generation": "wav", "sdk_error_code": None}],
        )

    def test_generate_tracks_failure_and_reraises(self):
        model = KittenTTS.__new__(KittenTTS)
        model.model = FailingModel()
        model.analytics = RecordingAnalytics()

        with self.assertRaises(ValueError):
            model.generate("hello", voice="Jasper")

        self.assertEqual(
            model.analytics.events,
            [{"selected_voice": "Jasper", "generation": "wav", "sdk_error_code": "VALUE_ERROR"}],
        )

    def test_generate_ignores_analytics_failure(self):
        model = KittenTTS.__new__(KittenTTS)
        model.model = DummyModel()
        model.analytics = FailingAnalytics()

        self.assertEqual(model.generate("hello", voice="Jasper"), "audio")

    def test_generate_stream_tracks_one_success_event(self):
        model = KittenTTS.__new__(KittenTTS)
        model.model = DummyStreamModel()
        model.analytics = RecordingAnalytics()

        self.assertEqual(list(model.generate_stream("hello", voice="Jasper")), ["chunk-1", "chunk-2"])
        self.assertEqual(
            model.analytics.events,
            [{"selected_voice": "Jasper", "generation": "stream", "sdk_error_code": None}],
        )

    def test_generate_stream_tracks_failure_and_reraises(self):
        model = KittenTTS.__new__(KittenTTS)
        model.model = FailingStreamModel()
        model.analytics = RecordingAnalytics()

        with self.assertRaises(ValueError):
            list(model.generate_stream("hello", voice="Jasper"))

        self.assertEqual(
            model.analytics.events,
            [{"selected_voice": "Jasper", "generation": "stream", "sdk_error_code": "VALUE_ERROR"}],
        )

    def test_anonymous_id_is_stable_across_clients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "analytics_id"
            first = self.make_client(lambda *args: None, anonymous_id_path=path)
            second = self.make_client(lambda *args: None, anonymous_id_path=path)

            self.assertEqual(first.anonymous_id, second.anonymous_id)
            self.assertEqual(first.anonymous_id, path.read_text(encoding="utf-8"))

    def test_sdk_does_not_import_or_reference_posthog(self):
        package_root = Path(__file__).resolve().parents[1] / "kittentts"

        for source_path in package_root.rglob("*.py"):
            with self.subTest(path=source_path):
                source = source_path.read_text(encoding="utf-8").lower()
                self.assertNotIn("posthog", source)

class DummyModel:
    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        return "audio"


class DummyStreamModel:
    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        yield "chunk-1"
        yield "chunk-2"


class FailingModel:
    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        raise ValueError("bad voice")


class FailingStreamModel:
    def generate_stream(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        raise ValueError("bad stream")
        yield


class RecordingAnalytics:
    def __init__(self):
        self.events = []

    def track_generation(self, selected_voice, generation, sdk_error_code=None):
        self.events.append({
            "selected_voice": selected_voice,
            "generation": generation,
            "sdk_error_code": sdk_error_code,
        })


class FailingAnalytics:
    def track_generation(self, selected_voice, generation, sdk_error_code=None):
        raise RuntimeError("analytics failed")


if __name__ == "__main__":
    unittest.main()
