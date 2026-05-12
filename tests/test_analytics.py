import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kittentts.analytics import AnalyticsClient, error_code, parse_model_name, post_json_request
from kittentts.get_model import KittenTTS


class AnalyticsTests(unittest.TestCase):
    def make_client(self, post_json, enabled=True, anonymous_id_path=None):
        return AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            enabled=enabled,
            anonymous_id_path=anonymous_id_path,
            post_json=post_json,
            async_delivery=False,
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
        self.assertEqual(endpoint, "https://kittentts-analytics.dewana-sl.workers.dev/v1/track")
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

    def test_async_delivery_uses_non_daemon_thread(self):
        client = AnalyticsClient(
            sdk_version="0.8.1",
            selected_model="kitten-tts-nano",
            model_version="0.8",
            asset_source="cache",
            post_json=lambda *args: None,
            async_delivery=True,
        )

        with patch("kittentts.analytics.threading.Thread") as thread_class:
            client.track_generation(selected_voice="Jasper", generation="wav")

        self.assertFalse(thread_class.call_args.kwargs["daemon"])
        thread_class.return_value.start.assert_called_once()

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


class FailingModel:
    def generate(self, text, voice="expr-voice-5-m", speed=1.0, clean_text=False):
        raise ValueError("bad voice")


class RecordingAnalytics:
    def __init__(self):
        self.events = []

    def track_generation(self, selected_voice, generation, sdk_error_code=None):
        self.events.append({
            "selected_voice": selected_voice,
            "generation": generation,
            "sdk_error_code": sdk_error_code,
        })


if __name__ == "__main__":
    unittest.main()
