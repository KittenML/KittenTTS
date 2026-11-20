"""Explicit credentials must reach the config and the selected model's downloads."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from kittenml.get_model import KittenTTS
from kittenml.kittentts_legacy.model import KittenTTSOnnx
from kittenml.repo import resolve_repo


class HubTokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "config.json"

    def config_download(self, config):
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

        def download(**kwargs):
            # An unauthenticated request to a private repo fails before loading a model.
            if kwargs.get("token") is not mock.sentinel.hf_token:
                raise PermissionError("private repository requires the explicit token")
            return str(self.config_path)

        return mock.patch("kittenml.repo.hf_hub_download", side_effect=download)

    def test_explicit_token_loads_private_kitten2_config(self):
        backend = mock.Mock()
        module = types.ModuleType("kittenml.kittentts2")
        module.KittenTTS2 = backend
        with self.config_download({"type": "KITTEN2"}), \
             mock.patch.dict("sys.modules", {"kittenml.kittentts2": module}), \
             mock.patch("kittenml.get_model.snapshot_download", return_value=self.tmp.name) as snapshot:
            model = KittenTTS("private/model", hf_token=mock.sentinel.hf_token)
        self.assertIs(model, backend.return_value)
        self.assertIs(snapshot.call_args.kwargs["token"], mock.sentinel.hf_token)

    def test_explicit_token_loads_private_onnx_config_and_artifacts(self):
        config = {"type": "ONNX1", "model_file": "model.onnx", "voices": "voices.npz"}

        def download(**kwargs):
            if kwargs.get("token") is not mock.sentinel.hf_token:
                raise PermissionError("private artifact requires the explicit token")
            return str(Path(self.tmp.name) / kwargs["filename"])

        for constructor in (KittenTTS, KittenTTSOnnx):
            with self.subTest(constructor=constructor.__name__), \
                 self.config_download(config), \
                 mock.patch("kittenml.kittentts_legacy.model.hf_hub_download", side_effect=download) as fetch, \
                 mock.patch("kittenml.kittentts_legacy.onnx_model.KittenTTS_1_Onnx") as backend:
                model = constructor("private/model", hf_token=mock.sentinel.hf_token)
                self.assertIs(model.model, backend.return_value)
                self.assertEqual([call.kwargs["filename"] for call in fetch.call_args_list],
                                 ["model.onnx", "voices.npz"])

    def test_local_repository_does_not_download_config(self):
        self.config_path.write_text('{"type": "ONNX1"}', encoding="utf-8")
        with mock.patch("kittenml.repo.hf_hub_download") as download:
            local, repo_id, config = resolve_repo(self.tmp.name, hf_token=mock.sentinel.hf_token)
        self.assertEqual(local, self.tmp.name)
        self.assertIsNone(repo_id)
        self.assertEqual(config["type"], "ONNX1")
        download.assert_not_called()

    def test_omitted_token_keeps_hub_default_authentication(self):
        self.config_path.write_text('{"type": "ONNX1"}', encoding="utf-8")
        with mock.patch("kittenml.repo.hf_hub_download", return_value=str(self.config_path)) as download:
            resolve_repo("public/model")
        self.assertIsNone(download.call_args.kwargs.get("token"))


if __name__ == "__main__":
    unittest.main()
