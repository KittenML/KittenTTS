import unittest
from types import SimpleNamespace

import torch

from kittenml.kittentts2.speaker import attach_speaker_head


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = torch.nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, cache=None,
                past_key_values=None, cache_position=None):
        return inputs_embeds


class SpeakerCacheTest(unittest.TestCase):
    def test_speaker_embedding_is_only_injected_at_prefill(self):
        projection = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.LayerNorm(4))
        state = {"spk_proj." + k: v for k, v in projection.state_dict().items()}
        model = attach_speaker_head(TinyLM(), "unused", "cpu", torch.float32,
                                    spk_dim=2, state=state)
        speaker = torch.tensor([[0.5, -0.3]])
        model._pending_speaker_embedding = speaker
        ids = torch.tensor([[0, 3, 1]])
        original = model.embedding(ids)
        projected = model.spk_proj(speaker)

        for key in ("cache", "past_key_values"):
            for length in (0, 3):
                with self.subTest(key=key, cache_length=length):
                    cache = SimpleNamespace(get_seq_length=lambda: length)
                    result = model(input_ids=ids, **{key: cache})
                    if length == 0:
                        torch.testing.assert_close(result[:, 0], projected)
                    else:
                        torch.testing.assert_close(result, original)


if __name__ == "__main__":
    unittest.main()
