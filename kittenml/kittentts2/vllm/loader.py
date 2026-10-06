"""vLLM plugin: selected KittenTTS weights, without an intermediate export."""

from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from ..weights import load_lm_weights


class KittenModelLoader(BaseModelLoader):
    def download_model(self, model_config):
        # KittenTTS resolves the selected files before constructing the engine.
        pass

    def load_weights(self, model, model_config):
        options = self.load_config.model_loader_extra_config
        state = load_lm_weights(options["repo_dir"], options["config"], options["weights"])
        loaded = model.load_weights((name, tensor) for name, tensor in state.items()
                                    if not name.startswith("spk_proj."))
        missing = set(dict(model.named_parameters())) - loaded
        if missing:
            raise ValueError(f"weights missing from the checkpoint: {sorted(missing)[:5]}")


def register():
    from vllm.model_executor.model_loader import register_model_loader
    register_model_loader("kittenml")(KittenModelLoader)
