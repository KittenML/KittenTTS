from pathlib import Path
from urllib.request import urlretrieve

from . import _ephonemizer


DEFAULT_RULES_URL = (
    "https://raw.githubusercontent.com/espeak-ng/espeak-ng/"
    "59eb19938f12e30881c81d86ce4a7de25414c9f4/dictsource/en_rules"
)
DEFAULT_LIST_URL = (
    "https://raw.githubusercontent.com/espeak-ng/espeak-ng/"
    "59eb19938f12e30881c81d86ce4a7de25414c9f4/dictsource/en_list"
)


class EPhonemizerBackend:
    """Python wrapper around the same C++ EPhonemizer used by the Swift SDK."""

    def __init__(
        self,
        rules_path=None,
        list_path=None,
        cache_dir=None,
        dialect="en-us",
    ):
        self.dialect = dialect
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "kittentts" / "ephonemizer")
        self.rules_path = Path(rules_path) if rules_path else self.cache_dir / "en_rules"
        self.list_path = Path(list_path) if list_path else self.cache_dir / "en_list"

    def phonemize(self, texts):
        self._ensure_data_files()
        return [
            _ephonemizer.phonemize(
                text,
                str(self.rules_path),
                str(self.list_path),
                self.dialect,
            )
            for text in texts
        ]

    def _ensure_data_files(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.rules_path.exists():
            urlretrieve(DEFAULT_RULES_URL, self.rules_path)
        if not self.list_path.exists():
            urlretrieve(DEFAULT_LIST_URL, self.list_path)
