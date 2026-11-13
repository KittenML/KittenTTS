# Vendored from misaki 0.9.4 (misaki/espeak.py, Apache-2.0, https://github.com/hexgrad/misaki): EspeakFallback and
# EspeakG2P unchanged. Only the espeak-ng setup differs: misaki calls EspeakWrapper.set_data_path, which only
# phonemizer-fork has; kittenml depends on phonemizer, so the data path also goes through ESPEAK_DATA_PATH
# (read by espeak-ng itself), as kittenml's other ONNX backend does.
import os
import re
from typing import Tuple

import espeakng_loader
import phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(espeakng_loader.get_library_path())
os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
if hasattr(EspeakWrapper, "set_data_path"):          # phonemizer-fork
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

# EspeakFallback is used as a last resort for English
class EspeakFallback:
    E2M = sorted({
        'ʔˌn\u0329':'ʔn', 'ʔn\u0329':'ʔn',# 'ʔn':'tn', 'ʔ':'t',
        'a^ɪ':'I', 'a^ʊ':'W',
        'd^ʒ':'ʤ',
        'e^ɪ':'A', 'e':'A',
        't^ʃ':'ʧ',
        'ɔ^ɪ':'Y',
        'ə^l':'ᵊl',
        'ʲo':'jo', 'ʲə':'jə', 'ʲ':'',
        'ɚ':'əɹ',
        'r':'ɹ',
        'x':'k', 'ç':'k',
        'ɐ':'ə',
        'ɬ':'l',
        '\u0303':'',
    }.items(), key=lambda kv: -len(kv[0]))

    def __init__(self, british, version=None):
        self.british = british
        self.version = version
        self.backend = phonemizer.backend.EspeakBackend(
            language=f"en-{'gb' if british else 'us'}",
            preserve_punctuation=True, with_stress=True, tie='^'
        )

    def __call__(self, token):
        ps = self.backend.phonemize([token.text])
        if not ps:
            return None, None
        ps = ps[0].strip()
        for old, new in type(self).E2M:
            ps = ps.replace(old, new)
        ps = re.sub(r'(\S)\u0329', r'ᵊ\1', ps).replace(chr(809), '')
        if self.british:
            ps = ps.replace('e^ə', 'ɛː')
            ps = ps.replace('iə', 'ɪə')
            ps = ps.replace('ə^ʊ', 'Q')
        else:
            ps = ps.replace('o^ʊ', 'O')
            ps = ps.replace('ɜːɹ', 'ɜɹ')
            ps = ps.replace('ɜː', 'ɜɹ')
            ps = ps.replace('ɪə', 'iə')
            ps = ps.replace('ː', '')
        ps = ps.replace('o', 'ɔ') # for espeak < 1.52
        if self.version != '2.0':
            ps = ps.replace('ɾ', 'T').replace('ʔ', 't')
        return ps.replace('^', ''), 2

# EspeakG2P used for most non-English/CJK languages
class EspeakG2P:
    def __init__(self, language, version=None):
        self.language = language
        self.version = version
        self.backend = phonemizer.backend.EspeakBackend(
            language=language, preserve_punctuation=True, with_stress=True,
            tie='^', language_switch='remove-flags'
        )
        self.e2m = {
            'a^ɪ':'I', 'a^ʊ':'W',
            'd^z':'ʣ', 'd^ʒ':'ʤ',
            'e^ɪ':'A',
            'o^ʊ':'O', 'ə^ʊ':'Q',
            's^s':'S',
            't^s':'ʦ', 't^ʃ':'ʧ',
            'ɔ^ɪ':'Y',
        }
        if version == '2.0':
            self.e2m.update({
                'œ̃':'B', 'ɔ̃':'C', 'ɑ̃':'D', 'ɛ̃':'E',
                'ʊ̃':'V', 'ũ':'U', 'õ':'X', 'ɐ̃':'Z',
            })
        self.e2m = sorted(self.e2m.items())

    def __call__(self, text) -> Tuple[str, None]:
        # Angles to curly quotes
        text = text.replace('«', chr(8220)).replace('»', chr(8221))
        # Parentheses to angles
        text = text.replace('(', '«').replace(')', '»')
        ps = self.backend.phonemize([text])
        if not ps:
            return '', None
        ps = ps[0].strip()
        for old, new in self.e2m:
            ps = ps.replace(old, new)
        # Delete any remaining tie characters, hyphens (not sure what they mean)
        ps = ps.replace('^', '')
        if self.version == '2.0':
            ps = ps.replace(chr(809), '').replace(chr(810), '')
            ps = re.sub(r'(\S)\u0329', r'ᵊ\1', ps)
        else:
            ps = ps.replace('-', '')
        # Angles back to parentheses
        ps = ps.replace('«', '(').replace('»', ')')
        return ps, None
