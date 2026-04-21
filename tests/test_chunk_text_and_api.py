"""
Regression tests for chunk_text punctuation preservation, clean_text API
default alignment, and version consistency.

Covers:
  - chunk_text preserving sentence-ending punctuation (.!?)
  - chunk_text handling long sentences (word-level splitting)
  - chunk_text handling text without punctuation (ensure_punctuation fallback)
  - chunk_text edge cases: empty text, single sentence, many sentences
  - KittenTTS public API clean_text default matches internal model default
  - generate_to_file exposes clean_text parameter
  - __version__ matches pyproject.toml and setup.py
  - Source audit: no residual re.split(r'[.!?]') pattern in chunk_text

All tests are lightweight (no model/ONNX/GPU needed).
"""

import ast
import inspect
import os
import re
import sys
import textwrap


# ── helpers ──────────────────────────────────────────────────────────

def _import_chunk_text():
    """Import chunk_text without triggering heavy ONNX / espeak deps."""
    src_path = os.path.join(
        os.path.dirname(__file__), "..", "kittentts", "onnx_model.py"
    )
    source = open(src_path).read()

    # Extract only the standalone functions we need (avoid module-level imports)
    tree = ast.parse(source)
    needed = {"ensure_punctuation", "chunk_text", "basic_english_tokenize"}
    fn_sources = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in needed:
                fn_sources.append(ast.get_source_segment(source, node))

    ns = {}
    for fn_src in fn_sources:
        exec(compile(fn_src, "<chunk_text>", "exec"), ns)
    return ns["chunk_text"], ns["ensure_punctuation"]


chunk_text, ensure_punctuation = _import_chunk_text()


# ── Tests: punctuation preservation ──────────────────────────────────


class TestChunkTextPunctuationPreservation:
    """chunk_text must keep original sentence-ending punctuation."""

    def test_period_preserved(self):
        chunks = chunk_text("Hello world. How are you.")
        # Both sentences should end with period (or at least with punctuation)
        for chunk in chunks:
            assert chunk[-1] in ".!?,;:", f"Chunk lost punctuation: {chunk!r}"
        # The first chunk should contain "world."
        assert any("world." in c for c in chunks), (
            f"Period after 'world' lost: {chunks}"
        )

    def test_question_mark_preserved(self):
        chunks = chunk_text("What is this? I don't know.")
        joined = " ".join(chunks)
        assert "?" in joined, f"Question mark lost: {chunks}"

    def test_exclamation_preserved(self):
        chunks = chunk_text("Amazing! This is great.")
        joined = " ".join(chunks)
        assert "!" in joined, f"Exclamation mark lost: {chunks}"

    def test_mixed_punctuation(self):
        text = "Hello. What? Wow! Really."
        chunks = chunk_text(text)
        joined = " ".join(chunks)
        # All three punctuation types should survive
        assert "." in joined, f"Period lost: {chunks}"
        assert "?" in joined, f"Question mark lost: {chunks}"
        assert "!" in joined, f"Exclamation lost: {chunks}"

    def test_no_spurious_commas_replacing_periods(self):
        """Original bug: all punctuation was stripped and replaced with commas."""
        chunks = chunk_text("I am fine. Thank you.")
        # Neither chunk should end with comma if original ended with period
        for chunk in chunks:
            # The original text had periods, not commas
            if "fine" in chunk:
                assert chunk.rstrip().endswith("."), (
                    f"Period replaced with wrong punctuation: {chunk!r}"
                )


class TestChunkTextLongSentences:
    """Long sentences should be split by words with punctuation added."""

    def test_long_sentence_split(self):
        long_sentence = "word " * 200  # ~1000 chars
        chunks = chunk_text(long_sentence, max_len=100)
        assert len(chunks) > 1, "Long sentence was not split"
        for chunk in chunks:
            assert len(chunk) <= 110, (  # small margin for added punctuation
                f"Chunk too long: {len(chunk)} chars"
            )

    def test_long_sentence_chunks_have_punctuation(self):
        long_sentence = "word " * 200
        chunks = chunk_text(long_sentence, max_len=100)
        for chunk in chunks:
            assert chunk[-1] in ".!?,;:", (
                f"Chunk missing punctuation: {chunk!r}"
            )


class TestChunkTextEdgeCases:
    """Edge cases for chunk_text."""

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   ") == []

    def test_single_word(self):
        chunks = chunk_text("Hello")
        assert len(chunks) == 1
        # Should have punctuation added
        assert chunks[0][-1] in ".!?,;:"

    def test_single_sentence_with_period(self):
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_text_without_punctuation(self):
        chunks = chunk_text("Hello world how are you")
        assert len(chunks) >= 1
        # ensure_punctuation should add punctuation
        for chunk in chunks:
            assert chunk[-1] in ".!?,;:"

    def test_multiple_consecutive_punctuation(self):
        """Text like 'Really?!' should not produce empty chunks."""
        chunks = chunk_text("Really?! Yes!!!")
        non_empty = [c for c in chunks if c.strip()]
        assert len(non_empty) >= 1
        # No empty chunks
        for chunk in chunks:
            assert chunk.strip(), f"Empty chunk produced: {chunks}"


class TestEnsurePunctuation:
    """ensure_punctuation should only add comma if no punctuation present."""

    def test_already_has_period(self):
        assert ensure_punctuation("Hello.") == "Hello."

    def test_already_has_question(self):
        assert ensure_punctuation("Hello?") == "Hello?"

    def test_already_has_exclamation(self):
        assert ensure_punctuation("Hello!") == "Hello!"

    def test_no_punctuation_adds_comma(self):
        assert ensure_punctuation("Hello") == "Hello,"

    def test_empty_string(self):
        assert ensure_punctuation("") == ""

    def test_strips_whitespace(self):
        result = ensure_punctuation("  Hello  ")
        assert result.startswith("Hello")


# ── Tests: clean_text API default alignment ──────────────────────────


class TestCleanTextDefaults:
    """KittenTTS wrapper clean_text defaults must match internal model."""

    def _get_default(self, cls_name, method_name, param_name):
        """Get the default value of a parameter via source inspection."""
        if cls_name == "KittenTTS":
            src_path = os.path.join(
                os.path.dirname(__file__), "..", "kittentts", "get_model.py"
            )
        else:
            src_path = os.path.join(
                os.path.dirname(__file__), "..", "kittentts", "onnx_model.py"
            )
        source = open(src_path).read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            defaults = {}
                            args = item.args
                            # Match defaults to args (defaults align to the end)
                            all_args = args.args
                            all_defaults = args.defaults
                            offset = len(all_args) - len(all_defaults)
                            for i, d in enumerate(all_defaults):
                                arg_name = all_args[offset + i].arg
                                if isinstance(d, ast.Constant):
                                    defaults[arg_name] = d.value
                                elif isinstance(d, ast.NameConstant):
                                    defaults[arg_name] = d.value
                            return defaults.get(param_name, "__NOT_FOUND__")
        return "__NOT_FOUND__"

    def test_generate_clean_text_default(self):
        wrapper = self._get_default("KittenTTS", "generate", "clean_text")
        internal = self._get_default("KittenTTS_1_Onnx", "generate", "clean_text")
        assert wrapper == internal == True, (
            f"clean_text default mismatch: KittenTTS={wrapper}, "
            f"KittenTTS_1_Onnx={internal}"
        )

    def test_generate_stream_clean_text_default(self):
        wrapper = self._get_default("KittenTTS", "generate_stream", "clean_text")
        internal = self._get_default(
            "KittenTTS_1_Onnx", "generate_stream", "clean_text"
        )
        assert wrapper == internal == True, (
            f"generate_stream clean_text default mismatch: KittenTTS={wrapper}, "
            f"KittenTTS_1_Onnx={internal}"
        )

    def test_generate_to_file_has_clean_text_param(self):
        """generate_to_file must expose clean_text so callers can control it."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "kittentts", "get_model.py"
        )
        source = open(src_path).read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "KittenTTS":
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "generate_to_file"
                    ):
                        param_names = [a.arg for a in item.args.args]
                        assert "clean_text" in param_names, (
                            f"generate_to_file missing clean_text param: {param_names}"
                        )
                        return
        raise AssertionError("KittenTTS.generate_to_file not found")


# ── Tests: version consistency ───────────────────────────────────────


class TestVersionConsistency:
    """__init__.py version must match pyproject.toml and setup.py."""

    def _read_init_version(self):
        src = os.path.join(
            os.path.dirname(__file__), "..", "kittentts", "__init__.py"
        )
        source = open(src).read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', source)
        return match.group(1) if match else None

    def _read_pyproject_version(self):
        path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        if not os.path.exists(path):
            return None
        source = open(path).read()
        match = re.search(r'^version\s*=\s*"([^"]+)"', source, re.MULTILINE)
        return match.group(1) if match else None

    def _read_setup_version(self):
        path = os.path.join(os.path.dirname(__file__), "..", "setup.py")
        if not os.path.exists(path):
            return None
        source = open(path).read()
        match = re.search(r'version\s*=\s*"([^"]+)"', source)
        return match.group(1) if match else None

    def test_init_matches_pyproject(self):
        init_v = self._read_init_version()
        pyproj_v = self._read_pyproject_version()
        assert init_v is not None, "__version__ not found in __init__.py"
        assert pyproj_v is not None, "version not found in pyproject.toml"
        assert init_v == pyproj_v, (
            f"__init__.py ({init_v}) != pyproject.toml ({pyproj_v})"
        )

    def test_init_matches_setup(self):
        init_v = self._read_init_version()
        setup_v = self._read_setup_version()
        assert init_v is not None
        assert setup_v is not None, "version not found in setup.py"
        assert init_v == setup_v, (
            f"__init__.py ({init_v}) != setup.py ({setup_v})"
        )


# ── Tests: source audit ──────────────────────────────────────────────


class TestSourceAudit:
    """Ensure the old punctuation-destroying pattern is gone."""

    def test_no_destructive_split_pattern(self):
        """chunk_text must not use re.split(r'[.!?]+', ...) which strips punctuation."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "kittentts", "onnx_model.py"
        )
        source = open(src_path).read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "chunk_text":
                fn_source = ast.get_source_segment(source, node)
                # The old pattern: re.split(r'[.!?]+', text) — destroys punctuation
                assert "re.split(r'[.!?]+" not in fn_source, (
                    "chunk_text still uses destructive re.split(r'[.!?]+') "
                    "which strips sentence-ending punctuation"
                )

    def test_no_stale_print_in_generate(self):
        """KittenTTS.generate should not print debugging output."""
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "kittentts", "get_model.py"
        )
        source = open(src_path).read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "KittenTTS":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "generate":
                        fn_src = ast.get_source_segment(source, item)
                        assert "print(" not in fn_src, (
                            "KittenTTS.generate still contains debug print()"
                        )


# ── Runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestChunkTextPunctuationPreservation,
        TestChunkTextLongSentences,
        TestChunkTextEdgeCases,
        TestEnsurePunctuation,
        TestCleanTextDefaults,
        TestVersionConsistency,
        TestSourceAudit,
    ]

    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            method = getattr(instance, name)
            try:
                method()
                passed += 1
                print(f"  PASS  {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, name, e))
                print(f"  FAIL  {cls.__name__}.{name}: {e}")

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        for cls_name, test_name, err in errors:
            print(f"\n  {cls_name}.{test_name}:")
            traceback.print_exception(type(err), err, err.__traceback__)
    sys.exit(1 if failed else 0)
