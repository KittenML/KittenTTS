"""Measure how expensive Indic text is for a candidate text tokenizer.

The question this answers is narrow: KittenTTS 2's sequence is text tokens followed
by audio tokens at 25 Hz, so a tokenizer that splits Hindi into six pieces per word
spends six times the context on the same sentence and gives the attention a six
times longer span to align across. Fertility is the number that quantifies that.

Measured on FLORES+, which is *parallel* -- the same 997 sentences translated into
every language. That matters more than it sounds: measuring Hindi on Hindi news and
English on English novels compares corpora, not tokenizers. Here the content is
held fixed and only the language varies, so the ratios are attributable.

    python -m indic.bench.tokenizer_fertility --out indic/data/fertility.json

Add candidates with `--tokenizer name=hf_repo_id`.
"""

import argparse
import json
import pathlib
import statistics
import sys
import unicodedata

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.langs import ALL, LANGUAGES

# The current tokenizer and the one the Sarvam experiment proposes. `kitten` is read
# from the TTS repo itself rather than from `Qwen/Qwen3-0.6B`, so what we measure is
# the vocabulary the checkpoint actually indexes, including its appended marker rows.
DEFAULT_TOKENIZERS = {
    "kitten-qwen3": ("KittenML/kitten-tts-2", "lm"),
    "sarvam-1": ("sarvamai/sarvam-1", None),
}


def load_tokenizer(repo, subfolder=None):
    from transformers import AutoTokenizer
    kwargs = {"subfolder": subfolder} if subfolder else {}
    return AutoTokenizer.from_pretrained(repo, **kwargs)


# FLEURS rather than FLORES: FLORES and FLORES+ are both gated, and FLEURS is the
# spoken version of the same FLORES sentences, so the text is the same corpus while
# the repo is open. It also carries the recordings, which the eval harness needs
# anyway -- one download serves both.
FLEURS_REPO = "google/fleurs"


def load_fleurs(code, split="dev"):
    """One language's FLEURS split as {sentence_id: raw transcription}.

    Keyed by sentence id because FLEURS records each sentence several times with
    different speakers; the text is identical across those rows, so keeping the
    first is enough and avoids weighting a sentence by how many people read it.

    `raw_transcription` rather than `transcription`: the latter is lowercased with
    punctuation stripped for ASR scoring, and a TTS front end never sees text in
    that shape.
    """
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FLEURS_REPO, f"data/{LANGUAGES[code].fleurs}/{split}.tsv",
                           repo_type="dataset")
    out = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3 and row[0] not in out:
                # NFC before anything touches it. FLEURS ships mixed normalization
                # forms, and both tokenizers NFC-normalize internally -- so without
                # this, decode() returns NFC text that differs byte-wise from a
                # composed-differently input and the round-trip check reports a
                # loss that is not real. It also keeps fertility honest: the same
                # Bengali syllable in NFD costs more pieces than in NFC, which
                # would otherwise be read as the tokenizer handling it badly.
                out[row[0]] = unicodedata.normalize("NFC", row[2])
    return out


def parallel_corpus(codes, split="dev", limit=None):
    """Sentences present in *every* requested language, aligned by id.

    Fertility is only attributable to the tokenizer if the content is held fixed.
    Comparing Hindi news against English novels compares corpora; comparing the
    same FLORES sentence in six languages compares tokenizers. FLEURS splits do
    not cover identical id sets per language, so the intersection is taken rather
    than assumed.
    """
    per_language = {code: load_fleurs(code, split) for code in codes}
    shared = set.intersection(*(set(d) for d in per_language.values()))
    ids = sorted(shared, key=int)
    if limit:
        ids = ids[:limit]
    return ids, {code: [per_language[code][i] for i in ids] for code in codes}


def measure(tokenizer, sentences):
    """Tokens per word, per character, and per sentence for one language.

    `add_special_tokens=False` because the inference path always passes it
    (model.py:315,356,421) -- counting a BOS that production never emits would
    inflate SentencePiece tokenizers against the BPE one by a token per sentence.
    """
    n_tokens = n_words = n_chars = 0
    per_sentence = []
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        n_tokens += len(ids)
        n_words += len(s.split())
        n_chars += len(s)
        per_sentence.append(len(ids))
    return {
        "tokens_per_word": n_tokens / max(n_words, 1),
        "tokens_per_char": n_tokens / max(n_chars, 1),
        "tokens_per_sentence": statistics.mean(per_sentence),
        "p95_tokens_per_sentence": sorted(per_sentence)[int(0.95 * len(per_sentence))],
        "n_sentences": len(sentences),
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_chars": n_chars,
    }


def check_roundtrip(tokenizer, sentences):
    """Fraction of sentences that decode back to exactly the input.

    A tokenizer that loses characters on Indic text is disqualified before
    fertility is even interesting, and byte-fallback vs. <unk> is exactly the
    difference that shows up here. Both candidates pass on NFC input; the check
    stays because it is what catches a candidate tokenizer that does not.
    """
    exact = 0
    for s in sentences:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if tokenizer.decode(ids, skip_special_tokens=True).strip() == s.strip():
            exact += 1
    return exact / max(len(sentences), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", action="append", default=[],
                    metavar="NAME=REPO", help="extra candidate, repeatable")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap sentences per language (0 = every shared sentence)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    candidates = dict(DEFAULT_TOKENIZERS)
    for spec in args.tokenizer:
        name, _, repo = spec.partition("=")
        candidates[name] = (repo, None)

    ids, corpora = parallel_corpus(ALL, args.split, args.limit)
    print(f"corpus: {len(ids)} FLEURS/{args.split} sentences present in all "
          f"{len(ALL)} languages\n")

    results = {}
    for name, (repo, subfolder) in candidates.items():
        tok = load_tokenizer(repo, subfolder)
        results[name] = {
            "repo": repo,
            "vocab_size": len(tok),
            "class": type(tok).__name__,
            "languages": {code: measure(tok, corpora[code]) for code in ALL},
            "roundtrip_exact": {code: check_roundtrip(tok, corpora[code][:100])
                                for code in ALL},
        }

    report(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


def report(results):
    names = list(results)
    for metric, label, fmt in [("tokens_per_word", "Tokens per word", "{:.2f}"),
                               ("tokens_per_char", "Tokens per character", "{:.3f}"),
                               ("tokens_per_sentence", "Tokens per sentence", "{:.1f}")]:
        print(f"\n== {label} ==")
        print(f"{'':<16}" + "".join(f"{c:>12}" for c in ALL))
        for name in names:
            row = results[name]["languages"]
            print(f"{name:<16}" + "".join(fmt.format(row[c][metric]).rjust(12) for c in ALL))
        if len(names) == 2:
            a, b = (results[n]["languages"] for n in names)
            print(f"{'ratio':<16}" + "".join(
                "{:.2f}x".format(a[c][metric] / b[c][metric]).rjust(12) for c in ALL))

    print("\n== Lossless round-trip (fraction of 100 sentences) ==")
    print(f"{'':<16}" + "".join(f"{c:>12}" for c in ALL))
    for name in names:
        rt = results[name]["roundtrip_exact"]
        print(f"{name:<16}" + "".join("{:.0%}".format(rt[c]).rjust(12) for c in ALL))

    print("\n== Vocabulary ==")
    for name in names:
        r = results[name]
        print(f"  {name:<16} {r['vocab_size']:>8,} pieces   {r['class']}   ({r['repo']})")


if __name__ == "__main__":
    main()
