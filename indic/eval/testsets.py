"""The evaluation sets: what we ask the model to say, and what we score it against.

Two sets, answering two different questions.

**Set A — read speech, from FLEURS.** Natural sentences with a human recording of
each one. Two things come out of that recording and both are load-bearing:

  * the **ASR floor**. Scoring the human reading of the same sentence through the
    same recognizer says how much of a TTS CER was never the TTS. Without it, 14%
    on Hindi and 22% on Telugu look like a model that is worse at Telugu, when the
    recognizer's own Telugu floor may account for all of the gap.
  * a **cloning reference**. A different FLEURS utterance, never the target's own
    recording: the target's recording *is* the answer, and handing it to a model
    that puts the reference audio in its prompt measures copying, not synthesis.

Set A is built on the sentences FLEURS has in all six languages, via
`parallel_corpus`. Content is then held fixed and only the language varies, so a
Hindi CER and a Tamil CER are about the same material — the same reason
`tokenizer_fertility.py` measures on a parallel corpus.

**Set C — robustness.** Numbers, currency, dates, times and over-long inputs, in
carrier sentences, generated rather than sampled so that every language gets the
same stress in the same places. Each item is emitted twice, once with ASCII digits
and once with the script's own digits, sharing a `pair_id`. That pairing is the
measurement: `kittenml/kittentts2/text.py` expands `2024` and leaves `२०२४` as four
rare codepoints, so the gap between a pair's two CERs is the size of that gap, and
it is a gap that `text/indic_text.py` exists to close.

A caveat on scoring set C: the recognizer writes numbers as words, the reference
holds them as digits, so the absolute CER on these items is inflated and is not
comparable with set A's. Both members of a pair are scored against the *same*
ASCII-digit reference, so the inflation is identical on both sides and cancels in
the difference. Read the delta; do not quote the level.

Tamil and Telugu native digits (௧, ౧) are archaic in modern writing, so those pairs
describe a rarer failure than the Devanagari and Bengali ones.

The carrier sentences are short and deliberately plain, but they are still
generated text in five languages and **need native-speaker review** before any
number taken on them is published — the same review `text/indic_text.py` asks for.

    python -m indic.eval.testsets --testset A --lang hi --limit 5
    python -m indic.eval.testsets --testset C --lang ta --manifest /tmp/c-ta.jsonl
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.bench.tokenizer_fertility import FLEURS_REPO, load_fleurs, parallel_corpus
from indic.langs import ALL, LANGUAGES
from indic.text.indic_text import DIGIT_BLOCK_STARTS, SENT_END

FLEURS_SAMPLE_RATE = 16000

# A cloning reference wants a few seconds of one speaker: too short and the voice
# is not determined, too long and it crowds the prompt (model.py budgets the
# reference clip in codec tokens at 25 Hz).
REFERENCE_SECONDS = (5.0, 30.0)
# Several reference clips rather than one, assigned round-robin, so a result is not
# an accident of one speaker's recording. FLEURS carries no speaker id, so this
# gives varied speakers without being able to say how many.
REFERENCE_POOL = 4

# Past text.py's 380-character chunk budget, which is where the splitter stops
# finding sentence ends and falls through to a hard `sentence[:max_chars]` cut.
LONG_CHARS = 420


@dataclasses.dataclass(frozen=True)
class Record:
    """One utterance to synthesize, and everything needed to score it later.

    `text` is what the model is asked to say; `ref_text` is what its transcription
    is scored against. They differ only for the native-digit half of a set C pair,
    where both halves are scored against the ASCII form so the pair's difference is
    attributable (see the module docstring).
    """

    id: str
    lang: str
    text: str
    ref_text: str
    category: str
    pair_id: str = None
    variant: str = None             # "ascii_digits" / "native_digits" for set C pairs
    ref_wav: str = None             # cloning prompt: a *different* recording
    ref_wav_text: str = None        # its transcript, so cloning needs no Whisper
    truth_wav: str = None           # human reading of `text`, for the ASR floor
    truth_seconds: float = None

    def to_dict(self):
        return dataclasses.asdict(self)


def cache_root(cache_dir=None):
    """Where extracted FLEURS audio lives. $KINDIC_CACHE overrides the default."""
    return pathlib.Path(cache_dir or os.environ.get("KINDIC_CACHE")
                        or pathlib.Path.home() / ".cache" / "kindic")


def fleurs_audio_rows(code, split="dev"):
    """{sentence_id: (file_name, seconds)} from the same TSV `load_fleurs` reads.

    A second pass over the file rather than a change to `load_fleurs`, which
    deliberately returns text only and is shared with the tokenizer benchmark. The
    columns are FLEURS's own: id, file name, raw transcription, transcription,
    unused, sample count, gender.
    """
    import csv

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(FLEURS_REPO, f"data/{LANGUAGES[code].fleurs}/{split}.tsv",
                           repo_type="dataset")
    rows = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            # First recording of each sentence wins, matching `load_fleurs` — the
            # text it returns for an id must belong to the clip picked here.
            if len(row) >= 6 and row[0] not in rows:
                try:
                    seconds = int(row[5]) / FLEURS_SAMPLE_RATE
                except ValueError:
                    seconds = None
                rows[row[0]] = (row[1], seconds)
    return rows


def ensure_fleurs_audio(code, split, file_names, cache_dir=None):
    """Extract the named wavs out of `data/<fleurs>/audio/<split>.tar.gz`, once.

    Only the files asked for, and only those not already extracted, so a run that
    wants twenty sentences does not unpack the whole split. Members are read and
    written by hand rather than through `extract()`, which removes path traversal
    as a thing to reason about, and each file lands via a rename so an interrupted
    job leaves no half-written wav that the next run would trust.
    """
    import tarfile

    from huggingface_hub import hf_hub_download

    target = cache_root(cache_dir) / "fleurs" / LANGUAGES[code].fleurs / split
    target.mkdir(parents=True, exist_ok=True)
    wanted = {name for name in file_names if not (target / name).exists()}
    if not wanted:
        return target

    archive = hf_hub_download(
        FLEURS_REPO, f"data/{LANGUAGES[code].fleurs}/audio/{split}.tar.gz",
        repo_type="dataset")
    # Streamed in one pass: the tarball is one gzip member, so seeking to a file
    # would decompress everything before it anyway.
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            name = pathlib.PurePosixPath(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            partial = target / (name + ".partial")
            partial.write_bytes(tar.extractfile(member).read())
            partial.rename(target / name)
            wanted.discard(name)
            if not wanted:
                break
    if wanted:
        raise FileNotFoundError(
            f"{len(wanted)} file(s) listed in the {code}/{split} TSV are absent from "
            f"its audio tarball, e.g. {sorted(wanted)[:3]} — the split's metadata and "
            "audio disagree, so a test set built from it would be silently short")
    return target


def fleurs_reference_pool(lang, split="dev", cache_dir=None, exclude=(), size=REFERENCE_POOL):
    """Cloning reference clips for a language: [(wav path, transcript), ...].

    Longest-first within the 5-30 s window, which is both what the cloning path
    wants and what makes the choice deterministic — a seeded random pick would
    still change whenever the seed or the split did.
    """
    text = load_fleurs(lang, split)
    rows = fleurs_audio_rows(lang, split)
    low, high = REFERENCE_SECONDS
    candidates = [
        (seconds, sentence_id, file_name)
        for sentence_id, (file_name, seconds) in rows.items()
        if sentence_id in text and sentence_id not in set(exclude)
        and seconds is not None and low <= seconds <= high]
    chosen = sorted(candidates, key=lambda c: (-c[0], int(c[1])))[:size]
    if not chosen:
        raise ValueError(f"no FLEURS {lang}/{split} clip is between {low} and {high} "
                         "seconds, so there is nothing to clone from")
    directory = ensure_fleurs_audio(lang, split, [c[2] for c in chosen], cache_dir)
    return [(str(directory / file_name), text[sentence_id])
            for _, sentence_id, file_name in chosen]


def set_a(lang, split="dev", limit=None, cache_dir=None, pool=REFERENCE_POOL):
    """Read speech from FLEURS, with a ground-truth recording and a clone reference."""
    ids, corpora = parallel_corpus(ALL, split)
    text = dict(zip(ids, corpora[lang]))
    rows = fleurs_audio_rows(lang, split)

    target_ids = [i for i in ids if i in rows]
    if limit:
        target_ids = target_ids[:limit]
    # References come from outside the scored set. Overlap would let a model be
    # handed the target sentence's own audio as its prompt.
    references = fleurs_reference_pool(lang, split, cache_dir,
                                       exclude=target_ids, size=pool)

    directory = ensure_fleurs_audio(lang, split, [rows[i][0] for i in target_ids],
                                    cache_dir)
    records = []
    for index, sentence_id in enumerate(target_ids):
        file_name, seconds = rows[sentence_id]
        reference_wav, reference_text = references[index % len(references)]
        records.append(Record(
            id=f"A-{lang}-{sentence_id}", lang=lang, text=text[sentence_id],
            ref_text=text[sentence_id], category="read",
            ref_wav=reference_wav, ref_wav_text=reference_text,
            truth_wav=str(directory / file_name), truth_seconds=seconds))
    return records


# ── set C ─────────────────────────────────────────────────────────────────────
# Carrier sentences, without their final terminator — `LANGUAGES[...].danda` decides
# whether that is a danda or a full stop, so the punctuation rule lives in one place.
# Short and plain on purpose: a carrier the model mispronounces for its own reasons
# would be charged to the number being carried.
CARRIERS = {
    "hi": {"count": "कुल {} लोग आए थे",
           "price": "इसकी कीमत {} रुपये है",
           "currency": "मैंने {} का भुगतान किया",
           "percent": "{} प्रतिशत लोग सहमत हैं",
           "date": "यह घटना {} को हुई थी",
           "time": "ट्रेन {} बजे आएगी",
           "year": "सन् {} में यह बना था"},
    "bn": {"count": "মোট {} জন এসেছিলেন",
           "price": "এর দাম {} টাকা",
           "currency": "আমি {} পরিশোধ করেছি",
           "percent": "{} শতাংশ মানুষ একমত",
           "date": "ঘটনাটি {} তারিখে ঘটেছিল",
           "time": "ট্রেনটি {} টায় আসবে",
           "year": "{} সালে এটি তৈরি হয়েছিল"},
    "ta": {"count": "மொத்தம் {} பேர் வந்தனர்",
           "price": "இதன் விலை {} ரூபாய்",
           "currency": "நான் {} செலுத்தினேன்",
           "percent": "{} சதவீதம் பேர் ஒப்புக்கொண்டனர்",
           "date": "இந்த நிகழ்வு {} அன்று நடந்தது",
           "time": "ரயில் {} மணிக்கு வரும்",
           "year": "{} ஆம் ஆண்டில் இது கட்டப்பட்டது"},
    "te": {"count": "మొత్తం {} మంది వచ్చారు",
           "price": "దీని ధర {} రూపాయలు",
           "currency": "నేను {} చెల్లించాను",
           "percent": "{} శాతం మంది అంగీకరించారు",
           "date": "ఈ సంఘటన {} న జరిగింది",
           "time": "రైలు {} గంటలకు వస్తుంది",
           "year": "{} సంవత్సరంలో ఇది నిర్మించబడింది"},
    "mr": {"count": "एकूण {} लोक आले होते",
           "price": "याची किंमत {} रुपये आहे",
           "currency": "मी {} भरले",
           "percent": "{} टक्के लोक सहमत आहेत",
           "date": "ही घटना {} रोजी घडली",
           "time": "गाडी {} वाजता येईल",
           "year": "{} साली हे बांधले गेले"},
    "en": {"count": "A total of {} people came",
           "price": "It costs {} rupees",
           "currency": "I paid {}",
           "percent": "{} percent of people agreed",
           "date": "The event happened on {}",
           "time": "The train arrives at {}",
           "year": "It was built in {}"},
}

# The values are the test. `2,50,000` is the Indian grouping, which a thousands
# separator written for English parses wrong; `12.5` is a decimal; the dates are
# day-first, which is the Indian convention and the opposite of what an
# English-trained normalizer assumes.
VALUES = {
    "count": ["7", "42", "108", "1250"],
    "price": ["25", "499", "2,50,000"],
    "currency": ["₹500", "₹1,250.50", "Rs. 99"],
    "percent": ["5", "12.5", "99"],
    "date": ["15/08/1947", "26/01/1950", "01/04/2024"],
    "time": ["10:30", "7:45", "18:00"],
    "year": ["1947", "2001", "2024"],
}


def _terminator(lang):
    return "।" if LANGUAGES[lang].danda else "."


def _to_native_digits(text, lang):
    """ASCII digits rewritten in the language's own script, or None for Latin.

    The inverse of `indic_text.indic_digits_to_ascii`, sharing its block table so
    the two cannot drift apart.
    """
    start = DIGIT_BLOCK_STARTS.get(LANGUAGES[lang].script)
    if start is None:
        return None
    return "".join(chr(start + int(ch)) if ch.isdigit() and ch.isascii() else ch
                   for ch in text)


def _long_items(clauses, lang):
    """Two over-long inputs built from the filled carriers.

    They probe different failures. `long_sentences` is several complete sentences
    in one call: the splitter has to find the danda to chunk it, and does not today.
    `long_unbroken` is one comma-joined chain with no internal terminator, which is
    what actually reaches `sentence[:max_chars]` and gets cut mid-grapheme.

    The chain is filtered through `SENT_END` itself, because a clause like
    "I paid Rs. 99" carries a sentence end in the middle of it — one such clause
    would let the splitter cut the chain in half and the record would quietly stop
    testing the thing it exists for.
    """
    end = _terminator(lang)
    sentences, chain = [], []
    for clause in clauses:
        if sum(len(s) for s in sentences) <= LONG_CHARS:
            sentences.append(clause + end)
        if sum(len(c) for c in chain) <= LONG_CHARS and not SENT_END.search(clause):
            chain.append(clause)
    return [("long_sentences", " ".join(sentences)),
            ("long_unbroken", ", ".join(chain) + end)]


def set_c(lang, reference=None):
    """Numbers, money, dates, times and over-long text, in both digit forms.

    `reference` is an optional (wav, transcript) pair applied to every record, for
    models that can only speak by cloning — there is no human recording of
    generated text, so these records have no ASR floor and `score.py` reports none.
    """
    carriers = CARRIERS[lang]
    end = _terminator(lang)

    clauses, items = [], []
    for category, values in VALUES.items():
        for value in values:
            clause = carriers[category].format(value)
            clauses.append(clause)
            items.append((category, clause + end))
    items.extend(_long_items(clauses, lang))

    reference_wav, reference_text = reference or (None, None)
    records = []
    for index, (category, ascii_text) in enumerate(items):
        pair_id = f"C-{lang}-{category}-{index:02d}"
        native_text = _to_native_digits(ascii_text, lang)
        # Both halves are scored against the ASCII form: see the module docstring on
        # why only the difference between them is a real quantity.
        variants = [("ascii_digits", ascii_text)]
        if native_text and native_text != ascii_text:
            variants.append(("native_digits", native_text))
        for variant, text in variants:
            records.append(Record(
                id=f"{pair_id}-{variant}", lang=lang, text=text, ref_text=ascii_text,
                category=category, pair_id=pair_id, variant=variant,
                ref_wav=reference_wav, ref_wav_text=reference_text))
    return records


TESTSETS = ("A", "C")


def build(name, lang, split="dev", limit=None, cache_dir=None, reference=None):
    """A test set by name. Records come back in a fixed order, which sharding relies on."""
    if lang not in LANGUAGES:
        raise ValueError(f"unknown language {lang!r}; known: {sorted(LANGUAGES)}")
    if name == "A":
        return set_a(lang, split, limit, cache_dir)
    if name == "C":
        records = set_c(lang, reference)
        return _thin(records, limit) if limit else records
    raise ValueError(f"unknown test set {name!r}; choose from {list(TESTSETS)}")


def _thin(records, limit):
    """Cut set C down without losing a category or half a pair.

    `records[:limit]` would be wrong twice over: the set is ordered by category, so
    a small limit would return nothing but counts, and it could keep a pair's ASCII
    half while dropping its native-digit half — and the pair is the whole
    measurement. So this strides over pairs and keeps each one entire.
    """
    groups, order = {}, []
    for record in records:
        key = record.pair_id or record.id
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    per_group = max(len(groups[key]) for key in order)
    wanted = max(1, limit // per_group)
    stride = max(1, len(order) // wanted)
    return [record for key in order[::stride][:wanted] for record in groups[key]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="A", choices=TESTSETS)
    ap.add_argument("--lang", default="hi", choices=sorted(LANGUAGES))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", help="default: $KINDIC_CACHE or ~/.cache/kindic")
    ap.add_argument("--manifest", type=pathlib.Path, help="write the records as JSONL")
    args = ap.parse_args()

    records = build(args.testset, args.lang, args.split, args.limit or None,
                    args.cache_dir)
    floor = sum(1 for r in records if r.truth_wav)
    print(f"set {args.testset}  {args.lang}  {len(records)} records   "
          f"{floor} with a ground-truth recording (ASR floor)\n")
    for record in records[:10]:
        print(f"  {record.id:<34}{record.category:<16}{record.text[:60]}")
    if len(records) > 10:
        print(f"  ... {len(records) - 10} more")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        print(f"\nwrote {args.manifest}")


if __name__ == "__main__":
    main()
