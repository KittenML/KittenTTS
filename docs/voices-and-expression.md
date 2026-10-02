# Voices and expression

The built-in voices, the expression markup the model understands, and what it does and
does not cover. See [api.md](api.md) for the arguments these go with.

## Expression controls

Three kinds of markup reach the model as direction rather than as words to pronounce. Using
any of them automatically switches on the model's `{emo: 1}` conditioning; pass
`advanced={"use_emotion": False}` to force it off, or `True` to force it on for untagged text.

Emotion support is **early beta**. It steers delivery rather than guaranteeing it, and the
effect varies by voice and by sentence.

### Emotion

A leading tag sets the emotion for the whole line.

```python
m.generate("[joyful] We actually won the grant!", voice="Kiki")
```

`[angry]` `[contemplative]` `[excited]` `[joyful]` `[mundane]` `[nervous]` `[sad]` `[stern]`
`[surprised]` `[tender]`

### Vocal events

An inline tag inserts a non-verbal beat anywhere in the line.

```python
m.generate("We won <laugh> I can hardly believe it.", voice="Kiki")
```

`<gasp>` `<giggle>` `<growl>` `<gulp>` `<laugh>` `<pause>` `<scoff>` `<sigh>` `<sob>` `<um>`

### Emphasis

Triple parentheses stress a word or short phrase.

```python
m.generate("I told you (((never))) to open that door.", voice="Victor")
```

### Why these lists and not others

The training data carries 2,684 distinct emotion values and 1,101 distinct vocal-event values.
The twenty above are the ten most common of each, covering roughly half of all tagged lines.
They are a recognition list, not the model's full vocabulary: a rarer tag such as `[reverent]`
is real in the data but is **not** recognised here, and will be normalized and spoken as
ordinary text.

That restraint is deliberate. Treating every bracketed span as markup would silently swallow
things people write for other reasons — a citation like `section [3]`, a comparison like
`x < 5`. Those are left as text.

Anything bracketed that is *not* one of these tags gets rewritten to plain comma-separated
text before synthesis, because the model destabilises on bracket punctuation it was not
trained to read. This was observed directly: a parenthetical list,
`(salons, dental offices, restaurants)`, generated abnormally fast — ~0.043 s/char against
0.056–0.093 for every other chunk of the same request, reproducibly — consistent with the
model rushing through it.

`EXPRESSION_TAGS` and `VOCAL_EVENT_TAGS` in `kittenml.kittentts2.text` are the live lists;
extend them if you want more tags protected.

## Built-in voices

Forty-seven voices, listed by `model.available_voices`: thirty-eight English, plus nine named
after the [languages](#languages) below. Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki and Leo
are the same speakers as in KittenTTS 0.8, so code written against the ONNX models keeps
working.

| Voice | Character | | Voice | Character |
|---|---|---|---|---|
| **Bella** | KittenTTS 0.8 voice, female | | **Herbert** | Wise elderly professor, male |
| **Jasper** | KittenTTS 0.8 voice, male | | **Diana** | Stern military officer, female |
| **Luna** | KittenTTS 0.8 voice, female | | **Laurence** | Dramatic stage actor, male |
| **Bruno** | KittenTTS 0.8 voice, male | | **Maeve** | Cozy warm storyteller, female |
| **Rosie** | KittenTTS 0.8 voice, female | | **Walter** | Warm grandfather, male |
| **Hugo** | KittenTTS 0.8 voice, male | | **Edith** | Soft elderly grandmother, female |
| **Kiki** | KittenTTS 0.8 voice, female | | **Miles** | Gentle librarian, male |
| **Leo** | KittenTTS 0.8 voice, male | | **Grace** | Soothing nurse, female |
| **Matthew** | Exhausted mechanic, male | | **Reginald** | Distinguished elderly diplomat, male |
| **Elliot** | Grieving young, male | | **Iris** | Hushed museum guide, female |
| **Willow** | Hushed young, female | | **Frank** | Gravelly grizzled war veteran, male |
| **Dolores** | Southern elderly, female | | **Serena** | Gentle yoga instructor, female |
| **Victor** | Controlled fury older, male | | **Julian** | Dreamy poet soft, male |
| **Dante** | Playful smooth young, male | | **Eleanor** | Somber historian, female |
| **Alfred** | Tender older, male | | **Otis** | Rumbling blues musician, male |
| **Saoirse** | Joyful irish, female | | **Vincent** | Hushed museum guide, male |
| **Claire** | Playful professional, female | | **Martha** | Weathered radio journalist, female |
| **Raven** | Controlled fury young, female | | **Sable** | Hushed conspirator, female |
| **Marcus** | Smooth radio DJ, male | | **Victoria** | Regal narrator, female |

Every reference clip is 8–18 seconds of a single speaker. A voice is the clip *and* its
transcript: both go into the prompt, which is why cloning a new voice wants a transcript too
(and transcribes one for you when you omit it).

## Languages

The model is multilingual. Its SFT mix is about 10% non-English across nine languages, and a
Hindi voice was added on top, giving ten in total. Each ships as one built-in voice, named
after its language:

| Voice | Language | Voice | Language |
|---|---|---|---|
| **Arabic** | العربية | **Italian** | Italiano |
| **Chinese** | 简体中文 | **Portuguese** | Português |
| **French** | Français | **Russian** | Русский |
| **German** | Deutsch | **Spanish** | Español |
| **Hindi** | हिन्दी | | |

```python
m.generate("Guten Morgen. Ich wünsche dir einen wunderschönen Tag.",
           voice="German", normalize=False)
```

The voice carries the language: these clips are what supply the accent and phonotactics, so
asking an English voice for German text will not work well. Cloning from your own non-English
recording works the same way.

**Pass `normalize=False` for non-English text.** The normalizer is English-tuned, and will
mangle numbers, dates and abbreviations in other languages. That is a real limitation, not an
oversight — the upstream demo carries the same caveat.

One reference clip per language is a modest base, so expect these to be less robust than the
English voices, and quality to vary by language. Measured by transcribing generated audio back
with Whisper, all ten reproduce their prompt: German, Spanish, French, Italian, Portuguese,
Russian and Chinese came back verbatim; Arabic and Hindi came back with small errors.

## Getting more expressive output

The `expressive` preset loosens sampling a little. To push further, use the
[advanced options](api.md#advanced-options): disable `top_k` by setting it to `0`, raise
`temperature` above `1.0`, and raise `top_p` to `0.9` or above. The further you push, the
livelier — and the less predictable — the delivery.

```python
m.generate("[excited] This is going to be good!", voice="Dante",
           preset="expressive", top_k=0, temperature=1.1, top_p=0.95)
```

## Known limitations

- Emotion conditioning is early beta; it biases delivery rather than guaranteeing it.
- Only the twenty tags above are recognised, not the full long tail in the data.
- The non-English voices rest on a single reference clip each; they are more fragile than the
  English set, and the text normalizer does not cover them.
- Voices far from the model's training distribution clone less well than the built-in ones.
  This is more pronounced on the quantised [decoders](decoders.md), whose student was trained
  on a 539-speaker pool.
