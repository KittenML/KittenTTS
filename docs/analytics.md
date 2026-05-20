# Analytics

Kitten TTS sends anonymous generation analytics to the KittenTTS ingest API at
`https://kittenmlanalytics.com/v1/track`. The SDK does not include PostHog or
any analytics-provider SDK, and it does not send input text or generated audio.

Events include SDK version, SDK type, platform, runtime version, selected model,
model version, selected/default voice, generation type (`wav`, `speak`, or
`stream`), asset source, and SDK error code for failed calls. IP address and
location are added server-side by Cloudflare.

Streaming calls send one `stream` analytics event per stream invocation, not one
event per generated chunk.

Disable analytics at model creation:

```python
model = KittenTTS("KittenML/kitten-tts-mini-0.8", analytics=False)
```

Analytics runs in the background with a short timeout. Network failures are
swallowed and do not block or fail TTS generation.
