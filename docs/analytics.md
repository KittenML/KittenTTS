# Analytics

Kitten TTS sends privacy-limited generation analytics to the KittenTTS ingest
API at `https://kittenmlanalytics.com/v1/track`. The SDK does not include
PostHog or another analytics-provider SDK, and it never records or sends input
text, generated audio, stack traces, file paths, hostnames, usernames, or
project names.

Each event contains a randomly generated installation ID and event ID, SDK and
Python versions, operating system, selected model and model version, selected
or default voice, generation type (`wav`, `speak`, or `stream`), asset source,
and a coarse SDK error code for failed calls. The server adds a country code.
It does not forward raw IP addresses, user agents, or city-level location to
PostHog. The installation ID is pseudonymous and remains stable until
`~/.kittentts/analytics_id` is deleted.

Streaming calls create one `stream` event per stream invocation, not one event
per generated chunk.

The first time an installation ID is created, the SDK prints a one-time notice
to stderr describing what is collected and how to opt out.

## Opt out

Disable analytics at model creation:

```python
model = KittenTTS("KittenML/kitten-tts-mini-0.8", analytics=False)
```

Or set any of these environment variables before creating the model:

```bash
export KITTENTTS_ANALYTICS=0
# KittenTTS also respects these ecosystem-wide controls:
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
```

Opting out removes unsent KittenTTS analytics events from the local pending
queue. Analytics failures never fail or delay speech generation.

## Offline behavior

Set either `KITTENTTS_OFFLINE=1` or `HF_HUB_OFFLINE=1` to prevent analytics
network requests. While analytics remains enabled, events containing only the
fields listed above are kept under `~/.kittentts/analytics_pending` and retried
in the background after explicit offline mode is removed.

The pending queue is capped at 1,000 events and 30 days. Oldest events are
discarded first, so permanently offline applications do not grow storage
without bound. Each event is persisted before a daemon delivery thread starts,
which makes short-lived scripts resilient to process exit without making TTS
wait for the network.

After a failed delivery attempt (an unreachable network, a rate-limited or
unavailable intake), the SDK waits 60 seconds before the next attempt instead
of retrying on every generation call. Events keep queueing locally during that
window and nothing is lost.
