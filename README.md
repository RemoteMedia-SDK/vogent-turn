# vogent-turn — Vogent-Turn-80M conversational turn-end detection

Standalone Path 3 Rust cdylib that registers `VogentTurnOnnxNode` into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

Wraps the [Vogent-Turn-80M](https://huggingface.co/vogent/Vogent-Turn-80M)
classifier (Whisper-Tiny encoder + SmolLM-135M + binary head) for
endpointing in conversational voice agents. Originally lived in
`remotemedia-core` under the `onnx` feature; extracted here so the host
crate doesn't drag in `ort` / `tokenizers` / `rustfft` just for this node.

License note: Vogent-Turn-80M weights are released under a modified
Apache-2.0 — operators agree not to set this model as the default option
in horizontal voice-agent platforms and to surface a "Vogent Turn
Detector" label when offered as an option.

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["vogent-turn@v0.1.0"],
  "nodes": [
    {
      "id": "turn",
      "node_type": "VogentTurnOnnxNode",
      "params": {
        "model_dir": "/path/to/vogent-turn-80m",
        "precision": "fp16",
        "device": "cuda:0",
        "threshold": 0.5,
        "throttle_ms": 100,
        "endpoint_hysteresis": 0.1,
        "veto_below_prob": 0.3
      }
    }
  ]
}
```

The SDK resolver expands `vogent-turn@v0.1.0` to
`github.com/RemoteMedia-SDK/vogent-turn`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

```bash
git clone https://github.com/RemoteMedia-SDK/vogent-turn
cd vogent-turn
cargo build --release
# → target/release/libvogent_turn_plugin.so
```

The Vogent-Turn-80M bundle is not vendored; fetch from Hugging Face
into `model_dir` with the expected layout:

```
<model_dir>/onnx-fp16/whisper-smol-lm-smaller-fp16.onnx
<model_dir>/onnx-fp32/whisper-smol-lm-smaller.onnx
<model_dir>/tokenizer.json
```

## What it exports

| Node type             | Input                                                                          | Output                                            |
|-----------------------|--------------------------------------------------------------------------------|---------------------------------------------------|
| `VogentTurnOnnxNode`  | `Audio` 16 kHz mono f32; `Json{prev,curr,is_speech_start,is_speech_end,reset}` | `Audio` (passthrough) + `Json{turn_classification \| turn_end}` |

### Audio path

Audio chunks are forwarded unchanged to downstream consumers, then
optionally trigger an inference (throttled by `throttle_ms`). The node
hard-requires **16 kHz mono f32 PCM** — wire `FastResampleNode` upstream
if your source rate differs.

### Json control events

* `{"reset": true}` — clear all per-session state.
* `{"prev": "...", "curr": "..."}` — update transcript context.
  Invalidates the tokenization cache; does not trigger inference.
* `{"is_speech_start": true}` — **arm** the gate and forward the event
  downstream so e.g. `AudioBufferAccumulatorNode` can begin buffering.
* `{"is_speech_end": true}` — apply veto logic:
  * If `veto_below_prob <= 0.0` (default), forward unchanged.
  * Otherwise run a fresh inference and forward only when
    `prob_endpoint >= veto_below_prob`.

This wiring (`vad → vogent_turn → accumulator`) lets VAD's
silence-counter run aggressively (e.g., `min_silence_duration_ms: 200`)
while Vogent-Turn holds the line on conversational pauses.

### Turn-end emission

On the rising edge of `prob_endpoint >= threshold` (only while armed),
the node emits:

```json
{"type": "turn_end", "is_speech_end": true, "prob_endpoint": 0.83}
```

which is a drop-in trigger for `AudioBufferAccumulatorNode`'s
`is_speech_end` check. Hysteresis (`endpoint_hysteresis`) prevents
re-firing while the model dithers around the threshold.

## License

See `LICENSE.md`. Governed by the RemoteMedia SDK Community License 1.0.
The Vogent-Turn-80M weights themselves are licensed separately
(modified Apache-2.0) — see the model card.
