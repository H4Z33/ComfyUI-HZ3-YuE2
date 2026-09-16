# HZ3 YuE2

Independent ComfyUI node pack for YuE2 workflow preparation.

## Nodes

- HZ3 YuE2 · Audio to Style (Discogs-EffNet)
- HZ3 YuE2 · Procedural Prosody
- HZ3 YuE2 · Lyrics Prosody (Ollama)
- HZ3 YuE2 · MixMash Style (Ollama)
- HZ3 YuE2 · ABC Piano Roll
- HZ3 YuE2 · Save Audio
- HZ3 YuE2 · Load Generation
- HZ3 YuE2 · Generate Token Stream
- HZ3 YuE2 · Music From Token Stream
- HZ3 YuE2 · Conditioning Cut
- HZ3 YuE2 · Conditioning Paste
- HZ3 YuE2 · Save Conditioning
- HZ3 YuE2 · Load Conditioning
- HZ3 YuE2 · Conditioning Timeline
- HZ3 YuE2 · Section Plan
- HZ3 YuE2 · Assemble Sections
- HZ3 YuE2 · Whisper
- HZ3 YuE2 · MixMash Genius (Ollama)
- HZ3 YuE2 · Section Plan
- HZ3 YuE2 · Assemble Sections

The audio classifier expects `discogs-effnet-bsdynamic-1.onnx` and its matching
JSON metadata under `models/audio_classifiers/discogs_effnet`. The Ollama nodes
expect a locally reachable Ollama server.

## Procedural prosody + agent

`Procedural Prosody` reads the actual `V: Vocal` bars, merges tied continuations,
counts written and connected Spanish syllables, detects sustain fillers, and
ranks reversible spacing experiments. It never changes letters, accents, word
order, or lyrical content. `analyze_only` reports candidates without modifying
the lyrics; `apply_best_spacing` applies at most `max_edits` spacing changes.

For cooperative analysis connect:

```text
ABC -> Procedural Prosody.score_abc
Procedural Prosody.lyrics -> Lyrics Prosody.lyrics
Procedural Prosody.agent_evidence -> Lyrics Prosody.procedural_evidence

`Procedural Prosody` can analyze only, apply the best boundary-spacing test, or apply the best conditioning test. Conditioning candidates are an explicit escalation ladder (joined boundary, equivalent-consonant respelling, stress cue, score-supported consonant sustain) and include token IDs/pieces read from an installed native YuE2 checkpoint. Tokenization is evidence about text conditioning, not proof of the resulting audio.

## Reusing semantic music tokens

`Generate Token Stream` separates YuE2's autoregressive semantic-token sampling from acoustic conditioning. Connect its `token_stream` output to `Music From Token Stream`; changing style, lyrics, or ABC on the second node rebuilds conditioning without resampling the music-token IDs. `start_seconds` and `duration` select a 25-token-per-second slice for sectional experiments. A slice is a controlled reuse test, not semantic infilling: it has no guarantee of seamless continuity with omitted audio before or after it.

Connect `token_stream` or `sliced_stream` to the optional input on `HZ3 YuE2 · Save Audio` to embed the semantic IDs in the audio metadata. `HZ3 YuE2 · Load Generation` exposes that stored stream as a connectable output. Older files remain loadable but report that no stream is available.

## Editing conditioning on a timeline

Native YuE2 conditioning stores a repeated text/ABC prefix plus one semantic KV position per 1/25 second audio frame in each `yue2_chunk`. `Conditioning Cut` rebuilds valid chunks for a selected interval. `Conditioning Paste` preserves the base duration and replaces only the selected base frames with the same number of donor frames, retaining each donor chunk's own prefix. Connect the returned `seconds` to `Empty YuE2 Latent Audio`. Cuts are hard chunk boundaries; crossfading should be performed on decoded audio until a context-aware conditioning blend is validated.

`Save Conditioning` writes the tensor losslessly to a `.safetensors` sidecar under the ComfyUI output directory. The tensor is intentionally not embedded in MP3/FLAC tags because a YuE2 KV conditioning can occupy hundreds of MiB. Connect its `asset_file` to the optional `conditioning_asset` input on `Save Audio`; the audio metadata then carries the sidecar reference. `Load Generation.conditioning_asset` connects directly to `Load Conditioning.asset_file`.

For normal use, connect the original `YuE2 Generate Music.conditioning` directly to the optional `conditioning` input on `HZ3 YuE2 · Save Audio`. Save Audio automatically writes a lossless sidecar, stores its relative path, SHA-256, and frame count in the song metadata, and returns the sidecar path as `conditioning_asset`. The separate Save Conditioning node remains useful when no audio is being saved.

`HZ3 YuE2 · Load Generation` has a `load_conditioning` switch. When enabled and the audio metadata references a sidecar, it returns the restored `conditioning` and `conditioning_seconds` directly for KSampler experiments. Disable it when inspecting metadata only, to avoid reading a potentially large tensor. Older audio without a sidecar returns an empty conditioning and zero seconds.

`Conditioning Timeline` accepts up to six loaded or live takes. Each non-comment line uses `baseStart-baseEnd: take@takeStart`; for example `12-18: 2@12` replaces seconds 12–18 of take 1 with seconds 12–18 from take 2. Omit `@takeStart` to use the same source and destination time. Unspecified regions remain from take 1. Connect the timeline's `conditioning` to KSampler and its `seconds` to `Empty YuE2 Latent Audio`.

## Section-by-section covers

`Section Plan` parses the native ABC into an ordered, timed section list (bars, seconds, and semantic frames per section). It merges optional per-section `style_overrides` lines (`Section: text`) into the plan and `sections` JSON. `Assemble Sections` is the complement: it concatenates one conditioning take per section into a single contiguous YuE2 conditioning, rebasing each chunk's KV offset and frame timestamps so the result feeds directly to KSampler with `seconds` for `Empty YuE2 Latent Audio`. Connect `Section Plan.sections` to label each take in the assembly report. Generate each section (optionally via `Conditioning Cut` on a single take) and tile the takes in order; sections must tile exactly.
ABC -> Lyrics Prosody.score_abc
```

The mapping is still a hypothesis because ordinary ABC lacks explicit `w:`
syllable-to-note anchors. The JSON output records that limitation and preserves
the measurements used by the agent.
