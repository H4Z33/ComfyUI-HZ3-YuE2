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
ABC -> Lyrics Prosody.score_abc
```

The mapping is still a hypothesis because ordinary ABC lacks explicit `w:`
syllable-to-note anchors. The JSON output records that limitation and preserves
the measurements used by the agent.
