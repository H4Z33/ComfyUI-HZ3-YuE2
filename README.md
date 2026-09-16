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
ABC -> Lyrics Prosody.score_abc
```

The mapping is still a hypothesis because ordinary ABC lacks explicit `w:`
syllable-to-note anchors. The JSON output records that limitation and preserves
the measurements used by the agent.
