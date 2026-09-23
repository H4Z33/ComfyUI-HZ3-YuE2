# HZ3 YuE2

Independent ComfyUI node pack for YuE2 workflow preparation.

## Nodes

- HZ3 YuE2 · Audio to Style (Discogs-EffNet)
- HZ3 YuE2 · Procedural Prosody
- HZ3 YuE2 · Lyrics Prosody (Ollama)
- HZ3 YuE2 · MixMash Style (Ollama)
- HZ3 YuE2 · Vocal Harmony
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
- HZ3 YuE2 · Karaoke & Audio Visualizer

The audio classifier expects `discogs-effnet-bsdynamic-1.onnx` and its matching
JSON metadata under `models/audio_classifiers/discogs_effnet`. The Ollama nodes
expect a locally reachable Ollama server.

## Vocal Harmony

Connect `MixMash.abc_repaired` to `Vocal Harmony.score_abc`. The node creates
`abc_lead`, `abc_tenor`, `abc_baritone`, and `abc_bass`: separate native two-track
ABCs with one singing part in `Vocal` and silence in `Ins`. The lead melody is
preserved. Chord annotations guide harmonization but are omitted from the solo
outputs, like MixMash's vocals-only output. No model calls or extra dependencies.

`close_harmony` balances chord coverage and smooth movement; `barbershop_inspired`
favors compact upper voices and complete existing seventh chords. It does not
reharmonize the song into a traditional barbershop arrangement. All supported
native chord qualities, slash chords, accidentals, ties, and key/meter changes
are handled. Missing chords are inferred from the active key and melody and
flagged in `report`.

Leave `active_sections` blank (or `all`) for the whole song. Use `chorus, outro`
to limit the harmonies; `chorus` matches numbered choruses, while `chorus 2`
selects just that section. Excluded sections and lead rests remain silent.
Ranges use MIDI note numbers (`C4 = 60`); the report lists actual ranges and any
voicing compromises. Harmonies follow the lead's attacks and rests, adapting at
chord/key boundaries during held notes. Sustained wordless backing is not included.

Route each ABC to a separate vocal generation branch with the corresponding solo
voice style. For full-song harmonies, reuse the same lyrics. For selected sections,
provide only those sections' sung words in the harmony branches, retaining the
other section markers without words. The node does not rewrite lyrics.
`duration_seconds` is the exact score duration as a float; use it wherever the
workflow accepts a duration, observing that sampler latents must still match the
actual conditioning length. All four scores retain the source's measures, section
boundaries, tempo, and duration; separate audio generations can still differ in
phrasing and need alignment before mixing. Start with a short chorus to audition.

Run isolated checks with `python -m unittest discover -s tests -v`.

`workflows/HZ3-YuE2_Karaoke_Harmonies.json` extends the existing cover workflow
with tenor, baritone, and bass generation branches. The lead vocal remains the
alignment reference; only the three harmony stems are mixed into the instrumental
soundtrack. Each sampler uses its own generator's actual `seconds`; the final
mix retains the instrumental length. Harmony stems and the mix are saved as FLAC,
and the Rolling Mode visualizer saves the karaoke MP4. Initial harmony levels are
-12, -14, and -12 dB; audition phrasing alignment before a final render.

## Karaoke & Audio Visualizer

`Karaoke & Audio Visualizer` is a single node: it times the clean lyrics against the
generated song and renders the karaoke video (word sweep, chords, section HUD, spectrum).
It also returns `timed_lyrics` (JSON with per-word times), `lrc_text` and an alignment
`report`.

Lyric timing sources, tried in this order in `Auto` mode:

1. **Forced alignment on audio** (`torchaudio.pipelines.MMS_FA`, multilingual CTC). Connect
   the generated song to `audio` and, strongly recommended, the separated vocal stem of the
   same audio (`AudioSeparation.Vocals`) to `vocals`. The lyrics text is never changed; each
   word receives the time it is actually sung. Unscripted vocals (ad-libs) are absorbed by a
   wildcard token at section boundaries; lyrics the generation never reached are flagged as
   `beyond_audio`. The ~1.2 GB model is downloaded once into `models/mms_fa`.
2. **Whisper segments**: `HZ3 YuE2 · Transcribe.segments` (or LRC text) on `whisper_segments`
   anchors the clean lines when forced alignment is unavailable.
3. **ABC score timing**: nominal sections, vocal phrases and note onsets from the ABC.

`lyrics_offset` fine-tunes every timestamp. Chords and the section HUD follow the measured
audio-vs-score offset when the lyrics were aligned on audio. The former `Score & Lyrics
Aligner` node was merged into this node; `score_lyric_aligner.py` remains as the library.

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
# Fetch lyrics from an audio input

The **HZ3 YuE2 · Fetch Lyrics from Audio** node takes a ComfyUI `AUDIO` input, creates a local Chromaprint fingerprint with `fpcalc`, identifies the recording through AcoustID, and retrieves plain or synchronized lyrics from LRCLIB. Its `lyrics` output is clean plain text that can be connected directly to MixMash; `synced_lyrics` preserves LRC timestamps separately, along with the matched title, artist, album, confidence, and status.

Setup:

1. Install Chromaprint so `fpcalc` is available on `PATH`, or enter the full executable path in the node's optional `fpcalc_path` field. On Windows, MusicBrainz Picard installs `fpcalc` with Chromaprint.
2. Register a free AcoustID application and enter its client ID in the node. AcoustID's public service is limited to non-commercial usage and asks clients to stay below three requests per second.
3. Connect your input audio. The node sends the fingerprint and duration to AcoustID, not the raw audio. LRCLIB does not require an API key.

This lookup depends on the recording being present in AcoustID and its lyrics being present in LRCLIB. It is intended for identifying released recordings; original or substantially rearranged AI-generated songs may not be recognized. For those, provide the lyrics directly or transcribe the vocal track separately.

References: [AcoustID API](https://acoustid.org/webservice), [LRCLIB API](https://lrclib.net/docs), [Chromaprint](https://github.com/acoustid/chromaprint).
