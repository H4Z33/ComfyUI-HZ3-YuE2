# HZ3 YuE2

Independent ComfyUI node pack for YuE2 workflow preparation.

## Installation

1. Use a ComfyUI with the native YuE2 nodes (`YuE2 Generate Music`, `YuE2 Generate ABC`).
2. Clone this repository into `ComfyUI/custom_nodes` and install its requirements with
   ComfyUI's own Python (Comfy Desktop: `<install>\ComfyUI\.venv\Scripts\python.exe`):
   ```
   git clone https://github.com/H4Z33/ComfyUI-HZ3-YuE2
   python -m pip install -r ComfyUI-HZ3-YuE2/requirements.txt
   ```
   `faster-whisper` is optional (the `fast` Whisper backend).
3. Install [audio-separation-nodes-comfyui](https://github.com/christian-byrne/audio-separation-nodes-comfyui)
   (`AudioSeparation`): HZ3 Studio and the cover workflows use it. Its Hybrid Demucs
   weights download on first use.
4. Models: a YuE2 checkpoint (e.g. `yue2_3b_int8_convrot.safetensors`) in `models/checkpoints`;
   `sheetsage2_bf16.safetensors` in `models/audio_encoders` for audio analysis; the Discogs
   EffNet files below for `Audio to Style`.
5. A local [Ollama](https://ollama.com) with a chat model for MixMash and the harmony
   arranger (default `deepseek-v4.1-flash:cloud`).
6. Restart ComfyUI and open **HZ3 → Abrir HZ3 Studio** in the main menu (or `/hz3/studio`
   on the same address as ComfyUI).

## HZ3 Studio

Open `/hz3/studio` on the running ComfyUI (same host and port as the ComfyUI page)
for a song editor on top of these nodes.

- Start from a song: `Audio original` adds the source audio and `Analizar audio`
  runs SheetSage2 (ABC), Whisper (lyrics, when the lyrics box is empty) and
  MixMash (style, lyrics, repaired ABC) through ComfyUI. Or write style and
  lyrics and let `Generar ABC (YuE2)` compose the score, or import a
  `{style, lyrics, abc}` JSON.
- The piano roll above the tracks shows the score under the original audio.
  Drag a section's left edge, or use `◀ Un compás antes` / `Un compás después ▶` in the
  inspector, to move its start by bars; rename, split at the cursor or merge sections there. ABC markers and lyric
  headers are rewritten together.
  `Terminar la canción aquí` ends the song at the cursor's bar: later bars, and sections left without any (with
  their lyrics), are removed, the old ABC kept in the versions. A section only cut short is not marked edited (its
  take still covers it) and the song fades out over its last second.
- `Generar canción` queues Generate Music Sections, KSampler (`dpmpp_2m`, 40 steps: as good as `dpm_2` in 79
  by ear and log-mel, 4x faster) and separation into lead and instrumental; the take is playable then. With
  `Armonías` on, its Vocal Harmonizer voices follow as a job of their own, shown as in progress on their lanes.
  Every render is a take of the whole song. Edit a section's lyrics or seed and
  `Regenerar sección`: only that section (and any other edited one) switches to
  the new take, with short crossfades at the boundaries.
  Unless the edited sections span most of the song, the KSampler renders only their stretch plus 4 s on each
  side (`window_start` / `window_seconds` of Generate Music Sections, built from the window's own tokens),
  placed at its time in the take: a section is rendered as close to a whole-song render as another noise seed
  is, in a fraction of the time.
  The other sections replay the exact tokens of their chosen take, and the new
  section's last bar is picked from several tries to lead into the next one.
- `Melodía libre` turns a section's vocal line into rests with the chords kept, so YuE2
  speaks or raps its words (with a `spoken word ... no singing` cue in that section's style).
  Speech runs at about 3.5–4 syllables per second and does not wait for the section's end:
  size a spoken section to its text, or YuE2 moves on to the next section's words early and
  repeats lines at the end. `Restaurar melodía escrita` brings the written melody back.
- Every track (lead, instrumental, harmonies, extra voices) is switched on or off per
  section by clicking it, e.g. for a duet;
  each track has mute, solo and volume. `Armonías con agente` asks the local
  Ollama (model under `Avanzado`, prompt in `prompts/harmony_arranger_system.txt`)
  which voices sing in each section; its reason shows in the section inspector.
- A section's own style (inspector, or `Estilos por sección desde [Sección]` from the
  `[Section] …` cue lines of the style) samples it under that style while continuing
  from the music before it. Section styles made from the cues follow later edits of the style, and a style
  change marks the sections singing it as edited.
- `+ Voz` adds another singer: a whole separate generation with its own style and
  seed, singing the lead melody or a Vocal Harmony line, optionally an octave up or
  down (the written register decides the kind of voice YuE2 sings, more than the
  style). Lead and voice are each mapped onto the score by pitch; their difference
  gives a shift per section, refined by rhythm where pitch is unreliable (rap),
  editable in ms. A role (equal, second voice −4 dB, choir −9 dB) sets its level against
  the lead section by section (against the lead's typical level where the lead is off). A
  limiter on the sum keeps playback and `Exportar FLAC` from clipping (output ceiling -1 dBFS). Each voice track is switched per section like the harmonies.
  A harmony track also takes free on/off spans: drag on its lane to switch it from any point to any
  other regardless of sections (a click on a span removes it; a plain click still switches the section).
- The transport shows a peak meter per channel in dBFS, how much the master limiter is reducing, and a CLIP light
  that stays on until clicked. A take's own `mix` file is YuE2's raw decode at full scale: export the song with
  `Exportar FLAC` rather than using that file.
- Classical instrumental lines: put public-domain MusicXML or MIDI scores (`.mxl`/`.musicxml`/`.mid`,
  e.g. CC0 [OpenScore String Quartets](https://github.com/OpenScore/StringQuartets) or
  [Mutopia](https://www.mutopiaproject.org) chorales and guitar studies) in
  `references/`, mark sections in the inspector as `Alta` (first violin: intros, endings,
  choruses) or `Baja` (viola/cello counter-line), and `Aplicar líneas clásicas`. Phrases
  from slow passages in 2/4, 4/4 or 3/4, matching the song's meter (the whole piece when it has no tempo words; a one-track MIDI
  splits into its upper and bass lines) are moved to the song's key by scale degree, their beats
  1 and 3 fitted to the song's chords, and written into the ABC `Ins` line; the vocal line
  and chords are untouched and the project keeps which bars of which work were used.
  YuE2's ABC has a single instrumental line, so this shapes melodic material; the rest of
  the accompaniment still follows the chords and style. `references/phrases.json` caches
  the phrases until a score changes.
- Our own dataset: `Catálogo` → *Agregar grabaciones de referencia* imports several audio
  files at once with their recording license (CC0, public domain, CC-BY, CC-BY-SA), work
  and source URL, and analyzes each with SheetSage2. Their melodies become `Tema` phrases
  for the classical line, with the reference and license in the report.
- `Exportar WAV` renders the current timeline.

The top bar's album menu only chooses which songs the song menu lists (a new song goes into that album);
moving a song to another album is done in `Catálogo`. `☰` folds the side panel.
`Duplicar` (top bar for the open song, or on any song in `Catálogo`) writes `<name> (copia)` in the same album
and opens it; the original package is only read.
`Guardar versión` freezes the song's state under a name; the versions menu opens one (saving the current state
first, or not) or deletes it into «Borradas» in the same menu, where it can be recovered.
`Catálogo` lists every project (date, source audio or composed, sections, takes, voices,
style) and opens it. Projects are `.mixmash` packages under `output/HZ3-YuE2/studio`: a zip with the
source audio, `lyrics.txt`, `score.abc`, `style.txt` and the whole studio project
(original analysis, takes, comping, harmonies, mixer). The catalog's `Abrir .mixmash` and each song's
`Descargar` move them between machines; take audio stays in the output folder, but the
package carries every take's sampled section tokens (a few KB each), restored on open (replacing
the local tokens of the same section key, since another GPU samples other tokens from the same inputs),
so another machine replays them instead of sampling the song again. It also carries the LoRAs of the song and of its singers (installed on open
when missing), and opening a song that sings with a voice LoRA adds that voice to the album's singers when the catalog lacks it.

## Nodes

- HZ3 YuE2 · Audio to Style (Discogs-EffNet)
- HZ3 YuE2 · Procedural Prosody
- HZ3 YuE2 · Lyrics Prosody (Ollama)
- HZ3 YuE2 · MixMash Style (Ollama)
- HZ3 YuE2 · Vocal Harmony
- HZ3 YuE2 · Vocal Harmonizer
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

`Audio to LoRA` learns from the reference's own YuE2 semantic tokens: either `music_tokens`
(e.g. a take's section tokens) or, when empty, the recording tokenized with the real-audio
tokenizer — `tokenizer_head_joint_v4.pt` from yue2-mothersuperior-realaudio-tokenizer-v4 and
`m-a-p/MERT-v2-FullSong` in `models/HZ3-YuE2/MERT-v2-FullSong`. Style mode trains the AR model
on those tokens; voice mode trains the NAR on the recording's latents conditioned on them.

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

`Vocal Harmonizer` makes harmonies from the generated lead itself instead of new
generations: separate the lead vocal, connect it with the generation's ABC, and
it returns one track per voice (tenor, baritone, low, bass, countertenor). Each
sung note is pitch-tracked, given chord tones from the ABC chords, and re-pitched
with PSOLA, so the voices keep the lead's timing, words and timbre. Mix the tracks
you want at your own levels; the ABC must start where the audio starts.

## Karaoke & Audio Visualizer

`Karaoke & Audio Visualizer` is a single node: it times the clean lyrics against the
generated song and renders the karaoke video (word sweep, chords, section HUD, spectrum).
It also returns `timed_lyrics` (JSON with per-word times), `lrc_text` and an alignment
`report`.

Optionally set `background_folder` to a directory of PNG, JPG, JPEG, WEBP, or BMP files.
The visualizer cycles through them alphabetically or in a shuffled order every
`background_interval` seconds. Choose `Cut`, `Crossfade`, `Fade Through Black`, or
`Wipe Left`, set the transition duration, and adjust `background_transparency` (0% is
opaque; 100% shows only the theme background). It keeps only a few resized images in
memory and returns a single preview frame; the MP4 contains the full animation.

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

`Generate Music Sections` samples the whole song as one continuous YuE2 pass, stopping at each ABC section boundary, and stores each section's tokens under `output/HZ3-YuE2/section_tokens`. Editing a section's lyrics or ABC, or rerolling it with `section_seeds` (`Chorus 2 = 1234`), resamples only that section; the rest is replayed from storage. Sung words come from the sampled tokens, so a lyric edit always needs that section resampled. `section_styles` (`Chorus 1: female soprano, Latin pop`) samples a section under its own style while still continuing from the earlier sections' music. When a resampled section is followed by a stored one, its last bar is sampled several times and the ending under which YuE2 finds the next section's opening most likely is kept. The `section_tokens` output in the node's UI result names each section's stored tokens; passing them back (`Chorus 1 = <id>`) replays that take's section exactly.
ABC -> Lyrics Prosody.score_abc
```

The mapping is still a hypothesis because ordinary ABC lacks explicit `w:`
syllable-to-note anchors. The JSON output records that limitation and preserves
the measurements used by the agent.
# Fetch lyrics from an audio input

The **HZ3 YuE2 · Fetch Lyrics from Audio** node takes a ComfyUI `AUDIO` input, creates a local Chromaprint fingerprint with `fpcalc`, identifies the recording through AcoustID, and retrieves plain or synchronized lyrics from LRCLIB. Its `lyrics` output is clean plain text that can be connected directly to MixMash; `synced_lyrics` preserves LRC timestamps separately, along with the matched title, artist, album, confidence, and status.

Setup:

1. The Windows x64 install on this machine includes Chromaprint `fpcalc` under the node's local `vendor` folder, which the node detects automatically. On other systems, install Chromaprint and add `fpcalc` to `PATH`, or enter its full executable path in the optional `fpcalc_path` field. The local binary is ignored by Git so it is not added to source commits.
2. Register a free AcoustID application and enter its client ID in the node. AcoustID's public service is limited to non-commercial usage and asks clients to stay below three requests per second.
3. Connect your input audio. The node sends the fingerprint and duration to AcoustID, not the raw audio. LRCLIB does not require an API key.

This lookup depends on the recording being present in AcoustID and its lyrics being present in LRCLIB. It is intended for identifying released recordings; original or substantially rearranged AI-generated songs may not be recognized. For those, provide the lyrics directly or transcribe the vocal track separately.

When either service returns no catalog match or no lyric record, the node completes with empty lyric outputs and a visible `WARNING` in its UI and `status` output. Authentication, network, and service errors still fail explicitly.

References: [AcoustID API](https://acoustid.org/webservice), [LRCLIB API](https://lrclib.net/docs), [Chromaprint](https://github.com/acoustid/chromaprint).
