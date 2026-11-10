"use strict";

// HZ3 Studio: song timeline over ComfyUI. A project starts from a source audio:
// analysis transcribes its ABC, lyrics and style, the piano roll corrects the
// sections, and every render is a "take" of the whole song. Each section plays
// from the take chosen for it (comping), with short crossfades at section
// boundaries. Harmony voices are tracks switched per section. Projects are
// saved as .mixmash packages and grouped in albums; an album keeps the style
// catalog its songs reference (singers per section, a group or genre as base
// style), the assistant chat and a log of what was done.

const TRACKS = [
  { id: "original", name: "Original", gain: 1, source: true },
  { id: "vocals", name: "Lead vocal", gain: 1 },
  { id: "instrumental", name: "Instrumental", gain: 1 },
  { id: "tenor", name: "Tenor", gain: 0.4, harmony: true },
  { id: "baritone", name: "Baritone", gain: 0.4, harmony: true },
  { id: "low", name: "Low", gain: 0.3, harmony: true },
  { id: "bass", name: "Bass", gain: 0.35, harmony: true },
  { id: "countertenor", name: "Countertenor", gain: 0.22, harmony: true },
];
const VOICES = TRACKS.filter((track) => track.harmony).map((track) => track.id);
const VOICE_LINES = { tenor: 1, baritone: 2, bass: 3 };

// Extra singers are whole separate generations, aligned to the lead by audio.
function allTracks() {
  return [...TRACKS, ...project.voices.map((voice) => ({ id: `voice-${voice.id}`, name: voice.name, gain: voice.autoGain ?? 0.8, voice })), ...instrumentTracks()];
}
const FADE = 0.3;
const PEAKS_PER_SECOND = 100;
const TICKS_PER_QUARTER = 256;
const SECTION_COLORS = ["#e0b04a", "#4aa3e0", "#7bc96f", "#d9714e", "#b07be0", "#4ec9b0", "#e07ba8", "#c9c24a"];
const RENDER_LABELS = { 1: "Loading model", 2: "YuE2 tokens", 4: "KSampler", 5: "Decoding audio", 7: "Separating vocals", 12: "Harmonies" };
const ANALYSIS_LABELS = { 3: "SheetSage2 (ABC)", 4: "Separating vocals", 5: "Whisper (lyrics)", 6: "MixMash (Ollama)" };
const COMPOSE_LABELS = { 1: "Loading model", 2: "YuE2 composes the ABC" };
const STYLE_KINDS = { singer: "Singer", group: "Band", genre: "Genre" };
// Median *written* Vocal note (MIDI) a singer's sections are moved to. YuE2 picks the voice from the melody's register far
// more than from the style text, and sings about an octave below the written ABC: in our test a verse written around 74
// was sung by a soprano near 62, and the same verse written around 62 by a baritone near 50. mid is between, untested.
const REGISTERS = { high: { label: "High (female)", center: 77 }, mid: { label: "Mid (tenor / mezzo)", center: 70 }, low: { label: "Low (male)", center: 62 } };
const DEFAULT_ALBUM = "General";
// Measured on a 3 min song: dpmpp_2m in 40 steps is 4x faster than dpm_2 in 79 (16 s vs 63 s) and sounds the same
// (0.11 dB mean log-mel difference; blind listening test).
const ACOUSTIC_SAMPLING = { steps: 40, sampler_name: "dpmpp_2m" };
const NEW_ALBUM = "*nuevo*";
const NEW_SONG = "*nueva*";
const CHAT_MEMORY = 24;
const ASSISTANT_STEPS = 8;

// Page tools the studio assistant can call; runAction carries them out.
const tool = (name, description, properties = {}, required = []) =>
  ({ type: "function", function: { name, description, parameters: { type: "object", properties, required } } });
const str = (description) => ({ type: "string", description });
const num = (description) => ({ type: "number", description });
const ASSISTANT_TOOLS = [
  tool("set_style", "Replace the song's global style text.", { text: str("YuE2 style text") }, ["text"]),
  tool("set_section_style", "Set a section's own style text; empty text makes it use the global style.", { section: str("section name"), text: str("style text") }, ["section", "text"]),
  tool("set_section_lyrics", "Replace the lyrics of one section.", { section: str("section name"), text: str("lyrics") }, ["section", "text"]),
  tool("set_section_seed", "Set the seed a section renders with; a new seed gives a different performance.", { section: str("section name"), seed: num("integer seed") }, ["section", "seed"]),
  tool("set_sampling", "Change the song's sampling settings for rendering (only the given values change).", {
    temperature: num("default 0.9; higher = more random"), top_p: num("default 0.95"), top_k: num("default 100; higher = more varied"),
    repetition_penalty: num("default 1.2"), cfg_scale: num("default 2.0; how strongly style and lyrics are followed"),
    seed: num("song seed, the default for every section"),
  }),
  tool("get_abc", "Read the song's ABC score, or one section of it.", { section: str("section name, omit for the whole score") }),
  tool("set_abc", "Replace the whole ABC score (it must keep the % section markers matching the lyrics headers).", { abc: str("complete ABC") }, ["abc"]),
  tool("set_compose_settings", "Change the settings compose_abc uses (only the given values change).", {
    seed: num("composition seed; a new one gives a different melody"), temperature: num("default 0.7; higher = further from the usual"),
    keep_key_meter_tempo: { type: "boolean", description: "keep the key, meter and tempo of the current ABC" },
  }),
  tool("compose_abc", "Queue YuE2 to compose a new ABC from the style and lyrics with the compose settings (the current ABC is kept in the ABC versions).", {
    from_section: str("keep the score before this section and compose this section and everything after it; omit to compose the whole score"),
  }),
  tool("free_melody", "Turn a section's written vocal melody into rests (chords and length kept), so YuE2 speaks or raps the words freely instead of singing a melody; for spoken word, trova recitation or rap.", {
    section: str("section name"),
  }, ["section"]),
  tool("set_base_style", "Set the song's base style to a catalog group or genre (null for none).", { style: str("catalog style name") }),
  tool("set_singer", "Choose who sings: a catalog singer for lead sections (moves that section's melody to the singer's register, tags its lyrics and declares the singers in the style), or the singer of a whole extra voice.", {
    track: str("'lead' or an extra voice name"), section: str("lead only: section name, omit for every section"), style: str("catalog singer name, omit for none"),
  }, ["track"]),
  tool("save_style", "Create or update a style in the album catalog.", {
    name: str("style name"), kind: { type: "string", enum: ["singer", "group", "genre"] }, text: str("YuE2 style text in English"),
    register: { type: "string", enum: ["high", "mid", "low"], description: "singers only: the register their sections are moved to" },
  }, ["name", "kind", "text"]),
  tool("delete_style", "Delete a style from the album catalog.", { name: str("style name") }, ["name"]),
  tool("add_voice", "Add an extra voice track (not generated yet).", {
    name: str("voice name"), source: { type: "string", enum: ["lead", "tenor", "baritone", "bass"] }, octave: num("-1, 0 or 1"),
    role: { type: "string", enum: ["0", "-4", "-9"] }, style: str("voice style text"), singer: str("catalog singer name"),
  }, ["name"]),
  tool("set_track", "Switch a track on or off in one section.", {
    track: str("vocals, instrumental, tenor, baritone, low, bass, countertenor or an extra voice name"), section: str("section name"), on: { type: "boolean" },
  }, ["track", "section", "on"]),
  tool("select_take", "Make a section play from one of its takes.", { section: str("section name"), take: num("take id") }, ["section", "take"]),
  tool("arrange_harmonies", "Let the harmony arranger switch harmony tracks per section.", { instructions: str("arranging instructions") }),
  tool("analyze_audio", "Queue the analysis of the source audio (replaces style, lyrics and ABC)."),
  tool("render_song", "Queue a take of the whole song (every section is sung again)."),
  tool("render_sections", "Queue a new take of these sections.", { sections: { type: "array", items: { type: "string" } } }, ["sections"]),
  tool("render_voice", "Queue the generation of an extra voice.", { voice: str("voice name") }, ["voice"]),
  tool("save_song", "Save the song."),
  ...dawTools(),
];

const $ = (id) => document.getElementById(id);
const clientId = crypto.randomUUID();

let project = newProject();
let album = { name: DEFAULT_ALBUM, styles: [], chat: [], log: [] };
let browseAlbum = DEFAULT_ALBUM;  // the album whose songs the top bar lists (the open song may belong to another)
let loras = [];  // YuE2 LoRAs ComfyUI can load: { name, trigger, type }
let catalog = [];
let sections = [];
let score = null;
let selected = null;
let selectedVoice = null;
let pxPerSecond = 6;
const buffers = new Map();
const pending = new Map();
let audioContext = null;
let playback = null;
let pausedAt = 0;

function newProject() {
  return {
    name: "", kind: "song", album: DEFAULT_ALBUM, notes: "", archived: false, license: null, source: null, sourceName: null, styleInstructions: "", analysis: null,
    style: "", negativeStyle: "", baseStyle: null, lora: null, lyrics: "", abc: "", seed: 60,
    compose: { seed: 60, temperature: 0.7, keep: true },
    ckpt: "yue2_3b_int8_convrot.safetensors",
    sampling: { temperature: 0.9, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, cfg_scale: 2.0 },
    harmonize: true, harmonyVoices: [...VOICES], history: null, sectionSeeds: {}, sectionStyles: {}, sectionNegatives: {}, sectionContexts: {}, sectionOnsets: {}, sectionSingers: {}, sectionLocks: {}, takes: [], comp: {}, trackOn: {}, trackSpans: {}, mixer: {}, voices: [], instruments: [], fx: {}, sectionOriginals: {},
    arranger: { model: "deepseek-v4.1-flash:cloud", instructions: "" }, arrangement: {}, classicalPlan: {}, classicalReport: [],
  };
}

function fmt(seconds) {
  seconds = Math.max(0, seconds);
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function status(text, progress = null, error = false) {
  $("status-text").textContent = text;
  $("status").classList.toggle("error", error);
  $("status-progress").value = progress ?? 0;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try { data = JSON.parse(text); } catch { data = null; }
  if (!response.ok) throw new Error(data?.error?.message || data?.error || text || response.statusText);
  return data;
}

function postJson(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function escapeHtml(text) {
  return String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

// ---------- form <-> project

const SAMPLING = ["temperature", "top_p", "top_k", "repetition_penalty", "cfg_scale"];

function readForm() {
  for (const key of ["style", "lyrics", "abc", "ckpt"]) project[key] = $(key).value;
  project.styleInstructions = $("style-instructions").value;
  project.negativeStyle = $("negative-style").value;
  project.baseStyle = $("base-style").value || null;
  const lora = loras.find((item) => item.name === $("song-lora").value);
  project.lora = lora ? { name: lora.name, trigger: lora.trigger, strength: Number.isFinite(parseFloat($("song-lora-strength").value)) ? Number($("song-lora-strength").value) : 1 } : null;
  project.compose = { seed: Number($("compose-seed").value) || 0, temperature: Number($("compose-temperature").value), keep: $("compose-keep").checked };
  project.seed = Number($("seed").value) || 0;
  project.harmonize = $("harmonize").checked;
  for (const key of SAMPLING) project.sampling[key] = Number($(key).value);
  project.arranger = { model: $("arranger-model").value.trim(), instructions: $("arranger-instructions").value };
}

const NEW_BRANCH = "+new-branch";

// The main branch first, then the others by their latest point, newest first.
function branchOrder(history) {
  const latest = (name) => history.commits[history.branches[name].head]?.at ?? 0;
  return Object.keys(history.branches).sort((a, b) => (b === MAIN_BRANCH) - (a === MAIN_BRANCH) || latest(b) - latest(a));
}

function drawBranchList() {
  const history = project.history;
  if (!history) {
    $("branch-list").innerHTML = `<option value="">— branch: ${MAIN_BRANCH} —</option>`;
    return;
  }
  const names = branchOrder(history).filter((name) => !history.branches[name].archived || name === history.current);
  $("branch-list").innerHTML = names.map((name) => `<option value="${escapeHtml(name)}"${name === history.current ? " selected" : ""}>Branch: ${escapeHtml(name)}</option>`).join("")
    + `<option value="${NEW_BRANCH}">+ New branch from here…</option>`;
}

function writeForm() {
  queueMicrotask(refreshAdvanced);
  // A LoRA installed after the page loaded (an imported song brings its own) is still the song's LoRA.
  if (project.lora?.name && !loras.some((lora) => lora.name === project.lora.name)) loras.push({ name: project.lora.name, trigger: project.lora.trigger });
  for (const key of ["style", "lyrics", "abc"]) $(key).value = project[key];
  $("style-instructions").value = project.styleInstructions;
  $("negative-style").value = project.negativeStyle;
  $("base-style").innerHTML = styleOptions(["group", "genre"], "— none", project.baseStyle);
  $("song-lora").innerHTML = `<option value="">— no LoRA (style only)</option>`
    + loras.map((lora) => `<option value="${escapeHtml(lora.name)}"${lora.name === project.lora?.name ? " selected" : ""}>${escapeHtml(loraFile(lora.name))}</option>`).join("");
  $("song-lora-strength").value = project.lora?.strength ?? 1;
  $("compose-seed").value = project.compose.seed;
  $("compose-temperature").value = project.compose.temperature;
  $("compose-keep").checked = project.compose.keep;
  drawBranchList();
  if ([...$("ckpt").options].some((option) => option.value === project.ckpt)) $("ckpt").value = project.ckpt;
  $("seed").value = project.seed;
  $("harmonize").checked = project.harmonize;
  for (const key of SAMPLING) $(key).value = project.sampling[key];
  $("arranger-model").value = project.arranger.model;
  $("arranger-instructions").value = project.arranger.instructions;
}

// ---------- sections, score and lyrics

let sectionsTimer = null;
function scheduleSections() {
  clearTimeout(sectionsTimer);
  sectionsTimer = setTimeout(() => refreshSections().catch((error) => status(error.message, null, true)), 400);
}

async function refreshSections() {
  readForm();
  if (!project.abc.trim() || !project.lyrics.trim()) { sections = []; score = null; draw(); return; }
  sections = (await postJson("/hz3/studio/sections", { abc: project.abc, lyrics: project.lyrics })).sections;
  score = await postJson("/hz3/yue2/abc_viewer/data", { abc: project.abc, lyrics: project.lyrics, sections: editableSections() });
  if (selected && !sections.some((section) => section.name === selected)) selected = null;
  status(`${sections.length} sections · ${fmt(duration())}`);
  draw();
}

function duration() {
  if (sections.length) return sections[sections.length - 1].end;
  return sourceEntry()?.buffer.duration ?? 0;
}

function barSeconds(bar) {
  const ticks = bar < score.bars.length ? score.bars[bar].start : score.total_ticks;
  return (ticks / TICKS_PER_QUARTER) * 60 / score.bpm;
}

function nearestBar(seconds) {
  let best = 0;
  score.bars.forEach((_, index) => {
    if (Math.abs(barSeconds(index) - seconds) < Math.abs(barSeconds(best) - seconds)) best = index;
  });
  return best;
}

function editableSections() {
  // The ABC section markers are the truth; the editor works on their bar ranges.
  let bar = 0;
  return sections.map((section, index) => {
    const start = bar;
    bar += section.bars;
    return { id: `section-${index + 1}`, name: section.name, start_bar: start, end_bar: bar - 1, lyrics: section.lyrics };
  });
}

// What the song keeps per section, for sections that no longer exist (renamed or removed long ago), is dropped.
// Takes and saved song versions keep theirs: they describe what was sung then.
function dropStaleSections() {
  if (!sections.length) return;
  const names = new Set(sections.map((section) => section.name));
  const maps = [project.sectionOriginals, project.comp, project.sectionSeeds, project.sectionStyles, project.sectionNegatives, project.sectionContexts ?? {}, project.sectionOnsets ?? {}, project.sectionSingers, project.sectionLocks ?? {}, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
  for (const voice of project.voices) maps.push(voice.on, voice.sectionOffsets, voice.sectionGains ?? {});
  for (const entry of Object.values(project.mixer)) if (entry.offsets) maps.push(entry.offsets);
  for (const map of maps) for (const name of Object.keys(map ?? {})) if (!names.has(name)) delete map[name];
}

// Renames are [from, to] pairs; several sections may share an old name (YuE2 repeats "% chorus"), so each new name
// gets a copy of what the old name held.
function renameKeys(renames) {
  const maps = [project.sectionOriginals, project.comp, project.sectionSeeds, project.sectionStyles, project.sectionNegatives, project.sectionContexts ?? {}, project.sectionOnsets ?? {}, project.sectionSingers, project.sectionLocks ?? {}, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
  for (const take of project.takes) maps.push(take.sectionLyrics, take.sectionSeeds, take.sectionStyles ?? {}, take.sectionAbc ?? {}, take.sectionTokens ?? {});
  for (const voice of project.voices) maps.push(voice.on, voice.sectionOffsets, voice.sectionGains ?? {});
  for (const entry of Object.values(project.mixer)) if (entry.offsets) maps.push(entry.offsets);
  for (const map of maps) {
    const values = renames.map(([from]) => map[from]);
    for (const [from] of renames) delete map[from];
    renames.forEach(([, to], index) => { if (values[index] !== undefined) map[to] = values[index]; });
  }
}

// The lyrics after a section edit, written here rather than taken from the ABC viewer: each section keeps its own
// lyric tag (with its "– who sings" cue; only the name changes when the section was renamed), and the blocks no ABC
// section sings stay after the section they followed. tagOf maps a section id to the id whose tag it takes (a merge).
function rebuiltLyrics(edited, tagOf = {}) {
  const blocks = lyricBlocks(project.lyrics);
  const ids = sections.map((_, index) => `section-${index + 1}`);
  const present = new Set(edited.map((section) => section.id));
  const tags = new Map();
  const after = new Map();
  let anchor = "";
  const owner = new Map(sections.map((section, index) => [section.lyrics_index, index]).filter(([block]) => block !== null));
  blocks.forEach((block, index) => {
    if (owner.has(index)) {
      const at = owner.get(index);
      const [, base, cue = ""] = block.name.match(/^(.*?)(\s+[–—-]\s+.*|\s*[:|(].*)?$/);
      tags.set(ids[at], { name: sections[at].name, base: base.trim(), cue });
      anchor = ids[at];
    } else {
      after.set(anchor, [...(after.get(anchor) ?? []), project.lyrics.slice(block.start, block.end).trim()]);
    }
  });
  // Blocks that followed a section merged away now follow the section before it.
  let kept = "";
  for (const id of ids) {
    if (present.has(id)) kept = id;
    else if (after.has(id)) after.set(kept, [...(after.get(kept) ?? []), ...after.get(id)]);
  }
  const parts = [...(after.get("") ?? [])];
  for (const section of edited) {
    const tag = tags.get(tagOf[section.id] ?? section.id);
    const body = section.lyrics.trim();
    if (tag || body) {
      const base = tag && tag.name === section.name ? tag.base : section.name;
      parts.push(`[${base}${tag?.cue ?? ""}]` + (body ? `\n${body}` : ""));
    }
    if (present.has(section.id)) parts.push(...(after.get(section.id) ?? []));
  }
  return parts.join("\n\n") + "\n";
}

// keepLyrics: only the ABC changes (renaming sections after their own lyric tags).
async function applySections(edited, renames = [], keepLyrics = false, tagOf = {}) {
  readForm();
  // Pin every section's seed by name so reshaping one section does not reroll the others.
  sections.forEach((section, index) => { project.sectionSeeds[section.name] ??= project.seed + index; });
  const lyrics = keepLyrics ? project.lyrics : rebuiltLyrics(edited, tagOf);
  renameKeys(renames);
  const data = await postJson("/hz3/yue2/abc_viewer/data", { abc: project.abc, lyrics: project.lyrics, sections: edited });
  project.abc = data.edited_abc;
  project.lyrics = lyrics;
  $("abc").value = project.abc;
  $("lyrics").value = project.lyrics;
  await refreshSections();
}

// Sections are keyed by name (takes, seeds, singers); YuE2 may repeat "% chorus". Each section takes the name of the
// lyric tag it is paired with (without a "– who sings" cue), or a number when that is taken or missing.
async function uniqueSectionNames() {
  if (!score || new Set(sections.map((section) => section.name)).size === sections.length) return [];
  const edited = editableSections();
  const blocks = lyricBlocks(project.lyrics);
  const seen = new Set();
  const renames = [];
  edited.forEach((section, index) => {
    const paired = sections[index].lyrics_index === null ? "" : blocks[sections[index].lyrics_index]?.name.split(/\s+[–—-]\s+|\s*[:|(]/)[0].trim() ?? "";
    let name = paired && !seen.has(paired.toLowerCase()) ? paired : section.name;
    for (let count = 2; seen.has(name.toLowerCase()); count++) name = `${section.name} ${count}`;
    seen.add(name.toLowerCase());
    if (name !== section.name) renames.push([section.name, name]);
    section.name = name;
  });
  await applySections(edited, renames, true);
  return renames;
}

function lyricBlocks(text) {
  const pattern = /^[ \t]*\[([^\]\n]+)\][ \t]*$/gm;
  const blocks = [];
  let match;
  while ((match = pattern.exec(text))) {
    if (blocks.length) blocks[blocks.length - 1].end = match.index;
    blocks.push({ name: match[1].trim(), start: match.index, bodyStart: match.index + match[0].length, end: text.length });
  }
  return blocks;
}

// Sections and lyric blocks are paired by the server (lyrics_index); an ABC section without lyrics
// gets a block of its own, written before the block of the next section that has one.
function setSectionLyrics(index, body) {
  const blocks = lyricBlocks(project.lyrics);
  const section = sections[index];
  if (section.lyrics_index === null) {
    if (!body.trim()) return;
    const next = sections.slice(index + 1).find((item) => item.lyrics_index !== null);
    const at = next ? blocks[next.lyrics_index].start : project.lyrics.length;
    const before = project.lyrics.slice(0, at).replace(/\s*$/, at ? "\n\n" : "");
    const name = section.name.replace(/^./, (letter) => letter.toUpperCase());
    project.lyrics = `${before}[${name}]\n${body.trim()}\n${next ? "\n" : ""}${project.lyrics.slice(at)}`;
  } else {
    const block = blocks[section.lyrics_index];
    const tail = section.lyrics_index === blocks.length - 1 ? "\n" : "\n\n";
    project.lyrics = project.lyrics.slice(0, block.bodyStart) + "\n" + body.trim() + tail + project.lyrics.slice(block.end);
  }
  $("lyrics").value = project.lyrics;
}

function sectionSeed(section, index) {
  return project.sectionSeeds[section.name] ?? project.seed + index;
}

function takeById(id) {
  return project.takes.find((take) => take.id === id);
}

// A kept section (🔒) plays its chosen take's tokens in «Generate song» instead of being sung again.
function isKept(section) {
  return Boolean(project.sectionLocks?.[section.name] && takeById(project.comp[section.name]));
}

function keepSections(names, keep) {
  project.sectionLocks ??= {};
  for (const name of names) {
    if (keep) project.sectionLocks[name] = true;
    else delete project.sectionLocks[name];
  }
}

function isEdited(section, index) {
  return editedParts(section, index).length > 0;
}

// What of a section differs from what its take sang (so a render would change it): "lyrics", "style", "seed", "score".
function editedParts(section, index) {
  const take = takeById(project.comp[section.name]);
  if (!take) return [];
  const abc = take.sectionAbc?.[section.name];
  // Takes since the global style was recorded: a section singing the global style changes when that style changed.
  const globalChanged = !project.sectionStyles[section.name] && take.style !== undefined && take.style !== globalStyle();
  return [
    (take.sectionLyrics[section.name] ?? "").trim() !== section.lyrics.trim() && "lyrics",
    (globalChanged || (take.sectionStyles?.[section.name] ?? "") !== sectionStyle(section)) && "style",
    take.sectionSeeds[section.name] !== sectionSeed(section, index) && "seed",
    Boolean(take.sectionAbc && abc !== section.abc && !trimmedFrom(abc, section.abc)) && "score",
  ].filter(Boolean);
}

// Bars of a score's music lines per voice, a multi-bar rest "Z3" counted as three bars.
function barsByVoice(abc) {
  const bars = { Vocal: [], Ins: [] };
  let voice = null;
  for (const line of (abc ?? "").split("\n")) {
    if (/^V:\s*(Vocal|Ins)\s*$/.test(line.trim())) voice = line.trim().slice(2).trim();
    else if (voice && line.trim().endsWith("|") && !/^[A-Za-z]:/.test(line)) {
      for (const bar of line.trim().slice(0, -1).split("|")) {
        const rest = bar.trim().match(/^Z(\d+)$/);
        bars[voice].push(...(rest ? Array(Number(rest[1])).fill("Z") : [bar.trim()]));
      }
    }
  }
  return bars;
}

// A section whose bars are the first bars of what its take sang: the take still covers it, it just ends sooner.
function trimmedFrom(original, abc) {
  if (!original) return false;
  const [before, after] = [barsByVoice(original), barsByVoice(abc)];
  return after.Vocal.length > 0 && after.Vocal.length < before.Vocal.length
    && ["Vocal", "Ins"].every((voice) => after[voice].every((bar, index) => bar.replace(/-$/, "") === before[voice][index].replace(/-$/, "")));
}

// The song keeps its first `keep` bars: later bars and the sections (with their lyrics) left without any are removed.
function truncateAbc(abc, keep) {
  const out = [];
  const counts = { Vocal: 0, Ins: 0 };
  let voice = null, switchLine = null, marker = null;
  for (const line of abc.split("\n")) {
    if (line.startsWith("% ")) { marker = line; continue; }
    if (/^V:\s*(Vocal|Ins)\s*$/.test(line.trim())) { voice = line.trim().slice(2).trim(); switchLine = line; continue; }
    if (voice && line.trim().endsWith("|") && !/^[A-Za-z]:/.test(line)) {
      const bars = barsByVoice(`V: ${voice}\n${line}`)[voice];
      const kept = bars.slice(0, Math.max(0, keep - counts[voice]));
      counts[voice] += bars.length;
      if (!kept.length) continue;
      if (counts[voice] >= keep) kept[kept.length - 1] = kept.at(-1).replace(/-\s*$/, "");
      if (marker) { out.push(marker); marker = null; }
      out.push(switchLine, kept.join("|") + "|");
      continue;
    }
    out.push(line);
  }
  return out.join("\n");
}

async function endSongHere() {
  const bar = nearestBar(position());
  if (bar < 1 || bar >= score.bars.length) throw new Error("Put the cursor on the bar where the song should end.");
  readForm();
  const removed = sections.filter((section) => section.start >= barSeconds(bar) - 1e-6);
  const blocks = lyricBlocks(project.lyrics);
  let lyrics = project.lyrics;
  for (const index of removed.map((section) => section.lyrics_index).filter((index) => index !== null).sort((a, b) => b - a)) {
    lyrics = lyrics.slice(0, blocks[index].start) + lyrics.slice(blocks[index].end);
  }
  keepAbcVersion(project, `before ending the song at bar ${bar + 1}`, true);
  project.abc = truncateAbc(project.abc, bar);
  project.lyrics = lyrics.trimEnd() + "\n";
  writeForm();
  await refreshSections();
  await saveProject();
  status(`The song now ends at ${fmt(duration())}${removed.length ? ` · removed: ${removed.map((section) => section.name).join(", ")}` : ""}. The previous ABC is in the versions.`, 1);
}

// A song cut inside its last section fades out over its last second instead of stopping dead.
function endsTrimmed() {
  const last = sections.at(-1);
  const take = last && takeById(project.comp[last.name]);
  return Boolean(take && trimmedFrom(take.sectionAbc?.[last.name], last.abc));
}

// MixMash-style cues: global lines, then "[Section] description" lines.
// A section's cue is found by its ABC name ("verse 2") or by the lyric tag it is paired with ("Verse 2"),
// since ABC sections are often named "verse" while the cues follow the lyrics.
function sectionCueStyle(section, style = project.style) {
  const lines = style.split("\n").filter((line) => line.trim());
  const global = lines.filter((line) => !line.trim().startsWith("[")).join("\n").trim();
  const header = section.lyrics_index === null ? "" : lyricBlocks(project.lyrics)[section.lyrics_index]?.name.split(/\s+[–—-]\s+|\s*[:|(]/)[0].trim() ?? "";
  const cues = lines.map((line) => line.match(/^\s*\[([^\]]+)\]\s*(.+)$/)).filter(Boolean);
  const cue = [section.name, header].filter(Boolean).map((name) => cues.find((match) => match[1].trim().toLowerCase() === name.toLowerCase())).find(Boolean);
  return { global, cue: cue?.[2].trim() };
}

// Section styles made from the cues follow later edits of the song's style: a section whose own style is still what
// the previous text gave it is rewritten from the new text (or goes back to the global style if its cue is gone).
function followStyle(before, after) {
  if (before === after) return;
  let count = 0;
  for (const section of sections) {
    const old = sectionCueStyle(section, before);
    if (!old.cue || project.sectionStyles[section.name] !== `${old.global}\n${old.cue}`) continue;
    const { global, cue } = sectionCueStyle(section, after);
    if (cue) project.sectionStyles[section.name] = `${global}\n${cue}`;
    else delete project.sectionStyles[section.name];
    count++;
  }
  if (count) status(`${count} section styles updated with the new style.`, 1);
  draw();
}

function stylesFromCues() {
  readForm();
  let count = 0;
  for (const section of sections) {
    const { global, cue } = sectionCueStyle(section);
    if (!cue) continue;
    project.sectionStyles[section.name] = `${global}\n${cue}`;
    count++;
  }
  status(count ? `${count} sections with their own style.` : "No «[Section] …» lines match the sections.", 1, !count);
  draw();
}

// ---------- album style catalog

function catalogStyle(id) {
  return album.styles.find((style) => style.id === id);
}

function styleOptions(kinds, empty, value) {
  return `<option value="">${escapeHtml(empty)}</option>` + album.styles.filter((style) => kinds.includes(style.kind))
    .map((style) => `<option value="${style.id}"${style.id === value ? " selected" : ""}>${escapeHtml(style.name)}</option>`).join("");
}

function oneLine(text) {
  return text.split("\n").map((line) => line.trim().replace(/[,;.]+$/, "")).filter(Boolean).join(", ");
}

function globalStyle() {
  return [catalogStyle(project.baseStyle)?.text, project.style].filter((text) => text?.trim()).join("\n");
}

// A section with its own style or its own singer sings the base style, its text (or the song's) and only its singer's
// description; "" means the global style. The inspector shows the result as sent.
function sectionStyle(section) {
  const own = project.sectionStyles[section.name];
  const singer = catalogStyle(project.sectionSingers[section.name])?.text;
  if (!own && !singer) return "";
  return [catalogStyle(project.baseStyle)?.text, own || project.style, singer].filter((text) => text?.trim()).join("\n");
}

// What YuE2 receives for a section: its tokens are sampled under the first text, the audio of the whole song under the
// second; `withLora` puts the LoRA trigger in front of both the same way.
function sectionNegative(section) {
  return (project.sectionNegatives[section.name] || project.negativeStyle).trim();
}

// Voice contrast, the recipe that gave the clearest voices: each section's CFG negative is its own style with the
// other singers of the song instead of its own. Written into the sections' negatives, where it can be read and edited.
function voiceContrast() {
  readForm();
  const used = [...new Set(sections.map((section) => project.sectionSingers[section.name]).filter((id) => catalogStyle(id)))];
  if (used.length < 2) throw new Error("Voice contrast needs two singers in the song: choose each section's singer first.");
  let count = 0;
  for (const section of sections) {
    const own = project.sectionSingers[section.name];
    if (!catalogStyle(own)) continue;
    const others = used.filter((id) => id !== own).map((id) => catalogStyle(id).text);
    project.sectionNegatives[section.name] = [catalogStyle(project.baseStyle)?.text, project.sectionStyles[section.name] || project.style, ...others]
      .filter((text) => text?.trim()).map(oneLine).join(", ");
    count++;
  }
  const warnings = [
    project.sampling.cfg_scale <= 1 && "CFG is 1, so the negative does nothing: raise it",
    loraUses().some((use) => use.lora?.name) && "a LoRA trained on a take outweighs the negatives (measured): the voices may not change",
  ].filter(Boolean);
  draw();
  status(`Voice contrast: ${count} sections have the other singer as negative (see «What YuE2 receives»).${warnings.length ? ` Note: ${warnings.join("; ")}.` : ""}`, 1, warnings.length > 0);
}

function clearVoiceContrast() {
  project.sectionNegatives = {};
  draw();
  status("Section negatives removed.", 1);
}

// The album's concept song: the song whose takes are the album's reference for voice and band. Its section with the
// same name, in its chosen take, becomes this section's context.
async function contextFromConcept(name) {
  if (!album.concept) throw new Error(`The album «${album.name}» has no concept song: choose it in Catalog, in the album.`);
  if (album.concept === project.name) throw new Error("This song is the album's concept song.");
  const source = await api(`/hz3/studio/project?name=${encodeURIComponent(album.concept)}`);
  const names = Object.keys(source.comp ?? {}).filter((item) => source.comp[item]);
  if (!names.length) throw new Error(`«${album.concept}» has no chosen takes yet.`);
  const key = names.find((item) => item.toLowerCase() === name.toLowerCase())
    ?? await choose(`Section of «${album.concept}»`, `The concept song has no «${name}». Which of its sections should YuE2 hear before this one?`,
      names.map((item) => ({ value: item, label: item })));
  if (!key) return;
  await setSectionContext(name, album.concept, source.comp[key], key);
}

function sentPrompt(section) {
  const trigger = loraUses().find((use) => use.lora?.name)?.lora.trigger;
  const withTrigger = (text) => trigger ? `${trigger}, ${text}` : text;
  const own = sectionStyle(section);
  const negativeText = sectionNegative(section);
  const negative = negativeText ? `\n\nCFG negative (with the same lyrics and ABC):\n${withTrigger(oneLine(negativeText))}` : "";
  const lead = project.sectionContexts?.[section.name];
  const context = lead ? `\n\nBefore sampling it, as if it had just played (not kept in the song):\n${contextLabel(lead)}` : "";
  const start = project.sectionOnsets?.[section.name];
  const onset = start ? `\n\nFixed opening (kept; the rest is sampled after it):\n${start.label} · ${start.seconds} s` : "";
  return `Tokens of this section:\n${withTrigger(own ? oneLine(own) : globalStyle())}${negative}${context}${onset}\n\nAudio of the whole song:\n${withTrigger(globalStyle())}`;
}

// A section's fixed opening: stored tokens shorter than the section, kept as its first frames while the rest of the
// section is sampled after them. `tokens` is the stored id, `label` says where they came from.
function setSectionOnset(name, tokens, label, seconds) {
  if (!tokens) delete project.sectionOnsets[name];
  else project.sectionOnsets[name] = { tokens, label, seconds };
  return project.sectionOnsets[name] ?? null;
}

// A section's context: the stored tokens of a section of another song's take, which YuE2 sees right before this
// section while sampling it, so that take's voice and band carry over.
function contextLabel(lead) {
  return `${lead.project} · take ${lead.take} · ${lead.section}`;
}

async function setSectionContext(name, sourceProject, takeId, sourceSection) {
  if (!sourceProject) {
    delete project.sectionContexts[name];
    return null;
  }
  const source = sourceProject === project.name ? project : await api(`/hz3/studio/project?name=${encodeURIComponent(sourceProject)}`);
  const take = source.takes?.find((item) => item.id === Number(takeId));
  if (!take) throw new Error(`«${sourceProject}» has no take ${takeId}.`);
  const key = Object.keys(take.sectionTokens ?? {}).find((item) => item.toLowerCase() === String(sourceSection).trim().toLowerCase());
  if (!key) throw new Error(`Take ${takeId} of «${sourceProject}» has no tokens for «${sourceSection}».`);
  project.sectionContexts[name] = { project: sourceProject, take: take.id, section: key, tokens: take.sectionTokens[key] };
  return project.sectionContexts[name];
}

// The current song's singer as a catalog singer: the vocal descriptors of its global style, its voice LoRA, and the
// register its ABC melody is written in. A LoRA names the singer after its trigger.
function songVoice() {
  const global = project.style.split("\n").filter((line) => line.trim() && !line.trim().startsWith("[")).join(", ");
  const vocal = /\b(vocals?|voice|singer|singing|sung|rap|rapper|spoken|tenor|baritone|bass voice|soprano|alto|mezzo|falsetto|croon|choir|male|female|breathy|raspy|husky)\b/i;
  const text = global.split(",").map((part) => part.trim()).filter((part) => vocal.test(part)).join(", ");
  if (!text && !project.lora) return null;
  const sung = score?.tracks.Vocal ?? [];
  const median = sung.length ? sung.map((note) => note.pitch).sort((a, b) => a - b)[Math.floor(sung.length / 2)] : null;
  const register = median === null ? null
    : Object.entries(REGISTERS).reduce((best, entry) => Math.abs(entry[1].center - median) < Math.abs(best[1].center - median) ? entry : best)[0];
  const named = project.lora?.trigger?.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
  const style = { id: crypto.randomUUID().slice(0, 8), name: (named || `Voice of ${project.name || "the song"}`).slice(0, 60), kind: "singer", text };
  if (register) style.register = register;
  if (project.lora) style.lora = { ...project.lora };
  return style;
}

// A singer already in the catalog: the same voice LoRA, or (without one) the same vocal description.
function knownVoice(style) {
  return album.styles.some((item) => item.kind === "singer"
    && (style.lora ? item.lora?.name === style.lora.name : !item.lora && item.text === style.text));
}

async function addSongVoice() {
  readForm();
  const style = songVoice();
  if (!style) throw new Error("The song's style does not describe the voice (e.g. «Spanish female lead vocal, warm») and it has no voice LoRA.");
  album.styles.push(style);
  await saveAlbum();
  remember(`Cantante «${style.name}» creado con la voz de la canción`);
  drawStyles();
  writeForm();
  status(`«${style.name}» added to the catalog${project.lora ? " with the song's LoRA" : ""}. Check its text and name.`, 1);
}

function singerRegister(singer) {
  return singer.register ?? (/\b(male|man|men|baritone|bass)\b/i.test(singer.text) ? "low" : /\b(tenor|mezzo)/i.test(singer.text) ? "mid" : "high");
}

function singerCue(singer) {
  return singer.text.split(",")[0].trim() || singer.name;
}

// One ABC line's notes moved by whole octaves (c -> C -> C,); chord symbols, inline keys and rests stay.
function shiftOctaves(line, octaves) {
  return line.replace(/"[^"]*"|\[K:[^\]]*\]|([A-Ga-g])([,']*)/g, (match, letter, marks) => {
    if (!letter) return match;
    const height = (letter === letter.toLowerCase() ? 1 : 0) + (marks.match(/'/g)?.length ?? 0) - (marks.match(/,/g)?.length ?? 0) + octaves;
    return height >= 1 ? letter.toLowerCase() + "'".repeat(height - 1) : letter.toUpperCase() + ",".repeat(-height);
  });
}

// The section's Vocal lines rewritten by `rewrite(line, barUnits)`. A note tied across the section's edges is struck
// again, since the two sides no longer match.
function rewriteSectionVocal(abc, name, rewrite) {
  const lines = abc.split("\n");
  const unit = Number(abc.match(/^L:1\/(\d+)/m)?.[1] ?? 32);
  let meter = abc.match(/^M:(\S+)/m)?.[1] ?? "4/4";
  const vocalLines = {};
  let section = null;
  let vocal = false;
  for (const [at, line] of lines.entries()) {
    if (line.startsWith("% ")) section = line.slice(2).trim();
    else if (line.startsWith("M:")) meter = line.slice(2).trim();
    else if (line.startsWith("V:")) vocal = line.trim() === "V: Vocal";
    else if (vocal && line.endsWith("|")) {
      (vocalLines[section] ??= []).push(at);
      const [beats, value] = meter.split("/").map(Number);
      if (section === name) lines[at] = rewrite(line, beats * unit / value);
    }
  }
  const names = Object.keys(vocalLines);
  for (const edge of [names[names.indexOf(name) - 1], name]) {
    const last = vocalLines[edge]?.at(-1);
    if (last !== undefined) lines[last] = lines[last].replace(/-\|$/, "|");
  }
  return lines.join("\n");
}

const REST_SIZES = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1];

function rests(units) {
  let text = "";
  for (let left = units; left > 0;) {
    const size = REST_SIZES.find((value) => value <= left);
    text += `z${size}`;
    left -= size;
  }
  return text;
}

// A sung bar turned into rests that keep its chord symbols and key changes where they were.
function freeBar(bar, units) {
  if (/^\s*Z\d?\s*$/.test(bar)) return bar.trim();
  const marks = [];
  let offset = 0;
  for (const match of bar.matchAll(/"([^"\n]*)"|(\[K:[^\]\n]+\])|(?:\^\^|__|\^|_|=)?[A-Ga-gz][,']*(\d*)-?/g)) {
    if (match[1] !== undefined) marks.push([offset, `"${match[1]}"`]);
    else if (match[2]) marks.push([offset, match[2]]);
    else offset += Number(match[3] || 1);
  }
  if (!marks.length) return "Z";
  let text = rests(marks[0][0]);
  marks.forEach(([start, mark], index) => {
    const end = marks.slice(index + 1).find(([next]) => next > start)?.[0] ?? units;
    text += mark + (marks[index + 1]?.[0] === start ? "" : rests(end - start));
  });
  return text;
}

// A section whose Vocal line is only rests (chords kept): YuE2 then delivers the words freely, spoken or rapped,
// in the style's own range, instead of singing a written melody.
// Line numbers of a section's music lines (Vocal and Ins), in order.
function sectionMusicLines(abc, name) {
  let section = null;
  return abc.split("\n").flatMap((line, at) => {
    if (line.startsWith("% ")) section = line.slice(2).trim();
    return section === name && line.endsWith("|") && !line.startsWith("V:") ? [at] : [];
  });
}

// What a section's score and lyrics were before "Melodía libre" or "Instrumental", so it can go back to them.
function keepSectionOriginal(name, kind) {
  if (project.sectionOriginals[name]) return;
  const lines = project.abc.split("\n");
  const section = sections.find((item) => item.name === name);
  project.sectionOriginals[name] = { kind, lines: sectionMusicLines(project.abc, name).map((at) => lines[at]), lyrics: section.lyrics };
}

// Sections changed before originals were kept: the ABC version saved just before the change.
function sectionOriginal(name) {
  if (name in project.sectionOriginals) return project.sectionOriginals[name] || null;  // null: already restored
  // Older songs carry the Spanish labels these versions were saved with.
  const labels = {
    [`antes de liberar la melodía de «${name}»`]: "free", [`antes de volver instrumental «${name}»`]: "instrumental",
    [`before freeing the melody of «${name}»`]: "free", [`before making «${name}» instrumental`]: "instrumental",
  };
  const version = (project.abcVersions ?? []).findLast((item) => item.label in labels);
  if (!version) return null;
  const lines = version.abc.split("\n");
  return { kind: labels[version.label], lines: sectionMusicLines(version.abc, name).map((at) => lines[at]), lyrics: null };
}

async function restoreSectionOriginal(name) {
  readForm();
  const original = sectionOriginal(name);
  const at = sectionMusicLines(project.abc, name);
  if (at.length !== original.lines.length) throw new Error(`«${name}» has changed bars since then: recover it from the ABC versions.`);
  const lines = project.abc.split("\n");
  at.forEach((line, index) => { lines[line] = original.lines[index]; });
  const abc = lines.join("\n");
  await postJson("/hz3/studio/sections", { abc, lyrics: project.lyrics });
  setAbc(abc, `before restoring the written melody of «${name}»`);
  await refreshSections();
  const index = sections.findIndex((section) => section.name === name);
  if (original.kind === "instrumental" && original.lyrics !== null && index >= 0) {
    setSectionLyrics(index, original.lyrics);
    await refreshSections();
  }
  project.sectionOriginals[name] = null;
  remember(`«${name}» vuelve a su melodía escrita`);
  status(`«${name}» got its written melody back${original.kind === "instrumental" && original.lyrics !== null ? " and its lyrics" : ""}.`, 1);
  draw();
}

async function setSectionFree(name) {
  readForm();
  keepSectionOriginal(name, "free");
  const abc = rewriteSectionVocal(project.abc, name, (line, units) => line.slice(0, -1).split("|").map((bar) => freeBar(bar, units)).join("|") + "|");
  if (abc === project.abc) {
    if (project.sectionOriginals[name]?.kind === "free") delete project.sectionOriginals[name];
    throw new Error(`«${name}» no longer has a written melody.`);
  }
  await postJson("/hz3/studio/sections", { abc, lyrics: project.lyrics });
  setAbc(abc, `before freeing the melody of «${name}»`);
  await refreshSections();
}

// A melody sung into the microphone becomes the section's Vocal line: a bar of clicks counts in, the song plays from
// the section (headphones keep it out of the recording) and the take ends with the section; chords stay.
async function recordSectionMelody(name) {
  readForm();
  const section = sections.find((item) => item.name === name);
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const audio = context();
  await audio.resume();
  pause();
  const bar = (section.end - section.start) / section.bars;
  const seconds = section.end - section.start;
  const begin = audio.currentTime + 0.3 + bar;
  // A click per beat: the count-in bar, and the whole section too when there is no take to sing along with.
  const clicks = project.takes.length ? 4 : 4 + 4 * section.bars;
  for (let beat = 0; beat < clicks; beat++) {
    const at = begin - bar + beat * bar / 4;
    const click = audio.createOscillator();
    const gain = audio.createGain();
    click.frequency.value = beat % 4 ? 1000 : 1500;
    gain.gain.setValueAtTime(0.3, at);
    gain.gain.exponentialRampToValueAtTime(0.001, at + 0.05);
    click.connect(gain).connect(audio.destination);
    click.start(at);
    click.stop(at + 0.06);
  }
  if (project.takes.length) {
    playback = { from: section.start, start: begin, ...buildGraph(audio, audio.destination, section.start, begin) };
    $("play").textContent = "❚❚";
    requestAnimationFrame(tick);
  }
  // The singer follows what they hear, which leaves the speakers outputLatency after it was scheduled.
  const heard = begin + (audio.outputLatency || audio.baseLatency || 0);
  const input = audio.createMediaStreamSource(stream);
  const recorder = audio.createScriptProcessor(1024, 1, 1);
  const mute = audio.createGain();
  mute.gain.value = 0;
  const samples = new Float32Array(Math.ceil(seconds * audio.sampleRate));
  recorder.onaudioprocess = (event) => {
    const data = event.inputBuffer.getChannelData(0);
    const at = Math.round((audio.currentTime - data.length / audio.sampleRate - heard) * audio.sampleRate);
    for (let index = Math.max(0, -at); index < data.length && at + index < samples.length; index++) samples[at + index] = data[index];
  };
  input.connect(recorder).connect(mute).connect(audio.destination);
  status(`One bar count-in, then sing «${name}» (${fmt(seconds)})…`, 0);
  while (audio.currentTime < heard + seconds + 0.2) await new Promise((resolve) => setTimeout(resolve, 100));
  recorder.disconnect();
  input.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  pause();
  const take = audio.createBuffer(1, samples.length, audio.sampleRate);
  take.copyToChannel(samples, 0);
  const form = new FormData();
  form.append("abc", project.abc);
  form.append("name", name);
  form.append("audio", new Blob([wav(take)], { type: "audio/wav" }), "voz.wav");
  const response = await fetch("/hz3/studio/sung_melody", { method: "POST", body: form });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  if (!result.notes) throw new Error("No sung note was heard: check the microphone and record again.");
  setAbc(result.abc, `before recording the melody of «${name}»`);
  await refreshSections();
  remember(`Melodía de «${name}» grabada con el micrófono (${result.notes} notas)`);
  status(`«${name}»: ${result.notes} notes from your voice in the score. Listen with «▶ Score» and regenerate the section so YuE2 sings them.`, 1);
}

// Lead singer of a section: its lyric tag says who sings and its style carries the singer's description. The melody
// stays where it was written unless `moveMelody`: another octave changes who sings, not only the register.
async function setSectionSinger(name, id, moveMelody = false) {
  const index = sections.findIndex((section) => section.name === name);
  if (index < 0) throw new Error(`there is no section «${name}»`);
  readForm();
  const singer = catalogStyle(id);
  let lyrics = project.lyrics;
  const block = sections[index].lyrics_index === null ? null : lyricBlocks(lyrics)[sections[index].lyrics_index];
  if (block) {
    const base = block.name.split(/\s+[–—-]\s+|\s*[:|(]/)[0].trim();
    const tag = singer ? `[${base} – ${singerCue(singer)}]` : `[${base}]`;
    lyrics = lyrics.slice(0, block.start) + tag + lyrics.slice(block.bodyStart);
  }
  const notes = moveMelody && singer && score ? (() => {
    const from = score.bars[score.sections[index].start_bar].start;
    const to = index + 1 < score.sections.length ? score.bars[score.sections[index + 1].start_bar].start : score.total_ticks;
    return (score.tracks.Vocal ?? []).filter((note) => note.start >= from && note.start < to).map((note) => note.pitch).sort((a, b) => a - b);
  })() : [];
  const octaves = notes.length ? Math.round((REGISTERS[singerRegister(singer)].center - notes[Math.floor(notes.length / 2)]) / 12) : 0;
  const abc = octaves ? rewriteSectionVocal(project.abc, name, (line) => shiftOctaves(line, octaves)) : project.abc;
  // The parser checks the result before it replaces the song's score and lyrics.
  await postJson("/hz3/studio/sections", { abc, lyrics });
  if (octaves) keepAbcVersion(project, `before moving «${name}» to the range of ${singer.name}`);
  setSinger(project.sectionSingers, name, id);
  project.abc = abc;
  project.lyrics = lyrics;
  writeForm();
  await refreshSections();
  return octaves;
}

// A LoRA patches the whole render (both the token model and the acoustic model), so one render carries at most
// one singer's LoRA; another singer with a LoRA sings as an extra voice, which is a render of its own.
function withLora(prompt, uses) {
  const distinct = [...new Map(uses.filter((use) => use.lora?.name).map((use) => [use.lora.name, use])).values()];
  if (distinct.length > 1) {
    throw new Error(`${distinct.map((use) => use.owner).join(" and ")} use different LoRAs: a render carries a single LoRA; sing one of them as an extra voice.`);
  }
  if (!distinct.length) return prompt;
  const { name, trigger, strength } = distinct[0].lora;
  prompt[21] = { class_type: "LoraLoader", inputs: { model: ["1", 0], clip: ["1", 1], lora_name: name, strength_model: strength, strength_clip: strength } };
  prompt[2].inputs.clip = ["21", 1];
  prompt[4].inputs.model = ["21", 0];
  if (trigger) {
    prompt[2].inputs.style = `${trigger}, ${prompt[2].inputs.style}`;
    if (prompt[2].inputs.negative_style) prompt[2].inputs.negative_style = `${trigger}, ${prompt[2].inputs.negative_style}`;
    if (prompt[2].inputs.section_negative_styles) {
      prompt[2].inputs.section_negative_styles = prompt[2].inputs.section_negative_styles.split("\n").map((line) => line.replace(/^([^:]+:\s*)/, `$1${trigger}, `)).join("\n");
    }
    prompt[2].inputs.section_styles = prompt[2].inputs.section_styles.split("\n").map((line) => line.replace(/^([^:]+:\s*)/, `$1${trigger}, `)).join("\n");
  }
  return prompt;
}

// ---------- freezing a voice: a LoRA trained on audio already generated for a singer

let freezing = null;

function freezeClips() {
  const clips = [];
  for (const take of project.takes) {
    if (take.files.mix) clips.push({ label: `Take ${take.id} · mix`, file: take.files.mix });
    if (take.files.vocals) clips.push({ label: `Take ${take.id} · vocals only`, file: take.files.vocals });
  }
  for (const voice of project.voices) {
    if (voice.mixFile) clips.push({ label: `${voice.name} · mix`, file: voice.mixFile });
    if (voice.file) clips.push({ label: `${voice.name} · vocals only`, file: voice.file });
  }
  return clips;
}

function openFreeze(singer) {
  freezing = singer;
  $("freeze-title").textContent = `Freeze the voice of ${singer.name}`;
  const clips = freezeClips();
  $("freeze-clips").innerHTML = clips.length ? "" : `<p class="note">This song has no generated audio yet: generate takes or voices with ${escapeHtml(singer.name)} and come back here.</p>`;
  for (const clip of clips) {
    const row = document.createElement("label");
    row.className = "switch";
    row.innerHTML = `<input type="checkbox"> <button type="button" title="Listen">▶</button> ${escapeHtml(clip.label)}`;
    row.querySelector("input").clip = clip;
    row.querySelector("button").onclick = (event) => { event.preventDefault(); new Audio(fileUrl(clip.file)).play(); };
    $("freeze-clips").append(row);
  }
  $("freeze").showModal();
}

async function freezeVoice() {
  const singer = freezing;
  const files = [...$("freeze-clips").querySelectorAll("input:checked")].map((box) => box.clip.file);
  if (!files.length) throw new Error("Choose at least one audio file with the voice you want to freeze.");
  readForm();
  const slug = singer.name.normalize("NFKD").replace(/[^\w]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase() || "voz";
  loras = (await api("/hz3/studio/loras")).loras;
  // A new version never overwrites an earlier LoRA of the same singer.
  const base = (version) => `hz3_${slug}${version > 1 ? `_v${version}` : ""}`;
  let version = 1;
  while (loras.some((lora) => lora.name.replace(/^.*[\\/]/, "") === `${base(version)}.safetensors`)) version++;
  const name = base(version);
  const trigger = `${slug}_voice`;
  const { filename } = await postJson("/hz3/studio/stage", { name: `lora_${name}`, files });
  const prompt = {
    1: { class_type: "LoadAudio", inputs: { audio: filename } },
    2: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    3: {
      class_type: "HZ3_YuE2_AudioToLoRA",
      inputs: {
        audio: ["1", 0], mode: $("freeze-mode").value, lora_name: name, trigger, steps: Number($("freeze-steps").value) || 100,
        rank: 32, learning_rate: 0.0005, save_to_loras_folder: true, alpha: 64, style_caption: globalStyle(), lyrics: project.lyrics,
        model: ["2", 0], clip: ["2", 1], vae: ["2", 2], custom_output_dir: "",
      },
    },
  };
  $("freeze").close();
  await queue(prompt, { 3: `Training the LoRA of ${singer.name}` }, async () => {
    loras = (await api("/hz3/studio/loras")).loras;
    const lora = loras.find((item) => item.name.replace(/^.*[\\/]/, "") === `${name}.safetensors`);
    if (!lora) throw new Error("Training finished without a LoRA.");
    singer.lora = { name: lora.name, trigger: lora.trigger || trigger, strength: 1 };
    await saveAlbum();
    remember(`Voz de «${singer.name}» congelada en ${lora.name} (${files.length} audios)`);
    if ($("styles").open) drawStyles();
    status(`LoRA ${lora.name} ready and assigned to ${singer.name} (trigger ${singer.lora.trigger}).`, 1);
  });
  status(`Training of the LoRA of ${singer.name} queued (${files.length} audio files).`, 0);
}

function referencedStyles() {
  const ids = new Set([project.baseStyle, ...Object.values(project.sectionSingers), ...project.voices.map((voice) => voice.singer)]);
  return album.styles.filter((style) => ids.has(style.id));
}

async function fetchAlbum(name) {
  return { styles: [], chat: [], log: [], ...(await api(`/hz3/studio/album?name=${encodeURIComponent(name)}`)), name };
}

// The open album is edited in memory and saved as it changes; any other one is read from disk.
async function albumData(name) {
  return name === album.name ? album : fetchAlbum(name);
}

// Songs carry a copy of the styles they use, so opening them in another album (or another install) keeps their singers.
async function addStylesToAlbum(name, styles) {
  const target = await albumData(name);
  const missing = (styles ?? []).filter((style) => !target.styles.some((item) => item.id === style.id));
  if (!missing.length) return;
  target.styles.push(...missing.map((style) => ({ ...style })));
  await postJson("/hz3/studio/album", target);
}

async function loadAlbum(name) {
  album = await fetchAlbum(name);
  drawChat();
}

async function saveAlbum() {
  await postJson("/hz3/studio/album", album);
}

async function listAlbums() {
  const { albums } = await api("/hz3/studio/albums");
  if (!albums.includes(browseAlbum)) albums.push(browseAlbum);
  $("album-list").innerHTML = albums.map((name) => `<option value="${escapeHtml(name)}"${name === browseAlbum ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")
    + `<option value="${NEW_ALBUM}">+ New album…</option>`;
}

// The top bar's album only chooses which songs are listed next to it; moving a song between albums is done in the catalog.
async function browseTo(name) {
  if (name === NEW_ALBUM) {
    name = (prompt("Name of the new album") ?? "").trim();
    if (!name) { $("album-list").value = browseAlbum; return; }
    const { albums } = await api("/hz3/studio/albums");
    if (!albums.includes(name)) await postJson("/hz3/studio/album", { name, notes: "", styles: [], chat: [], log: [] });
  }
  browseAlbum = name;
  await listAlbums();
  await listProjects();
  status(`Album «${name}»: choose a song or «+ New song».`, 1);
}

async function moveToAlbum(name) {
  readForm();
  const used = referencedStyles();
  project.album = name;
  browseAlbum = name;
  await loadAlbum(name);
  await addStylesToAlbum(name, used);
  await saveAlbum();
  if (project.name) await saveProject();
  await listAlbums();
  writeForm();
  draw();
  status(project.name ? `«${project.name}» is now in the album «${name}».` : `Album «${name}».`, 1);
}

function remember(text) {
  album.log.push({ at: Date.now(), song: project.name, text });
  album.log = album.log.slice(-300);
  saveAlbum().catch((error) => status(error.message, null, true));
}

function drawStyles() {
  $("styles-album").textContent = `· ${album.name}`;
  const rows = $("style-rows");
  rows.innerHTML = album.styles.length ? "" : `<tr><td colspan="6" class="style">No styles yet: add singers, bands and genres, or ask the assistant for them.</td></tr>`;
  for (const style of album.styles) {
    const row = document.createElement("tr");
    row.innerHTML = `<td><input maxlength="60"></td>
      <td><select class="kind">${Object.entries(STYLE_KINDS).map(([kind, label]) => `<option value="${kind}">${label}</option>`).join("")}</select></td>
      <td><select class="register" title="Range the melody of the sections they sing is moved to">${Object.entries(REGISTERS).map(([key, { label }]) => `<option value="${key}">${label}</option>`).join("")}</select></td>
      <td><textarea rows="2" placeholder="e.g. Spanish female soprano lead vocal, warm and breathy"></textarea></td>
      <td class="lora"><select class="lora-name" title="Voice LoRA applied when rendering with this singer">
          <option value="">— no LoRA</option>${loras.map((lora) => `<option value="${escapeHtml(lora.name)}">${escapeHtml(lora.name.replace(/^.*[\\/]/, ""))}</option>`).join("")}</select>
        <input class="lora-strength" type="number" min="0" max="2" step="0.05" title="LoRA strength">
        <button class="freeze" title="Trains a LoRA on audio already generated for this singer">Freeze voice…</button></td>
      <td><button class="delete">Delete</button></td>`;
    const [name, kind, register, text] = [row.querySelector("input"), row.querySelector(".kind"), row.querySelector(".register"), row.querySelector("textarea")];
    const [loraName, loraStrength] = [row.querySelector(".lora-name"), row.querySelector(".lora-strength")];
    loraName.value = style.lora?.name ?? "";
    loraStrength.value = style.lora?.strength ?? 1;
    for (const input of [loraName, loraStrength, row.querySelector(".freeze")]) input.disabled = style.kind !== "singer";
    row.querySelector(".freeze").onclick = () => openFreeze(style);
    name.value = style.name;
    kind.value = style.kind;
    text.value = style.text;
    register.value = style.kind === "singer" ? singerRegister(style) : "";
    register.disabled = style.kind !== "singer";
    const update = guard(async () => {
      Object.assign(style, { name: name.value.trim() || style.name, kind: kind.value, text: text.value.trim() });
      if (style.kind === "singer") style.register = register.value || singerRegister(style);
      else delete style.register;
      // The trigger comes from the LoRA file itself, written into it when it was trained.
      const lora = style.kind === "singer" && loras.find((item) => item.name === loraName.value);
      if (lora) style.lora = { name: lora.name, trigger: lora.trigger, strength: Number(loraStrength.value) || 1 };
      else delete style.lora;
      remember(`Estilo «${style.name}» (${STYLE_KINDS[style.kind]}) editado`);
      drawStyles();
      writeForm();
      draw();
    });
    for (const input of [name, kind, register, text, loraName, loraStrength]) input.onchange = update;
    row.querySelector(".delete").onclick = guard(async () => {
      album.styles = album.styles.filter((item) => item !== style);
      remember(`Estilo «${style.name}» eliminado`);
      drawStyles();
      writeForm();
      draw();
    });
    rows.append(row);
  }
}

function loraFile(name) {
  return name.replace(/^.*[\\/]/, "").replace(/\.safetensors$/, "");
}

// The default singer of every section: the song's own style, and its LoRA when the song has one.
function songSingerLabel() {
  return project.lora ? `Song singer · ${loraFile(project.lora.name)}` : "Song singer (its style)";
}

function singerSelect(value, empty, change) {
  const select = document.createElement("select");
  select.className = "singer";
  select.title = "Who sings this section";
  select.innerHTML = styleOptions(["singer"], empty, value);
  select.onclick = (event) => event.stopPropagation();
  select.onchange = guard(async () => { await change(select.value || null); draw(); });
  return select;
}

function setSinger(map, name, id) {
  if (id) map[name] = id;
  else delete map[name];
}

// ---------- studio assistant

function drawChat() {
  $("assistant-album").textContent = `album «${album.name}»`;
  const log = $("chat-log");
  log.innerHTML = "";
  for (const message of album.chat) {
    const item = document.createElement("div");
    item.className = `msg ${message.role}${message.error ? " error" : ""}`;
    item.textContent = message.content;
    if (message.done?.length) {
      const list = document.createElement("ul");
      list.className = "actions";
      for (const line of message.done) list.append(Object.assign(document.createElement("li"), { textContent: line }));
      item.append(list);
    }
    log.append(item);
  }
  log.scrollTop = log.scrollHeight;
}

function assistantState() {
  const styleName = (id) => catalogStyle(id)?.name ?? null;
  return {
    album: {
      name: album.name, notes: album.notes ?? "",
      styles: album.styles.map(({ name, kind, text }) => ({ name, type: kind, text })),
      songs: catalog.filter((entry) => entry.album === album.name).map((entry) => entry.archived ? `${entry.name} (archivada)` : entry.name),
      log: album.log.slice(-40).map((entry) => `${new Date(entry.at).toLocaleString()} · ${entry.song || "—"} · ${entry.text}`),
    },
    song: {
      name: project.name, notes: project.notes, base_style: styleName(project.baseStyle), style: project.style, style_instructions: project.styleInstructions,
      seed: project.seed, sampling: project.sampling, compose: project.compose, history_commits: Object.keys(project.history?.commits ?? {}).length,
      lyrics: sections.length ? undefined : project.lyrics, has_abc: Boolean(project.abc.trim()), source_audio: Boolean(project.source),
      harmonize: project.harmonize, jobs_running: pending.size, takes: project.takes.map((take) => take.id),
      sections: sections.map((section, index) => ({
        name: section.name, start: fmt(section.start), end: fmt(section.end), bars: section.bars, lyrics: section.lyrics,
        own_style: project.sectionStyles[section.name] ?? "", singer: styleName(project.sectionSingers[section.name]),
        seed: sectionSeed(section, index), take: project.comp[section.name] ?? null, edited: isEdited(section, index),
        tracks_on: allTracks().filter((track) => !track.source && (track.voice ? voiceEnabled(track.voice, section) : trackEnabled(track.id, section)))
          .map((track) => track.voice ? track.voice.name : track.id),
      })),
      voices: project.voices.map((voice) => ({
        name: voice.name, source: voice.source, octave: voice.octave, role: voice.role, style: voice.style, generated: Boolean(voice.file),
        singer: styleName(voice.singer),
      })),
      ...dawState(),
    },
  };
}

async function runAction(action) {
  const sectionNamed = (name) => {
    const found = sections.find((section) => section.name.toLowerCase() === String(name ?? "").toLowerCase());
    if (!found) throw new Error(`there is no section «${name}»`);
    return found;
  };
  const voiceNamed = (name) => {
    const found = project.voices.find((voice) => voice.name.toLowerCase() === String(name ?? "").toLowerCase());
    if (!found) throw new Error(`there is no voice «${name}»`);
    return found;
  };
  const styleNamed = (name, kinds) => {
    if (!name) return null;
    const found = album.styles.find((style) => kinds.includes(style.kind) && style.name.toLowerCase() === String(name).toLowerCase());
    if (!found) throw new Error(`«${name}» is not in the catalog`);
    return found.id;
  };
  switch (action.type) {
    case "set_style":
      project.style = String(action.text ?? "");
      writeForm();
      return "Global style updated";
    case "set_section_style": {
      const section = sectionNamed(action.section);
      const text = String(action.text ?? "").trim();
      if (text) project.sectionStyles[section.name] = text;
      else delete project.sectionStyles[section.name];
      draw();
      return `Style of «${section.name}» ${text ? "updated" : "= global"}`;
    }
    case "set_section_lyrics": {
      const section = sectionNamed(action.section);
      setSectionLyrics(sections.indexOf(section), String(action.text ?? ""));
      await refreshSections();
      return `Lyrics of «${section.name}» updated`;
    }
    case "set_section_seed": {
      const section = sectionNamed(action.section);
      project.sectionSeeds[section.name] = Number(action.seed) || 0;
      draw();
      return `Seed of «${section.name}»: ${project.sectionSeeds[section.name]}`;
    }
    case "free_melody": {
      const section = sectionNamed(action.section);
      await setSectionFree(section.name);
      return `«${section.name}» with free melody (the previous one is in the ABC versions)`;
    }
    case "set_base_style":
      project.baseStyle = styleNamed(action.style, ["group", "genre"]);
      writeForm();
      draw();
      return `Base style: ${catalogStyle(project.baseStyle)?.name ?? "none"}`;
    case "set_singer": {
      const id = styleNamed(action.style, ["singer"]);
      if (action.track && action.track !== "lead") {
        const voice = voiceNamed(action.track);
        if (action.section) throw new Error("an extra voice is a separate generation: it has a single singer for the whole song");
        voice.singer = id;
        draw();
        return `${catalogStyle(id)?.name ?? "No singer"} → ${voice.name}`;
      }
      const moved = [];
      for (const section of action.section ? [sectionNamed(action.section)] : [...sections]) {
        const octaves = await setSectionSinger(section.name, id);
        if (octaves) moved.push(`${section.name} ${octaves > 0 ? "+" : ""}${octaves} oct.`);
      }
      draw();
      return `${catalogStyle(id)?.name ?? "No singer"} → lead vocal · ${action.section || "all sections"}${moved.length ? ` · melody moved: ${moved.join(", ")}` : ""}`;
    }
    case "save_style": {
      const name = String(action.name ?? "").trim().slice(0, 60);
      if (!name) throw new Error("style without a name");
      const kind = STYLE_KINDS[action.kind] ? action.kind : "singer";
      let style = album.styles.find((item) => item.name.toLowerCase() === name.toLowerCase());
      if (!style) album.styles.push(style = { id: crypto.randomUUID().slice(0, 8) });
      Object.assign(style, { name, kind, text: String(action.text ?? "").trim() });
      if (kind === "singer" && REGISTERS[action.register]) style.register = action.register;
      await saveAlbum();
      writeForm();
      draw();
      return `Style «${name}» (${STYLE_KINDS[kind]}) saved in the catalog`;
    }
    case "delete_style": {
      const id = styleNamed(action.name, Object.keys(STYLE_KINDS));
      album.styles = album.styles.filter((style) => style.id !== id);
      await saveAlbum();
      writeForm();
      draw();
      return `Style «${action.name}» deleted from the catalog`;
    }
    case "add_voice": {
      addVoice();
      const voice = project.voices[project.voices.length - 1];
      if (action.name) voice.name = String(action.name).slice(0, 40);
      if (["lead", ...Object.keys(VOICE_LINES)].includes(action.source)) voice.source = action.source;
      if ([-1, 0, 1].includes(Number(action.octave))) voice.octave = Number(action.octave);
      if (["0", "-4", "-9"].includes(String(action.role))) voice.role = String(action.role);
      if (action.style) voice.style = String(action.style);
      voice.singer = styleNamed(action.singer, ["singer"]);
      draw();
      return `Voice «${voice.name}» added (not generated)`;
    }
    case "set_track": {
      const section = sectionNamed(action.section);
      const voice = project.voices.find((item) => item.name.toLowerCase() === String(action.track ?? "").toLowerCase());
      if (voice) voice.on[section.name] = Boolean(action.on);
      else if (TRACKS.some((track) => track.id === action.track && !track.source)) (project.trackOn[action.track] ??= {})[section.name] = Boolean(action.on);
      else throw new Error(`there is no track «${action.track}»`);
      draw();
      restartIfPlaying();
      return `${action.track} ${action.on ? "on" : "off"} in «${section.name}»`;
    }
    case "select_take": {
      const section = sectionNamed(action.section);
      const take = takeById(Number(action.take));
      if (!take) throw new Error(`there is no take ${action.take}`);
      project.comp[section.name] = take.id;
      draw();
      restartIfPlaying();
      return `«${section.name}» plays from T${take.id}`;
    }
    case "arrange_harmonies":
      if (action.instructions) {
        project.arranger.instructions = String(action.instructions);
        writeForm();
      }
      await arrange();
      return "Harmonies arranged";
    case "set_sampling": {
      for (const key of SAMPLING) if (Number.isFinite(Number(action[key])) && action[key] !== undefined) project.sampling[key] = Number(action[key]);
      if (Number.isFinite(Number(action.seed)) && action.seed !== undefined) project.seed = Number(action.seed);
      writeForm();
      draw();
      return `Sampling: ${SAMPLING.map((key) => `${key} ${project.sampling[key]}`).join(", ")} · seed ${project.seed}`;
    }
    case "get_abc":
      return action.section ? sectionNamed(action.section).abc : project.abc;
    case "set_abc":
      // The section parser rejects a broken score before it replaces the current one.
      await postJson("/hz3/studio/sections", { abc: String(action.abc ?? ""), lyrics: project.lyrics });
      setAbc(String(action.abc ?? ""), "before the assistant edited the ABC");
      await refreshSections();
      return `ABC replaced · ${sections.length} sections`;
    case "set_compose_settings":
      if (Number.isFinite(Number(action.seed)) && action.seed !== undefined) project.compose.seed = Number(action.seed);
      if (Number.isFinite(Number(action.temperature)) && action.temperature !== undefined) project.compose.temperature = Number(action.temperature);
      if (typeof action.keep_key_meter_tempo === "boolean") project.compose.keep = action.keep_key_meter_tempo;
      writeForm();
      return `Compose: seed ${project.compose.seed}, temperature ${project.compose.temperature}, ${project.compose.keep ? "keeps" : "does not keep"} key, meter and tempo`;
    case "compose_abc": {
      const from = action.from_section ? sectionNamed(action.from_section).name : null;
      await compose(from);
      return `ABC composition${from ? ` from «${from}»` : ""} queued (seed ${project.compose.seed}, temperature ${project.compose.temperature})`;
    }
    case "analyze_audio":
      await analyze();
      return "Audio analysis queued";
    case "render_song":
      await render(null);
      return "Song take queued";
    case "render_sections": {
      const names = (Array.isArray(action.sections) ? action.sections : []).map((name) => sectionNamed(name).name);
      if (!names.length) throw new Error("sections are missing");
      await render(names);
      return `Take queued: ${names.join(", ")}`;
    }
    case "render_voice": {
      const voice = voiceNamed(action.voice);
      await renderVoice(voice);
      return `${voice.name} queued`;
    }
    case "save_song":
      await saveProject();
      return "Song saved";
    default: {
      const result = await runDawAction(action);
      if (result === undefined) throw new Error("unknown action");
      return result;
    }
  }
}

async function sendChat(text) {
  text = text.trim();
  if (!text) return;
  readForm();
  album.chat.push({ role: "user", content: text, at: Date.now(), song: project.name });
  drawChat();
  $("chat-send").disabled = true;
  status("The assistant is thinking…");
  try {
    // Earlier tool results reach the model through the action log in the state, not as text it could imitate.
    const history = album.chat.filter((message) => !message.error).slice(-CHAT_MEMORY).map(({ role, content }) => ({ role, content }));
    const message = { role: "assistant", content: "", at: Date.now(), song: project.name, done: [] };
    album.chat.push(message);
    // The assistant calls page tools and sees their results until it answers in words.
    const turn = [];
    for (let step = 0; step < ASSISTANT_STEPS && !message.content; step++) {
      const answer = await postJson("/hz3/studio/assistant", {
        model: project.arranger.model, tools: ASSISTANT_TOOLS, state: assistantState(), messages: [...history, ...turn],
      });
      if (!answer.tool_calls.length) {
        message.content = answer.content.trim() || "(no answer)";
        break;
      }
      turn.push({ role: "assistant", content: answer.content, tool_calls: answer.tool_calls.map((call) => ({ function: call })) });
      for (const call of answer.tool_calls) {
        const args = typeof call.arguments === "string" ? JSON.parse(call.arguments || "{}") : call.arguments ?? {};
        let result;
        try {
          result = await runAction({ ...args, type: call.name });
        } catch (error) {
          result = `✗ ${call.name}: ${error.message}`;
        }
        turn.push({ role: "tool", tool_name: call.name, content: result });
        message.done.push(call.name === "get_abc" ? `ABC read${args.section ? ` · ${args.section}` : ""}` : result);
      }
      drawChat();
    }
    message.content ||= "I stopped after several actions without a final answer.";
    if (message.done.length) remember(`Asistente: ${message.done.join("; ")}`);
    if (project.name && message.done.length) await saveProject();
    status("Assistant ready.", 1);
  } catch (error) {
    album.chat.push({ role: "assistant", content: error.message, error: true, at: Date.now() });
    throw error;
  } finally {
    $("chat-send").disabled = false;
    album.chat = album.chat.slice(-200);
    await saveAlbum();
    drawChat();
  }
}

function trackEnabled(track, section) {
  // Every track can be switched per section; harmonies start on only in choruses and bridges.
  const value = project.trackOn[track]?.[section.name];
  return value ?? (!VOICES.includes(track) || /^(chorus|coro|bridge|puente)/i.test(section.name));
}

async function applyClassical() {
  readForm();
  const plan = Object.fromEntries(Object.entries(project.classicalPlan).filter(([name, role]) => role && sections.some((section) => section.name === name)));
  if (!Object.keys(plan).length) throw new Error("Choose in the inspector which sections get a classical line (high, low or theme).");
  status("Looking for phrases in the reference scores…");
  const { abc, report } = await postJson("/hz3/studio/classical", { abc: project.abc, plan });
  project.abc = abc;
  project.classicalReport = report;
  $("abc").value = abc;
  await refreshSections();
  await saveProject();
  status(`Classical lines: ${report.join(" · ")}`, 1);
}

async function arrange() {
  readForm();
  if (!sections.length) throw new Error("Lyrics and an ABC with sections are needed.");
  status("The agent is arranging the harmonies…");
  $("arrange").disabled = true;
  try {
    const { plan } = await postJson("/hz3/studio/arrange", {
      style: project.style, instructions: project.arranger.instructions, model: project.arranger.model,
      sections: sections.map(({ name, lyrics, start, end }) => ({ name, lyrics, start: fmt(start), end: fmt(end) })),
    });
    for (const section of sections) {
      const entry = plan[section.name];
      if (!entry) continue;
      for (const voice of VOICES) (project.trackOn[voice] ??= {})[section.name] = entry.voices.includes(voice);
      project.arrangement[section.name] = entry.reason;
    }
    const summary = sections.filter((section) => plan[section.name]?.voices.length)
      .map((section) => `${section.name}: ${plan[section.name].voices.length} voices`).join(" · ");
    remember(`Armonías del agente · ${summary || "sin armonías"}`);
    status(`Agent arrangement · ${summary || "no harmonies"}`, 1);
    draw();
    restartIfPlaying();
  } finally {
    $("arrange").disabled = false;
  }
}

// ---------- jobs through ComfyUI

// A job can finish after another project was opened: it must write into the project that queued it.
async function updateProject(owner, change) {
  owner = currentName(owner);
  if (project.name === owner) {
    // Keep what is being typed, then apply the result; saving reads the form back.
    readForm();
    change(project);
    writeForm();
    await saveProject();
    return true;
  }
  const stored = await api(`/hz3/studio/project?name=${encodeURIComponent(owner)}`);
  change(stored);
  await postJson("/hz3/studio/project", stored);
  for (const [id, state] of Object.entries(stored._snapshots ?? {})) snapshotCache.set(id, state);
  await listProjects();
  status(`«${owner}» updated in the background.`, 1);
  return false;
}

async function queue(prompt, labels, finish) {
  const data = await postJson("/prompt", { prompt, client_id: clientId });
  pending.set(data.prompt_id, { labels, finish });
  setBusy(true);
  return data.prompt_id;
}

async function finishJob(promptId) {
  const job = pending.get(promptId);
  pending.delete(promptId);
  setBusy(pending.size > 0);
  const history = await api(`/history/${promptId}`);
  await job.finish(history[promptId]?.outputs ?? {});
}

function setBusy(busy) {
  for (const id of ["render-song", "render-section", "analyze", "compose", "section-recompose", "voice-render", "abc-reimagine"]) $(id).disabled = busy;
}

function connectSocket() {
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${clientId}`);
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    const { type, data } = JSON.parse(event.data);
    const job = data?.prompt_id && pending.get(data.prompt_id);
    if (!job) return;
    if (type === "progress") {
      status(`${job.labels[data.node] ?? "Processing"} · ${data.value}/${data.max}`, data.value / data.max);
    } else if (type === "executing" && data.node) {
      status(`${job.labels[data.node] ?? "Processing"}…`);
    } else if (type === "executing") {
      // Sent after the queue stores the history; execution_success comes before it.
      finishJob(data.prompt_id).catch((error) => status(error.message, null, true));
    } else if (type === "execution_error" || type === "execution_interrupted") {
      pending.delete(data.prompt_id);
      setBusy(pending.size > 0);
      status(type === "execution_error" ? `Error in ${data.node_type}: ${data.exception_message}` : "Job interrupted.", null, true);
    }
  };
  socket.onclose = () => setTimeout(connectSocket, 2000);
}

async function uploadSource(file) {
  readForm();
  if (!project.name) {
    project.name = audioName(file, new Set(catalog.map((entry) => entry.name)));
    }
  const [name] = await addAudios([file], { kind: "source" });
  await useSource(name, { audio: true });
  status(`Audio «${name}» added to the library and assigned to the song. Press «Analyze audio».`, 1);
}

// Analyses the open song's audio, or the library audio / song named `owner` without opening it.
async function analyze(owner = project.name) {
  readForm();
  const target = owner === project.name ? project : await api(`/hz3/studio/project?name=${encodeURIComponent(owner)}`);
  if (!target.source) throw new Error("Open the source audio first.");
  const lyrics = target.lyrics.trim();
  const prompt = {
    1: { class_type: "LoadAudio", inputs: { audio: target.source.filename } },
    2: { class_type: "AudioEncoderLoader", inputs: { audio_encoder_name: "sheetsage2_bf16.safetensors" } },
    3: { class_type: "HZ3_YuE2_SheetSage2Sections", inputs: { audio_encoder: ["2", 0], audio: ["1", 0], mode: "full" } },
    4: { class_type: "AudioSeparation", inputs: { audio: ["1", 0] } },
    5: {
      class_type: "HZ3_YuE2_Whisper",
      inputs: { audio: ["4", 3], backend: "fast", language: "auto", task: "transcribe", device: "cpu", beam_size: 5, force_rerun: false },
    },
    6: {
      class_type: "HZ3_YuE2_MixMashStyle",
      inputs: {
        model: project.arranger.model, endpoint: "http://127.0.0.1:11434", temperature: 0.35, timeout: 180, force_redo: false,
        lora_trigger: "", abc_report: ["3", 4], abc: ["3", 0], instructions: target.styleInstructions,
        lyrics: lyrics || ["5", 0], extend_abc: false, karaoke_mode: false,
      },
    },
    10: { class_type: "PreviewAny", inputs: { source: ["6", 0] } },
    11: { class_type: "PreviewAny", inputs: { source: ["6", 1] } },
    12: { class_type: "PreviewAny", inputs: { source: ["6", 2] } },
    13: { class_type: "PreviewAny", inputs: { source: ["3", 0] } },
    14: { class_type: "PreviewAny", inputs: { source: ["5", 0] } },
  };
  await queue(prompt, ANALYSIS_LABELS, async (outputs) => {
    const text = (node) => outputs[node]?.text?.[0] ?? "";
    if (!text(12).trim()) throw new Error("The analysis finished without an ABC.");
    const current = await updateProject(owner, (target) => {
      target.analysis = {
        at: Date.now(), sheetsage_abc: text(13), whisper_lyrics: text(14),
        mixmash: { style: text(10), lyrics: text(11), abc: text(12) },
      };
      target.style = text(10);
      target.lyrics = text(11);
      keepAbcVersion(target, "before the analysis");
      target.abc = text(12);
    });
    if (!current) return;
    writeForm();
    await refreshSections();
    await uniqueSectionNames();
    await saveProject();
    remember("Audio analizado: estilo, letra y ABC nuevos");
    status("Analysis ready: check the sections in the score, the lyrics and the ABC.", 1);
  });
  status("Analysis queued…", 0);
}

// Before a change that rewrites the score (a composition, an analysis, a reimagining...), the song as it is is
// committed with what is about to happen as its name, so the history can always go back to it.
function keepAbcVersion(target, label) {
  if (!target.abc.trim()) return;
  commitSong(target, label);
}

// A section as an instrumental: where the voice sang, its melody moves to the instrument line; the voice keeps only
// its chords over rests; and the section's lyrics become "(instrumental)", or YuE2 would still say the words.
async function sectionToInstrumental(name) {
  readForm();
  const section = sections.find((item) => item.name === name);
  const lines = project.abc.split("\n");
  const unit = Number(project.abc.match(/^L:1\/(\d+)/m)?.[1] ?? 32);
  let meter = project.abc.match(/^M:(\S+)/m)?.[1] ?? "4/4";
  const bars = (line) => line.slice(0, -1).split("|").flatMap((bar) => {
    const rest = bar.trim().match(/^Z(\d)?$/);
    return rest ? Array(Number(rest[1] || 1)).fill("Z") : [bar.trim()];
  });
  const sung = (bar) => /[A-Ga-g]/.test(bar.replace(/"[^"]*"|\[K:[^\]]*\]/g, ""));
  const groups = [];
  let current = null;
  let vocal = null;
  let pending = null;
  for (const [at, line] of lines.entries()) {
    if (line.startsWith("% ")) current = line.slice(2).trim();
    else if (line.startsWith("M:")) meter = line.slice(2).trim();
    else if (line.startsWith("V:")) vocal = line.trim() === "V: Vocal" ? true : line.trim() === "V: Ins" ? false : vocal;
    else if (vocal !== null && line.endsWith("|")) {
      const [beats, value] = meter.split("/").map(Number);
      if (vocal) pending = { vocal: at, units: beats * unit / value, inside: current === name };
      else if (pending) {
        groups.push({ ...pending, ins: at });
        pending = null;
      }
    }
  }
  // Every bar of the song, so ties across the section's edges can be checked on both voices.
  const voice = [];
  const played = [];
  for (const group of groups) {
    const instrument = bars(lines[group.ins]);
    group.from = voice.length;
    bars(lines[group.vocal]).forEach((bar, index) => {
      const moved = group.inside && sung(bar);
      voice.push({ text: moved ? freeBar(bar, group.units) : bar, silenced: moved });
      played.push(moved ? { text: bar.replace(/"[^"]*"/g, ""), moved } : { text: instrument[index], moved });
    });
    group.count = voice.length - group.from;
  }
  if (!played.some((bar) => bar.moved)) throw new Error(`«${name}» no longer has a sung vocal.`);
  // A tie into a silenced voice bar, or between a moved instrument bar and a kept one, would join notes that no
  // longer follow each other: those notes are struck again.
  voice.forEach((bar, index) => {
    if (voice[index + 1]?.silenced && !bar.silenced) bar.text = bar.text.replace(/-\s*$/, "");
  });
  played.forEach((bar, index) => {
    if (index + 1 < played.length && played[index + 1].moved !== bar.moved) bar.text = bar.text.replace(/-\s*$/, "");
  });
  for (const group of groups) {
    lines[group.vocal] = voice.slice(group.from, group.from + group.count).map((bar) => bar.text).join("|") + "|";
    lines[group.ins] = played.slice(group.from, group.from + group.count).map((bar) => bar.text).join("|") + "|";
  }
  const abc = lines.join("\n");
  let lyrics = project.lyrics;
  const block = section.lyrics_index === null ? null : lyricBlocks(lyrics)[section.lyrics_index];
  if (block) {
    const last = section.lyrics_index === lyricBlocks(lyrics).length - 1;
    lyrics = lyrics.slice(0, block.bodyStart) + "\n(instrumental)" + (last ? "\n" : "\n\n") + lyrics.slice(block.end);
  }
  await postJson("/hz3/studio/sections", { abc, lyrics });
  keepSectionOriginal(name, "instrumental");
  keepAbcVersion(project, `before making «${name}» instrumental`, true);
  project.abc = abc;
  project.lyrics = lyrics;
  writeForm();
  await refreshSections();
  remember(`«${name}» convertida en instrumental (melodía a la línea instrumental, voz en silencio)`);
  status(`«${name}» is instrumental: its melody moved to the instrumental line. The previous version (ABC and lyrics) is in the versions.`, 1);
}

// ---------- history: commits and branches, like git

// A commit freezes everything a song is made from (its state); the song's catalog data, its takes (shared by every
// branch) and the history itself stay out. A state is stored once, under the hash of its content, in the package's
// history/ folder; commits and branches only point at it. Nothing in the history is ever deleted.
const NOT_STATE = new Set(["name", "kind", "album", "notes", "archived", "license", "takes", "history", "styles", "analysis",
  "_snapshots", "songVersions", "deletedSongVersions", "abcVersions"]);
const MAIN_BRANCH = "principal";
const snapshotCache = new Map();

function stateOf(source) {
  const keys = new Set([...Object.keys(newProject()), ...Object.keys(source)].filter((key) => !NOT_STATE.has(key)));
  return JSON.parse(JSON.stringify(Object.fromEntries([...keys].sort().map((key) => [key, source[key] ?? null]))));
}

// 104-bit content hash (two 53-bit cyrb53 rounds): stable, synchronous, and plenty to tell song states apart.
function stateHash(state) {
  const text = JSON.stringify(state);
  const round = (seed) => {
    let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      h1 = Math.imul(h1 ^ code, 2654435761);
      h2 = Math.imul(h2 ^ code, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16).padStart(14, "0").slice(-13);
  };
  return round(1) + round(2);
}

function storeState(target, state) {
  const id = stateHash(state);
  snapshotCache.set(id, state);
  target._snapshots = { ...(target._snapshots ?? {}), [id]: state };
  return id;
}

function addCommit(target, branch, state, label, auto, at = Date.now()) {
  const history = target.history;
  const parent = history.branches[branch]?.head ?? null;
  const stateId = storeState(target, state);
  const id = stateHash({ parent, stateId, at, label });
  history.commits[id] = { parent, at, label, auto, state: stateId, branch };
  history.branches[branch] = { ...(history.branches[branch] ?? { created: at, archived: false }), head: id };
  return id;
}

// Songs saved before branches existed: their saved versions, deleted versions and earlier ABCs become commits on
// their own branches (a version held only part of the song: the rest comes from the song as it is now), and the
// song as it is now starts the main branch.
function ensureHistory(target) {
  if (target.history) return;
  target.history = { current: MAIN_BRANCH, branches: {}, commits: {} };
  const now = stateOf(target);
  const imported = [
    ["saved versions", target.songVersions ?? [], (version) => ({ ...now, ...stateOf({ ...target, ...version }) })],
    ["deleted versions", target.deletedSongVersions ?? [], (version) => ({ ...now, ...stateOf({ ...target, ...version }) })],
    ["earlier ABCs", target.abcVersions ?? [], (version) => ({ ...now, abc: version.abc, lyrics: version.lyrics ?? now.lyrics })],
  ];
  for (const [branch, versions, toState] of imported) {
    for (const version of [...versions].sort((a, b) => a.at - b.at)) {
      const { at, label, ...fields } = version;
      addCommit(target, branch, toState(fields), label || "unnamed", false, at);
    }
  }
  // "Restore written melody" read old melodies from the ABC versions: they move into the sections' originals.
  // Their labels are [kind, before the name, after the name]; older songs carry the Spanish ones.
  const patterns = [
    ["free", "antes de liberar la melodía de «", "»"], ["instrumental", "antes de volver instrumental «", "»"],
    ["free", "before freeing the melody of «", "»"], ["instrumental", "before making «", "» instrumental"],
  ];
  for (const version of target.abcVersions ?? []) {
    for (const [kind, prefix, suffix] of patterns) {
      if (!version.label?.startsWith(prefix) || !version.label.endsWith(suffix)) continue;
      const name = version.label.slice(prefix.length, version.label.length - suffix.length);
      if (target.sectionOriginals?.[name] !== undefined) continue;
      const lines = version.abc.split("\n");
      target.sectionOriginals = { ...(target.sectionOriginals ?? {}), [name]: { kind, lines: sectionMusicLines(version.abc, name).map((at) => lines[at]), lyrics: null } };
    }
  }
  addCommit(target, MAIN_BRANCH, stateOf(target), "the song when it moved to branches", true);
  delete target.songVersions;
  delete target.deletedSongVersions;
  delete target.abcVersions;
}

function headCommit(target) {
  const history = target.history;
  return history.commits[history.branches[history.current]?.head] ?? null;
}

// Commits the song as it is now on its current branch, unless nothing changed since the branch's last commit.
// A name given to an unchanged state names that commit instead.
function commitSong(target, label, auto = true) {
  ensureHistory(target);
  const state = stateOf(target);
  const head = headCommit(target);
  if (head && head.state === stateHash(state)) {
    if (!auto && label) Object.assign(head, { label, auto: false });
    return null;
  }
  return addCommit(target, target.history.current, state, label || "changes", auto);
}

function songChanged() {
  if (!savedName) return false;
  readForm();
  ensureHistory(project);
  return headCommit(project)?.state !== stateHash(stateOf(project));
}

async function snapshot(stateId, owner = project) {
  if (snapshotCache.has(stateId)) return snapshotCache.get(stateId);
  const state = owner._snapshots?.[stateId]
    ?? await api(`/hz3/studio/snapshot?name=${encodeURIComponent(owner.name)}&id=${stateId}`);
  snapshotCache.set(stateId, state);
  return state;
}

async function loadState(stateId) {
  const state = await snapshot(stateId);
  const fresh = newProject();
  for (const key of new Set([...Object.keys(stateOf(project)), ...Object.keys(state)])) {
    project[key] = JSON.parse(JSON.stringify(state[key] ?? fresh[key] ?? null));
  }
  writeForm();
  await refreshSections();
  draw();
  restartIfPlaying();
}

function branchName(base) {
  const taken = new Set(Object.keys(project.history.branches));
  let name = base.slice(0, 60);
  for (let index = 2; taken.has(name); index++) name = `${base.slice(0, 54)} (${index})`;
  return name;
}

// Opening an old commit never moves a branch back: what you have now is committed first, and the old commit becomes
// the start of a new branch (the head of a branch just opens it).
async function openCommit(id) {
  readForm();
  commitSong(project, "before opening another version");
  const history = project.history;
  const commit = history.commits[id];
  const owning = Object.entries(history.branches).find(([, branch]) => branch.head === id)?.[0];
  if (owning) {
    history.current = owning;
  } else {
    const name = branchName(`from «${commit.label}» ${new Date(commit.at).toLocaleDateString()}`);
    history.branches[name] = { head: id, created: Date.now(), archived: false };
    history.current = name;
  }
  await loadState(commit.state);
  await saveProject();
  status(`Opened «${commit.label}» on the branch «${history.current}». What you had stays on its branch.`, 1);
}

async function switchBranch(name) {
  readForm();
  commitSong(project, "before switching branches");
  project.history.current = name;
  const head = headCommit(project);
  if (head) await loadState(head.state);
  await saveProject();
  status(`Branch «${name}».`, 1);
}

async function newBranch(name) {
  readForm();
  commitSong(project, "before creating a branch");
  name = branchName(name.trim() || "new branch");
  project.history.branches[name] = { head: project.history.branches[project.history.current].head, created: Date.now(), archived: false };
  project.history.current = name;
  await saveProject();
  status(`Branch «${name}» created from here: what you change now goes into it.`, 1);
}

function renameBranch(from, to) {
  const branches = project.history.branches;
  to = to.trim();
  if (!to || to === from) return;
  if (branches[to]) throw new Error(`There is already a branch «${to}».`);
  branches[to] = branches[from];
  delete branches[from];
  for (const commit of Object.values(project.history.commits)) if (commit.branch === from) commit.branch = to;
  if (project.history.current === from) project.history.current = to;
}

// The commits of a branch, newest first: its head and every ancestor (also those made on the branch it came from).
function branchCommits(name) {
  const history = project.history;
  const list = [];
  for (let id = history.branches[name]?.head; id && list.length < 5000; id = history.commits[id]?.parent) list.push([id, history.commits[id]]);
  return list;
}

const STATE_NAMES = {
  style: "style", negativeStyle: "negative style", baseStyle: "base style", lora: "LoRA", lyrics: "lyrics", abc: "ABC",
  seed: "seed", sampling: "sampling", ckpt: "checkpoint", compose: "compose settings", harmonize: "harmonies",
  harmonyVoices: "harmony voices", sectionSeeds: "section seeds", sectionStyles: "section styles",
  sectionLocks: "kept sections", sectionNegatives: "section negatives", sectionContexts: "contexts", sectionOnsets: "fixed openings",
  sectionSingers: "singers", comp: "chosen takes", trackOn: "active tracks", trackSpans: "harmony spans",
  mixer: "mix", voices: "extra voices", instruments: "instruments", fx: "effects", sectionOriginals: "original melodies",
  classicalPlan: "classical lines", arrangement: "agent arrangement", styleInstructions: "style directions",
  source: "source audio", sourceName: "source audio",
};

function stateChanges(before, after) {
  if (!before) return ["start"];
  const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
  const changed = [...keys].filter((key) => JSON.stringify(before[key] ?? null) !== JSON.stringify(after[key] ?? null));
  return [...new Set(changed.map((key) => STATE_NAMES[key] ?? key))];
}

function setAbc(abc, label) {
  if (abc !== project.abc) keepAbcVersion(project, label);
  project.abc = abc;
  writeForm();
}

// The written beginning YuE2 continues: the current header (key, meter, tempo), or the whole score
// before a section's "% name" marker plus the marker itself.
function abcStart(fromSection) {
  const lines = project.abc.split("\n");
  if (fromSection) {
    const marker = lines.findIndex((line) => line.trim().startsWith("% ") && line.trim().slice(2).trim() === fromSection);
    if (marker < 0) throw new Error(`The ABC has no «% ${fromSection}» marker.`);
    return lines.slice(0, marker + 1).join("\n") + "\n";
  }
  const key = lines.findIndex((line) => line.startsWith("K:"));
  return project.compose.keep && key >= 0 ? lines.slice(0, key + 1).join("\n") + "\n" : "";
}

// `name`'s block (its "% name" marker up to the next marker) taken from `composed` into `abc`.
function spliceSection(abc, composed, name) {
  const block = (text) => {
    const lines = text.split("\n");
    const start = lines.findIndex((line) => line.trim().startsWith("% ") && line.trim().slice(2).trim() === name);
    const end = lines.findIndex((line, index) => index > start && line.trim().startsWith("% "));
    return { lines, start, end: end < 0 ? lines.length : end };
  };
  const old = block(abc);
  const fresh = block(composed);
  if (fresh.start < 0) throw new Error(`YuE2 did not return «${name}».`);
  return [...old.lines.slice(0, old.start), ...fresh.lines.slice(fresh.start, fresh.end), ...old.lines.slice(old.end)].join("\n");
}

// only: YuE2 composes `fromSection` after the song before it, and the sections after it stay as they are.
async function compose(fromSection = null, only = false) {
  readForm();
  if (!project.style.trim() || !project.lyrics.trim()) throw new Error("Write a style and lyrics with [Section] headers before generating the ABC.");
  const { seed, temperature } = project.compose;
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "HZ3_YuE2_ContinueABC",
      inputs: {
        clip: ["1", 1], style: globalStyle(), lyrics: project.lyrics, abc_start: abcStart(fromSection), seed, mode: "full",
        max_abc_tokens: 8192, temperature, top_p: 0.9, top_k: 30, repetition_penalty: 1.005,
      },
    },
    3: { class_type: "PreviewAny", inputs: { source: ["2", 0] } },
  };
  const owner = project.name;
  await queue(prompt, COMPOSE_LABELS, async (outputs) => {
    const abc = outputs[3]?.text?.[0] ?? "";
    if (!abc.trim()) throw new Error("YuE2 returned no ABC.");
    const replaced = (target) => {
      keepAbcVersion(target, only ? `before recomposing «${fromSection}»` : fromSection ? `before recomposing from «${fromSection}»` : `before composing with seed ${seed}`);
      target.abc = only ? spliceSection(target.abc, abc, fromSection) : abc;
    };
    if (!await updateProject(owner, replaced)) return;
    await refreshSections().catch((error) => status(`ABC generated; check the sections: ${error.message}`, null, true));
    await uniqueSectionNames();
    await saveProject();
    remember(only ? `YuE2 recompuso «${fromSection}»` : fromSection ? `YuE2 recompuso el ABC desde «${fromSection}»` : "YuE2 compuso un ABC nuevo");
    status("ABC generated: check the sections in the score.", 1);
  });
  status("Composition queued…", 0);
}

const WINDOW_MARGIN = 4;  // seconds rendered around regenerated sections; the comp crossfades inside them

// The stretch to render for these sections, or null when it is most of the song anyway.
// Measured: a 4 s margin renders a section as close to a whole-song render as another noise seed does.
function renderWindow(names) {
  const chosen = sections.filter((section) => names.has(section.name));
  if (!chosen.length) return null;
  const end = sections.at(-1).end;
  const start = Math.max(0, Math.min(...chosen.map((section) => section.start)) - WINDOW_MARGIN);
  const stop = Math.min(end, Math.max(...chosen.map((section) => section.end)) + WINDOW_MARGIN);
  return stop - start < 0.7 * end ? [Math.round(start * 25) / 25, Math.round(stop * 25) / 25] : null;
}

// The KSampler renders [start, stop) only; silence ahead of it keeps every file on the song's timeline.
function renderOnly(prompt, start, stop) {
  Object.assign(prompt[2].inputs, { window_start: start, window_seconds: stop - start });
  prompt[40] = { class_type: "EmptyAudio", inputs: { duration: start, sample_rate: 48000, channels: 2 } };
  prompt[41] = { class_type: "AudioConcat", inputs: { audio1: ["5", 0], audio2: ["40", 0], direction: "before" } };
  for (const [id, node] of Object.entries(prompt)) {
    if (id === "41") continue;
    for (const [key, value] of Object.entries(node.inputs)) if (Array.isArray(value) && value[0] === "5") node.inputs[key] = ["41", 0];
  }
}

function buildPrompt(prefix, kept) {
  const overrides = Object.entries(project.sectionSeeds)
    .filter(([name]) => sections.some((section) => section.name === name))
    .map(([name, seed]) => `${name} = ${seed}`).join("\n");
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "HZ3_YuE2_GenerateMusicSections",
      inputs: {
        clip: ["1", 1], style: globalStyle(), lyrics: project.lyrics, abc: project.abc, seed: project.seed,
        mode: "full", ...project.sampling, section_seeds: overrides,
        section_styles: sections.map((section) => [section.name, sectionStyle(section)]).filter(([, text]) => text.trim())
          .map(([name, text]) => `${name}: ${oneLine(text)}`).join("\n"),
        // A rendered section with a fixed opening sends it in place of kept tokens; the node samples the rest.
        section_tokens: [...Object.entries(kept), ...Object.entries(project.sectionOnsets ?? {})
          .filter(([name]) => !kept[name] && sections.some((section) => section.name === name)).map(([name, start]) => [name, start.tokens])]
          .map(([name, tokens]) => `${name} = ${tokens}`).join("\n"),
        negative_style: project.negativeStyle.trim(),
        section_negative_styles: sections.filter((section) => project.sectionNegatives[section.name]?.trim())
          .map((section) => `${section.name}: ${oneLine(project.sectionNegatives[section.name])}`).join("\n"),
        section_contexts: sections.filter((section) => project.sectionContexts?.[section.name])
          .map((section) => `${section.name} = ${project.sectionContexts[section.name].tokens}`).join("\n"),
      },
    },
    3: { class_type: "EmptyYuE2LatentAudio", inputs: { seconds: ["2", 1], batch_size: 1 } },
    4: {
      class_type: "KSampler",
      inputs: {
        model: ["1", 0], seed: 42, ...ACOUSTIC_SAMPLING, cfg: 4.2, scheduler: "sgm_uniform",
        positive: ["2", 0], negative: ["2", 0], latent_image: ["3", 0], denoise: 1,
      },
    },
    5: { class_type: "VAEDecodeAudioTiled", inputs: { samples: ["4", 0], vae: ["1", 2], tile_size: 256, overlap: 32 } },
    6: { class_type: "SaveAudio", inputs: { audio: ["5", 0], filename_prefix: `${prefix}/mix` } },
    7: { class_type: "AudioSeparation", inputs: { audio: ["5", 0] } },
    8: { class_type: "SaveAudio", inputs: { audio: ["7", 3], filename_prefix: `${prefix}/vocals` } },
    9: { class_type: "AudioMerge", inputs: { audio1: ["7", 0], audio2: ["7", 1], merge_method: "add" } },
    10: { class_type: "AudioMerge", inputs: { audio1: ["9", 0], audio2: ["7", 2], merge_method: "add" } },
    11: { class_type: "SaveAudio", inputs: { audio: ["10", 0], filename_prefix: `${prefix}/instrumental` } },
  };
  const saves = { 6: "mix", 8: "vocals", 11: "instrumental" };
  withLora(prompt, loraUses());
  return { prompt, saves };
}

function loraUses() {
  return sections.map((section) => {
    const singer = catalogStyle(project.sectionSingers[section.name]);
    return singer ? { lora: singer.lora, owner: singer.name } : { lora: project.lora, owner: "the song's singer" };
  });
}

function hasChords(abc) {
  return melodyOnly(abc) !== abc;
}

// SheetSage2 transcribing a song's audio (melody and chords), as nodes `at` (encoder) and `at + 1` (transcription).
function hearChords(audio, at) {
  return {
    [at]: { class_type: "AudioEncoderLoader", inputs: { audio_encoder_name: "sheetsage2_bf16.safetensors" } },
    [at + 1]: { class_type: "HZ3_YuE2_SheetSage2Sections", inputs: { audio_encoder: [String(at), 0], audio, mode: "full" } },
  };
}

// The score's melody without its chord symbols (only music lines: the V: definitions hold quoted names too).
function melodyOnly(abc) {
  return abc.split("\n").map((line) => line.endsWith("|") && !line.startsWith("V:") ? line.replace(/"[^"\n]*"/g, "") : line).join("\n");
}

// The score with its instrument line silenced bar by bar (inline fields and multi-bar rests kept), so YuE2 plays its own.
function withoutIns(abc) {
  let ins = false;
  return abc.split("\n").map((line) => {
    if (line.startsWith("V:")) ins = line.slice(2).trim() === "Ins";
    if (!ins || line.startsWith("V:") || !line.endsWith("|")) return line;
    return line.slice(0, -1).split("|").map((bar) => /^\s*Z\d*\s*$/.test(bar) ? bar : `${(bar.match(/\[[^\]]*\]/g) ?? []).join("")}Z`).join("|") + "|";
  }).join("\n");
}

const LETTERS = "CDEFGAB";

// A note's diatonic step (C4 = 28) and back; written without accidentals, the key signature keeps it in the key.
function noteStep(letter, marks) {
  const octave = (letter === letter.toLowerCase() ? 5 : 4) + [...marks].reduce((sum, mark) => sum + (mark === "'" ? 1 : -1), 0);
  return octave * 7 + LETTERS.indexOf(letter.toUpperCase());
}

function stepNote(step) {
  const octave = Math.floor(step / 7);
  const letter = LETTERS[step % 7];
  return octave >= 5 ? letter.toLowerCase() + "'".repeat(octave - 5) : letter + ",".repeat(4 - octave);
}

// Light variations of the vocal melody of sections `names`: about `amount` of its notes move a step or a third (to a
// tone of the bar's chord when one is that near), keeping rhythm, ties, syllables, range and the note closing each
// phrase. Notes with accidentals, and bars holding any, are left alone (an accidental lasts the bar in ABC).
function varyMelody(abc, names, amount, seed) {
  let state = seed >>> 0;
  const random = () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
  const lines = abc.split("\n");
  const events = [];
  let section = null;
  let vocal = false;
  let body = false;
  let chord = null;
  lines.forEach((line, index) => {
    const text = line.trim();
    if (text.startsWith("K:")) body = true;
    if (!body) return;
    if (text.startsWith("% ")) { section = text.slice(2).trim(); chord = null; return; }
    if (text.startsWith("V:")) { vocal = text.slice(2).trim() === "Vocal"; return; }
    if (!vocal || !names.has(section) || !text.endsWith("|")) return;
    let offset = 0;
    for (const bar of line.split("|")) {
      const accidental = /[\^_=]/.test(bar.replace(/"[^"]*"/g, ""));
      for (const match of bar.matchAll(/"([^"]*)"|\[[^\]]*\]|([\^_=]*)([A-Ga-g])([,']*)[\d/]*(-?)|[zZx]/g)) {
        if (match[1] !== undefined) chord = match[1];
        else if (match[3]) {
          events.push({ section, line: index, at: offset + match.index + match[2].length, length: match[3].length + match[4].length,
            step: noteStep(match[3], match[4]), tie: Boolean(match[5]), fixed: Boolean(match[2]) || accidental, chord });
        } else if (!match[0].startsWith("[")) events.push({ section, rest: true });
      }
      offset += bar.length + 1;
    }
  });
  // Tied notes move together; a rest closes a phrase.
  const groups = [];
  events.forEach((event, index) => {
    const previous = events[index - 1];
    if (event.rest) groups.push(null);
    else if (previous && !previous.rest && previous.tie && previous.section === event.section) groups.at(-1).push(event);
    else groups.push([event]);
  });
  const range = {};
  for (const event of events.filter((item) => !item.rest)) {
    const [low, high] = range[event.section] ?? [event.step, event.step];
    range[event.section] = [Math.min(low, event.step), Math.max(high, event.step)];
  }
  let changed = 0;
  groups.forEach((group, index) => {
    const next = groups[index + 1];
    const previous = groups[index - 1];
    if (!group || !next || next[0].section !== group[0].section || group.some((event) => event.fixed) || random() >= amount) return;
    const { step, section: name, chord: symbol } = group[0];
    const [low, high] = range[name];
    const near = [-2, -1, 1, 2].map((delta) => step + delta).filter((candidate) => candidate >= low && candidate <= high
      && Math.abs(candidate - next[0].step) <= 4 && (!previous || previous[0].section !== name || Math.abs(candidate - previous[0].step) <= 4));
    const root = LETTERS.indexOf(symbol?.[0]);
    const tones = root < 0 ? [] : [root, (root + 2) % 7, (root + 4) % 7];
    const chordTones = near.filter((candidate) => tones.includes(candidate % 7));
    const pool = chordTones.length ? chordTones : near.filter((candidate) => Math.abs(candidate - step) === 1);
    if (!pool.length) return;
    const moved = pool[Math.floor(random() * pool.length)];
    for (const event of group) { event.step = moved; event.changed = true; }
    changed++;
  });
  for (const event of events.filter((item) => item.changed).sort((a, b) => b.line - a.line || b.at - a.at)) {
    const line = lines[event.line];
    lines[event.line] = line.slice(0, event.at) + stepNote(event.step) + line.slice(event.at + event.length);
  }
  return { abc: lines.join("\n"), changed };
}

// «Reimaginar ABC»: new chords (melody kept), a new instrument line and chords (vocal line kept), light variations of
// the melody, or one section composed again after the song before it.
async function reimagineAbc() {
  readForm();
  await refreshSections();
  const section = sections.some((item) => item.name === selected) ? selected : null;
  const action = await choose("Reimagine ABC", "The previous ABC stays in the ABC versions.", [
    { value: "chords", label: "Chords: the melody stays the same; YuE2 reinvents harmony and arrangement (generates the song)" },
    { value: "instrumental", label: "New instrumental: the vocal stays the same in the score; the instrumental line and the chords are reinvented (generates the song)" },
    ...(section ? [{ value: "vary-section", label: `Vary the melody of «${section}»: changes some notes, same rhythm and syllables (instant)` }] : []),
    { value: "vary", label: "Vary the melody of the whole song: changes some notes, same rhythm and syllables (instant)" },
    ...(section ? [{ value: "section", label: `Recompose only «${section}»: YuE2 composes it again after what comes before; the rest stays the same` }] : []),
  ]);
  if (!action) return;
  if (action === "chords" || action === "instrumental") return render(null, action);
  if (action === "section") return compose(section, true);
  const answer = prompt("What percentage of the notes should change? (the ones closing each phrase are not touched)", "20");
  if (answer === null) return;
  const amount = Math.min(100, Math.max(0, Number(answer) || 0)) / 100;
  const names = new Set(action === "vary" ? sections.map((item) => item.name) : [section]);
  const { abc, changed } = varyMelody(project.abc, names, amount, project.compose.seed);
  if (!changed) {
    status("No notes left to vary there.", 1);
    return;
  }
  keepAbcVersion(project, `before varying the melody (${Math.round(amount * 100)} %)`);
  project.abc = abc;
  writeForm();
  await refreshSections();
  await saveProject();
  remember(`Melodía variada (${changed} notas, semilla ${project.compose.seed})${action === "vary" ? "" : ` en «${section}»`}`);
  status(`${changed} notes varied. Listen with «▶ Score» and regenerate the marked sections to sing them; 🎲 next to «Compose seed» gives another variation.`, 1);
  draw();
}

// The stored tokens each section not rendered again keeps: its take's own, or, when its bars changed (a moved
// boundary, a split), the frames under its new bounds cut from its take's tokens, which is what plays there now.
async function keptTokens(comped) {
  const kept = {};
  const slices = [];
  for (const section of sections.filter((item) => !comped.has(item.name))) {
    const take = takeById(project.comp[section.name]);
    const own = take.sectionTokens?.[section.name];
    if (own && take.sectionAbc?.[section.name] === section.abc) kept[section.name] = own;
    else if (take.sectionTokens) slices.push({ name: section.name, tokens: takeStream(take), start: section.start_frame, frames: section.frames });
  }
  if (slices.length) {
    const { tokens } = await postJson("/hz3/studio/slice_tokens", { slices });
    slices.forEach((slice, index) => { if (tokens[index]) kept[slice.name] = tokens[index]; });
  }
  return kept;
}

function takeStream(take) {
  if (take.tokenStream) return take.tokenStream;
  // Older takes: their sections in the song's current order.
  const at = (name) => { const index = sections.findIndex((section) => section.name === name); return index < 0 ? Infinity : index; };
  return Object.keys(take.sectionTokens).sort((a, b) => at(a) - at(b)).map((name) => take.sectionTokens[name]);
}

// reimagine: render the whole song in melody mode (YuE2 keeps the melody and invents harmony and arrangement; with
// "instrumental" the instrument line is emptied as well), have SheetSage2 hear the chords it played, and freeze them
// into the song's ABC, which then sings in full mode.
async function render(targets, reimagine = null) {
  readForm();
  ensureName();
  await refreshSections();
  if (!sections.length) throw new Error("Lyrics and an ABC with sections are needed.");
  const id = Math.max(0, ...project.takes.map((take) => take.id)) + 1;
  const take = { id, created: Date.now(), harmonized: project.harmonize, files: {}, sectionLyrics: {}, sectionSeeds: {} };
  if (project.history) take.branch = project.history.current;
  take.sectionStyles = {};
  take.style = globalStyle();
  if (Object.keys(project.sectionContexts ?? {}).length) take.sectionContexts = structuredClone(project.sectionContexts);
  if (Object.keys(project.sectionOnsets ?? {}).length) take.sectionOnsets = structuredClone(project.sectionOnsets);
  take.sectionAbc = Object.fromEntries(sections.map((section) => [section.name, section.abc]));
  sections.forEach((section, index) => {
    take.sectionLyrics[section.name] = section.lyrics;
    take.sectionSeeds[section.name] = sectionSeed(section, index);
    if (sectionStyle(section)) take.sectionStyles[section.name] = sectionStyle(section);
  });
  // "Generate song" renders every section not kept (🔒); "Regenerate section" only the chosen ones, kept or not.
  // A section never rendered is always sung.
  const comped = new Set(sections.filter((section) =>
    reimagine || (targets ? targets.includes(section.name) : !isKept(section)) || !takeById(project.comp[section.name])
  ).map((section) => section.name));
  if (!comped.size) throw new Error("Every section keeps its take (🔒): unlock one in its Takes tab, or use «Regenerate section».");
  const kept = reimagine ? {} : await keptTokens(comped);
  const missing = sections.filter((section) => !comped.has(section.name) && !kept[section.name]).map((section) => section.name);
  if (missing.length) {
    const go = await choose(`${missing.length} more sections would also be regenerated`,
      `No stored tokens cover ${missing.map((name) => `«${name}»`).join(", ")} with their current bars, so YuE2 would sing them again.`,
      [{ value: true, label: `Regenerate those ${missing.length} too` }]);
    if (!go) return;
    for (const name of missing) comped.add(name);
  }
  const { prompt, saves } = buildPrompt(`HZ3-Studio/${project.name}/take-${id}`, kept);
  // Regenerating some sections renders only their stretch (plus a margin), placed at its time in the take's files.
  take.window = !reimagine && targets ? renderWindow(comped) : null;
  if (take.window) renderOnly(prompt, ...take.window);
  const melody = reimagine === "instrumental" ? withoutIns(melodyOnly(project.abc)) : melodyOnly(project.abc);
  if (reimagine) {
    Object.assign(prompt[2].inputs, { mode: "melody", abc: melody });
    Object.assign(prompt, hearChords(["5", 0], 30));
    prompt[32] = { class_type: "PreviewAny", inputs: { source: ["31", 0] } };
  }
  const owner = project.name;
  // The take is playable once voice and instrumental are saved; its harmonies follow as their own job (in progress
  // on their lanes meanwhile), so the song can be reviewed while they are made.
  const harmonies = project.harmonize ? [...project.harmonyVoices] : [];
  const renderedAbc = project.abc;
  await queue(prompt, { ...RENDER_LABELS, 31: "SheetSage2 (chords of the render)" }, async (outputs) => {
    for (const [node, track] of Object.entries(saves)) {
      const file = outputs[node]?.audio?.[0];
      if (file) take.files[track] = file;
    }
    if (!take.files.vocals) throw new Error("The generation finished without audio.");
    take.sectionTokens = outputs[2]?.section_tokens?.[0] ?? {};
    if (outputs[32]?.text?.[0]) take.heardAbc = outputs[32].text[0];
    // The take's tokens in song order (section renames reorder the keys above).
    take.tokenStream = Object.values(take.sectionTokens);
    if (harmonies.length) take.pending = harmonies;
    const current = await updateProject(owner, (target) => {
      target.takes.push(take);
      for (const name of comped) target.comp[name] = take.id;
    });
    if (harmonies.length) await harmonizeTake(take, owner, harmonies, renderedAbc);
    if (!current) return;
    await loadTake(take);
    if (reimagine) {
      const { abc } = await postJson("/hz3/studio/freeze_chords", { abc: melody, transcription: outputs[32]?.text?.[0] ?? "" });
      keepAbcVersion(project, reimagine === "instrumental" ? "before reimagining the instrumental" : "before reimagining the ABC");
      project.abc = abc;
      writeForm();
      await refreshSections();
      // The take is what this frozen score describes, so its sections are not shown as edited.
      take.sectionAbc = Object.fromEntries(sections.map((section) => [section.name, section.abc]));
      await saveProject();
      remember(`ABC reimaginado con el take ${take.id}: ${reimagine === "instrumental" ? "línea instrumental vacía, " : ""}acordes congelados, modo full`);
      status(`Take ${take.id} reimagined: its chords are now in the ABC (full mode)${reimagine === "instrumental" ? " and the instrumental line is left free for YuE2" : ""}. «Regenerate» harmonies to use them.`, 1);
      draw();
      return;
    }
    remember(`Take ${take.id} · ${[...comped].join(", ")}`);
    status(`Take ${take.id} ready · ${[...comped].join(", ")}${harmonies.length ? " · harmonies in progress…" : ""}`, harmonies.length ? 0 : 1);
    draw();
  });
  status(`Take ${id} queued…`, 0);
}

// ---------- audio

function fileUrl(file) {
  const query = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder, type: file.type });
  return `/view?${query}`;
}

function sourceUrl() {
  return project.source ? fileUrl({ filename: project.source.filename, subfolder: "", type: "input" }) : null;
}

function context() {
  audioContext ??= new AudioContext();
  return audioContext;
}

// A song opens with over a hundred files. Firefox refuses some of them ("unknown content type") when they are all
// decoded at once, so only a few are decoded at a time, and a file that still fails leaves its lane empty instead
// of stopping the song from opening.
const DECODES_AT_ONCE = 4;
const unreadableAudio = new Set();
let decoding = 0;
const decodeQueue = [];

async function decodeSlot() {
  if (decoding >= DECODES_AT_ONCE) await new Promise((resolve) => decodeQueue.push(resolve));
  else decoding++;
}

function releaseDecodeSlot() {
  const next = decodeQueue.shift();
  if (next) next();
  else decoding--;
}

async function loadUrl(url) {
  if (buffers.has(url)) return;
  await decodeSlot();
  try {
    if (buffers.has(url)) return;
    for (let attempt = 0; ; attempt++) {
      const response = await fetch(url);
      if (!response.ok) return;
      try {
        const buffer = await context().decodeAudioData(await response.arrayBuffer());
        buffers.set(url, { buffer, peaks: peaks(buffer) });
        unreadableAudio.delete(url);
        return;
      } catch (error) {
        if (attempt) {
          unreadableAudio.add(url);
          console.warn(`Audio not decoded: ${url}`, error);
          return;
        }
      }
    }
  } finally {
    releaseDecodeSlot();
  }
}

async function loadTake(take) {
  await Promise.all(Object.entries(take.files).filter(([track]) => track !== "mix" && track !== "vocalsOriginal").map(([, file]) => loadUrl(fileUrl(file))));
}

async function loadSource() {
  if (project.source) await loadUrl(sourceUrl());
}

function sourceEntry() {
  return project.source ? buffers.get(sourceUrl()) : undefined;
}

function peaks(buffer) {
  const step = Math.max(1, Math.floor(buffer.sampleRate / PEAKS_PER_SECOND));
  const result = new Float32Array(Math.ceil(buffer.length / step));
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < result.length; index++) {
      let peak = 0;
      const end = Math.min(data.length, (index + 1) * step);
      for (let sample = index * step; sample < end; sample++) peak = Math.max(peak, Math.abs(data[sample]));
      result[index] = Math.max(result[index], peak);
    }
  }
  return result;
}

function trackBuffer(track, takeId) {
  const file = takeById(takeId)?.files[track];
  return file ? buffers.get(fileUrl(file)) : undefined;
}

// Free on/off spans of a track ({ start, end, on } in seconds), drawn by dragging on its lane: where one lies it wins
// over the section switches, so a harmony can start or stop anywhere inside a section.
function trackOnAt(track, time) {
  const span = (project.trackSpans[track] ?? []).findLast((item) => item.start <= time && time < item.end);
  return span ? span.on : trackEnabled(track, sections.findLast((section) => section.start <= time) ?? sections[0]);
}

// Gain breakpoints [[time, value]] of one take of a track: on where the comp plays that take and the track is on.
function trackGate(track, takeId) {
  const times = [...new Set([0, ...sections.map((section) => section.start),
    ...(project.trackSpans[track] ?? []).flatMap((span) => [span.start, span.end])])].sort((a, b) => a - b);
  return times.map((time) => {
    const section = sections.findLast((item) => item.start <= time) ?? sections[0];
    return [time, project.comp[section.name] === takeId && trackOnAt(track, time) ? 1 : 0];
  });
}

function scheduleGain(param, points, start, from) {
  const at = (time) => start + time - from;
  param.setValueAtTime((points.findLast(([time]) => time <= from) ?? points[0])[1], start);
  points.forEach(([time, value], index) => {
    if (!index || value === points[index - 1][1] || time + FADE / 2 <= from) return;
    const fadeStart = Math.max(at(time - FADE / 2), start);
    param.setValueAtTime(points[index - 1][1], fadeStart);
    param.linearRampToValueAtTime(value, Math.max(at(time + FADE / 2), fadeStart + 0.001));
  });
}

function muted(track) {
  // The source is the reference while correcting; once takes exist it starts muted.
  return project.mixer[track.id]?.mute ?? Boolean(track.source && project.takes.length);
}

function mixerGain(track) {
  const soloed = allTracks().some((other) => project.mixer[other.id]?.solo);
  if (muted(track) || (soloed && !project.mixer[track.id]?.solo)) return 0;
  return project.mixer[track.id]?.gain ?? track.gain;
}

function buildGraph(target, destination, from, start) {
  const sources = [];
  const trackGains = {};
  const takeIds = new Set(Object.values(project.comp));
  // A limiter on the sum, so stacked voices never clip in playback or export.
  const limiter = target.createDynamicsCompressor();
  limiter.threshold.value = -3;
  limiter.knee.value = 0;
  limiter.ratio.value = 20;
  limiter.attack.value = 0.002;
  limiter.release.value = 0.12;
  // The compressor still overshoots on stacked peaks: a ceiling that is transparent below
  // 0.9 and bends softly into 1.0 (curve spans ±2, so the signal is halved around it).
  const ceiling = target.createWaveShaper();
  ceiling.curve = Float32Array.from({ length: 4097 }, (_, index) => {
    const value = (index / 2048 - 1) * 2;
    const size = Math.abs(value);
    return Math.sign(value) * (size <= 0.9 ? size : 0.9 + 0.1 * Math.tanh((size - 0.9) / 0.1)) / 2;
  });
  const into = target.createGain();
  const out = target.createGain();
  into.gain.value = 0.5;
  const end = duration();
  if (endsTrimmed() && end - 1 > from) {
    into.gain.setValueAtTime(0.5, start + end - 1 - from);
    into.gain.linearRampToValueAtTime(0.0001, start + end - from);
  }
  // The ceiling's 1.0 lands at -1 dBFS, so exports keep headroom for encoders and editors (true peaks).
  out.gain.value = 2 * CEILING;
  limiter.connect(into).connect(ceiling).connect(out).connect(destination);
  // Live playback feeds the level meter from what reaches the speakers.
  const meters = target === audioContext ? levelMeters(target, out) : null;
  const buses = effectBuses(target, limiter);
  const connect = (entry, trackGain, values) => {
    if (!entry || from >= entry.buffer.duration) return;
    const source = target.createBufferSource();
    source.buffer = entry.buffer;
    const gain = target.createGain();
    if (values) scheduleGain(gain.gain, values, start, from);
    source.connect(gain).connect(trackGain);
    source.start(start, from);
    sources.push(source);
  };
  // Section by section, each read at its own shift (seconds into the file ahead of the timeline).
  const connectSections = (entry, trackGain, parts) => {
    if (!entry) return;
    const at = (time) => start + time - from;
    for (const { section, shift, level } of parts) {
      if (section.end + FADE / 2 <= from) continue;
      const begin = Math.max(section.start - FADE / 2, from);
      const end = section.end + FADE / 2;
      const offset = begin + shift;
      if (offset >= entry.buffer.duration) continue;
      const source = target.createBufferSource();
      source.buffer = entry.buffer;
      const gain = target.createGain();
      gain.gain.setValueAtTime(0, at(begin));
      gain.gain.linearRampToValueAtTime(level, at(Math.min(begin + FADE, end)));
      gain.gain.setValueAtTime(level, at(Math.max(end - FADE, begin + FADE)));
      gain.gain.linearRampToValueAtTime(0, at(end));
      source.connect(gain).connect(trackGain);
      source.start(at(begin) + Math.max(0, -offset), Math.max(0, offset), end - begin);
      sources.push(source);
    }
  };
  // A voice plays at its own measured shift per section.
  const connectVoice = (voice, trackGain) => connectSections(voice.file && buffers.get(fileUrl(voice.file)), trackGain,
    sections.filter((section) => voiceEnabled(voice, section))
      .map((section) => ({ section, shift: voiceShift(voice, section), level: voice.sectionGains?.[section.name] ?? 1 })));
  for (const track of allTracks()) {
    const trackGain = target.createGain();
    trackGain.gain.value = mixerGain(track);
    trackChain(target, track, trackGain, limiter, buses, start, from);
    trackGains[track.id] = trackGain;
    if (track.source) connect(sourceEntry(), trackGain, null);
    else if (track.voice) connectVoice(track.voice, trackGain);
    else if (track.instrument) connect(instrumentEntry(track.instrument), trackGain, null);
    else {
      // A take track nudged in time (ms per section, + = later) plays section by section.
      const offsets = trackFx(track.id).offsets ?? {};
      for (const takeId of takeIds) {
        if (!Object.values(offsets).some(Boolean)) connect(trackBuffer(track.id, takeId), trackGain, trackGate(track.id, takeId));
        else connectSections(trackBuffer(track.id, takeId), trackGain, sections.filter((section) => project.comp[section.name] === takeId && trackEnabled(track.id, section))
          .map((section) => ({ section, shift: -(offsets[section.name] ?? 0), level: 1 })));
      }
    }
  }
  return { sources, trackGains, meters, limiter };
}

function voiceEnabled(voice, section) {
  return voice.on[section.name] ?? !/^\(?instrumental\)?$/i.test(section.lyrics.trim() || "instrumental");
}

function voiceShift(voice, section) {
  return voice.offset + (voice.sectionOffsets[section.name] ?? 0);
}

const CEILING = 10 ** (-1 / 20);
let peakHold = [0, 0];
let clipped = false;

// One analyser per channel after the master ceiling.
function levelMeters(target, node) {
  const split = target.createChannelSplitter(2);
  node.connect(split);
  return [0, 1].map((channel) => {
    const analyser = target.createAnalyser();
    analyser.fftSize = 2048;
    split.connect(analyser, channel);
    return analyser;
  });
}

const dbfs = (value) => (value > 0 ? 20 * Math.log10(value) : -Infinity);

// Peak bars (-48..0 dBFS: green, yellow above -12, red above -3), a held peak per channel, the loudest peak so far and
// a CLIP light that stays on until clicked: a sample at or above -0.1 dBFS.
function drawMeter() {
  const canvas = $("meter-bars");
  const draw = canvas.getContext("2d");
  draw.clearRect(0, 0, canvas.width, canvas.height);
  const data = new Float32Array(2048);
  const scale = (db) => Math.max(0, Math.min(1, (db + 48) / 48)) * canvas.width;
  (playback?.meters ?? []).forEach((analyser, channel) => {
    analyser.getFloatTimeDomainData(data);
    const peak = data.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
    peakHold[channel] = Math.max(peak, peakHold[channel] * 0.995);
    if (peak >= 0.989) clipped = true;
    const y = channel * (canvas.height / 2), h = canvas.height / 2 - 1, width = scale(dbfs(peak));
    for (const [from, color] of [[-48, "#4caf50"], [-12, "#e0c34a"], [-3, "#e05a4a"]]) {
      const left = scale(from);
      if (width > left) { draw.fillStyle = color; draw.fillRect(left, y, width - left, h); }
    }
    draw.fillStyle = "#e6e6ea";
    draw.fillRect(scale(dbfs(peakHold[channel])) - 1, y, 2, h);
  });
  const loudest = Math.max(...peakHold);
  $("meter-peak").textContent = loudest > 0 ? `${dbfs(loudest).toFixed(1)} dB` : "−∞";
  // How hard the master limiter is pressing: the mix itself would clip without it.
  const reduction = playback?.limiter.reduction ?? 0;
  $("meter-limit").textContent = reduction < -0.5 ? `limiter ${reduction.toFixed(1)} dB` : "";
  $("meter-limit").classList.toggle("hot", reduction < -6);
  $("meter-clip").classList.toggle("on", clipped);
}

function position() {
  return playback ? playback.from + audioContext.currentTime - playback.start : pausedAt;
}

async function play() {
  stopAudition();
  if (playback) { pause(); return; }
  await context().resume();
  const from = pausedAt >= duration() ? 0 : pausedAt;
  const start = audioContext.currentTime + 0.05;
  playback = { from, start, ...buildGraph(audioContext, audioContext.destination, from, start) };
  $("play").textContent = "❚❚";
  requestAnimationFrame(tick);
}

function pause() {
  if (!playback) return;
  pausedAt = position();
  playback.sources.forEach((source) => source.stop());
  playback = null;
  $("play").textContent = "▶";
}

function seek(seconds) {
  const wasPlaying = Boolean(playback);
  pause();
  pausedAt = Math.min(Math.max(0, seconds), duration());
  if (wasPlaying) play();
  updatePlayhead();
}

function restartIfPlaying() {
  if (playback) seek(position());
}

function tick() {
  if (!playback) return;
  if (position() >= duration()) { pause(); pausedAt = 0; }
  updatePlayhead();
  drawMeter();
  if (playback) requestAnimationFrame(tick);
}

function headWidth() {
  return document.querySelector(".lane-head")?.offsetWidth ?? 0;
}

function updatePlayhead() {
  $("playhead").style.left = `${headWidth() + position() * pxPerSecond}px`;
  $("clock").textContent = `${fmt(position())} / ${fmt(duration())}`;
}

async function exportMix() {
  if (!duration()) throw new Error("There is nothing to export yet.");
  const sampleRate = 48000;
  const offline = new OfflineAudioContext(2, Math.ceil(duration() * sampleRate), sampleRate);
  buildGraph(offline, offline.destination, 0, 0);
  status("Exporting mix…");
  const rendered = await offline.startRendering();
  const peak = Math.max(...[...Array(rendered.numberOfChannels).keys()].map((channel) =>
    rendered.getChannelData(channel).reduce((max, value) => Math.max(max, Math.abs(value)), 0)));
  const response = await fetch("/hz3/studio/flac", { method: "POST", body: wav(rendered) });
  if (!response.ok) throw new Error(`Could not encode the FLAC: ${await response.text()}`);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(await response.blob());
  link.download = `${project.name || "hz3-studio"}.flac`;
  link.click();
  URL.revokeObjectURL(link.href);
  status(`Mix exported · peak ${dbfs(peak).toFixed(1)} dBFS.`, 1);
}

function wav(buffer) {
  const channels = buffer.numberOfChannels;
  const frames = buffer.length;
  const view = new DataView(new ArrayBuffer(44 + frames * channels * 2));
  const text = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, 36 + frames * channels * 2, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, channels, true);
  view.setUint32(24, buffer.sampleRate, true); view.setUint32(28, buffer.sampleRate * channels * 2, true);
  view.setUint16(32, channels * 2, true); view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, frames * channels * 2, true);
  const data = [...Array(channels).keys()].map((channel) => buffer.getChannelData(channel));
  let offset = 44;
  for (let frame = 0; frame < frames; frame++) {
    for (let channel = 0; channel < channels; channel++) {
      view.setInt16(offset, Math.max(-1, Math.min(1, data[channel][frame])) * 0x7fff, true);
      offset += 2;
    }
  }
  return view.buffer;
}

// ---------- timeline drawing

function lane(className, head) {
  const row = document.createElement("div");
  row.className = `lane ${className}`;
  const headCell = document.createElement("div");
  headCell.className = "lane-head";
  if (head) headCell.append(head);
  const body = document.createElement("div");
  body.className = "lane-body";
  body.style.width = `${duration() * pxPerSecond}px`;
  row.append(headCell, body);
  $("lanes").append(row);
  return { row, headCell, body };
}

// ---------- simple and advanced mode

// What the simple mode hides: engine settings, alternative routes and the lab. A hidden field that holds a value is
// shown anyway (marked), so nothing that reaches YuE2 is ever out of sight.
const ADVANCED = ["open-styles", "arrange", "source-bring", "import-json",
  "style-instructions", "apply-style-instructions", "negative-style", "section-styles-from-cues", "abc", "compose-seed",
  "compose-dice", "compose-temperature", "compose-keep", "apply-classical", "abc-reimagine", "seed",
  "section-classical", "section-negative", "section-context", "section-seed", "section-dice", "section-free",
  "section-record", "section-recompose", "voice-seed", "voice-offset"];
const SHOWN_WHEN_SET = ["negative-style", "section-negative", "section-context", "section-classical"];

function markAdvanced() {
  for (const id of ADVANCED) {
    const element = $(id);
    (element.closest("label") ?? element).classList.add("adv");
  }
  for (const element of document.querySelectorAll("[data-advanced]")) element.classList.add("adv");
  let advanced = false;
  try { advanced = localStorage.getItem("hz3.advanced") === "1"; } catch { /* simple mode */ }
  $("advanced-mode").checked = advanced;
  document.body.classList.toggle("advanced", advanced);
  $("advanced-mode").addEventListener("change", (event) => {
    document.body.classList.toggle("advanced", event.target.checked);
    if (!event.target.checked && document.querySelector('[data-song-tab="engine"].active')) showSongTab("song");
    try { localStorage.setItem("hz3.advanced", event.target.checked ? "1" : "0"); } catch { /* not remembered */ }
  });
  for (const id of SHOWN_WHEN_SET) $(id).addEventListener("input", refreshAdvanced);
}

function refreshAdvanced() {
  for (const id of SHOWN_WHEN_SET) $(id).closest("label").classList.toggle("in-use", Boolean($(id).value.trim()));
  // A tab that holds an advanced field in use says so, even while another tab is open.
  for (const tab of document.querySelectorAll("[data-insp-tab]")) {
    tab.classList.toggle("has-advanced", Boolean(document.querySelector(`[data-insp-pane="${tab.dataset.inspTab}"] .adv.in-use`)));
  }
}

// The section inspector's tabs; the last one chosen stays chosen.
function showInspectorTab(name) {
  for (const tab of document.querySelectorAll("[data-insp-tab]")) tab.classList.toggle("active", tab.dataset.inspTab === name);
  for (const pane of document.querySelectorAll("[data-insp-pane]")) pane.classList.toggle("active", pane.dataset.inspPane === name);
  try { localStorage.setItem("hz3.inspectorTab", name); } catch { /* not remembered */ }
}

function setupInspectorTabs() {
  for (const tab of document.querySelectorAll("[data-insp-tab]")) tab.onclick = () => showInspectorTab(tab.dataset.inspTab);
  let last = "lyrics";
  try { last = localStorage.getItem("hz3.inspectorTab") || last; } catch { /* first tab */ }
  showInspectorTab(document.querySelector(`[data-insp-tab="${last}"]`) ? last : "lyrics");
  for (const tab of document.querySelectorAll("[data-song-tab]")) tab.onclick = () => showSongTab(tab.dataset.songTab);
  let song = "song";
  try { song = localStorage.getItem("hz3.songTab") || song; } catch { /* first tab */ }
  showSongTab(song);
}

// The side panel's tabs (Song, Score, Source, Engine); Engine only exists in Advanced.
function showSongTab(name) {
  if (name === "engine" && !document.body.classList.contains("advanced")) name = "song";
  if (!document.querySelector(`[data-song-tab="${name}"]`)) name = "song";
  for (const tab of document.querySelectorAll("[data-song-tab]")) tab.classList.toggle("active", tab.dataset.songTab === name);
  for (const pane of document.querySelectorAll("[data-song-pane]")) pane.classList.toggle("active", pane.dataset.songPane === name);
  try { localStorage.setItem("hz3.songTab", name); } catch { /* not remembered */ }
}

function draw() {
  queueMicrotask(refreshAdvanced);
  $("lanes").innerHTML = "";
  if (!duration()) { updatePlayhead(); drawInspector(); drawVoiceInspector(); drawTrackInspector(); return; }
  const ruler = lane("ruler", "m:ss");
  const step = pxPerSecond >= 20 ? 5 : pxPerSecond >= 8 ? 10 : 30;
  for (let time = 0; time < duration(); time += step) {
    const mark = document.createElement("div");
    mark.className = "tick";
    mark.style.left = `${time * pxPerSecond}px`;
    mark.textContent = fmt(time);
    ruler.body.append(mark);
  }
  ruler.body.onclick = (event) => seek(event.offsetX / pxPerSecond);
  if (sections.length) drawSections();
  if (sections.length) drawLyrics();
  if (score) drawScore();
  for (const track of allTracks().filter((track) => !track.harmony && !track.voice && !track.instrument)) drawTrack(track);
  drawHarmonyHeader();
  if (!harmoniesCollapsed) for (const track of TRACKS.filter((track) => track.harmony)) drawTrack(track);
  for (const track of allTracks().filter((track) => track.voice)) drawTrack(track);
  for (const track of instrumentTracks()) drawTrack(track);
  updatePlayhead();
  drawInspector();
  drawVoiceInspector();
  drawTrackInspector();
}

// Each section's lyrics, editable in place, so words can be moved to the ABC section that should sing them.
function drawLyrics() {
  const used = new Set(sections.map((section) => section.lyrics_index));
  const orphans = lyricBlocks(project.lyrics).filter((_, index) => !used.has(index)).map((block) => block.name);
  const head = document.createElement("span");
  head.className = "name";
  head.textContent = orphans.length ? `Lyrics · ${orphans.length} without section` : "Lyrics";
  head.title = orphans.length
    ? `Lyrics blocks without an ABC section (YuE2 still receives them, in the full lyrics): ${orphans.join(", ")}`
    : "Lyrics of each section: edit them here to fit them to the ABC sections";
  const { row, body } = lane("lyrics", head);
  // Drag the lane's corner to give the lyric boxes more room; the height is a view preference of this browser.
  try { row.style.height = `${Number(localStorage.getItem("hz3.lyricsHeight")) || 96}px`; } catch { /* default height */ }
  new ResizeObserver(() => {
    if (!row.isConnected || !row.offsetHeight) return;  // a redraw removed this lane
    try { localStorage.setItem("hz3.lyricsHeight", String(row.offsetHeight)); } catch { /* not remembered */ }
  }).observe(row);
  sections.forEach((section, index) => {
    const area = document.createElement("textarea");
    area.className = `lyric-block${section.lyrics_index === null ? " empty" : ""}`;
    area.style.left = `${section.start * pxPerSecond}px`;
    area.style.width = `${(section.end - section.start) * pxPerSecond}px`;
    area.value = section.lyrics;
    area.placeholder = "no lyrics";
    area.title = `${section.name}: ${section.lyrics_index === null ? "no block in the lyrics (type here to create it)" : "lyrics of this section"}`;
    area.onchange = guard(async () => {
      setSectionLyrics(index, area.value);
      await refreshSections();
    });
    body.append(area);
    if (index > 0) {
      // Moving the boundary between two sections in the lyrics moves one line across it.
      const arrows = document.createElement("div");
      arrows.className = "lyric-arrows";
      arrows.style.left = `${section.start * pxPerSecond}px`;
      arrows.innerHTML = `<button data-way="back" title="Moves the first line of «${escapeHtml(section.name)}» to the previous section">⇠</button><button data-way="forward" title="Brings the last line of the previous section into «${escapeHtml(section.name)}»">⇢</button>`;
      arrows.querySelectorAll("button").forEach((button) => { button.onclick = guard(() => moveLyricLine(index, button.dataset.way)); });
      body.append(arrows);
    }
  });
}

// One lyric line across the boundary before section `index`: "back" gives its first line to the previous section,
// "forward" brings the previous section's last line into it.
async function moveLyricLine(index, way) {
  const [earlier, later] = [sections[index - 1].lyrics.split("\n"), sections[index].lyrics.split("\n")];
  const source = way === "back" ? later : earlier;
  const at = way === "back" ? source.findIndex((line) => line.trim()) : source.findLastIndex((line) => line.trim());
  if (at < 0) throw new Error("There are no lines to move in that section.");
  const [line] = source.splice(at, 1);
  if (way === "back") earlier.push(line);
  else later.unshift(line);
  // The later section first, then the earlier one with fresh pairings, since writing may create a lyric block.
  setSectionLyrics(index, later.join("\n"));
  await refreshSections();
  setSectionLyrics(index - 1, earlier.join("\n"));
  await refreshSections();
}

// The harmony tracks fold under one header; only the checked ones are generated or regenerated.
let harmoniesCollapsed = (() => { try { return localStorage.getItem("hz3.harmoniesCollapsed") === "1"; } catch { return false; } })();

function drawHarmonyHeader() {
  const head = document.createElement("div");
  head.innerHTML = `<span class="name link">${harmoniesCollapsed ? "▸" : "▾"} Harmonies · ${project.harmonyVoices.length}/${VOICES.length}</span>
    <span class="controls"><button data-action="regenerate" title="Harmonizes the lead vocal of the playing takes again, only for the checked harmonies, without rendering the song">Regenerate</button></span>`;
  head.querySelector(".name").onclick = () => {
    harmoniesCollapsed = !harmoniesCollapsed;
    try { localStorage.setItem("hz3.harmoniesCollapsed", harmoniesCollapsed ? "1" : "0"); } catch { /* the fold is only a convenience */ }
    draw();
  };
  head.querySelector('[data-action="regenerate"]').onclick = guard(regenerateHarmonies);
  lane("group", head);
}

// Harmonies come from the lead vocal alone, so they are rebuilt from each playing take's vocals.
async function regenerateHarmonies() {
  readForm();
  if (!project.harmonyVoices.length) throw new Error("First check which harmonies to generate.");
  const takes = [...new Set(Object.values(project.comp))].map(takeById).filter((take) => take?.files.vocals);
  if (!takes.length) throw new Error("There are no takes with a lead vocal yet.");
  for (const take of takes) await harmonizeTake(take, project.name, project.harmonyVoices, project.abc);
  status(`Harmonies queued for ${takes.length} take(s).`, 0);
}

// Queues the harmonies of one take from its saved lead vocal, on the score it was rendered from.
async function harmonizeTake(take, owner, voices, abc) {
  const { filename } = await postJson("/hz3/studio/stage", { name: `harmony_${take.id}_${Date.now().toString(36)}`, files: [take.files.vocals] });
  const prompt = {
    1: { class_type: "LoadAudio", inputs: { audio: filename } },
    2: { class_type: "HZ3_YuE2_VocalHarmonizer", inputs: { vocals: ["1", 0], score_abc: abc } },
  };
  // A melody-only score harmonizes on the chords heard in the take's own mix.
  if (!hasChords(abc) && take.files.mix) {
    const mix = await postJson("/hz3/studio/stage", { name: `harmony_mix_${take.id}_${Date.now().toString(36)}`, files: [take.files.mix] });
    Object.assign(prompt, { 20: { class_type: "LoadAudio", inputs: { audio: mix.filename } } }, hearChords(["20", 0], 21));
    prompt[2].inputs.heard_abc = ["22", 0];
    prompt[23] = { class_type: "PreviewAny", inputs: { source: ["22", 0] } };
  }
  const saves = {};
  VOICES.forEach((voice, index) => {
    if (!voices.includes(voice)) return;
    prompt[3 + index] = { class_type: "SaveAudio", inputs: { audio: ["2", index], filename_prefix: `HZ3-Studio/${owner}/take-${take.id}/${voice}` } };
    saves[3 + index] = voice;
  });
  await queue(prompt, { 2: `Harmonies of take ${take.id}`, 22: "SheetSage2 (chords of the take)" }, async (outputs) => {
    const files = Object.fromEntries(Object.entries(saves).map(([node, voice]) => [voice, outputs[node]?.audio?.[0]]).filter(([, file]) => file));
    const current = await updateProject(owner, (target) => {
      const stored = target.takes.find((item) => item.id === take.id);
      if (stored) Object.assign(stored.files, files);
      if (stored) delete stored.pending;
      if (stored && outputs[23]?.text?.[0]) stored.heardAbc = outputs[23].text[0];
    });
    if (!current) return;
    await loadTake(takeById(take.id));
    remember(`Armonías del take ${take.id}: ${Object.keys(files).join(", ")}`);
    status(`Harmonies of take ${take.id} ready.`, 1);
    draw();
    restartIfPlaying();
  });
}

function drawSections() {
  const { body } = lane("sections", "Sections");
  sections.forEach((section, index) => {
    const block = document.createElement("div");
    block.className = `section-block${section.name === selected ? " selected" : ""}`;
    block.style.left = `${section.start * pxPerSecond}px`;
    block.style.width = `${(section.end - section.start) * pxPerSecond}px`;
    block.style.borderLeftColor = SECTION_COLORS[index % SECTION_COLORS.length];
    block.title = `${section.name} · ${fmt(section.start)}–${fmt(section.end)}`;
    block.textContent = section.name;
    const takeId = project.comp[section.name];
    if (takeId) block.insertAdjacentHTML("beforeend", isKept(section)
      ? `<span class="take" title="Kept: «Generate song» does not sing it again">🔒T${takeId}</span>` : `<span class="take">T${takeId}</span>`);
    const parts = editedParts(section, index);
    if (parts.length) block.insertAdjacentHTML("beforeend", `<span class="edited" title="${escapeHtml(`The ${parts.join(", ")} of this section is not what take ${project.comp[section.name]} sang: «Regenerate section» sings it this way. If you only corrected the text, you can leave it.`)}">edited: ${parts.join(", ")}</span>`);
    const fit = melodyFit(section);
    if (fit) block.insertAdjacentHTML("beforeend", `<span class="edited" title="${escapeHtml(fit.title)}">${fit.label}</span>`);
    block.onclick = () => { selected = section.name; selectedVoice = null; selectedTrack = null; draw(); };
    if (index > 0 && score) {
      const handle = document.createElement("div");
      handle.className = "handle";
      handle.title = "Drag to move the start of the section (snaps to the bar)";
      handle.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); dragBoundary(index, body); };
      block.append(handle);
    }
    body.append(block);
  });
}

// YuE2 lays the song's syllables on its notes in order: a section short of notes sings its leftover words on the
// next section's melody, and spare notes get invented words. A free section (no notes) has no such limit.
function melodyFit(section) {
  const { notes, syllables } = section;
  if (!notes || !syllables) return null;
  if (syllables > notes + 2) {
    return { label: `${syllables - notes} notes short`, title: `${notes} notes for about ${syllables} syllables: what does not fit is sung over the next section's melody. Give it more bars or notes, shorten the lyrics or leave it as free melody.` };
  }
  if (notes > syllables * 1.25 + 4) {
    return { label: `${notes - syllables} notes over`, title: `${notes} notes for about ${syllables} syllables: on the spare notes YuE2 stretches vowels or invents words.` };
  }
  return null;
}

function dragBoundary(index, body) {
  const edited = editableSections();
  const low = edited[index - 1].start_bar + 1;
  const high = edited[index].end_bar;
  const line = document.createElement("div");
  line.className = "drag-line";
  $("lanes").append(line);
  let bar = edited[index].start_bar;
  const move = (event) => {
    const seconds = (event.clientX - body.getBoundingClientRect().left) / pxPerSecond;
    bar = Math.min(high, Math.max(low, nearestBar(seconds)));
    line.style.left = `${headWidth() + barSeconds(bar) * pxPerSecond}px`;
    status(`«${edited[index].name}» will start at bar ${bar + 1} · ${fmt(barSeconds(bar))}`);
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    line.remove();
    if (bar === edited[index].start_bar) return;
    edited[index].start_bar = bar;
    edited[index - 1].end_bar = bar - 1;
    guard(() => applySections(edited))();
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
}

function drawScore() {
  const head = document.createElement("div");
  head.innerHTML = `<span class="name">Score · ${score.key} · ${score.bpm} BPM</span><span class="controls">
    <button id="score-lane-play" title="Play / pause the score from its cursor (click the lane to move it)">${scorePlayback ? "❚❚" : "▶"}</button>
    <button id="score-lane-stop" title="Stop and go back to the marked point">■</button></span>`;
  head.querySelector("#score-lane-play").onclick = guard(async () => {
    if (scorePlayback) return pauseScore();
    await context().resume();
    playScoreLane();
  });
  head.querySelector("#score-lane-stop").onclick = stopScore;
  const { body } = lane("score", head);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(duration() * pxPerSecond));
  canvas.height = 119;
  const cursor = Object.assign(document.createElement("div"), { id: "score-cursor", className: "score-cursor" });
  cursor.style.left = `${scorePosition() * pxPerSecond}px`;
  body.append(canvas, cursor);
  body.onclick = (event) => {
    scoreAt = scoreMark = Math.min(Math.max(0, (event.clientX - body.getBoundingClientRect().left) / pxPerSecond), duration());
    if (scorePlayback) playScoreLane();
    else updateScoreCursor();
  };
  const graphics = canvas.getContext("2d");
  const x = (ticks) => (ticks / TICKS_PER_QUARTER) * 60 / score.bpm * pxPerSecond;
  const notes = [...(score.tracks.Vocal ?? []), ...(score.tracks.Ins ?? [])];
  if (!notes.length) return;
  const low = Math.min(...notes.map((note) => note.pitch)) - 1;
  const high = Math.max(...notes.map((note) => note.pitch)) + 1;
  const top = 14;
  const row = (canvas.height - top) / (high - low + 1);
  const starts = score.sections.map((section) => section.start_bar);
  score.bars.forEach((bar, index) => {
    graphics.fillStyle = starts.includes(index) ? "#6a6d78" : "#262930";
    graphics.fillRect(Math.round(x(bar.start)), 0, 1, canvas.height);
  });
  const sectionAt = (ticks) => {
    let found = 0;
    score.sections.forEach((section, index) => { if (score.bars[section.start_bar].start <= ticks) found = index; });
    return found;
  };
  for (const [name, color] of [["Ins", null], ["Vocal", true]]) {
    for (const note of score.tracks[name] ?? []) {
      graphics.fillStyle = color ? SECTION_COLORS[sectionAt(note.start) % SECTION_COLORS.length] : "#4a4d57";
      graphics.fillRect(x(note.start), top + (high - note.pitch) * row, Math.max(1, x(note.duration) - 1), Math.max(3, row - 1));
    }
  }
  graphics.font = "10px system-ui, sans-serif";
  graphics.fillStyle = "#9a9ca5";
  let lastEnd = -Infinity;
  for (const chord of score.chords) {
    const left = x(chord.start);
    if (left < lastEnd + 4) continue;
    graphics.fillText(chord.symbol, left + 2, 10);
    lastEnd = left + graphics.measureText(chord.symbol).width;
  }
}

function drawTrack(track) {
  const head = document.createElement("div");
  const generate = track.harmony ? `<label class="generate" title="Generate this harmony in renders and when regenerating"><input type="checkbox"${project.harmonyVoices.includes(track.id) ? " checked" : ""}> generate</label>` : "";
  const singer = track.voice && catalogStyle(track.voice.singer);
  const who = track.voice ? (singer ? `${singer.name} · ${singerCue(singer)}` : track.voice.style.split("\n").filter((line) => line.trim()).at(-1) ?? "") : "";
  const inProgress = project.takes.some((take) => take.pending?.includes(track.id));
  head.innerHTML = `<span class="name">${escapeHtml(track.name)}</span>${inProgress ? `<span class="who">in progress…</span>` : ""}${who ? `<span class="who" title="${escapeHtml(who)}">${escapeHtml(who)}</span>` : ""}${generate}<span class="controls">
    <button data-action="mute" class="${muted(track) ? "active" : ""}" title="Mute">M</button>
    <button data-action="solo" class="${project.mixer[track.id]?.solo ? "active" : ""}" title="Solo">S</button>
    <input type="range" min="0" max="1.5" step="0.01" value="${project.mixer[track.id]?.gain ?? track.gain}" title="Volume"></span>`;
  if (!track.voice) {
    const name = head.querySelector(".name");
    name.classList.add("link");
    name.title = "Volume, pan, effects and settings of this track";
    name.onclick = () => { selectedTrack = track.id; selected = null; selectedVoice = null; draw(); };
  }
  head.querySelector(".generate input")?.addEventListener("change", (event) => {
    project.harmonyVoices = VOICES.filter((voice) => voice === track.id ? event.target.checked : project.harmonyVoices.includes(voice));
    draw();
  });
  head.querySelector('[data-action="mute"]').onclick = () => {
    (project.mixer[track.id] ??= {}).mute = !muted(track);
    applyMixer();
    draw();
  };
  head.querySelector('[data-action="solo"]').onclick = () => {
    const entry = project.mixer[track.id] ??= {};
    entry.solo = !entry.solo;
    applyMixer();
    draw();
  };
  head.querySelector("input").oninput = (event) => {
    (project.mixer[track.id] ??= {}).gain = Number(event.target.value);
    applyMixer();
  };
  const { body } = lane(track.instrument ? "track instrument" : "track", head);
  if (track.instrument) {
    drawPianoRoll(track.instrument, body);
    return;
  }
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(duration() * pxPerSecond));
  canvas.height = 63;
  body.append(canvas);
  const graphics = canvas.getContext("2d");
  const middle = canvas.height / 2;
  const paint = (entry, from, to, shift = 0) => {
    for (let column = Math.floor(from * pxPerSecond); column < to * pxPerSecond; column++) {
      const first = Math.max(0, Math.floor((column / pxPerSecond + shift) * PEAKS_PER_SECOND));
      const last = Math.min(entry.peaks.length, Math.floor(((column + 1) / pxPerSecond + shift) * PEAKS_PER_SECOND));
      let peak = 0;
      for (let index = first; index < last; index++) peak = Math.max(peak, entry.peaks[index]);
      const height = Math.min(1, peak) * (middle - 2);
      graphics.fillRect(column, middle - height, 1, height * 2 || 1);
    }
  };
  if (track.source) {
    const entry = sourceEntry();
    graphics.fillStyle = "#8a8d96";
    if (entry) paint(entry, 0, Math.min(duration(), entry.buffer.duration));
    return;
  }
  if (track.voice) {
    const voice = track.voice;
    const name = head.querySelector(".name");
    name.classList.add("link");
    name.title = "Edit, generate or align this voice";
    name.onclick = () => { selectedVoice = voice.id; selected = null; selectedTrack = null; draw(); };
    const entry = voice.file && buffers.get(fileUrl(voice.file));
    for (const section of sections) {
      const enabled = voiceEnabled(voice, section);
      graphics.fillStyle = enabled ? "#b07be0" : "#3a3d45";
      if (entry) paint(entry, section.start, section.end, voiceShift(voice, section));
      const toggle = document.createElement("div");
      toggle.className = `toggle${enabled ? "" : " off"}`;
      toggle.style.left = `${section.start * pxPerSecond}px`;
      toggle.style.width = `${(section.end - section.start) * pxPerSecond}px`;
      toggle.title = `${voice.name} · ${section.name}: ${enabled ? "on" : "off"} (click to toggle)`;
      toggle.onclick = () => {
        voice.on[section.name] = !enabled;
        draw();
        restartIfPlaying();
      };
      body.append(toggle);
    }
    return;
  }
  sections.forEach((section) => {
    const entry = trackBuffer(track.id, project.comp[section.name]);
    const enabled = trackEnabled(track.id, section);
    graphics.fillStyle = enabled ? (track.harmony ? "#4aa3e0" : "#c9a14a") : "#3a3d45";
    if (entry) paint(entry, section.start, section.end);
    const toggle = document.createElement("div");
    toggle.className = `toggle${enabled ? "" : " off"}`;
    toggle.style.left = `${section.start * pxPerSecond}px`;
    toggle.style.width = `${(section.end - section.start) * pxPerSecond}px`;
    toggle.title = `${track.name} · ${section.name}: ${enabled ? "on" : "off"} (click to toggle)`;
    toggle.onclick = () => {
      if (body.dataset.dragged) return;
      (project.trackOn[track.id] ??= {})[section.name] = !enabled;
      draw();
      restartIfPlaying();
    };
    if (track.id === "vocals" && album.styles.some((style) => style.kind === "singer") && (section.end - section.start) * pxPerSecond >= 50) {
      toggle.append(singerSelect(project.sectionSingers[section.name] ?? "", songSingerLabel(), (id) => setSectionSinger(section.name, id)));
    }
    body.append(toggle);
  });
  if (track.harmony) drawTrackSpans(track, body);
}

// A harmony lane takes free on/off spans: dragging draws one that flips the state found where the drag began,
// a click on a span removes it, and a plain click still switches the whole section.
function drawTrackSpans(track, body) {
  const spans = project.trackSpans[track.id] ??= [];
  for (const span of spans) {
    const block = document.createElement("div");
    block.className = `span${span.on ? " on" : " off"}`;
    block.style.left = `${span.start * pxPerSecond}px`;
    block.style.width = `${(span.end - span.start) * pxPerSecond}px`;
    block.title = `${track.name} ${span.on ? "on" : "off"} from ${fmt(span.start)} to ${fmt(span.end)} (click to remove)`;
    block.onclick = (event) => {
      event.stopPropagation();
      spans.splice(spans.indexOf(span), 1);
      draw();
      restartIfPlaying();
    };
    body.append(block);
  }
  const timeAt = (event) => Math.min(duration(), Math.max(0, (event.clientX - body.getBoundingClientRect().left) / pxPerSecond));
  // The gesture is followed on the window, not captured by the lane: a plain click must still reach the section
  // switch under the pointer.
  body.onpointerdown = (event) => {
    if (event.button !== 0 || event.target.closest(".span, select")) return;
    const start = timeAt(event);
    const preview = document.createElement("div");
    delete body.dataset.dragged;
    const move = (moved) => {
      if (Math.abs(moved.clientX - event.clientX) < 4) return;
      body.dataset.dragged = "1";
      const [from, to] = [start, timeAt(moved)].sort((a, b) => a - b);
      preview.className = `span preview${trackOnAt(track.id, start) ? " off" : " on"}`;
      preview.style.left = `${from * pxPerSecond}px`;
      preview.style.width = `${(to - from) * pxPerSecond}px`;
      if (!preview.isConnected) body.append(preview);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", (up) => {
      window.removeEventListener("pointermove", move);
      preview.remove();
      if (!body.dataset.dragged) return;
      const [from, to] = [start, timeAt(up)].sort((a, b) => a - b);
      spans.push({ start: Math.round(from * 100) / 100, end: Math.round(to * 100) / 100, on: !trackOnAt(track.id, start) });
      draw();
      restartIfPlaying();
      // The click that ends a drag must not also switch the section under it.
      setTimeout(() => delete body.dataset.dragged, 0);
    }, { once: true });
  };
}

function applyMixer() {
  if (!playback) return;
  for (const track of allTracks()) playback.trackGains[track.id].gain.value = mixerGain(track);
}

let styleDraft = null;  // { section, text } while a section's style has unsaved edits

// What a section sings without a style of its own: the global lines and its "[Section] …" cue.
function defaultSectionStyle(section) {
  const { global, cue } = sectionCueStyle(section);
  return [global, cue].filter(Boolean).join("\n");
}

function saveSectionStyle() {
  const section = sections.find((item) => item.name === selected);
  const text = $("section-style").value.trim();
  if (!text || text === defaultSectionStyle(section).trim()) delete project.sectionStyles[selected];
  else project.sectionStyles[selected] = text;
  styleDraft = null;
  draw();
}

// ---------- score preview: the ABC's notes and chords through a small synth

const CHORD_SHAPES = {
  "": [0, 4, 7], m: [0, 3, 7], dim: [0, 3, 6], aug: [0, 4, 8], 7: [0, 4, 7, 10], maj7: [0, 4, 7, 11], m7: [0, 3, 7, 10],
  dim7: [0, 3, 6, 9], m7b5: [0, 3, 6, 10], sus4: [0, 5, 7], sus2: [0, 2, 7], 6: [0, 4, 7, 9], m6: [0, 3, 7, 9],
  "7sus4": [0, 5, 7, 10], "m(maj7)": [0, 3, 7, 11],
};
const PITCH_CLASSES = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function pitchClass(name) {
  return PITCH_CLASSES[name[0]] + (name.slice(1).match(/#/g)?.length ?? 0) - (name.slice(1).match(/b/g)?.length ?? 0);
}

function chordPitches(symbol) {
  const match = symbol.match(/^([A-G](?:bb|##|b|#)?)(.*?)(?:\/([A-G](?:bb|##|b|#)?))?$/);
  if (!match || !(match[2] in CHORD_SHAPES)) return [];
  const root = 48 + pitchClass(match[1]);
  const pitches = CHORD_SHAPES[match[2]].map((step) => root + step);
  return match[3] ? [36 + pitchClass(match[3]), ...pitches] : pitches;
}

let scorePlayback = null;
// The score lane's own cursor, in seconds: where its ▶ starts, and scoreMark where ■ returns (set by clicking the lane).
let scoreAt = 0;
let scoreMark = 0;

function scorePosition() {
  return scorePlayback ? scorePlayback.from + audioContext.currentTime - scorePlayback.begin : scoreAt;
}

function haltScore() {
  if (!scorePlayback) return;
  scorePlayback.sources.forEach((source) => source.stop());
  clearTimeout(scorePlayback.timer);
  scorePlayback = null;
  $("score-play").textContent = "♪ Score";
  $("section-score-play").textContent = "▶ Score";
  if ($("score-lane-play")) $("score-lane-play").textContent = "▶";
}

function stopScore() {
  haltScore();
  scoreAt = scoreMark;
  updateScoreCursor();
}

function pauseScore() {
  scoreAt = scorePosition();
  haltScore();
  updateScoreCursor();
}

function playScoreLane() {
  if (scoreAt >= duration()) scoreAt = 0;
  const ticks = (seconds) => Math.min(score.total_ticks, Math.round(seconds * score.bpm / 60 * TICKS_PER_QUARTER));
  playScore(ticks(scoreAt), score.total_ticks, $("score-lane-play"), "❚❚");
}

function updateScoreCursor() {
  const cursor = $("score-cursor");
  if (cursor) cursor.style.left = `${scorePosition() * pxPerSecond}px`;
  if (scorePlayback) requestAnimationFrame(updateScoreCursor);
}

function playScore(fromTick, toTick, button, playingLabel = "■ Stop") {
  haltScore();
  const target = context();
  const begin = target.currentTime + 0.05;
  const time = (ticks) => (ticks / TICKS_PER_QUARTER) * 60 / score.bpm;
  const out = target.createGain();
  out.gain.value = 0.2;
  out.connect(target.destination);
  const sources = [];
  const note = (pitch, start, length, type, level) => {
    const from = Math.max(start, fromTick);
    const to = Math.min(start + length, toTick);
    if (to <= from) return;
    const oscillator = target.createOscillator();
    const gain = target.createGain();
    oscillator.type = type;
    oscillator.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
    const at = begin + time(from - fromTick);
    const end = begin + time(to - fromTick);
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(level, at + 0.01);
    gain.gain.setValueAtTime(level, Math.max(at + 0.01, end - 0.04));
    gain.gain.linearRampToValueAtTime(0, end);
    oscillator.connect(gain).connect(out);
    oscillator.start(at);
    oscillator.stop(end + 0.01);
    sources.push(oscillator);
  };
  for (const item of score.tracks.Vocal ?? []) note(item.pitch, item.start, item.duration, "triangle", 0.5);
  for (const item of score.tracks.Ins ?? []) note(item.pitch, item.start, item.duration, "sine", 0.3);
  score.chords.forEach((chord, index) => {
    const end = score.chords[index + 1]?.start ?? score.total_ticks;
    for (const pitch of chordPitches(chord.symbol)) note(pitch, chord.start, end - chord.start, "sine", 0.1);
  });
  button.textContent = playingLabel;
  scorePlayback = { sources, from: time(fromTick), begin, timer: setTimeout(stopScore, (time(toTick - fromTick) + 0.3) * 1000) };
  requestAnimationFrame(updateScoreCursor);
}

function sectionTicks(index) {
  const from = score.bars[score.sections[index].start_bar].start;
  const to = index + 1 < score.sections.length ? score.bars[score.sections[index + 1].start_bar].start : score.total_ticks;
  return [from, to];
}

function drawInspector() {
  const index = sections.findIndex((section) => section.name === selected);
  $("inspector").classList.toggle("hidden", index < 0);
  if (index < 0) return;
  const section = sections[index];
  $("inspector-title").textContent = `${section.name} · ${fmt(section.start)}–${fmt(section.end)} · ${section.bars} bars · ` +
    (section.notes ? `${section.notes} notes, ~${section.syllables} syllables` : "free melody");
  $("section-arrangement").textContent = project.arrangement[section.name] ? `Agent: ${project.arrangement[section.name]}` : "";
  if (document.activeElement !== $("section-name")) $("section-name").value = section.name;
  if (document.activeElement !== $("section-lyrics")) $("section-lyrics").value = section.lyrics;
  const ownStyle = project.sectionStyles[section.name];
  if (styleDraft?.section !== section.name) {
    styleDraft = null;
    $("section-style").value = ownStyle ?? defaultSectionStyle(section);
  }
  $("section-style-label").textContent = ownStyle ? "Section style (its own)" : "Section style (the global one; updates itself)";
  $("section-style-save").disabled = $("section-style-undo").disabled = !styleDraft;
  $("section-style-original").disabled = !ownStyle && !styleDraft;
  $("section-negative").value = project.sectionNegatives[section.name] ?? "";
  $("section-negative").placeholder = project.negativeStyle.trim() || "no negative";
  const lead = project.sectionContexts?.[section.name];
  if (document.activeElement !== $("section-context")) $("section-context").value = lead ? contextLabel(lead) : "";
  $("section-classical").value = project.classicalPlan[section.name] ?? "";
  $("section-singer").innerHTML = styleOptions(["singer"], songSingerLabel(), project.sectionSingers[section.name]);
  $("section-prompt").textContent = sentPrompt(section);
  if (document.activeElement !== $("section-seed")) $("section-seed").value = sectionSeed(section, index);
  $("section-merge").disabled = index === sections.length - 1;
  const original = sectionOriginal(section.name);
  $("section-restore").classList.toggle("hidden", !original);
  $("section-restore").textContent = original?.kind === "instrumental" ? "Restore vocal and lyrics" : "Restore written melody";
  $("section-earlier").disabled = $("section-later").disabled = index === 0;
  const takes = $("section-takes");
  takes.innerHTML = project.takes.length ? "<span>Take of this section</span>" : "<span>No takes yet.</span>";
  for (const take of project.takes) {
    const row = document.createElement("div");
    row.className = "take-row";
    const changed = (take.sectionLyrics[section.name] ?? "").trim() !== section.lyrics.trim() ? " · other lyrics" : "";
    const sung = takeCovers(take, section);
    const playing = audition?.take === take.id && audition.section === section.name;
    row.innerHTML = `<label><input type="radio" name="take" ${project.comp[section.name] === take.id ? "checked" : ""}>
      T${take.id} · seed ${take.sectionSeeds[section.name]}${changed}</label>
      <button type="button" class="take-play" ${sung ? "" : "disabled"} title="${sung ? "Listen to this section in this take, without choosing it" : "This take did not render this section"}">${playing ? "■" : "▶"}</button>
      <button type="button" class="take-in-song" ${sung ? "" : "disabled"} title="Listen to it inside the song, from 4 s before the section, with the other sections as chosen. The chosen take does not change.">▶ in song</button>`;
    row.querySelector("input").onchange = () => {
      project.comp[section.name] = take.id;
      draw();
      restartIfPlaying();
    };
    row.querySelector(".take-play").onclick = guard(async () => { await auditionTake(take, section); drawInspector(); });
    row.querySelector(".take-in-song").onclick = guard(() => auditionInSong(take, section));
    takes.append(row);
  }
  $("section-keep").checked = Boolean(project.sectionLocks?.[section.name]);
  $("section-keep").disabled = !takeById(project.comp[section.name]);
}

// Whether a take's files hold this section (a take that regenerated some sections only rendered their stretch).
function takeCovers(take, section) {
  if (!take.files?.vocals) return false;
  return !take.window || (take.window[0] < section.end && take.window[1] > section.start);
}

// One take's voice and instrumental over a section, without the mixer and without choosing the take.
let audition = null;

function stopAudition() {
  if (!audition) return;
  audition.sources.forEach((source) => { source.onended = null; source.stop(); });
  audition = null;
}

async function auditionTake(take, section) {
  const same = audition?.take === take.id && audition.section === section.name;
  stopAudition();
  if (same) return;
  pause();
  await context().resume();
  await loadTake(take);
  const sources = ["vocals", "instrumental"].map((track) => take.files[track] && buffers.get(fileUrl(take.files[track]))?.buffer).filter(Boolean).map((buffer) => {
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(0, section.start, section.end - section.start);
    return source;
  });
  if (!sources.length) throw new Error(`The audio of take ${take.id} could not be read.`);
  audition = { take: take.id, section: section.name, sources };
  sources[0].onended = () => { audition = null; if (selected === section.name) drawInspector(); };
}

// The same take heard inside the song: the section plays this take until the playback stops or restarts; the chosen
// take is not changed.
async function auditionInSong(take, section) {
  pause();
  await loadTake(take);
  const chosen = project.comp[section.name];
  project.comp[section.name] = take.id;
  pausedAt = Math.max(0, section.start - 4);
  try {
    await play();
  } finally {
    project.comp[section.name] = chosen;
  }
}

async function renameSection(name) {
  const index = sections.findIndex((section) => section.name === selected);
  name = name.replace(/[\r\n[\]]+/g, " ").trim();
  if (!name || name === selected) return;
  if (sections.some((section) => section.name === name)) throw new Error(`A section «${name}» already exists.`);
  const edited = editableSections();
  edited[index].name = name;
  const from = selected;
  selected = name;
  await applySections(edited, [[from, name]]);
}

async function splitSection() {
  const index = sections.findIndex((section) => section.name === selected);
  const edited = editableSections();
  const bar = nearestBar(position());
  if (bar <= edited[index].start_bar || bar > edited[index].end_bar) {
    throw new Error("Put the cursor inside the section, on the bar where the new one should start.");
  }
  let name = `${edited[index].name} b`;
  while (sections.some((section) => section.name === name)) name += "b";
  edited.splice(index + 1, 0, { id: `section-${Date.now()}`, name, start_bar: bar, end_bar: edited[index].end_bar, lyrics: "" });
  edited[index].end_bar = bar - 1;
  // The new part keeps what the section had: its take (the audio under its bars), style, singer, and which
  // tracks and voices play there.
  const maps = [project.comp, project.sectionStyles, project.sectionSingers, project.sectionLocks ?? {}, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
  for (const voice of project.voices) maps.push(voice.on, voice.sectionOffsets, voice.sectionGains ?? {});
  for (const map of maps) if (selected in map) map[name] = map[selected];
  selected = name;
  await applySections(edited);
}

async function moveSectionStart(bars) {
  const index = sections.findIndex((section) => section.name === selected);
  if (index < 1) throw new Error("The first section always starts at bar 1.");
  const edited = editableSections();
  const start = edited[index].start_bar + bars;
  if (start <= edited[index - 1].start_bar || start > edited[index].end_bar) throw new Error("There are not enough bars to move that boundary.");
  edited[index].start_bar = start;
  edited[index - 1].end_bar = start - 1;
  await applySections(edited);
}

async function mergeSection() {
  const index = sections.findIndex((section) => section.name === selected);
  const [first, second] = [sections[index], sections[index + 1]];
  const describe = (section) => {
    const style = project.sectionStyles[section.name];
    const singer = catalogStyle(project.sectionSingers[section.name]);
    return `${style ? `own style («${style.split("\n").at(-1).slice(0, 50)}»)` : "global style"} · ${singer ? `sung by ${singer.name}` : songSingerLabel().toLowerCase()}`;
  };
  // The merged section can only sing one style and one singer.
  let keep = first;
  if ((project.sectionStyles[first.name] ?? "") !== (project.sectionStyles[second.name] ?? "")
      || (project.sectionSingers[first.name] ?? null) !== (project.sectionSingers[second.name] ?? null)) {
    keep = await choose(`Merge «${first.name}» and «${second.name}»`,
      "They have a different style or singer, and the merged section can only have one. Which one does it keep?",
      [{ value: first, label: `«${first.name}»: ${describe(first)}` }, { value: second, label: `«${second.name}»: ${describe(second)}` }]);
    if (!keep) return;
  }
  for (const map of [project.sectionStyles, project.sectionSingers]) {
    if (keep.name in map) map[first.name] = map[keep.name];
    else delete map[first.name];
    delete map[second.name];
  }
  const edited = editableSections();
  const [next] = edited.splice(index + 1, 1);
  edited[index].end_bar = next.end_bar;
  edited[index].lyrics = [edited[index].lyrics.trim(), next.lyrics.trim()].filter(Boolean).join("\n");
  await applySections(edited, [], false, keep === second ? { [edited[index].id]: next.id } : {});
}

// A small modal choice; resolves to the chosen option's value, or null when cancelled.
function choose(title, text, options) {
  return new Promise((resolve) => {
    const dialog = $("choice");
    $("choice-title").textContent = title;
    $("choice-text").textContent = text;
    const answer = (value) => { dialog.onclose = null; dialog.close(); resolve(value); };
    $("choice-options").replaceChildren(...options.map(({ value, label }) => {
      const button = document.createElement("button");
      button.textContent = label;
      button.onclick = () => answer(value);
      return button;
    }), Object.assign(document.createElement("button"), { textContent: "Cancel", onclick: () => answer(null) }));
    dialog.onclose = () => resolve(null);
    dialog.showModal();
  });
}

// A name typed in our own dialog (the browser's prompt() is awkward on a phone); null when cancelled.
function askText(title, text, value) {
  return new Promise((resolve) => {
    const dialog = $("ask");
    $("ask-title").textContent = title;
    $("ask-text").textContent = text;
    $("ask-input").value = value ?? "";
    const answer = (result) => { dialog.onclose = null; dialog.close(); resolve(result); };
    $("ask-ok").onclick = () => answer($("ask-input").value.trim());
    $("ask-cancel").onclick = () => answer(null);
    $("ask-input").onkeydown = (event) => { if (event.key === "Enter") answer($("ask-input").value.trim()); };
    dialog.onclose = () => resolve(null);
    dialog.showModal();
    $("ask-input").select();
  });
}

// ---------- history window

let historyBranch = null;

async function openHistory() {
  if (!savedName) throw new Error("Save the song first: its history starts with the first save.");
  readForm();
  ensureHistory(project);
  historyBranch = project.history.current;
  $("history-song").textContent = `«${project.name}»`;
  $("history").showModal();
  await drawHistory();
}

async function drawHistory() {
  const history = project.history;
  const showArchived = $("history-archived").checked;
  const names = branchOrder(history).filter((name) => showArchived || !history.branches[name].archived || name === historyBranch);
  $("history-branch").innerHTML = names.map((name) => {
    const branch = history.branches[name];
    const tags = [name === history.current ? "in use" : "", branch.archived ? "archived" : ""].filter(Boolean).join(", ");
    return `<option value="${escapeHtml(name)}"${name === historyBranch ? " selected" : ""}>${escapeHtml(name)}${tags ? ` (${tags})` : ""}</option>`;
  }).join("");
  $("history-archive-branch").textContent = history.branches[historyBranch]?.archived ? "Unarchive branch" : "Archive branch";
  $("history-archive-branch").disabled = historyBranch === history.current;
  const heads = {};
  for (const [name, branch] of Object.entries(history.branches)) (heads[branch.head] ??= []).push(name);
  const current = stateHash(stateOf(project));
  const unsaved = historyBranch === history.current && headCommit(project)?.state !== current;
  const rows = branchCommits(historyBranch);
  $("history-rows").replaceChildren(...(unsaved ? [Object.assign(document.createElement("div"), {
    className: "history-row current", textContent: "Now: changes not saved yet (they save themselves when you switch branch or song).",
  })] : []), ...rows.map(([id, commit]) => {
    const row = document.createElement("div");
    row.className = "history-row" + (commit.state === current ? " current" : "");
    const branches = (heads[id] ?? []).map((name) => `<span class="chip">${escapeHtml(name)}</span>`).join(" ");
    row.innerHTML = `<div class="history-head"><span class="history-date">${new Date(commit.at).toLocaleString()}</span>
      <input class="history-label" maxlength="80" value="${escapeHtml(commit.label)}" title="Name of this point; change it to find it later">
      ${commit.auto ? '<span class="note">automatic</span>' : ""} ${branches}
      ${commit.branch !== historyBranch ? `<span class="note">from the branch «${escapeHtml(commit.branch)}»</span>` : ""}</div>
      <div class="history-changes note">…</div>
      <div class="row"><button data-action="open">${commit.state === current ? "This is what you have open" : "Open"}</button>
      <button data-action="compare">Compare with current</button></div>`;
    const open = row.querySelector('[data-action="open"]');
    open.disabled = commit.state === current;
    open.onclick = guard(async () => { await openCommit(id); historyBranch = project.history.current; await drawHistory(); });
    row.querySelector('[data-action="compare"]').onclick = guard(async () => {
      const changes = stateChanges(await snapshot(commit.state), stateOf(project));
      row.querySelector(".history-changes").textContent = changes.length
        ? `The current state differs from this point in: ${changes.join(", ")}.` : "The current state is the same as this point.";
    });
    row.querySelector(".history-label").onchange = guard(async (event) => {
      commit.label = event.target.value.trim() || commit.label;
      commit.auto = false;
      await saveProject();
    });
    return row;
  }));
  // What each point changed, filled in as the snapshots arrive (newest first, a few at a time).
  const boxes = [...$("history-rows").querySelectorAll(".history-row:not(.current), .history-row.current")].filter((row) => row.querySelector(".history-changes"));
  for (const [index, [, commit]] of rows.entries()) {
    const box = boxes[index]?.querySelector(".history-changes");
    if (!box) continue;
    try {
      const parent = history.commits[commit.parent];
      const changes = stateChanges(parent ? await snapshot(parent.state) : null, await snapshot(commit.state));
      box.textContent = changes[0] === "start" ? "First point." : changes.length ? `Changed: ${changes.join(", ")}.` : "No changes to the song (only the name).";
    } catch {
      box.textContent = "";
    }
  }
}

// ---------- extra voices

function selectedVoiceEntry() {
  return project.voices.find((voice) => voice.id === selectedVoice);
}

function addVoice() {
  readForm();
  const id = Math.max(0, ...project.voices.map((voice) => voice.id)) + 1;
  const global = project.style.split("\n").find((line) => line.trim() && !line.trim().startsWith("[")) ?? "";
  project.voices.push({
    id, name: `Voice ${id + 1}`, source: "lead", octave: 0, role: "-4", seed: project.seed + 1000 * id,
    style: `${global.trim()}\nSpanish female soprano lead vocal, clear and bright`.trim(),
    singer: null, file: null, offset: 0, sectionOffsets: {}, on: {},
  });
  selectedVoice = id;
  selected = null;
  draw();
}

async function renderVoice(voice) {
  readForm();
  ensureName();
  await refreshSections();
  const octave = Number(voice.octave) || 0;
  // Harmony lines and octave moves come from Vocal Harmony (melody-only ABC); the plain lead keeps the song ABC.
  const line = VOICE_LINES[voice.source] ?? (octave ? 0 : undefined);
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "HZ3_YuE2_GenerateMusicSections",
      inputs: {
        clip: ["1", 1], style: [voice.style, catalogStyle(voice.singer)?.text].filter((text) => text?.trim()).join("\n"),
        lyrics: project.lyrics, abc: line !== undefined ? ["20", line] : project.abc, seed: voice.seed,
        mode: line !== undefined ? "melody" : "full", ...project.sampling, section_seeds: "", section_styles: "",
      },
    },
    3: { class_type: "EmptyYuE2LatentAudio", inputs: { seconds: ["2", 1], batch_size: 1 } },
    4: {
      class_type: "KSampler",
      inputs: {
        model: ["1", 0], seed: 42, ...ACOUSTIC_SAMPLING, cfg: 4.2, scheduler: "sgm_uniform",
        positive: ["2", 0], negative: ["2", 0], latent_image: ["3", 0], denoise: 1,
      },
    },
    5: { class_type: "VAEDecodeAudioTiled", inputs: { samples: ["4", 0], vae: ["1", 2], tile_size: 256, overlap: 32 } },
    7: { class_type: "AudioSeparation", inputs: { audio: ["5", 0] } },
    8: { class_type: "SaveAudio", inputs: { audio: ["7", 3], filename_prefix: `HZ3-Studio/${project.name}/voice-${voice.id}/vocals` } },
    9: { class_type: "SaveAudio", inputs: { audio: ["5", 0], filename_prefix: `HZ3-Studio/${project.name}/voice-${voice.id}/mix` } },
  };
  const singer = catalogStyle(voice.singer);
  withLora(prompt, [{ lora: singer?.lora, owner: singer?.name }]);
  if (line !== undefined) {
    // The harmony line keeps the song's bars and section markers, so its sections match the lead's.
    prompt[20] = {
      class_type: "HZ3_YuE2_VocalHarmony",
      inputs: { score_abc: project.abc, arrangement: "close_harmony", active_sections: "", tenor_low: 60, tenor_high: 84,
                baritone_low: 48, baritone_high: 76, bass_low: 36, bass_high: 64, octave },
    };
  }
  const owner = project.name;
  await queue(prompt, { ...RENDER_LABELS, 2: `YuE2 tokens (${voice.name})`, 20: "Harmony line" }, async (outputs) => {
    const file = outputs[8]?.audio?.[0];
    if (!file) throw new Error("The voice finished without audio.");
    const current = await updateProject(owner, (target) => {
      const stored = target.voices.find((item) => item.id === voice.id);
      if (stored) Object.assign(stored, { file, mixFile: outputs[9]?.audio?.[0] ?? null });
    });
    if (!current) return;  // aligned when that project is open again ("Align with the lead vocal")
    await loadUrl(fileUrl(file));
    await alignVoice(project.voices.find((item) => item.id === voice.id));
    await saveProject();
    remember(`${voice.name} generada y alineada`);
    draw();
  });
  status(`${voice.name} queued…`, 0);
}

function interpolate(x, xs, ys) {
  if (x <= xs[0]) return ys[0] + (x - xs[0]);
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1] + (x - xs[xs.length - 1]);
  let high = 1;
  while (xs[high] < x) high++;
  const low = high - 1;
  const span = xs[high] - xs[low];
  return span ? ys[low] + (ys[high] - ys[low]) * (x - xs[low]) / span : ys[high];
}

function onsets(values, from, to) {
  const result = new Float32Array(to - from);
  for (let index = Math.max(1, from); index < to; index++) result[index - from] = Math.max(0, (values[index] ?? 0) - (values[index - 1] ?? 0));
  return result;
}

function refineShift(leadEntry, voiceEntry, section, shift) {
  // Rhythm settles what pitch cannot (rap, spoken lines): best onset match within ±1.5 s.
  const from = Math.floor(section.start * PEAKS_PER_SECOND);
  const to = Math.floor(section.end * PEAKS_PER_SECOND);
  const reference = onsets(leadEntry.peaks, from, to);
  const radius = 1.5 * PEAKS_PER_SECOND;
  const center = Math.round(shift * PEAKS_PER_SECOND);
  const signal = onsets(voiceEntry.peaks, from + center - radius, to + center + radius);
  let best = 0;
  let bestScore = -Infinity;
  for (let lag = 0; lag <= 2 * radius; lag++) {
    let score = 0;
    for (let index = 0; index < reference.length; index++) score += reference[index] * signal[index + lag];
    if (score > bestScore) { bestScore = score; best = lag; }
  }
  return (center + best - radius) / PEAKS_PER_SECOND;
}

async function lineWarp(file, line) {
  return postJson("/hz3/studio/warp", { file, abc: project.abc, line });
}

async function alignVoice(voice) {
  // Lead and voice are each mapped onto the score by pitch; their difference is the shift.
  if (!voice.file) throw new Error("Generate the voice before aligning it.");
  status(`Aligning ${voice.name} with the lead vocal…`);
  for (const take of new Set(Object.values(project.comp).map(takeById))) {
    if (take?.files.vocals && !take.warp) take.warp = await lineWarp(take.files.vocals, "lead");
  }
  voice.warp = await lineWarp(voice.file, voice.source);
  const shifts = {};
  for (const section of sections) {
    const lead = takeById(project.comp[section.name])?.warp;
    if (!lead || !voiceEnabled(voice, section)) continue;
    const samples = [];
    for (let time = section.start; time < section.end; time += 0.5) {
      const scoreTime = interpolate(time, lead.audio, lead.abc);
      samples.push(interpolate(scoreTime, voice.warp.abc, voice.warp.audio) - time);
    }
    samples.sort((a, b) => a - b);
    const shift = samples[Math.floor(samples.length / 2)];
    // Trust pitch where both lines matched the score; elsewhere (rap, spoken parts) refine by rhythm.
    const sure = (warp, from, to) => {
      const knots = warp.audio.map((time, index) => [time, warp.sure[index]]).filter(([time]) => time >= from && time < to);
      return knots.length > 0 && knots.filter(([, flag]) => flag).length >= 0.6 * knots.length;
    };
    const voiceFrom = section.start + shift;
    shifts[section.name] = sure(lead, section.start, section.end) && sure(voice.warp, voiceFrom, voiceFrom + section.end - section.start)
      ? shift
      : refineShift(trackBuffer("vocals", project.comp[section.name]), buffers.get(fileUrl(voice.file)), section, shift);
  }
  // Drift between two singers is gradual; a jump of seconds is a false match, so search near the previous section.
  let previous = null;
  for (const section of sections) {
    if (!(section.name in shifts)) continue;
    if (previous !== null && Math.abs(shifts[section.name] - previous) > 3) {
      shifts[section.name] = refineShift(trackBuffer("vocals", project.comp[section.name]), buffers.get(fileUrl(voice.file)), section, previous);
    }
    previous = shifts[section.name];
  }
  const values = Object.values(shifts).sort((a, b) => a - b);
  voice.offset = values.length ? values[Math.floor(values.length / 2)] : 0;
  voice.sectionOffsets = Object.fromEntries(Object.entries(shifts).map(([name, shift]) => [name, shift - voice.offset]));
  levelVoice(voice, buffers.get(fileUrl(voice.file)));
  status(`${voice.name} aligned: ${Math.round(voice.offset * 1000)} ms global · level ${voice.role} dB relative to the lead.`, 1);
  draw();
  restartIfPlaying();
}

function levelVoice(voice, entry) {
  // Section by section, sit the voice at its role's level against the lead's energy there;
  // where the lead is switched off, against the lead's typical level.
  // RMS of the real samples (every 16th, first channel); peaks would understate loudness gaps.
  const energy = (target, from, to, shift = 0) => {
    const data = target.buffer.getChannelData(0);
    const rate = target.buffer.sampleRate / PEAKS_PER_SECOND;
    const first = Math.max(0, Math.floor((from + shift) * rate));
    const last = Math.min(data.length, Math.floor((to + shift) * rate));
    let sum = 0;
    let count = 0;
    for (let index = first; index < last; index += 16) { sum += data[index] ** 2; count++; }
    return Math.sqrt(sum / Math.max(1, count));
  };
  const leadLevels = {};
  for (const section of sections) {
    const leadEntry = trackBuffer("vocals", project.comp[section.name]);
    if (leadEntry && trackEnabled("vocals", section)) {
      leadLevels[section.name] = energy(leadEntry, Math.floor(section.start * PEAKS_PER_SECOND), Math.floor(section.end * PEAKS_PER_SECOND));
    }
  }
  const typical = Object.values(leadLevels).sort((a, b) => a - b)[Math.floor(Object.keys(leadLevels).length / 2)] ?? 0;
  voice.sectionGains = {};
  for (const section of sections) {
    if (!voiceEnabled(voice, section)) continue;
    const own = energy(entry, Math.floor(section.start * PEAKS_PER_SECOND), Math.floor(section.end * PEAKS_PER_SECOND),
                       Math.round(voiceShift(voice, section) * PEAKS_PER_SECOND));
    const lead = leadLevels[section.name] ?? typical;
    voice.sectionGains[section.name] = own ? Math.min(2, (lead / own) * 10 ** (Number(voice.role) / 20)) : 1;
  }
  voice.autoGain = 1;
  delete project.mixer[`voice-${voice.id}`]?.gain;
}

function drawVoiceInspector() {
  const voice = selectedVoiceEntry();
  $("voice-inspector").classList.toggle("hidden", !voice);
  if (!voice) return;
  $("voice-title").textContent = `${voice.name}${voice.file ? "" : " · not generated"}`;
  for (const [id, key] of [["voice-name", "name"], ["voice-source", "source"], ["voice-octave", "octave"], ["voice-role", "role"], ["voice-style", "style"], ["voice-seed", "seed"]]) {
    if (document.activeElement !== $(id)) $(id).value = voice[key];
  }
  if (document.activeElement !== $("voice-offset")) $("voice-offset").value = Math.round(voice.offset * 1000);
  $("voice-singer").innerHTML = styleOptions(["singer"], "— (no catalog singer)", voice.singer);
  const list = $("voice-sections");
  list.innerHTML = "<span>Fine adjustment per section (ms)</span>";
  for (const section of sections.filter((item) => voiceEnabled(voice, item))) {
    const label = document.createElement("label");
    label.innerHTML = `${section.name} <input type="number" step="10" value="${Math.round((voice.sectionOffsets[section.name] ?? 0) * 1000)}">`;
    label.querySelector("input").onchange = (event) => {
      voice.sectionOffsets[section.name] = Number(event.target.value) / 1000;
      draw();
      restartIfPlaying();
    };
    list.append(label);
  }
}

// ---------- reference recordings (our own dataset)

function audioName(file, taken) {
  let name = file.name.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N} _()-]+/gu, "_").slice(0, 56).trim() || "audio";
  for (let count = 2; taken.has(name); count++) name = `${name.replace(/ \(\d+\)$/, "")} (${count})`;
  taken.add(name);
  return name;
}

// Each file becomes a package of its own (a library audio or a reference) holding the audio.
async function addAudios(files, fields) {
  const { projects } = await api("/hz3/studio/projects");
  const taken = new Set(projects.map(({ name }) => name));
  const names = [];
  for (const file of files) {
    const name = audioName(file, taken);
    const form = new FormData();
    form.append("file", file);
    const source = await api(`/hz3/studio/source?project=${encodeURIComponent(name)}`, { method: "POST", body: form });
    await postJson("/hz3/studio/project", { ...newProject(), name, ...fields, source });
    names.push(name);
  }
  await listProjects();
  return names;
}

async function importReferences(files, license) {
  for (const name of await addAudios(files, { kind: "reference", license })) {
    const { source } = await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`);
    // Melody and chords only: a reference is a source of themes, not a song to sing.
    const prompt = {
      1: { class_type: "LoadAudio", inputs: { audio: source.filename } },
      2: { class_type: "AudioEncoderLoader", inputs: { audio_encoder_name: "sheetsage2_bf16.safetensors" } },
      3: { class_type: "HZ3_YuE2_SheetSage2Sections", inputs: { audio_encoder: ["2", 0], audio: ["1", 0], mode: "full" } },
      13: { class_type: "PreviewAny", inputs: { source: ["3", 0] } },
      14: { class_type: "PreviewAny", inputs: { source: ["3", 4] } },
    };
    await queue(prompt, { 3: `SheetSage2 · ${name}` }, async (outputs) => {
      const text = (node) => outputs[node]?.text?.[0] ?? "";
      const reference = await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`);
      reference.abc = text(13);
      reference.analysis = { at: Date.now(), sheetsage_abc: text(13), sheetsage_report: text(14) };
      await postJson("/hz3/studio/project", reference);
      await listProjects();
      status(`Reference «${name}» analyzed.`, 1);
    });
  }
  status(`${files.length} references queued for analysis.`, 0);
}

// ---------- projects (.mixmash packages)

let savedName = null;
// Songs renamed since the page loaded, old name -> new name: a job queued under the old name still finds its song.
const renamedTo = {};
const currentName = (name) => { while (renamedTo[name] && renamedTo[name] !== name) name = renamedTo[name]; return name; };

// Songs are renamed in the catalog; a song without a name yet asks for one when you save or render it.
function ensureName() {
  if (project.name) return;
  const name = (prompt("Song name") ?? "").trim();
  if (!name) throw new Error("The song needs a name to be saved.");
  project.name = name;
}

// Every save is a commit on the song's current branch (when something changed); `label` names it.
async function saveProject(label = null) {
  readForm();
  if (!project.name) throw new Error("Give the song a name.");
  if (!savedName) {
    // A song never saved must not take the name of another one: saving would overwrite it.
    const { projects } = await api("/hz3/studio/projects");
    const taken = projects.find(({ name }) => name.toLowerCase() === project.name.toLowerCase());
    if (taken) {
      const name = project.name;
      project.name = "";
      throw new Error(`A song called «${taken.name}» already exists. Choose another name for «${name}».`);
    }
  }
  // A new name on an opened project renames its package instead of copying it.
  if (savedName && savedName !== project.name) {
    try {
      await postJson("/hz3/studio/rename", { from: savedName, to: project.name });
    } catch (error) {
      // The song keeps the name it has on disk, so later saves and finished jobs still find it.
      project.name = savedName;
      writeForm();
      throw error;
    }
    renamedTo[savedName] = project.name;
  }
  project.styles = referencedStyles();
  dropStaleSections();
  commitSong(project, label ?? "saved changes", !label);
  await postJson("/hz3/studio/project", project);
  delete project._snapshots;
  savedName = project.name;
  await listProjects();
  status(`Song «${project.name}» saved.`, 1);
}

async function listProjects() {
  const { projects } = await api("/hz3/studio/projects");
  catalog = projects;
  // Only the browsed album's songs; archived ones stay out of the quick list (unless open), the catalog shows them.
  const songs = projects.filter((entry) => entry.album === browseAlbum).filter((entry) => entry.kind !== "source" && (!entry.archived || entry.name === project.name));
  $("project-list").innerHTML = `<option value="">— ${songs.length} songs —</option><option value="${NEW_SONG}">+ New song</option>`
    + songs.map((entry) => `<option value="${escapeHtml(entry.name)}"${entry.name === project.name ? " selected" : ""}>${escapeHtml(entry.name)}</option>`).join("");
  const sources = projects.filter((entry) => entry.kind === "source");
  $("source-pick").innerHTML = `<option value="">— no library audio —</option>`
    + sources.map((entry) => `<option value="${escapeHtml(entry.name)}"${entry.name === project.sourceName ? " selected" : ""}>${escapeHtml(entry.name)}</option>`).join("");
  if ($("catalog").open) await drawCatalog();
  if ($("catalog").open && !$("tab-sources").classList.contains("hidden")) await drawSources();
}

// ---------- audio library: each audio is analysed once; songs take its audio, ABC, lyrics or style

async function useSource(name, parts) {
  const source = await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`);
  readForm();
  if (parts.audio) {
    project.source = source.source;
    project.sourceName = name;
    await loadSource();
  }
  if (parts.style && source.style.trim()) project.style = source.style;
  if (parts.lyrics && source.lyrics.trim()) project.lyrics = source.lyrics;
  if (parts.abc && source.abc.trim()) setAbc(source.abc, `before bringing the ABC of «${name}»`);
  writeForm();
  await refreshSections().catch((error) => status(`Check the sections: ${error.message}`, null, true));
  if (parts.abc || parts.lyrics) await uniqueSectionNames();
  if (project.name) await saveProject();
  const taken = Object.entries(parts).filter(([, on]) => on).map(([part]) => ({ audio: "audio", abc: "ABC", lyrics: "lyrics", style: "style" })[part]);
  remember(`De la biblioteca «${name}»: ${taken.join(", ")}`);
  status(`«${name}» → ${taken.join(", ")}.`, 1);
  draw();
}

async function songFromSource(name) {
  await newSong();
  project.name = audioName({ name }, new Set(catalog.map((entry) => entry.name)));
  await useSource(name, { audio: true, abc: true, lyrics: true, style: true });
}

const openSources = new Set();

async function drawSources() {
  const rows = $("source-rows");
  const sources = catalog.filter((entry) => entry.kind === "source");
  rows.innerHTML = sources.length ? "" : `<p class="note">The library is empty.</p>`;
  for (const entry of sources) {
    const users = catalog.filter((other) => other.kind !== "source" && other.source_file === entry.source_file).map((other) => other.name);
    const item = document.createElement("details");
    item.className = "source";
    item.innerHTML = `<summary><b>${escapeHtml(entry.name)}</b> · ${escapeHtml(entry.source ?? "")} ·
        ${entry.analyzed ? `analyzed, ${entry.sections} sections` : "not analyzed"}${users.length ? ` · used in ${users.map(escapeHtml).join(", ")}` : ""}</summary>
      <div class="row">
        <button data-action="analyze">${entry.analyzed ? "Analyze again" : "Analyze"}</button>
        <span class="parts">Bring into the open song:
          <label class="switch"><input type="checkbox" value="audio" checked> audio</label>
          <label class="switch"><input type="checkbox" value="abc" checked> ABC</label>
          <label class="switch"><input type="checkbox" value="lyrics" checked> lyrics</label>
          <label class="switch"><input type="checkbox" value="style" checked> style</label>
          <button data-action="use">Bring</button></span>
        <button data-action="song" class="primary">New song with this audio</button>
        <button data-action="delete">Delete</button>
      </div>
      <div class="fields">
        <label>Style<textarea data-field="style" rows="4"></textarea></label>
        <label>Lyrics<textarea data-field="lyrics" rows="8"></textarea></label>
        <label>ABC<textarea data-field="abc" rows="8" spellcheck="false"></textarea></label>
      </div>`;
    // The texts load when the entry is opened, so the list stays light.
    const fill = async () => {
      const stored = await api(`/hz3/studio/project?name=${encodeURIComponent(entry.name)}`);
      item.querySelectorAll("textarea").forEach((area) => { area.value = stored[area.dataset.field] ?? ""; });
    };
    item.ontoggle = guard(async () => {
      if (item.open) {
        openSources.add(entry.name);
        await fill();
      } else {
        openSources.delete(entry.name);
      }
    });
    item.querySelectorAll("textarea").forEach((area) => {
      area.onchange = guard(() => updateProject(entry.name, (stored) => {
        if (area.dataset.field === "abc") keepAbcVersion(stored, "before editing it in the library");
        stored[area.dataset.field] = area.value;
      }));
    });
    item.querySelector('[data-action="analyze"]').onclick = guard(() => analyze(entry.name));
    item.querySelector('[data-action="use"]').onclick = guard(() => useSource(entry.name,
      Object.fromEntries([...item.querySelectorAll(".parts input")].map((box) => [box.value, box.checked]))));
    item.querySelector('[data-action="song"]').onclick = guard(async () => {
      $("catalog").close();
      await songFromSource(entry.name);
    });
    item.querySelector('[data-action="delete"]').onclick = guard(() => deleteSong(entry.name));
    rows.append(item);
    if (openSources.has(entry.name)) item.open = true;
  }
}

// ---------- catalog: albums and their songs

let managedAlbum = DEFAULT_ALBUM;

async function drawCatalog() {
  const { albums } = await api("/hz3/studio/albums");
  if (!albums.includes(managedAlbum)) managedAlbum = DEFAULT_ALBUM;
  const info = await albumData(managedAlbum);
  const songs = catalog.filter((entry) => entry.album === managedAlbum && entry.kind !== "source");
  $("album-nav").innerHTML = albums.map((name) => `<button data-album="${escapeHtml(name)}" class="${name === managedAlbum ? "active" : ""}">${escapeHtml(name)}
    <small>${catalog.filter((entry) => entry.album === name && !entry.archived).length}</small></button>`).join("")
    + `<button data-album="${NEW_ALBUM}">+ New album</button>`;
  $("album-nav").querySelectorAll("button").forEach((button) => {
    button.onclick = guard(async () => {
      let name = button.dataset.album;
      if (name === NEW_ALBUM) {
        name = (prompt("Name of the new album") ?? "").trim();
        if (!name) return;
        if (!albums.includes(name)) await postJson("/hz3/studio/album", { name, notes: "", styles: [], chat: [], log: [] });
        await listAlbums();
      }
      managedAlbum = name;
      await drawCatalog();
    });
  });
  $("album-name").value = managedAlbum;
  $("album-notes").value = info.notes ?? "";
  $("album-concept").innerHTML = `<option value="">— none</option>` + songs.filter((entry) => !entry.archived || entry.name === info.concept)
    .map((entry) => `<option value="${escapeHtml(entry.name)}"${entry.name === info.concept ? " selected" : ""}>${escapeHtml(entry.name)}</option>`).join("");
  $("album-delete").disabled = managedAlbum === DEFAULT_ALBUM;
  $("album-styles").innerHTML = info.styles.map((style) => {
    const users = songs.filter((entry) => entry.styles.some((used) => used.id === style.id)).map((entry) => escapeHtml(entry.name));
    return `<span class="chip" title="${escapeHtml(style.text)}"><b>${escapeHtml(style.name)}</b> · ${STYLE_KINDS[style.kind]} · ${users.length ? users.join(", ") : "unused"}</span>`;
  }).join("") || `<span class="note">No styles: create them in «Styles» with a song of this album open, or ask the assistant for them.</span>`;
  const albumOptions = (selectedName) => albums.map((name) => `<option value="${escapeHtml(name)}"${name === selectedName ? " selected" : ""}>${escapeHtml(name)}</option>`).join("");
  $("catalog-bulk-album").innerHTML = albumOptions(managedAlbum);
  drawCatalogCards(songs, albumOptions);
}

// The songs of the managed album as cards: search (name and lyrics), a filter, an order, a play button with the newest
// take, and every action of a song in its «⋯» menu. Several cards can be selected to move or archive them together.
const catalogSelection = new Set();
const catalogPreview = new Audio();

function drawCatalogCards(songs, albumOptions) {
  const query = $("catalog-search").value.trim().toLowerCase();
  const filter = $("catalog-filter").value;
  const shown = songs
    .filter((entry) => ({ active: !entry.archived, takes: !entry.archived && entry.takes > 0, empty: !entry.archived && !entry.takes,
      archived: entry.archived, all: true })[filter])
    .filter((entry) => !query || `${entry.name} ${entry.first_line} ${entry.lyrics_text ?? ""} ${entry.notes}`.toLowerCase().includes(query))
    .sort({ recent: (a, b) => b.updated - a.updated, name: (a, b) => a.name.localeCompare(b.name), takes: (a, b) => b.takes - a.takes }[$("catalog-sort").value]);
  for (const name of [...catalogSelection]) if (!songs.some((entry) => entry.name === name)) catalogSelection.delete(name);
  $("catalog-bulk").classList.toggle("hidden", !catalogSelection.size);
  $("catalog-selected").textContent = `${catalogSelection.size} selected`;
  const cards = $("catalog-cards");
  cards.replaceChildren();
  if (!shown.length) {
    cards.innerHTML = `<p class="note">${songs.length ? "No song matches the search or the filter." : "This album has no songs yet."}</p>`;
    return;
  }
  for (const entry of shown) {
    const card = document.createElement("article");
    card.className = "song-card" + (entry.archived ? " archived" : "") + (entry.name === project.name ? " open" : "");
    const tags = [entry.name === project.name ? "open now" : "", entry.archived ? "archived" : "", entry.kind === "reference" ? "reference" : "",
      entry.source ? "from audio" : ""].filter(Boolean);
    card.innerHTML = `<header class="song-card-head">
        <input type="checkbox" aria-label="Select ${escapeHtml(entry.name)}"${catalogSelection.has(entry.name) ? " checked" : ""}>
        <button class="play" aria-label="Play the newest take of ${escapeHtml(entry.name)}" ${entry.preview ? "" : "disabled"}>▶</button>
        <strong class="song-card-name" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</strong>
      </header>
      <p class="song-card-line">${escapeHtml(entry.first_line || entry.style || "—")}</p>
      <p class="song-card-meta">${entry.takes} takes · ${entry.sections} sections · ${entry.voices} voices${entry.branches > 1 ? ` · ${entry.branches} branches` : ""}</p>
      <p class="song-card-meta">${new Date(entry.updated * 1000).toLocaleString()}${tags.length ? ` · ${tags.join(" · ")}` : ""}</p>
      ${entry.notes ? `<p class="song-card-notes">${escapeHtml(entry.notes)}</p>` : ""}
      <footer class="row">
        <button data-action="open" class="primary">Open</button>
        <details class="song-card-menu"><summary aria-label="More actions for ${escapeHtml(entry.name)}">⋯</summary>
          <div class="menu">
            <button data-action="rename">Rename…</button>
            <label>Notes<textarea rows="3"></textarea></label>
            <button data-action="duplicate" title="Creates «${escapeHtml(entry.name)} (copy)» in the same album and opens it; the original does not change">Copy as new song</button>
            <label>Album<select title="Move to another album">${albumOptions(entry.album)}</select></label>
            <button data-action="download" title="Downloads its .mixmash: audio, lyrics, ABC, tokens, LoRAs, history and app state">Download .mixmash</button>
            <button data-action="archive">${entry.archived ? "Unarchive" : "Archive"}</button>
            <button data-action="delete" class="danger" title="The package moves to the deleted folder">Delete…</button>
          </div></details>
      </footer>`;
    const act = (name) => card.querySelector(`[data-action="${name}"]`);
    card.querySelector('input[type="checkbox"]').onchange = (event) => {
      if (event.target.checked) catalogSelection.add(entry.name); else catalogSelection.delete(entry.name);
      $("catalog-bulk").classList.toggle("hidden", !catalogSelection.size);
      $("catalog-selected").textContent = `${catalogSelection.size} selected`;
    };
    card.querySelector(".play").onclick = () => {
      const url = `/view?filename=${encodeURIComponent(entry.preview.filename)}&subfolder=${encodeURIComponent(entry.preview.subfolder)}&type=output`;
      const playing = !catalogPreview.paused && catalogPreview.dataset.song === entry.name;
      catalogPreview.pause();
      cards.querySelectorAll(".play").forEach((button) => { button.textContent = "▶"; });
      if (playing) return;
      catalogPreview.src = url;
      catalogPreview.dataset.song = entry.name;
      catalogPreview.play();
      card.querySelector(".play").textContent = "■";
    };
    act("open").onclick = guard(async () => {
      catalogPreview.pause();
      $("catalog").close();
      await openProject(entry.name);
    });
    act("rename").onclick = guard(async () => {
      const name = await askText("Rename song", `New name for «${entry.name}»:`, entry.name);
      if (name !== null) await renameSong(entry.name, name);
    });
    const notes = card.querySelector("textarea");
    notes.value = entry.notes;
    notes.onchange = guard(() => updateProject(entry.name, (stored) => { stored.notes = notes.value; }));
    card.querySelector(".menu select").onchange = guard((event) => moveSong(entry.name, event.target.value));
    act("duplicate").onclick = guard(async () => {
      $("catalog").close();
      await duplicateSong(entry.name);
    });
    act("download").onclick = guard(async () => {
      if (entry.name === project.name) await saveProject();
      location.href = `/hz3/studio/package?name=${encodeURIComponent(entry.name)}`;
    });
    act("archive").onclick = guard(() => updateProject(entry.name, (stored) => { stored.archived = !entry.archived; }));
    act("delete").onclick = guard(() => deleteSong(entry.name));
    cards.append(card);
  }
}

async function catalogBulk(change, done) {
  const names = [...catalogSelection];
  for (const name of names) await change(name);
  catalogSelection.clear();
  await listProjects();
  status(done(names.length), 1);
}

// A copy of a song beside it ("<name> (copy)"): the original is only read, the copy opens.
async function duplicateSong(name) {
  if (name === project.name) await saveProject();
  const { name: copy } = await postJson("/hz3/studio/duplicate", { name });
  await listProjects();
  await openProject(copy);
  status(`«${copy}» is a copy of «${name}»: what you change here does not touch the original.`, 1);
}

async function renameSong(from, to) {
  to = to.trim();
  if (!to || to === from) return;
  if (from === project.name) {
    project.name = to;
    await saveProject();
  } else {
    await postJson("/hz3/studio/rename", { from, to });
    renamedTo[from] = to;
    await listProjects();
  }
  status(`«${from}» is now called «${to}».`, 1);
}

async function moveSong(name, target) {
  if (name === project.name) return moveToAlbum(target);
  let styles = [];
  await updateProject(name, (stored) => {
    stored.album = target;
    styles = stored.styles;
  });
  await addStylesToAlbum(target, styles);
  status(`«${name}» is now in the album «${target}».`, 1);
}

async function deleteSong(name) {
  if (!confirm(`Delete «${name}»? The package moves to the studio's deleted folder (output/HZ3-YuE2/studio/deleted).`)) return;
  await postJson("/hz3/studio/delete", { name });
  if (name === project.name) await newSong();
  await listProjects();
  status(`«${name}» deleted (it is in the deleted folder).`, 1);
}

async function renameAlbum(to) {
  const from = managedAlbum;
  to = to.trim();
  if (!to || to === from) return;
  await postJson("/hz3/studio/album/rename", { from, to });
  if (project.album === from) project.album = to;
  if (album.name === from) await loadAlbum(to);
  if (browseAlbum === from) browseAlbum = to;
  managedAlbum = to;
  await listAlbums();
  await listProjects();
  status(`The album «${from}» is now called «${to}».`, 1);
}

async function deleteAlbum() {
  const name = managedAlbum;
  if (!confirm(`Delete the album «${name}»? Its songs move to «${DEFAULT_ALBUM}» and the album (styles, chat and log) moves to the deleted folder.`)) return;
  const used = referencedStyles();
  await postJson("/hz3/studio/album/delete", { name });
  if (project.album === name) project.album = DEFAULT_ALBUM;
  if (album.name === name) {
    await loadAlbum(DEFAULT_ALBUM);
    await addStylesToAlbum(DEFAULT_ALBUM, used);
  }
  if (browseAlbum === name) browseAlbum = DEFAULT_ALBUM;
  managedAlbum = DEFAULT_ALBUM;
  await listAlbums();
  await listProjects();
  writeForm();
  draw();
  status(`Album «${name}» deleted; its songs are in «${DEFAULT_ALBUM}».`, 1);
}

async function saveAlbumNotes(notes) {
  const info = await albumData(managedAlbum);
  info.notes = notes;
  await postJson("/hz3/studio/album", info);
}

async function newSong() {
  if (songChanged()) await saveProject("before opening another song");
  pause();
  pausedAt = 0;
  if (album.name !== browseAlbum) await loadAlbum(browseAlbum);
  project = { ...newProject(), album: album.name };
  savedName = null;
  selected = null;
  selectedVoice = null;
  writeForm();
  await refreshSections();
  await listProjects();
  draw();
  status(`New song in the album «${album.name}»: give it a name, a style and lyrics.`, 1);
}

async function openProject(name) {
  if (songChanged() && savedName !== name) await saveProject("before opening another song");
  pause();
  pausedAt = 0;
  project = { ...newProject(), ...(await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`)) };
  savedName = project.name;
  ensureHistory(project);
  selected = null;
  selectedTrack = null;
  instrumentAudio.clear();
  // Decoded takes of the previous song would otherwise stay in memory for the whole session (gigabytes over an album).
  buffers.clear();
  if (album.name !== project.album) await loadAlbum(project.album);
  browseAlbum = project.album;
  await addStylesToAlbum(album.name, project.styles);
  await listAlbums();
  await listProjects();
  writeForm();
  await refreshSections();
  // A song from another machine brings its own voice LoRA: it joins the album's singers unless one already has it.
  // (A voice only described in the style is added by hand with «+ Voice of the current song», or every test song would add one.)
  const voice = songVoice();
  if (voice?.lora && !knownVoice(voice)) {
    album.styles.push(voice);
    await saveAlbum();
    drawStyles();
  }
  // Songs saved with repeated section names get unique ones, or every same-named section would be edited at once.
  const renamed = await uniqueSectionNames();
  status("Loading audio…");
  selectedVoice = null;
  for (const voice of project.voices) {
    voice.octave ??= 0;
    voice.role ??= "-4";
  }
  // Sections, lyrics and score do not wait for the audio.
  draw();
  unreadableAudio.clear();
  await Promise.all([loadSource(), ...project.takes.map(loadTake), ...project.voices.filter((voice) => voice.file).map((voice) => loadUrl(fileUrl(voice.file)))]);
  status(renamed.length
    ? `Song «${project.name}» · sections with repeated names renamed (${renamed.map(([, to]) => to).join(", ")}): save to keep it.`
    : `Song «${project.name}» · ${project.takes.length} takes${unreadableAudio.size ? ` · ${unreadableAudio.size} audio files could not be read by this browser (open the song again to retry)` : ""}`, 1, unreadableAudio.size > 0);
  draw();
  guard(renderInstruments)();
}

// ---------- wiring

function guard(action) {
  return (...args) => Promise.resolve(action(...args)).catch((error) => status(error.message, null, true));
}

async function init() {
  markAdvanced();
  setupInspectorTabs();
  const info = await api("/object_info/CheckpointLoaderSimple");
  const names = info.CheckpointLoaderSimple.input.required.ckpt_name[0];
  $("ckpt").innerHTML = names.map((name) => `<option>${name}</option>`).join("");
  loras = (await api("/hz3/studio/loras")).loras;
  await loadAlbum(DEFAULT_ALBUM);
  await listAlbums();
  writeForm();
  await listProjects();
  connectSocket();

  for (const id of ["lyrics", "abc"]) $(id).addEventListener("input", scheduleSections);
  let styleBefore = "";
  $("style").addEventListener("focus", () => { styleBefore = $("style").value; });
  $("style").addEventListener("change", () => { readForm(); followStyle(styleBefore, project.style); styleBefore = project.style; });
  $("project-list").onchange = guard((event) => event.target.value === NEW_SONG ? newSong() : event.target.value && openProject(event.target.value));
  $("save-project").onclick = guard(async () => {
    ensureName();
    await saveProject("saved");
  });
  $("duplicate-project").onclick = guard(async () => {
    if (!project.name) throw new Error("Save the song first.");
    await duplicateSong(project.name);
  });
  $("open-catalog").onclick = guard(async () => {
    managedAlbum = project.album;
    $("catalog").showModal();
    await listProjects();
  });
  document.querySelectorAll("#catalog .tabs button").forEach((button) => {
    button.onclick = () => {
      document.querySelectorAll("#catalog .tabs button").forEach((other) => other.classList.toggle("active", other === button));
      for (const tab of ["manage", "sources", "references"]) $(`tab-${tab}`).classList.toggle("hidden", button.dataset.tab !== tab);
      if (button.dataset.tab === "sources") guard(drawSources)();
    };
  });
  for (const id of ["catalog-search", "catalog-filter", "catalog-sort"]) $(id).addEventListener(id === "catalog-search" ? "input" : "change", guard(drawCatalog));
  $("catalog-bulk-clear").onclick = guard(async () => { catalogSelection.clear(); await drawCatalog(); });
  $("catalog-bulk-move").onclick = guard(() => catalogBulk((name) => moveSong(name, $("catalog-bulk-album").value),
    (count) => `${count} songs moved to «${$("catalog-bulk-album").value}».`));
  $("catalog-bulk-archive").onclick = guard(() => catalogBulk((name) => updateProject(name, (stored) => { stored.archived = true; }),
    (count) => `${count} songs archived (Show › Archived lists them).`));
  $("catalog-bulk-unarchive").onclick = guard(() => catalogBulk((name) => updateProject(name, (stored) => { stored.archived = false; }),
    (count) => `${count} songs unarchived.`));
  $("catalog").addEventListener("close", () => catalogPreview.pause());
  $("import-sources").onclick = guard(async () => {
    const files = [...$("source-files").files];
    if (!files.length) throw new Error("Choose one or more audio files.");
    const names = await addAudios(files, { kind: "source" });
    $("source-files").value = "";
    if ($("source-analyze").checked) for (const name of names) await analyze(name);
    await drawSources();
    status(`${names.length} audio files in the library${$("source-analyze").checked ? ", analysis queued" : ""}.`, $("source-analyze").checked ? 0 : 1);
  });
  $("source-pick").onchange = guard(async (event) => {
    if (event.target.value) await useSource(event.target.value, { audio: true });
  });
  $("source-bring").onclick = guard(async () => {
    readForm();
    if (!project.sourceName) throw new Error("Choose a library audio first.");
    await useSource(project.sourceName, { abc: true, lyrics: true, style: true });
  });
  $("album-rename").onclick = guard(() => renameAlbum($("album-name").value));
  $("album-delete").onclick = guard(deleteAlbum);
  $("album-notes").onchange = guard((event) => saveAlbumNotes(event.target.value));
  $("close-catalog").onclick = () => $("catalog").close();
  $("close-history").onclick = () => $("history").close();
  $("history-branch").onchange = guard(async (event) => { historyBranch = event.target.value; await drawHistory(); });
  $("history-archived").onchange = guard(drawHistory);
  $("history-new-branch").onclick = guard(async () => {
    if (historyBranch !== project.history.current) await switchBranch(historyBranch);
    const name = await askText("New branch", `Branch name (it starts where «${historyBranch}» is):`, "");
    if (name === null) return;
    await newBranch(name);
    historyBranch = project.history.current;
    writeForm();
    await drawHistory();
  });
  $("history-rename-branch").onclick = guard(async () => {
    const name = await askText("Rename branch", `New name for «${historyBranch}»:`, historyBranch);
    if (name === null) return;
    renameBranch(historyBranch, name);
    historyBranch = name;
    await saveProject();
    writeForm();
    await drawHistory();
  });
  $("history-archive-branch").onclick = guard(async () => {
    const branch = project.history.branches[historyBranch];
    branch.archived = !branch.archived;
    await saveProject();
    writeForm();
    await drawHistory();
  });
  // Closing or reloading the page with changes not yet in the history asks first.
  window.addEventListener("beforeunload", (event) => { if (songChanged()) event.preventDefault(); });
  $("album-list").onchange = guard((event) => browseTo(event.target.value));
  $("open-styles").onclick = guard(async () => {
    loras = (await api("/hz3/studio/loras")).loras;
    drawStyles();
    $("styles").showModal();
  });
  $("close-freeze").onclick = () => $("freeze").close();
  $("freeze-train").onclick = guard(freezeVoice);
  $("close-styles").onclick = () => $("styles").close();
  $("add-song-voice").onclick = guard(addSongVoice);
  $("add-style").onclick = guard(async () => {
    album.styles.push({ id: crypto.randomUUID().slice(0, 8), name: `Singer ${album.styles.length + 1}`, kind: "singer", text: "" });
    await saveAlbum();
    drawStyles();
  });
  $("base-style").onchange = () => { readForm(); draw(); };
  $("toggle-assistant").onclick = () => { $("assistant").classList.toggle("hidden"); drawChat(); };
  const send = guard(async () => {
    const text = $("chat-input").value;
    $("chat-input").value = "";
    await sendChat(text);
  });
  $("chat-send").onclick = send;
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.ctrlKey) { event.preventDefault(); send(); }
  });
  $("apply-style-instructions").onclick = guard(async () => {
    readForm();
    if (!project.styleInstructions.trim()) throw new Error("Write the style directions first.");
    $("assistant").classList.remove("hidden");
    await sendChat(`Aplica estas indicaciones de estilo a la canción: ${project.styleInstructions.trim()}`);
  });
  $("import-references").onclick = guard(async () => {
    const files = [...$("reference-files").files];
    if (!files.length) throw new Error("Choose one or more reference audio files.");
    if (!$("reference-license").value) throw new Error("Choose the license of the recordings.");
    await importReferences(files, { name: $("reference-license").value, work: $("reference-work").value.trim(), url: $("reference-url").value.trim() });
    $("reference-files").value = "";
  });
  $("toggle-song").onclick = () => document.querySelector(".song").classList.toggle("hidden");
  // On a phone the side panel is a drawer: it starts closed so the timeline is what you see.
  if (matchMedia("(max-width: 760px)").matches) document.querySelector(".song").classList.add("hidden");
  $("source-audio").onchange = guard(async (event) => {
    const [file] = event.target.files;
    event.target.value = "";
    if (file) await uploadSource(file);
  });
  $("analyze").onclick = guard(() => analyze());
  $("compose").onclick = guard(() => compose());
  $("section-recompose").onclick = guard(() => compose(selected));
  $("section-free").onclick = guard(() => setSectionFree(selected));
  $("section-record").onclick = guard(() => recordSectionMelody(selected));
  $("section-restore").onclick = guard(() => restoreSectionOriginal(selected));
  $("section-instrumental").onclick = guard(() => sectionToInstrumental(selected));
  $("compose-dice").onclick = () => { $("compose-seed").value = Math.floor(Math.random() * 2 ** 31); readForm(); };
  $("branch-list").onchange = guard(async (event) => {
    const name = event.target.value;
    if (name === NEW_BRANCH) {
      $("branch-list").value = project.history.current;
      const typed = await askText("New branch", "Branch name (it starts at the current state):", "");
      if (typed !== null) await newBranch(typed);
    } else if (name && name !== project.history.current) await switchBranch(name);
    writeForm();
  });
  $("open-history").onclick = guard(openHistory);
  $("close-inspector").onclick = () => { selected = null; draw(); };
  $("close-voice-inspector").onclick = () => { selectedVoice = null; draw(); };
  $("render-song").onclick = guard(() => render(null));
  $("abc-reimagine").onclick = guard(reimagineAbc);
  $("render-section").onclick = guard(() => render([selected]));
  $("export-mix").onclick = guard(exportMix);
  $("arrange").onclick = guard(arrange);
  $("play").onclick = guard(play);
  $("stop").onclick = () => { pause(); pausedAt = 0; updatePlayhead(); };
  $("meter-clip").onclick = () => { clipped = false; peakHold = [0, 0]; drawMeter(); };
  $("zoom").oninput = (event) => { pxPerSecond = Number(event.target.value); draw(); };
  $("import-json").onchange = guard(async (event) => {
    const data = JSON.parse(await event.target.files[0].text());
    for (const key of ["style", "lyrics", "abc"]) if (typeof data[key] === "string") $(key).value = data[key];
    event.target.value = "";
    await refreshSections();
  });
  $("import-package").onchange = guard(async (event) => {
    const form = new FormData();
    form.append("file", event.target.files[0]);
    event.target.value = "";
    const { name } = await api("/hz3/studio/package", { method: "POST", body: form });
    $("catalog").close();
    await listProjects();
    await openProject(name);
  });
  $("section-name").addEventListener("change", guard((event) => renameSection(event.target.value)));
  $("section-split").onclick = guard(splitSection);
  $("song-end").onclick = guard(endSongHere);
  $("section-merge").onclick = guard(mergeSection);
  $("section-earlier").onclick = guard(() => moveSectionStart(-1));
  $("section-later").onclick = guard(() => moveSectionStart(1));
  $("section-lyrics").addEventListener("input", guard((event) => {
    setSectionLyrics(sections.findIndex((section) => section.name === selected), event.target.value);
    scheduleSections();
  }));
  $("section-context").addEventListener("change", guard(async (event) => {
    const parts = event.target.value.split("·").map((part) => part.trim());
    if (!event.target.value.trim()) await setSectionContext(selected, "");
    else if (parts.length !== 3 || !/^((toma|take)\s*)?\d+$/i.test(parts[1])) throw new Error("Write «Project · take N · Section».");
    else await setSectionContext(selected, parts[0], parts[1].replace(/\D/g, ""), parts[2]);
    draw();
  }));
  $("section-negative").addEventListener("change", (event) => {
    const text = event.target.value.trim();
    if (text) project.sectionNegatives[selected] = text;
    else delete project.sectionNegatives[selected];
    draw();
  });
  $("section-style").addEventListener("input", (event) => {
    styleDraft = { section: selected, text: event.target.value };
    $("section-style-save").disabled = $("section-style-undo").disabled = $("section-style-original").disabled = false;
  });
  $("section-style-save").onclick = saveSectionStyle;
  $("section-style-undo").onclick = () => { styleDraft = null; draw(); };
  $("section-style-original").onclick = () => { delete project.sectionStyles[selected]; styleDraft = null; draw(); };
  $("section-score-play").onclick = guard(async () => {
    if (scorePlayback) return stopScore();
    if (!score) throw new Error("There is no score to play.");
    await context().resume();
    playScore(...sectionTicks(sections.findIndex((section) => section.name === selected)), $("section-score-play"));
  });
  $("score-play").onclick = guard(async () => {
    if (scorePlayback) return stopScore();
    if (!score) throw new Error("There is no score to play.");
    await context().resume();
    playScore(Math.min(score.total_ticks, Math.round(position() * score.bpm / 60 * TICKS_PER_QUARTER)), score.total_ticks, $("score-play"));
  });
  for (const id of ["song-lora", "song-lora-strength"]) $(id).addEventListener("change", () => { readForm(); draw(); });
  $("section-styles-from-cues").onclick = guard(stylesFromCues);
  $("apply-classical").onclick = guard(applyClassical);
  $("section-classical").addEventListener("change", (event) => { project.classicalPlan[selected] = event.target.value; });
  $("section-singer").addEventListener("change", guard(async (event) => { await setSectionSinger(selected, event.target.value || null); draw(); }));
  $("section-singer-range").onclick = guard(async () => {
    const id = project.sectionSingers[selected];
    if (!id) throw new Error("Choose the section's singer first.");
    const octaves = await setSectionSinger(selected, id, true);
    draw();
    status(octaves ? `«${selected}»: melody moved ${octaves > 0 ? "+" : ""}${octaves} octave(s); the earlier score is in the ABC versions.` : `«${selected}» is already in the singer's range.`, 1);
  });
  $("voice-contrast").onclick = guard(voiceContrast);
  $("voice-contrast-clear").onclick = guard(clearVoiceContrast);
  $("section-context-concept").onclick = guard(async () => { await contextFromConcept(selected); draw(); });
  $("section-keep").addEventListener("change", (event) => { keepSections([selected], event.target.checked); draw(); });
  $("keep-all").onclick = () => { keepSections(sections.filter((section) => takeById(project.comp[section.name])).map((section) => section.name), true); draw(); };
  $("keep-none").onclick = () => { keepSections(sections.map((section) => section.name), false); draw(); };
  $("album-concept").onchange = guard(async (event) => {
    const info = await albumData(managedAlbum);
    info.concept = event.target.value || null;
    await postJson("/hz3/studio/album", info);
    if (album.name === managedAlbum) album.concept = info.concept;
    status(info.concept ? `«${info.concept}» is the concept song of «${managedAlbum}».` : `«${managedAlbum}» has no concept song.`, 1);
  });
  $("voice-singer").addEventListener("change", (event) => {
    const voice = selectedVoiceEntry();
    voice.singer = event.target.value || null;
    // A voice still under its default name takes its singer's name.
    if (voice.singer && /^(Voz|Voice) \d+$/.test(voice.name)) voice.name = catalogStyle(voice.singer).name.slice(0, 40);
    draw();
  });
  $("add-voice").onclick = addVoice;
  wireTrackInspector();
  for (const [id, key] of [["voice-name", "name"], ["voice-source", "source"], ["voice-octave", "octave"], ["voice-role", "role"], ["voice-style", "style"], ["voice-seed", "seed"]]) {
    $(id).addEventListener("change", (event) => {
      const voice = selectedVoiceEntry();
      voice[key] = ["seed", "octave"].includes(key) ? Number(event.target.value) || 0 : event.target.value;
      const entry = voice.file && buffers.get(fileUrl(voice.file));
      if (key === "role" && entry) levelVoice(voice, entry);
      draw();
      restartIfPlaying();
    });
  }
  $("voice-offset").addEventListener("change", (event) => {
    selectedVoiceEntry().offset = Number(event.target.value) / 1000;
    draw();
    restartIfPlaying();
  });
  $("voice-render").onclick = guard(() => renderVoice(selectedVoiceEntry()));
  $("voice-align").onclick = guard(() => alignVoice(selectedVoiceEntry()));
  $("voice-delete").onclick = () => {
    project.voices = project.voices.filter((voice) => voice.id !== selectedVoice);
    selectedVoice = null;
    draw();
    restartIfPlaying();
  };
  $("section-seed").addEventListener("change", (event) => {
    project.sectionSeeds[selected] = Number(event.target.value);
    draw();
  });
  $("section-dice").onclick = () => {
    project.sectionSeeds[selected] = Math.floor(Math.random() * 2 ** 31);
    draw();
  };
  document.addEventListener("keydown", (event) => {
    if (event.code === "Space" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      event.preventDefault();
      guard(play)();
    }
  });
}

init().catch((error) => status(error.message, null, true));
