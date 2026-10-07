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
  { id: "vocals", name: "Voz principal", gain: 1 },
  { id: "instrumental", name: "Instrumental", gain: 1 },
  { id: "tenor", name: "Tenor", gain: 0.4, harmony: true },
  { id: "baritone", name: "Barítono", gain: 0.4, harmony: true },
  { id: "low", name: "Grave", gain: 0.3, harmony: true },
  { id: "bass", name: "Bajo", gain: 0.35, harmony: true },
  { id: "countertenor", name: "Contratenor", gain: 0.22, harmony: true },
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
const RENDER_LABELS = { 1: "Cargando modelo", 2: "Tokens YuE2", 4: "KSampler", 5: "Decodificando audio", 7: "Separando voz", 12: "Armonías" };
const ANALYSIS_LABELS = { 3: "SheetSage2 (ABC)", 4: "Separando voz", 5: "Whisper (letra)", 6: "MixMash (Ollama)" };
const COMPOSE_LABELS = { 1: "Cargando modelo", 2: "YuE2 compone el ABC" };
const STYLE_KINDS = { singer: "Cantante", group: "Grupo", genre: "Género" };
// Median *written* Vocal note (MIDI) a singer's sections are moved to. YuE2 picks the voice from the melody's register far
// more than from the style text, and sings about an octave below the written ABC: in our test a verse written around 74
// was sung by a soprano near 62, and the same verse written around 62 by a baritone near 50. mid is between, untested.
const REGISTERS = { high: { label: "Aguda (femenina)", center: 77 }, mid: { label: "Media (tenor / mezzo)", center: 70 }, low: { label: "Grave (masculina)", center: 62 } };
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
    style: "", baseStyle: null, lora: null, lyrics: "", abc: "", abcVersions: [], seed: 60,
    compose: { seed: 60, temperature: 0.7, keep: true },
    ckpt: "yue2_3b_int8_convrot.safetensors",
    sampling: { temperature: 0.9, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, cfg_scale: 2.0 },
    harmonize: true, harmonyVoices: [...VOICES], songVersions: [], sectionSeeds: {}, sectionStyles: {}, sectionSingers: {}, takes: [], comp: {}, trackOn: {}, trackSpans: {}, mixer: {}, voices: [], instruments: [], fx: {}, sectionOriginals: {},
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
  project.baseStyle = $("base-style").value || null;
  const lora = loras.find((item) => item.name === $("song-lora").value);
  project.lora = lora ? { name: lora.name, trigger: lora.trigger, strength: Number($("song-lora-strength").value) || 1 } : null;
  project.compose = { seed: Number($("compose-seed").value) || 0, temperature: Number($("compose-temperature").value), keep: $("compose-keep").checked };
  project.seed = Number($("seed").value) || 0;
  project.harmonize = $("harmonize").checked;
  for (const key of SAMPLING) project.sampling[key] = Number($(key).value);
  project.arranger = { model: $("arranger-model").value.trim(), instructions: $("arranger-instructions").value };
}

function writeForm() {
  for (const key of ["style", "lyrics", "abc"]) $(key).value = project[key];
  $("style-instructions").value = project.styleInstructions;
  $("base-style").innerHTML = styleOptions(["group", "genre"], "— ninguno", project.baseStyle);
  $("song-lora").innerHTML = `<option value="">— sin LoRA (solo el estilo)</option>`
    + loras.map((lora) => `<option value="${escapeHtml(lora.name)}"${lora.name === project.lora?.name ? " selected" : ""}>${escapeHtml(loraFile(lora.name))}</option>`).join("");
  $("song-lora-strength").value = project.lora?.strength ?? 1;
  $("compose-seed").value = project.compose.seed;
  $("compose-temperature").value = project.compose.temperature;
  $("compose-keep").checked = project.compose.keep;
  $("song-versions").innerHTML = `<option value="">${project.songVersions.length ? `— ${project.songVersions.length} versiones de la canción —` : "— sin versiones de la canción —"}</option>`
    + project.songVersions.map((version, index) => `<option value="${index}">${new Date(version.at).toLocaleString()} · ${escapeHtml(version.label)}</option>`).reverse().join("");
  $("abc-versions").innerHTML = `<option value="">${project.abcVersions.length ? `— ${project.abcVersions.length} versiones anteriores —` : "— sin versiones anteriores —"}</option>`
    + project.abcVersions.map((version, index) => `<option value="${index}">${new Date(version.at).toLocaleString()} · ${escapeHtml(version.label)}</option>`).reverse().join("");
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
  status(`${sections.length} secciones · ${fmt(duration())}`);
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

// Renames are [from, to] pairs; several sections may share an old name (YuE2 repeats "% chorus"), so each new name
// gets a copy of what the old name held.
function renameKeys(renames) {
  const maps = [project.sectionOriginals, project.comp, project.sectionSeeds, project.sectionStyles, project.sectionSingers, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
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

function isEdited(section, index) {
  const take = takeById(project.comp[section.name]);
  if (!take) return false;
  return (take.sectionLyrics[section.name] ?? "").trim() !== section.lyrics.trim()
    || take.sectionSeeds[section.name] !== sectionSeed(section, index)
    || (take.sectionStyles?.[section.name] ?? "") !== sectionStyle(section)
    || Boolean(take.sectionAbc && take.sectionAbc[section.name] !== section.abc);
}

// MixMash-style cues: global lines, then "[Section] description" lines.
// A section's cue is found by its ABC name ("verse 2") or by the lyric tag it is paired with ("Verse 2"),
// since ABC sections are often named "verse" while the cues follow the lyrics.
function sectionCueStyle(section) {
  const lines = project.style.split("\n").filter((line) => line.trim());
  const global = lines.filter((line) => !line.trim().startsWith("[")).join("\n").trim();
  const header = section.lyrics_index === null ? "" : lyricBlocks(project.lyrics)[section.lyrics_index]?.name.split(/\s+[–—-]\s+|\s*[:|(]/)[0].trim() ?? "";
  const cues = lines.map((line) => line.match(/^\s*\[([^\]]+)\]\s*(.+)$/)).filter(Boolean);
  const cue = [section.name, header].filter(Boolean).map((name) => cues.find((match) => match[1].trim().toLowerCase() === name.toLowerCase())).find(Boolean);
  return { global, cue: cue?.[2].trim() };
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
  status(count ? `${count} secciones con estilo propio.` : "No hay líneas «[Sección] …» que coincidan con las secciones.", 1, !count);
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

// The lead's singers are declared up front, as YuE2's own duets do; who sings each section is in its lyric tag.
function singersStyle() {
  const singers = [...new Set(sections.map((section) => project.sectionSingers[section.name]).filter(Boolean))].map(catalogStyle).filter(Boolean);
  if (singers.length < 2) return singers[0]?.text ?? "";
  return `${singers.length === 2 ? "duet" : "several lead singers"}, ${singers.map((singer) => singer.text).join(" and ")}`;
}

function globalStyle() {
  return [catalogStyle(project.baseStyle)?.text, project.style, singersStyle()].filter((text) => text?.trim()).join("\n");
}

// A section with its own style text sings the base style, that text and the singers; "" means the global style.
function sectionStyle(section) {
  const own = project.sectionStyles[section.name];
  if (!own) return "";
  return [catalogStyle(project.baseStyle)?.text, own, singersStyle()].filter((text) => text?.trim()).join("\n");
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
  const style = { id: crypto.randomUUID().slice(0, 8), name: (named || `Voz de ${project.name || "la canción"}`).slice(0, 60), kind: "singer", text };
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
  if (!style) throw new Error("El estilo de la canción no describe la voz (p. ej. «Spanish female lead vocal, warm») ni tiene LoRA de voz.");
  album.styles.push(style);
  await saveAlbum();
  remember(`Cantante «${style.name}» creado con la voz de la canción`);
  drawStyles();
  writeForm();
  status(`«${style.name}» agregado al catálogo${project.lora ? " con el LoRA de la canción" : ""}. Revisa su texto y nombre.`, 1);
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
  const labels = { [`antes de liberar la melodía de «${name}»`]: "free", [`antes de volver instrumental «${name}»`]: "instrumental" };
  const version = project.abcVersions.findLast((item) => item.label in labels);
  if (!version) return null;
  const lines = version.abc.split("\n");
  return { kind: labels[version.label], lines: sectionMusicLines(version.abc, name).map((at) => lines[at]), lyrics: null };
}

async function restoreSectionOriginal(name) {
  readForm();
  const original = sectionOriginal(name);
  const at = sectionMusicLines(project.abc, name);
  if (at.length !== original.lines.length) throw new Error(`«${name}» cambió de compases desde entonces: recupérala con las versiones del ABC.`);
  const lines = project.abc.split("\n");
  at.forEach((line, index) => { lines[line] = original.lines[index]; });
  const abc = lines.join("\n");
  await postJson("/hz3/studio/sections", { abc, lyrics: project.lyrics });
  setAbc(abc, `antes de restaurar la melodía escrita de «${name}»`);
  await refreshSections();
  const index = sections.findIndex((section) => section.name === name);
  if (original.kind === "instrumental" && original.lyrics !== null && index >= 0) {
    setSectionLyrics(index, original.lyrics);
    await refreshSections();
  }
  project.sectionOriginals[name] = null;
  remember(`«${name}» vuelve a su melodía escrita`);
  status(`«${name}» recuperó su melodía escrita${original.kind === "instrumental" && original.lyrics !== null ? " y su letra" : ""}.`, 1);
  draw();
}

async function setSectionFree(name) {
  readForm();
  keepSectionOriginal(name, "free");
  const abc = rewriteSectionVocal(project.abc, name, (line, units) => line.slice(0, -1).split("|").map((bar) => freeBar(bar, units)).join("|") + "|");
  if (abc === project.abc) {
    if (project.sectionOriginals[name]?.kind === "free") delete project.sectionOriginals[name];
    throw new Error(`«${name}» ya no tiene melodía escrita.`);
  }
  await postJson("/hz3/studio/sections", { abc, lyrics: project.lyrics });
  setAbc(abc, `antes de liberar la melodía de «${name}»`);
  await refreshSections();
}

// Lead singer of a section: its melody moves to the singer's register and its lyric tag says who sings.
async function setSectionSinger(name, id) {
  const index = sections.findIndex((section) => section.name === name);
  if (index < 0) throw new Error(`no hay sección «${name}»`);
  readForm();
  const singer = catalogStyle(id);
  let lyrics = project.lyrics;
  const block = sections[index].lyrics_index === null ? null : lyricBlocks(lyrics)[sections[index].lyrics_index];
  if (block) {
    const base = block.name.split(/\s+[–—-]\s+|\s*[:|(]/)[0].trim();
    const tag = singer ? `[${base} – ${singerCue(singer)}]` : `[${base}]`;
    lyrics = lyrics.slice(0, block.start) + tag + lyrics.slice(block.bodyStart);
  }
  const notes = singer && score ? (() => {
    const from = score.bars[score.sections[index].start_bar].start;
    const to = index + 1 < score.sections.length ? score.bars[score.sections[index + 1].start_bar].start : score.total_ticks;
    return (score.tracks.Vocal ?? []).filter((note) => note.start >= from && note.start < to).map((note) => note.pitch).sort((a, b) => a - b);
  })() : [];
  const octaves = notes.length ? Math.round((REGISTERS[singerRegister(singer)].center - notes[Math.floor(notes.length / 2)]) / 12) : 0;
  const abc = octaves ? rewriteSectionVocal(project.abc, name, (line) => shiftOctaves(line, octaves)) : project.abc;
  // The parser checks the result before it replaces the song's score and lyrics.
  await postJson("/hz3/studio/sections", { abc, lyrics });
  if (octaves) keepAbcVersion(project, `antes de llevar «${name}» al registro de ${singer.name}`);
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
    throw new Error(`${distinct.map((use) => use.owner).join(" y ")} usan LoRAs distintos: un render lleva un solo LoRA; canta a uno como voz extra.`);
  }
  if (!distinct.length) return prompt;
  const { name, trigger, strength } = distinct[0].lora;
  prompt[21] = { class_type: "LoraLoader", inputs: { model: ["1", 0], clip: ["1", 1], lora_name: name, strength_model: strength, strength_clip: strength } };
  prompt[2].inputs.clip = ["21", 1];
  prompt[4].inputs.model = ["21", 0];
  if (trigger) {
    prompt[2].inputs.style = `${trigger}, ${prompt[2].inputs.style}`;
    prompt[2].inputs.section_styles = prompt[2].inputs.section_styles.split("\n").map((line) => line.replace(/^([^:]+:\s*)/, `$1${trigger}, `)).join("\n");
  }
  return prompt;
}

// ---------- freezing a voice: a LoRA trained on audio already generated for a singer

let freezing = null;

function freezeClips() {
  const clips = [];
  for (const take of project.takes) {
    if (take.files.mix) clips.push({ label: `Take ${take.id} · mezcla`, file: take.files.mix });
    if (take.files.vocals) clips.push({ label: `Take ${take.id} · solo voz`, file: take.files.vocals });
  }
  for (const voice of project.voices) {
    if (voice.mixFile) clips.push({ label: `${voice.name} · mezcla`, file: voice.mixFile });
    if (voice.file) clips.push({ label: `${voice.name} · solo voz`, file: voice.file });
  }
  return clips;
}

function openFreeze(singer) {
  freezing = singer;
  $("freeze-title").textContent = `Congelar la voz de ${singer.name}`;
  const clips = freezeClips();
  $("freeze-clips").innerHTML = clips.length ? "" : `<p class="note">Esta canción no tiene audios generados todavía: genera takes o voces con ${escapeHtml(singer.name)} y vuelve aquí.</p>`;
  for (const clip of clips) {
    const row = document.createElement("label");
    row.className = "switch";
    row.innerHTML = `<input type="checkbox"> <button type="button" title="Escuchar">▶</button> ${escapeHtml(clip.label)}`;
    row.querySelector("input").clip = clip;
    row.querySelector("button").onclick = (event) => { event.preventDefault(); new Audio(fileUrl(clip.file)).play(); };
    $("freeze-clips").append(row);
  }
  $("freeze").showModal();
}

async function freezeVoice() {
  const singer = freezing;
  const files = [...$("freeze-clips").querySelectorAll("input:checked")].map((box) => box.clip.file);
  if (!files.length) throw new Error("Elige al menos un audio con la voz que quieres congelar.");
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
  await queue(prompt, { 3: `Entrenando el LoRA de ${singer.name}` }, async () => {
    loras = (await api("/hz3/studio/loras")).loras;
    const lora = loras.find((item) => item.name.replace(/^.*[\\/]/, "") === `${name}.safetensors`);
    if (!lora) throw new Error("El entrenamiento terminó sin LoRA.");
    singer.lora = { name: lora.name, trigger: lora.trigger || trigger, strength: 1 };
    await saveAlbum();
    remember(`Voz de «${singer.name}» congelada en ${lora.name} (${files.length} audios)`);
    if ($("styles").open) drawStyles();
    status(`LoRA ${lora.name} listo y asignado a ${singer.name} (disparador ${singer.lora.trigger}).`, 1);
  });
  status(`Entrenamiento del LoRA de ${singer.name} en cola (${files.length} audios).`, 0);
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
  $("album-list").innerHTML = albums.map((name) => `<option${name === browseAlbum ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")
    + `<option value="${NEW_ALBUM}">+ Nuevo álbum…</option>`;
}

// The top bar's album only chooses which songs are listed next to it; moving a song between albums is done in the catalog.
async function browseTo(name) {
  if (name === NEW_ALBUM) {
    name = (prompt("Nombre del nuevo álbum") ?? "").trim();
    if (!name) { $("album-list").value = browseAlbum; return; }
    const { albums } = await api("/hz3/studio/albums");
    if (!albums.includes(name)) await postJson("/hz3/studio/album", { name, notes: "", styles: [], chat: [], log: [] });
  }
  browseAlbum = name;
  await listAlbums();
  await listProjects();
  status(`Álbum «${name}»: elige una canción o «+ Canción nueva».`, 1);
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
  status(project.name ? `«${project.name}» está ahora en el álbum «${name}».` : `Álbum «${name}».`, 1);
}

function remember(text) {
  album.log.push({ at: Date.now(), song: project.name, text });
  album.log = album.log.slice(-300);
  saveAlbum().catch((error) => status(error.message, null, true));
}

function drawStyles() {
  $("styles-album").textContent = `· ${album.name}`;
  const rows = $("style-rows");
  rows.innerHTML = album.styles.length ? "" : `<tr><td colspan="6" class="style">Sin estilos todavía: agrega cantantes, grupos y géneros, o pídeselos al asistente.</td></tr>`;
  for (const style of album.styles) {
    const row = document.createElement("tr");
    row.innerHTML = `<td><input maxlength="60"></td>
      <td><select class="kind">${Object.entries(STYLE_KINDS).map(([kind, label]) => `<option value="${kind}">${label}</option>`).join("")}</select></td>
      <td><select class="register" title="Registro al que se lleva la melodía de las secciones que canta">${Object.entries(REGISTERS).map(([key, { label }]) => `<option value="${key}">${label}</option>`).join("")}</select></td>
      <td><textarea rows="2" placeholder="p. ej. Spanish female soprano lead vocal, warm and breathy"></textarea></td>
      <td class="lora"><select class="lora-name" title="LoRA de voz que se aplica al renderizar con este cantante">
          <option value="">— sin LoRA</option>${loras.map((lora) => `<option value="${escapeHtml(lora.name)}">${escapeHtml(lora.name.replace(/^.*[\\/]/, ""))}</option>`).join("")}</select>
        <input class="lora-strength" type="number" min="0" max="2" step="0.05" title="Fuerza del LoRA">
        <button class="freeze" title="Entrena un LoRA con audios ya generados de este cantante">Congelar voz…</button></td>
      <td><button class="delete">Eliminar</button></td>`;
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
  return project.lora ? `Cantante de la canción · ${loraFile(project.lora.name)}` : "Cantante de la canción (su estilo)";
}

function singerSelect(value, empty, change) {
  const select = document.createElement("select");
  select.className = "singer";
  select.title = "Quién canta esta sección";
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
  $("assistant-album").textContent = `álbum «${album.name}»`;
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
      seed: project.seed, sampling: project.sampling, compose: project.compose, abc_versions: project.abcVersions.length,
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
    if (!found) throw new Error(`no hay sección «${name}»`);
    return found;
  };
  const voiceNamed = (name) => {
    const found = project.voices.find((voice) => voice.name.toLowerCase() === String(name ?? "").toLowerCase());
    if (!found) throw new Error(`no hay voz «${name}»`);
    return found;
  };
  const styleNamed = (name, kinds) => {
    if (!name) return null;
    const found = album.styles.find((style) => kinds.includes(style.kind) && style.name.toLowerCase() === String(name).toLowerCase());
    if (!found) throw new Error(`no hay «${name}» en el catálogo`);
    return found.id;
  };
  switch (action.type) {
    case "set_style":
      project.style = String(action.text ?? "");
      writeForm();
      return "Estilo general actualizado";
    case "set_section_style": {
      const section = sectionNamed(action.section);
      const text = String(action.text ?? "").trim();
      if (text) project.sectionStyles[section.name] = text;
      else delete project.sectionStyles[section.name];
      draw();
      return `Estilo de «${section.name}» ${text ? "actualizado" : "= general"}`;
    }
    case "set_section_lyrics": {
      const section = sectionNamed(action.section);
      setSectionLyrics(sections.indexOf(section), String(action.text ?? ""));
      await refreshSections();
      return `Letra de «${section.name}» actualizada`;
    }
    case "set_section_seed": {
      const section = sectionNamed(action.section);
      project.sectionSeeds[section.name] = Number(action.seed) || 0;
      draw();
      return `Semilla de «${section.name}»: ${project.sectionSeeds[section.name]}`;
    }
    case "free_melody": {
      const section = sectionNamed(action.section);
      await setSectionFree(section.name);
      return `«${section.name}» con melodía libre (la anterior quedó en las versiones del ABC)`;
    }
    case "set_base_style":
      project.baseStyle = styleNamed(action.style, ["group", "genre"]);
      writeForm();
      draw();
      return `Estilo base: ${catalogStyle(project.baseStyle)?.name ?? "ninguno"}`;
    case "set_singer": {
      const id = styleNamed(action.style, ["singer"]);
      if (action.track && action.track !== "lead") {
        const voice = voiceNamed(action.track);
        if (action.section) throw new Error("una voz extra es una generación aparte: tiene un solo cantante para toda la canción");
        voice.singer = id;
        draw();
        return `${catalogStyle(id)?.name ?? "Sin cantante"} → ${voice.name}`;
      }
      const moved = [];
      for (const section of action.section ? [sectionNamed(action.section)] : [...sections]) {
        const octaves = await setSectionSinger(section.name, id);
        if (octaves) moved.push(`${section.name} ${octaves > 0 ? "+" : ""}${octaves} oct.`);
      }
      draw();
      return `${catalogStyle(id)?.name ?? "Sin cantante"} → voz principal · ${action.section || "todas las secciones"}${moved.length ? ` · melodía movida: ${moved.join(", ")}` : ""}`;
    }
    case "save_style": {
      const name = String(action.name ?? "").trim().slice(0, 60);
      if (!name) throw new Error("estilo sin nombre");
      const kind = STYLE_KINDS[action.kind] ? action.kind : "singer";
      let style = album.styles.find((item) => item.name.toLowerCase() === name.toLowerCase());
      if (!style) album.styles.push(style = { id: crypto.randomUUID().slice(0, 8) });
      Object.assign(style, { name, kind, text: String(action.text ?? "").trim() });
      if (kind === "singer" && REGISTERS[action.register]) style.register = action.register;
      await saveAlbum();
      writeForm();
      draw();
      return `Estilo «${name}» (${STYLE_KINDS[kind]}) guardado en el catálogo`;
    }
    case "delete_style": {
      const id = styleNamed(action.name, Object.keys(STYLE_KINDS));
      album.styles = album.styles.filter((style) => style.id !== id);
      await saveAlbum();
      writeForm();
      draw();
      return `Estilo «${action.name}» eliminado del catálogo`;
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
      return `Voz «${voice.name}» agregada (sin generar)`;
    }
    case "set_track": {
      const section = sectionNamed(action.section);
      const voice = project.voices.find((item) => item.name.toLowerCase() === String(action.track ?? "").toLowerCase());
      if (voice) voice.on[section.name] = Boolean(action.on);
      else if (TRACKS.some((track) => track.id === action.track && !track.source)) (project.trackOn[action.track] ??= {})[section.name] = Boolean(action.on);
      else throw new Error(`no hay pista «${action.track}»`);
      draw();
      restartIfPlaying();
      return `${action.track} ${action.on ? "encendida" : "apagada"} en «${section.name}»`;
    }
    case "select_take": {
      const section = sectionNamed(action.section);
      const take = takeById(Number(action.take));
      if (!take) throw new Error(`no hay take ${action.take}`);
      project.comp[section.name] = take.id;
      draw();
      restartIfPlaying();
      return `«${section.name}» suena desde T${take.id}`;
    }
    case "arrange_harmonies":
      if (action.instructions) {
        project.arranger.instructions = String(action.instructions);
        writeForm();
      }
      await arrange();
      return "Armonías arregladas";
    case "set_sampling": {
      for (const key of SAMPLING) if (Number.isFinite(Number(action[key])) && action[key] !== undefined) project.sampling[key] = Number(action[key]);
      if (Number.isFinite(Number(action.seed)) && action.seed !== undefined) project.seed = Number(action.seed);
      writeForm();
      draw();
      return `Muestreo: ${SAMPLING.map((key) => `${key} ${project.sampling[key]}`).join(", ")} · semilla ${project.seed}`;
    }
    case "get_abc":
      return action.section ? sectionNamed(action.section).abc : project.abc;
    case "set_abc":
      // The section parser rejects a broken score before it replaces the current one.
      await postJson("/hz3/studio/sections", { abc: String(action.abc ?? ""), lyrics: project.lyrics });
      setAbc(String(action.abc ?? ""), "antes de que el asistente editara el ABC");
      await refreshSections();
      return `ABC reemplazado · ${sections.length} secciones`;
    case "set_compose_settings":
      if (Number.isFinite(Number(action.seed)) && action.seed !== undefined) project.compose.seed = Number(action.seed);
      if (Number.isFinite(Number(action.temperature)) && action.temperature !== undefined) project.compose.temperature = Number(action.temperature);
      if (typeof action.keep_key_meter_tempo === "boolean") project.compose.keep = action.keep_key_meter_tempo;
      writeForm();
      return `Composición: semilla ${project.compose.seed}, temperatura ${project.compose.temperature}, ${project.compose.keep ? "conserva" : "no conserva"} tonalidad, compás y tempo`;
    case "compose_abc": {
      const from = action.from_section ? sectionNamed(action.from_section).name : null;
      await compose(from);
      return `Composición del ABC${from ? ` desde «${from}»` : ""} en cola (semilla ${project.compose.seed}, temperatura ${project.compose.temperature})`;
    }
    case "analyze_audio":
      await analyze();
      return "Análisis del audio en cola";
    case "render_song":
      await render(null);
      return "Take de la canción en cola";
    case "render_sections": {
      const names = (Array.isArray(action.sections) ? action.sections : []).map((name) => sectionNamed(name).name);
      if (!names.length) throw new Error("faltan secciones");
      await render(names);
      return `Take en cola: ${names.join(", ")}`;
    }
    case "render_voice": {
      const voice = voiceNamed(action.voice);
      await renderVoice(voice);
      return `${voice.name} en cola`;
    }
    case "save_song":
      await saveProject();
      return "Canción guardada";
    default: {
      const result = await runDawAction(action);
      if (result === undefined) throw new Error("acción desconocida");
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
  status("El asistente está pensando…");
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
        message.content = answer.content.trim() || "(sin respuesta)";
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
        message.done.push(call.name === "get_abc" ? `ABC leído${args.section ? ` · ${args.section}` : ""}` : result);
      }
      drawChat();
    }
    message.content ||= "Me detuve tras varias acciones sin una respuesta final.";
    if (message.done.length) remember(`Asistente: ${message.done.join("; ")}`);
    if (project.name && message.done.length) await saveProject();
    status("Asistente listo.", 1);
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
  if (!Object.keys(plan).length) throw new Error("Elige en el inspector qué secciones llevan línea clásica (alta, baja o tema).");
  status("Buscando frases en las partituras de referencia…");
  const { abc, report } = await postJson("/hz3/studio/classical", { abc: project.abc, plan });
  project.abc = abc;
  project.classicalReport = report;
  $("abc").value = abc;
  await refreshSections();
  await saveProject();
  status(`Líneas clásicas: ${report.join(" · ")}`, 1);
}

async function arrange() {
  readForm();
  if (!sections.length) throw new Error("Hacen falta letra y ABC con secciones.");
  status("El agente está arreglando las armonías…");
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
      .map((section) => `${section.name}: ${plan[section.name].voices.length} voces`).join(" · ");
    remember(`Armonías del agente · ${summary || "sin armonías"}`);
    status(`Arreglo del agente · ${summary || "sin armonías"}`, 1);
    draw();
    restartIfPlaying();
  } finally {
    $("arrange").disabled = false;
  }
}

// ---------- jobs through ComfyUI

// A job can finish after another project was opened: it must write into the project that queued it.
async function updateProject(owner, change) {
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
  await listProjects();
  status(`«${owner}» actualizado en segundo plano.`, 1);
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
      status(`${job.labels[data.node] ?? "Procesando"} · ${data.value}/${data.max}`, data.value / data.max);
    } else if (type === "executing" && data.node) {
      status(`${job.labels[data.node] ?? "Procesando"}…`);
    } else if (type === "executing") {
      // Sent after the queue stores the history; execution_success comes before it.
      finishJob(data.prompt_id).catch((error) => status(error.message, null, true));
    } else if (type === "execution_error" || type === "execution_interrupted") {
      pending.delete(data.prompt_id);
      setBusy(pending.size > 0);
      status(type === "execution_error" ? `Error en ${data.node_type}: ${data.exception_message}` : "Trabajo interrumpido.", null, true);
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
  status(`Audio «${name}» agregado a la biblioteca y asignado a la canción. Pulsa «Analizar audio».`, 1);
}

// Analyses the open song's audio, or the library audio / song named `owner` without opening it.
async function analyze(owner = project.name) {
  readForm();
  const target = owner === project.name ? project : await api(`/hz3/studio/project?name=${encodeURIComponent(owner)}`);
  if (!target.source) throw new Error("Abre primero el audio original.");
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
    if (!text(12).trim()) throw new Error("El análisis terminó sin ABC.");
    const current = await updateProject(owner, (target) => {
      target.analysis = {
        at: Date.now(), sheetsage_abc: text(13), whisper_lyrics: text(14),
        mixmash: { style: text(10), lyrics: text(11), abc: text(12) },
      };
      target.style = text(10);
      target.lyrics = text(11);
      keepAbcVersion(target, "antes del análisis");
      target.abc = text(12);
    });
    if (!current) return;
    writeForm();
    await refreshSections();
    await uniqueSectionNames();
    await saveProject();
    remember("Audio analizado: estilo, letra y ABC nuevos");
    status("Análisis listo: revisa las secciones en la partitura, la letra y el ABC.", 1);
  });
  status("Análisis en cola…", 0);
}

// Replaced scores stay in the project, so a new composition or analysis can be undone.
// A version that also changed the lyrics keeps them too, so restoring it brings both back.
function keepAbcVersion(target, label, withLyrics = false) {
  if (!target.abc.trim()) return;
  const version = { at: Date.now(), label, abc: target.abc, ...(withLyrics ? { lyrics: target.lyrics } : {}) };
  target.abcVersions = [...(target.abcVersions ?? []), version].slice(-20);
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
  if (!played.some((bar) => bar.moved)) throw new Error(`«${name}» ya no tiene voz cantada.`);
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
  keepAbcVersion(project, `antes de volver instrumental «${name}»`, true);
  project.abc = abc;
  project.lyrics = lyrics;
  writeForm();
  await refreshSections();
  remember(`«${name}» convertida en instrumental (melodía a la línea instrumental, voz en silencio)`);
  status(`«${name}» es instrumental: su melodía pasó a la línea instrumental. La versión anterior (ABC y letra) quedó en las versiones.`, 1);
}

// What a song version holds: everything a render is made from, frozen only when the producer asks ("Guardar versión").
const SONG_STATE = ["style", "baseStyle", "lora", "lyrics", "abc", "seed", "sampling", "harmonize", "harmonyVoices",
  "sectionSeeds", "sectionStyles", "sectionSingers", "classicalPlan", "arrangement", "comp", "trackOn", "trackSpans", "mixer", "voices", "instruments", "fx"];

function songSnapshot(source, label) {
  return { at: Date.now(), label, ...JSON.parse(JSON.stringify(Object.fromEntries(SONG_STATE.map((key) => [key, source[key]])))) };
}

async function pickSongVersion(index) {
  readForm();
  const version = project.songVersions[index];
  const automatic = project.songVersions.filter((item) => /^take \d+/.test(item.label) || item.label === "antes de restaurar una versión");
  let action = await choose(`Versión «${version.label}»`, `Guardada el ${new Date(version.at).toLocaleString()}.`, [
    { value: "restore", label: "Restaurarla (lo que tienes ahora se pierde si no lo guardaste como versión)" },
    { value: "keep", label: "Guardar lo actual como versión y luego restaurarla" },
    { value: "delete", label: "Borrar esta versión" },
    // Earlier builds froze a version on every render and every restore.
    ...(automatic.length ? [{ value: "automatic", label: `Borrar las ${automatic.length} versiones automáticas (de cada render o restauración)` }] : []),
  ]);
  if (action === "delete" || action === "automatic") {
    const gone = action === "delete" ? [version] : automatic;
    project.songVersions = project.songVersions.filter((item) => !gone.includes(item));
    writeForm();
    status(`${gone.length === 1 ? `Versión «${version.label}» borrada` : `${gone.length} versiones borradas`}. Guarda el proyecto para que quede así.`, 1);
    return;
  }
  if (action === "keep" && !saveSongVersion()) action = null;
  if (!action) { writeForm(); return; }
  for (const key of SONG_STATE) if (key in version) project[key] = JSON.parse(JSON.stringify(version[key]));
  writeForm();
  await refreshSections();
  draw();
  restartIfPlaying();
  status(`Versión «${version.label}» restaurada. Guarda el proyecto para conservarla.`, 1);
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
    if (marker < 0) throw new Error(`El ABC no tiene la marca «% ${fromSection}».`);
    return lines.slice(0, marker + 1).join("\n") + "\n";
  }
  const key = lines.findIndex((line) => line.startsWith("K:"));
  return project.compose.keep && key >= 0 ? lines.slice(0, key + 1).join("\n") + "\n" : "";
}

async function compose(fromSection = null) {
  readForm();
  if (!project.style.trim() || !project.lyrics.trim()) throw new Error("Escribe estilo y letra con encabezados [Sección] antes de generar el ABC.");
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
    if (!abc.trim()) throw new Error("YuE2 no devolvió ABC.");
    const replaced = (target) => {
      keepAbcVersion(target, fromSection ? `antes de recomponer desde «${fromSection}»` : `antes de componer con semilla ${seed}`);
      target.abc = abc;
    };
    if (!await updateProject(owner, replaced)) return;
    $("abc").value = abc;
    await refreshSections().catch((error) => status(`ABC generado; revisa las secciones: ${error.message}`, null, true));
    await uniqueSectionNames();
    await saveProject();
    remember(fromSection ? `YuE2 recompuso el ABC desde «${fromSection}»` : "YuE2 compuso un ABC nuevo");
    status("ABC generado: revisa las secciones en la partitura.", 1);
  });
  status("Composición en cola…", 0);
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
        section_tokens: Object.entries(kept).map(([name, tokens]) => `${name} = ${tokens}`).join("\n"),
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
  withLora(prompt, sections.map((section) => {
    const singer = catalogStyle(project.sectionSingers[section.name]);
    return singer ? { lora: singer.lora, owner: singer.name } : { lora: project.lora, owner: "el cantante de la canción" };
  }));
  return { prompt, saves };
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

function saveSongVersion() {
  readForm();
  const label = prompt("Nombre de la versión:", `versión ${project.songVersions.length + 1}`);
  if (label === null) return false;
  project.songVersions = [...project.songVersions, songSnapshot(project, label.trim() || "sin nombre")].slice(-30);
  writeForm();
  status(`Versión «${label.trim() || "sin nombre"}» guardada. Guarda el proyecto para conservarla.`, 1);
  return true;
}

// reimagine: render the whole song in melody mode (YuE2 keeps the melody and invents harmony and arrangement), have
// SheetSage2 hear the chords it played, and freeze them into the song's ABC, which then sings in full mode.
async function render(targets, reimagine = false) {
  readForm();
  ensureName();
  await refreshSections();
  if (!sections.length) throw new Error("Hacen falta letra y ABC con secciones.");
  const id = Math.max(0, ...project.takes.map((take) => take.id)) + 1;
  const take = { id, created: Date.now(), harmonized: project.harmonize, files: {}, sectionLyrics: {}, sectionSeeds: {} };
  take.sectionStyles = {};
  take.sectionAbc = Object.fromEntries(sections.map((section) => [section.name, section.abc]));
  sections.forEach((section, index) => {
    take.sectionLyrics[section.name] = section.lyrics;
    take.sectionSeeds[section.name] = sectionSeed(section, index);
    if (sectionStyle(section)) take.sectionStyles[section.name] = sectionStyle(section);
  });
  // "Generar canción" renders every section; "Regenerar sección" only the chosen ones (and any never rendered).
  const comped = new Set(sections.filter((section) =>
    reimagine || !targets || targets.includes(section.name) || !takeById(project.comp[section.name])
  ).map((section) => section.name));
  const kept = reimagine ? {} : await keptTokens(comped);
  const missing = sections.filter((section) => !comped.has(section.name) && !kept[section.name]).map((section) => section.name);
  if (missing.length) {
    const go = await choose(`También se regenerarían ${missing.length} secciones más`,
      `No quedan tokens guardados que cubran ${missing.map((name) => `«${name}»`).join(", ")} con sus compases actuales, así que YuE2 las volvería a cantar.`,
      [{ value: true, label: `Regenerar también esas ${missing.length}` }]);
    if (!go) return;
    for (const name of missing) comped.add(name);
  }
  const { prompt, saves } = buildPrompt(`HZ3-Studio/${project.name}/take-${id}`, kept);
  const melody = melodyOnly(project.abc);
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
  await queue(prompt, { ...RENDER_LABELS, 31: "SheetSage2 (acordes del render)" }, async (outputs) => {
    for (const [node, track] of Object.entries(saves)) {
      const file = outputs[node]?.audio?.[0];
      if (file) take.files[track] = file;
    }
    if (!take.files.vocals) throw new Error("La generación terminó sin audio.");
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
      keepAbcVersion(project, "antes de reimaginar el ABC");
      project.abc = abc;
      writeForm();
      await refreshSections();
      // The take is what this frozen score describes, so its sections are not shown as edited.
      take.sectionAbc = Object.fromEntries(sections.map((section) => [section.name, section.abc]));
      await saveProject();
      remember(`ABC reimaginado con el take ${take.id}: acordes congelados, modo full`);
      status(`Take ${take.id} reimaginado: sus acordes quedaron en el ABC (modo full). «Regenerar» armonías para usarlos.`, 1);
      draw();
      return;
    }
    remember(`Take ${take.id} · ${[...comped].join(", ")}`);
    status(`Take ${take.id} listo · ${[...comped].join(", ")}${harmonies.length ? " · armonías en proceso…" : ""}`, harmonies.length ? 0 : 1);
    draw();
  });
  status(`Take ${id} en cola…`, 0);
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

async function loadUrl(url) {
  if (buffers.has(url)) return;
  const response = await fetch(url);
  if (!response.ok) return;
  const buffer = await context().decodeAudioData(await response.arrayBuffer());
  buffers.set(url, { buffer, peaks: peaks(buffer) });
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
  $("meter-limit").textContent = reduction < -0.5 ? `limitador ${reduction.toFixed(1)} dB` : "";
  $("meter-limit").classList.toggle("hot", reduction < -6);
  $("meter-clip").classList.toggle("on", clipped);
}

function position() {
  return playback ? playback.from + audioContext.currentTime - playback.start : pausedAt;
}

async function play() {
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
  if (!duration()) throw new Error("No hay nada que exportar todavía.");
  const sampleRate = 48000;
  const offline = new OfflineAudioContext(2, Math.ceil(duration() * sampleRate), sampleRate);
  buildGraph(offline, offline.destination, 0, 0);
  status("Exportando mezcla…");
  const rendered = await offline.startRendering();
  const peak = Math.max(...[...Array(rendered.numberOfChannels).keys()].map((channel) =>
    rendered.getChannelData(channel).reduce((max, value) => Math.max(max, Math.abs(value)), 0)));
  const response = await fetch("/hz3/studio/flac", { method: "POST", body: wav(rendered) });
  if (!response.ok) throw new Error(`No se pudo codificar el FLAC: ${await response.text()}`);
  const link = document.createElement("a");
  link.href = URL.createObjectURL(await response.blob());
  link.download = `${project.name || "hz3-studio"}.flac`;
  link.click();
  URL.revokeObjectURL(link.href);
  status(`Mezcla exportada · pico ${dbfs(peak).toFixed(1)} dBFS.`, 1);
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

function draw() {
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
  head.textContent = orphans.length ? `Letra · ${orphans.length} sin sección` : "Letra";
  head.title = orphans.length
    ? `Bloques de la letra sin sección del ABC (YuE2 los recibe igual, en la letra completa): ${orphans.join(", ")}`
    : "Letra de cada sección: edítala aquí para acomodarla con las secciones del ABC";
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
    area.placeholder = "sin letra";
    area.title = `${section.name}: ${section.lyrics_index === null ? "sin bloque en la letra (escribe aquí para crearlo)" : "letra de esta sección"}`;
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
      arrows.innerHTML = `<button data-way="back" title="Pasa la primera línea de «${escapeHtml(section.name)}» a la sección anterior">⇠</button><button data-way="forward" title="Trae la última línea de la sección anterior a «${escapeHtml(section.name)}»">⇢</button>`;
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
  if (at < 0) throw new Error("No hay líneas que mover en esa sección.");
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
  head.innerHTML = `<span class="name link">${harmoniesCollapsed ? "▸" : "▾"} Armonías · ${project.harmonyVoices.length}/${VOICES.length}</span>
    <span class="controls"><button data-action="regenerate" title="Vuelve a armonizar la voz principal de los takes que suenan, solo para las armonías marcadas, sin renderizar la canción">Regenerar</button></span>`;
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
  if (!project.harmonyVoices.length) throw new Error("Marca primero qué armonías generar.");
  const takes = [...new Set(Object.values(project.comp))].map(takeById).filter((take) => take?.files.vocals);
  if (!takes.length) throw new Error("Todavía no hay takes con voz principal.");
  for (const take of takes) await harmonizeTake(take, project.name, project.harmonyVoices, project.abc);
  status(`Armonías en cola para ${takes.length} take(s).`, 0);
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
  await queue(prompt, { 2: `Armonías del take ${take.id}`, 22: "SheetSage2 (acordes del take)" }, async (outputs) => {
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
    status(`Armonías del take ${take.id} listas.`, 1);
    draw();
    restartIfPlaying();
  });
}

function drawSections() {
  const { body } = lane("sections", "Secciones");
  sections.forEach((section, index) => {
    const block = document.createElement("div");
    block.className = `section-block${section.name === selected ? " selected" : ""}`;
    block.style.left = `${section.start * pxPerSecond}px`;
    block.style.width = `${(section.end - section.start) * pxPerSecond}px`;
    block.style.borderLeftColor = SECTION_COLORS[index % SECTION_COLORS.length];
    block.title = `${section.name} · ${fmt(section.start)}–${fmt(section.end)}`;
    block.textContent = section.name;
    const takeId = project.comp[section.name];
    if (takeId) block.insertAdjacentHTML("beforeend", `<span class="take">T${takeId}</span>`);
    if (isEdited(section, index)) block.insertAdjacentHTML("beforeend", `<span class="edited">editado</span>`);
    block.onclick = () => { selected = section.name; selectedVoice = null; selectedTrack = null; draw(); };
    if (index > 0 && score) {
      const handle = document.createElement("div");
      handle.className = "handle";
      handle.title = "Arrastra para mover el inicio de la sección (se ajusta al compás)";
      handle.onmousedown = (event) => { event.preventDefault(); event.stopPropagation(); dragBoundary(index, body); };
      block.append(handle);
    }
    body.append(block);
  });
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
    status(`«${edited[index].name}» empezará en el compás ${bar + 1} · ${fmt(barSeconds(bar))}`);
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
  head.innerHTML = `<span class="name">Partitura · ${score.key} · ${score.bpm} BPM</span><span class="controls">
    <button id="score-lane-play" title="Reproducir / pausar la partitura desde su cursor (clic en la pista para moverlo)">${scorePlayback ? "❚❚" : "▶"}</button>
    <button id="score-lane-stop" title="Detener y volver al punto marcado">■</button></span>`;
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
      graphics.fillRect(x(note.start), top + (high - note.pitch) * row, Math.max(1, x(note.duration) - 1), Math.max(2, row - 1));
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
  const generate = track.harmony ? `<label class="generate" title="Generar esta armonía en los renders y al regenerar"><input type="checkbox"${project.harmonyVoices.includes(track.id) ? " checked" : ""}> generar</label>` : "";
  const singer = track.voice && catalogStyle(track.voice.singer);
  const who = track.voice ? (singer ? `${singer.name} · ${singerCue(singer)}` : track.voice.style.split("\n").filter((line) => line.trim()).at(-1) ?? "") : "";
  const inProgress = project.takes.some((take) => take.pending?.includes(track.id));
  head.innerHTML = `<span class="name">${escapeHtml(track.name)}</span>${inProgress ? `<span class="who">en proceso…</span>` : ""}${who ? `<span class="who" title="${escapeHtml(who)}">${escapeHtml(who)}</span>` : ""}${generate}<span class="controls">
    <button data-action="mute" class="${muted(track) ? "active" : ""}" title="Silenciar">M</button>
    <button data-action="solo" class="${project.mixer[track.id]?.solo ? "active" : ""}" title="Solo">S</button>
    <input type="range" min="0" max="1.5" step="0.01" value="${project.mixer[track.id]?.gain ?? track.gain}" title="Volumen"></span>`;
  if (!track.voice) {
    const name = head.querySelector(".name");
    name.classList.add("link");
    name.title = "Volumen, paneo, efectos y ajustes de esta pista";
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
    name.title = "Editar, generar o alinear esta voz";
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
      toggle.title = `${voice.name} · ${section.name}: ${enabled ? "encendida" : "apagada"} (clic para cambiar)`;
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
    toggle.title = `${track.name} · ${section.name}: ${enabled ? "encendido" : "apagado"} (clic para cambiar)`;
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
    block.title = `${track.name} ${span.on ? "encendido" : "apagado"} de ${fmt(span.start)} a ${fmt(span.end)} (clic para quitar)`;
    block.onclick = (event) => {
      event.stopPropagation();
      spans.splice(spans.indexOf(span), 1);
      draw();
      restartIfPlaying();
    };
    body.append(block);
  }
  const timeAt = (event) => Math.min(duration(), Math.max(0, (event.clientX - body.getBoundingClientRect().left) / pxPerSecond));
  body.onpointerdown = (event) => {
    if (event.button !== 0 || event.target.closest(".span, select")) return;
    const start = timeAt(event);
    const preview = document.createElement("div");
    delete body.dataset.dragged;
    body.setPointerCapture(event.pointerId);
    body.onpointermove = (move) => {
      if (Math.abs(move.clientX - event.clientX) < 4) return;
      body.dataset.dragged = "1";
      const [from, to] = [start, timeAt(move)].sort((a, b) => a - b);
      preview.className = `span preview${trackOnAt(track.id, start) ? " off" : " on"}`;
      preview.style.left = `${from * pxPerSecond}px`;
      preview.style.width = `${(to - from) * pxPerSecond}px`;
      if (!preview.isConnected) body.append(preview);
    };
    body.onpointerup = (up) => {
      body.onpointermove = body.onpointerup = null;
      preview.remove();
      if (!body.dataset.dragged) return;
      const [from, to] = [start, timeAt(up)].sort((a, b) => a - b);
      spans.push({ start: Math.round(from * 100) / 100, end: Math.round(to * 100) / 100, on: !trackOnAt(track.id, start) });
      draw();
      restartIfPlaying();
      // The click that ends a drag must not also switch the section under it.
      setTimeout(() => delete body.dataset.dragged, 0);
    };
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
  $("score-play").textContent = "♪ Partitura";
  $("section-score-play").textContent = "▶ Partitura";
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

function playScore(fromTick, toTick, button, playingLabel = "■ Detener") {
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
  $("inspector-title").textContent = `${section.name} · ${fmt(section.start)}–${fmt(section.end)} · ${section.bars} compases`;
  $("section-arrangement").textContent = project.arrangement[section.name] ? `Agente: ${project.arrangement[section.name]}` : "";
  if (document.activeElement !== $("section-name")) $("section-name").value = section.name;
  if (document.activeElement !== $("section-lyrics")) $("section-lyrics").value = section.lyrics;
  const ownStyle = project.sectionStyles[section.name];
  if (styleDraft?.section !== section.name) {
    styleDraft = null;
    $("section-style").value = ownStyle ?? defaultSectionStyle(section);
  }
  $("section-style-label").textContent = ownStyle ? "Estilo de la sección (propio)" : "Estilo de la sección (el general; se actualiza solo)";
  $("section-style-save").disabled = $("section-style-undo").disabled = !styleDraft;
  $("section-style-original").disabled = !ownStyle && !styleDraft;
  $("section-classical").value = project.classicalPlan[section.name] ?? "";
  $("section-singer").innerHTML = styleOptions(["singer"], songSingerLabel(), project.sectionSingers[section.name]);
  if (document.activeElement !== $("section-seed")) $("section-seed").value = sectionSeed(section, index);
  $("section-merge").disabled = index === sections.length - 1;
  const original = sectionOriginal(section.name);
  $("section-restore").classList.toggle("hidden", !original);
  $("section-restore").textContent = original?.kind === "instrumental" ? "Restaurar voz y letra" : "Restaurar melodía escrita";
  $("section-earlier").disabled = $("section-later").disabled = index === 0;
  const takes = $("section-takes");
  takes.innerHTML = project.takes.length ? "<span>Take de esta sección</span>" : "<span>Sin takes todavía.</span>";
  for (const take of project.takes) {
    const label = document.createElement("label");
    const changed = (take.sectionLyrics[section.name] ?? "").trim() !== section.lyrics.trim() ? " · otra letra" : "";
    label.innerHTML = `<input type="radio" name="take" ${project.comp[section.name] === take.id ? "checked" : ""}>
      T${take.id} · semilla ${take.sectionSeeds[section.name]}${changed}`;
    label.querySelector("input").onchange = () => {
      project.comp[section.name] = take.id;
      draw();
      restartIfPlaying();
    };
    takes.append(label);
  }
}

async function renameSection(name) {
  const index = sections.findIndex((section) => section.name === selected);
  name = name.replace(/[\r\n[\]]+/g, " ").trim();
  if (!name || name === selected) return;
  if (sections.some((section) => section.name === name)) throw new Error(`Ya existe una sección «${name}».`);
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
    throw new Error("Pon el cursor dentro de la sección, en el compás donde debe empezar la nueva.");
  }
  let name = `${edited[index].name} b`;
  while (sections.some((section) => section.name === name)) name += "b";
  edited.splice(index + 1, 0, { id: `section-${Date.now()}`, name, start_bar: bar, end_bar: edited[index].end_bar, lyrics: "" });
  edited[index].end_bar = bar - 1;
  // The new part keeps what the section had: its take (the audio under its bars), style, singer, and which
  // tracks and voices play there.
  const maps = [project.comp, project.sectionStyles, project.sectionSingers, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
  for (const voice of project.voices) maps.push(voice.on, voice.sectionOffsets, voice.sectionGains ?? {});
  for (const map of maps) if (selected in map) map[name] = map[selected];
  selected = name;
  await applySections(edited);
}

async function moveSectionStart(bars) {
  const index = sections.findIndex((section) => section.name === selected);
  if (index < 1) throw new Error("La primera sección empieza siempre en el compás 1.");
  const edited = editableSections();
  const start = edited[index].start_bar + bars;
  if (start <= edited[index - 1].start_bar || start > edited[index].end_bar) throw new Error("No hay compases suficientes para mover ese límite.");
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
    return `estilo ${style ? `propio («${style.split("\n").at(-1).slice(0, 50)}»)` : "general"} · ${singer ? `canta ${singer.name}` : songSingerLabel().toLowerCase()}`;
  };
  // The merged section can only sing one style and one singer.
  let keep = first;
  if ((project.sectionStyles[first.name] ?? "") !== (project.sectionStyles[second.name] ?? "")
      || (project.sectionSingers[first.name] ?? null) !== (project.sectionSingers[second.name] ?? null)) {
    keep = await choose(`Unir «${first.name}» y «${second.name}»`,
      "Tienen distinto estilo o cantante, y la sección unida solo puede tener uno. ¿Con cuál se queda?",
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
    }), Object.assign(document.createElement("button"), { textContent: "Cancelar", onclick: () => answer(null) }));
    dialog.onclose = () => resolve(null);
    dialog.showModal();
  });
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
    id, name: `Voz ${id + 1}`, source: "lead", octave: 0, role: "-4", seed: project.seed + 1000 * id,
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
  await queue(prompt, { ...RENDER_LABELS, 2: `Tokens YuE2 (${voice.name})`, 20: "Línea de armonía" }, async (outputs) => {
    const file = outputs[8]?.audio?.[0];
    if (!file) throw new Error("La voz terminó sin audio.");
    const current = await updateProject(owner, (target) => {
      const stored = target.voices.find((item) => item.id === voice.id);
      if (stored) Object.assign(stored, { file, mixFile: outputs[9]?.audio?.[0] ?? null });
    });
    if (!current) return;  // aligned when that project is open again ("Alinear con la voz principal")
    await loadUrl(fileUrl(file));
    await alignVoice(project.voices.find((item) => item.id === voice.id));
    await saveProject();
    remember(`${voice.name} generada y alineada`);
    draw();
  });
  status(`${voice.name} en cola…`, 0);
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
  if (!voice.file) throw new Error("Genera la voz antes de alinearla.");
  status(`Alineando ${voice.name} con la voz principal…`);
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
  status(`${voice.name} alineada: ${Math.round(voice.offset * 1000)} ms global · nivel ${voice.role} dB respecto a la principal.`, 1);
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
  $("voice-title").textContent = `${voice.name}${voice.file ? "" : " · sin generar"}`;
  for (const [id, key] of [["voice-name", "name"], ["voice-source", "source"], ["voice-octave", "octave"], ["voice-role", "role"], ["voice-style", "style"], ["voice-seed", "seed"]]) {
    if (document.activeElement !== $(id)) $(id).value = voice[key];
  }
  if (document.activeElement !== $("voice-offset")) $("voice-offset").value = Math.round(voice.offset * 1000);
  $("voice-singer").innerHTML = styleOptions(["singer"], "— (sin cantante del catálogo)", voice.singer);
  const list = $("voice-sections");
  list.innerHTML = "<span>Ajuste fino por sección (ms)</span>";
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
      status(`Referencia «${name}» analizada.`, 1);
    });
  }
  status(`${files.length} referencias en cola para análisis.`, 0);
}

// ---------- projects (.mixmash packages)

let savedName = null;

// Songs are renamed in the catalog; a song without a name yet asks for one when you save or render it.
function ensureName() {
  if (project.name) return;
  const name = (prompt("Nombre de la canción") ?? "").trim();
  if (!name) throw new Error("La canción necesita un nombre para guardarse.");
  project.name = name;
}

async function saveProject() {
  readForm();
  if (!project.name) throw new Error("Ponle nombre al proyecto.");
  // A new name on an opened project renames its package instead of copying it.
  if (savedName && savedName !== project.name) await postJson("/hz3/studio/rename", { from: savedName, to: project.name });
  project.styles = referencedStyles();
  await postJson("/hz3/studio/project", project);
  savedName = project.name;
  await listProjects();
  status(`Proyecto «${project.name}» guardado.`, 1);
}

async function listProjects() {
  const { projects } = await api("/hz3/studio/projects");
  catalog = projects;
  // Only the browsed album's songs; archived ones stay out of the quick list (unless open), the catalog shows them.
  const songs = projects.filter((entry) => entry.album === browseAlbum).filter((entry) => entry.kind !== "source" && (!entry.archived || entry.name === project.name));
  $("project-list").innerHTML = `<option value="">— ${songs.length} canciones —</option><option value="${NEW_SONG}">+ Canción nueva</option>`
    + songs.map((entry) => `<option${entry.name === project.name ? " selected" : ""}>${escapeHtml(entry.name)}</option>`).join("");
  const sources = projects.filter((entry) => entry.kind === "source");
  $("source-pick").innerHTML = `<option value="">— sin audio de la biblioteca —</option>`
    + sources.map((entry) => `<option${entry.name === project.sourceName ? " selected" : ""}>${escapeHtml(entry.name)}</option>`).join("");
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
  if (parts.abc && source.abc.trim()) setAbc(source.abc, `antes de traer el ABC de «${name}»`);
  writeForm();
  await refreshSections().catch((error) => status(`Revisa las secciones: ${error.message}`, null, true));
  if (parts.abc || parts.lyrics) await uniqueSectionNames();
  if (project.name) await saveProject();
  const taken = Object.entries(parts).filter(([, on]) => on).map(([part]) => ({ audio: "audio", abc: "ABC", lyrics: "letra", style: "estilo" })[part]);
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
  rows.innerHTML = sources.length ? "" : `<p class="note">La biblioteca está vacía.</p>`;
  for (const entry of sources) {
    const users = catalog.filter((other) => other.kind !== "source" && other.source_file === entry.source_file).map((other) => other.name);
    const item = document.createElement("details");
    item.className = "source";
    item.innerHTML = `<summary><b>${escapeHtml(entry.name)}</b> · ${escapeHtml(entry.source ?? "")} ·
        ${entry.analyzed ? `analizado, ${entry.sections} secciones` : "sin analizar"}${users.length ? ` · usado en ${users.map(escapeHtml).join(", ")}` : ""}</summary>
      <div class="row">
        <button data-action="analyze">${entry.analyzed ? "Analizar de nuevo" : "Analizar"}</button>
        <span class="parts">Traer a la canción abierta:
          <label class="switch"><input type="checkbox" value="audio" checked> audio</label>
          <label class="switch"><input type="checkbox" value="abc" checked> ABC</label>
          <label class="switch"><input type="checkbox" value="lyrics" checked> letra</label>
          <label class="switch"><input type="checkbox" value="style" checked> estilo</label>
          <button data-action="use">Traer</button></span>
        <button data-action="song" class="primary">Nueva canción con este audio</button>
        <button data-action="delete">Eliminar</button>
      </div>
      <div class="fields">
        <label>Estilo<textarea data-field="style" rows="4"></textarea></label>
        <label>Letra<textarea data-field="lyrics" rows="8"></textarea></label>
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
        if (area.dataset.field === "abc") keepAbcVersion(stored, "antes de editarlo en la biblioteca");
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
    + `<button data-album="${NEW_ALBUM}">+ Nuevo álbum</button>`;
  $("album-nav").querySelectorAll("button").forEach((button) => {
    button.onclick = guard(async () => {
      let name = button.dataset.album;
      if (name === NEW_ALBUM) {
        name = (prompt("Nombre del nuevo álbum") ?? "").trim();
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
  $("album-delete").disabled = managedAlbum === DEFAULT_ALBUM;
  $("album-styles").innerHTML = info.styles.map((style) => {
    const users = songs.filter((entry) => entry.styles.some((used) => used.id === style.id)).map((entry) => escapeHtml(entry.name));
    return `<span class="chip" title="${escapeHtml(style.text)}"><b>${escapeHtml(style.name)}</b> · ${STYLE_KINDS[style.kind]} · ${users.length ? users.join(", ") : "sin usar"}</span>`;
  }).join("") || `<span class="note">Sin estilos: créalos en «Estilos» con una canción de este álbum abierta, o pídeselos al asistente.</span>`;
  const albumOptions = albums.map((name) => `<option${name === managedAlbum ? " selected" : ""}>${escapeHtml(name)}</option>`).join("");
  const rows = $("catalog-rows");
  rows.innerHTML = songs.length ? "" : `<tr><td colspan="7" class="style">Este álbum no tiene canciones.</td></tr>`;
  for (const entry of songs.filter((item) => $("show-archived").checked || !item.archived)) {
    const row = document.createElement("tr");
    row.className = entry.archived ? "archived" : "";
    row.innerHTML = `<td><input maxlength="64" title="Cambia el nombre y pulsa Enter"><button data-action="open">Abrir</button>${entry.archived ? " <small>archivada</small>" : ""}</td>
      <td><textarea rows="2" placeholder="Notas de la canción"></textarea></td>
      <td class="style">${entry.styles.map((style) => `${escapeHtml(style.name)} <small>(${STYLE_KINDS[style.kind] ?? style.kind})</small>`).join(", ") || "—"}
        <br><small>${escapeHtml(entry.style)}</small></td>
      <td>${new Date(entry.updated * 1000).toLocaleString()}<br><small>${entry.kind === "reference" ? "Referencia" : "Canción"} ·
        ${entry.source ? `audio: ${escapeHtml(entry.source)}` : "compuesta"}${entry.license ? ` · ${escapeHtml(entry.license.name)}` : ""}</small></td>
      <td>${entry.sections} secciones<br>${entry.takes} takes · ${entry.voices} voces</td>
      <td><select title="Mover a otro álbum">${albumOptions}</select></td>
      <td class="actions"><button data-action="duplicate" title="Crea «${escapeHtml(entry.name)} (copia)» en el mismo álbum y la abre; el original no cambia">Duplicar</button>
        <button data-action="download" title="Descarga su .mixmash: audio, letra, ABC, tokens, LoRAs y estado de la app">Descargar</button>
        <button data-action="archive">${entry.archived ? "Desarchivar" : "Archivar"}</button>
        <button data-action="delete" title="El paquete se mueve a la carpeta deleted">Eliminar</button></td>`;
    const [name, notes, target] = [row.querySelector("input"), row.querySelector("textarea"), row.querySelector("select")];
    name.value = entry.name;
    notes.value = entry.notes;
    row.querySelector('[data-action="open"]').onclick = guard(async () => {
      $("catalog").close();
      await openProject(entry.name);
    });
    name.onchange = guard(() => renameSong(entry.name, name.value));
    notes.onchange = guard(() => updateProject(entry.name, (stored) => { stored.notes = notes.value; }));
    target.onchange = guard(() => moveSong(entry.name, target.value));
    row.querySelector('[data-action="duplicate"]').onclick = guard(async () => {
      $("catalog").close();
      await duplicateSong(entry.name);
    });
    row.querySelector('[data-action="download"]').onclick = guard(async () => {
      if (entry.name === project.name) await saveProject();
      location.href = `/hz3/studio/package?name=${encodeURIComponent(entry.name)}`;
    });
    row.querySelector('[data-action="archive"]').onclick = guard(() => updateProject(entry.name, (stored) => { stored.archived = !entry.archived; }));
    row.querySelector('[data-action="delete"]').onclick = guard(() => deleteSong(entry.name));
    rows.append(row);
  }
}

// A copy of a song beside it ("<name> (copia)"): the original is only read, the copy opens.
async function duplicateSong(name) {
  if (name === project.name) await saveProject();
  const { name: copy } = await postJson("/hz3/studio/duplicate", { name });
  await listProjects();
  await openProject(copy);
  status(`«${copy}» es una copia de «${name}»: lo que cambies aquí no toca el original.`, 1);
}

async function renameSong(from, to) {
  to = to.trim();
  if (!to || to === from) return;
  if (from === project.name) {
    project.name = to;
    await saveProject();
  } else {
    await postJson("/hz3/studio/rename", { from, to });
    await listProjects();
  }
  status(`«${from}» ahora se llama «${to}».`, 1);
}

async function moveSong(name, target) {
  if (name === project.name) return moveToAlbum(target);
  let styles = [];
  await updateProject(name, (stored) => {
    stored.album = target;
    styles = stored.styles;
  });
  await addStylesToAlbum(target, styles);
  status(`«${name}» está ahora en el álbum «${target}».`, 1);
}

async function deleteSong(name) {
  if (!confirm(`¿Eliminar «${name}»? El paquete se mueve a la carpeta deleted del estudio (output/HZ3-YuE2/studio/deleted).`)) return;
  await postJson("/hz3/studio/delete", { name });
  if (name === project.name) await newSong();
  await listProjects();
  status(`«${name}» eliminada (está en la carpeta deleted).`, 1);
}

async function renameAlbum(to) {
  const from = managedAlbum;
  to = to.trim();
  if (!to || to === from) return;
  await postJson("/hz3/studio/album/rename", { from, to });
  if (project.album === from) project.album = to;
  if (album.name === from) await loadAlbum(to);
  managedAlbum = to;
  await listAlbums();
  await listProjects();
  status(`El álbum «${from}» ahora se llama «${to}».`, 1);
}

async function deleteAlbum() {
  const name = managedAlbum;
  if (!confirm(`¿Eliminar el álbum «${name}»? Sus canciones pasan a «${DEFAULT_ALBUM}» y el álbum (estilos, chat y bitácora) se mueve a la carpeta deleted.`)) return;
  const used = referencedStyles();
  await postJson("/hz3/studio/album/delete", { name });
  if (project.album === name) project.album = DEFAULT_ALBUM;
  if (album.name === name) {
    await loadAlbum(DEFAULT_ALBUM);
    await addStylesToAlbum(DEFAULT_ALBUM, used);
  }
  managedAlbum = DEFAULT_ALBUM;
  await listAlbums();
  await listProjects();
  writeForm();
  draw();
  status(`Álbum «${name}» eliminado; sus canciones están en «${DEFAULT_ALBUM}».`, 1);
}

async function saveAlbumNotes(notes) {
  const info = await albumData(managedAlbum);
  info.notes = notes;
  await postJson("/hz3/studio/album", info);
}

async function newSong() {
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
  status(`Canción nueva en el álbum «${album.name}»: ponle nombre, estilo y letra.`, 1);
}

async function openProject(name) {
  pause();
  pausedAt = 0;
  project = { ...newProject(), ...(await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`)) };
  savedName = project.name;
  selected = null;
  selectedTrack = null;
  instrumentAudio.clear();
  if (album.name !== project.album) await loadAlbum(project.album);
  browseAlbum = project.album;
  await addStylesToAlbum(album.name, project.styles);
  await listAlbums();
  await listProjects();
  writeForm();
  await refreshSections();
  // A song from another machine brings its own voice LoRA: it joins the album's singers unless one already has it.
  // (A voice only described in the style is added by hand with «Voz de la canción», or every test song would add one.)
  const voice = songVoice();
  if (voice?.lora && !knownVoice(voice)) {
    album.styles.push(voice);
    await saveAlbum();
    drawStyles();
  }
  // Songs saved with repeated section names get unique ones, or every same-named section would be edited at once.
  const renamed = await uniqueSectionNames();
  status("Cargando audio…");
  selectedVoice = null;
  for (const voice of project.voices) {
    voice.octave ??= 0;
    voice.role ??= "-4";
  }
  await Promise.all([loadSource(), ...project.takes.map(loadTake), ...project.voices.filter((voice) => voice.file).map((voice) => loadUrl(fileUrl(voice.file)))]);
  status(renamed.length
    ? `Proyecto «${project.name}» · secciones con nombre repetido renombradas (${renamed.map(([, to]) => to).join(", ")}): guarda para conservarlo.`
    : `Proyecto «${project.name}» · ${project.takes.length} takes`, 1);
  draw();
  guard(renderInstruments)();
}

// ---------- wiring

function guard(action) {
  return (...args) => Promise.resolve(action(...args)).catch((error) => status(error.message, null, true));
}

async function init() {
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
  $("project-list").onchange = guard((event) => event.target.value === NEW_SONG ? newSong() : event.target.value && openProject(event.target.value));
  $("save-project").onclick = guard(async () => {
    ensureName();
    await saveProject();
  });
  $("duplicate-project").onclick = guard(async () => {
    if (!project.name) throw new Error("Guarda primero la canción.");
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
  $("show-archived").onchange = guard(drawCatalog);
  $("import-sources").onclick = guard(async () => {
    const files = [...$("source-files").files];
    if (!files.length) throw new Error("Elige uno o más audios.");
    const names = await addAudios(files, { kind: "source" });
    $("source-files").value = "";
    if ($("source-analyze").checked) for (const name of names) await analyze(name);
    await drawSources();
    status(`${names.length} audios en la biblioteca${$("source-analyze").checked ? ", análisis en cola" : ""}.`, $("source-analyze").checked ? 0 : 1);
  });
  $("source-pick").onchange = guard(async (event) => {
    if (event.target.value) await useSource(event.target.value, { audio: true });
  });
  $("source-bring").onclick = guard(async () => {
    readForm();
    if (!project.sourceName) throw new Error("Elige primero un audio de la biblioteca.");
    await useSource(project.sourceName, { abc: true, lyrics: true, style: true });
  });
  $("album-rename").onclick = guard(() => renameAlbum($("album-name").value));
  $("album-delete").onclick = guard(deleteAlbum);
  $("album-notes").onchange = guard((event) => saveAlbumNotes(event.target.value));
  $("close-catalog").onclick = () => $("catalog").close();
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
    album.styles.push({ id: crypto.randomUUID().slice(0, 8), name: `Cantante ${album.styles.length + 1}`, kind: "singer", text: "" });
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
    if (!project.styleInstructions.trim()) throw new Error("Escribe primero las indicaciones de estilo.");
    $("assistant").classList.remove("hidden");
    await sendChat(`Aplica estas indicaciones de estilo a la canción: ${project.styleInstructions.trim()}`);
  });
  $("import-references").onclick = guard(async () => {
    const files = [...$("reference-files").files];
    if (!files.length) throw new Error("Elige uno o más audios de referencia.");
    if (!$("reference-license").value) throw new Error("Indica la licencia de las grabaciones.");
    await importReferences(files, { name: $("reference-license").value, work: $("reference-work").value.trim(), url: $("reference-url").value.trim() });
    $("reference-files").value = "";
  });
  $("toggle-song").onclick = () => document.querySelector(".song").classList.toggle("hidden");
  $("source-audio").onchange = guard(async (event) => {
    const [file] = event.target.files;
    event.target.value = "";
    if (file) await uploadSource(file);
  });
  $("analyze").onclick = guard(() => analyze());
  $("compose").onclick = guard(() => compose());
  $("section-recompose").onclick = guard(() => compose(selected));
  $("section-free").onclick = guard(() => setSectionFree(selected));
  $("section-restore").onclick = guard(() => restoreSectionOriginal(selected));
  $("section-instrumental").onclick = guard(() => sectionToInstrumental(selected));
  $("compose-dice").onclick = () => { $("compose-seed").value = Math.floor(Math.random() * 2 ** 31); readForm(); };
  $("song-version-save").onclick = saveSongVersion;
  $("song-versions").onchange = guard(async (event) => {
    if (event.target.value !== "") await pickSongVersion(Number(event.target.value));
  });
  $("close-inspector").onclick = () => { selected = null; draw(); };
  $("close-voice-inspector").onclick = () => { selectedVoice = null; draw(); };
  $("abc-versions").onchange = guard(async (event) => {
    if (event.target.value === "") return;
    readForm();
    const version = project.abcVersions[Number(event.target.value)];
    const withLyrics = version.lyrics !== undefined;
    keepAbcVersion(project, "antes de restaurar una versión", withLyrics);
    project.abc = version.abc;
    if (withLyrics) project.lyrics = version.lyrics;
    writeForm();
    await refreshSections();
    status("Versión del ABC restaurada; la que había quedó en las versiones.", 1);
  });
  $("render-song").onclick = guard(() => render(null));
  $("abc-reimagine").onclick = guard(() => render(null, true));
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
  $("section-merge").onclick = guard(mergeSection);
  $("section-earlier").onclick = guard(() => moveSectionStart(-1));
  $("section-later").onclick = guard(() => moveSectionStart(1));
  $("section-lyrics").addEventListener("input", guard((event) => {
    setSectionLyrics(sections.findIndex((section) => section.name === selected), event.target.value);
    scheduleSections();
  }));
  $("section-style").addEventListener("input", (event) => {
    styleDraft = { section: selected, text: event.target.value };
    $("section-style-save").disabled = $("section-style-undo").disabled = $("section-style-original").disabled = false;
  });
  $("section-style-save").onclick = saveSectionStyle;
  $("section-style-undo").onclick = () => { styleDraft = null; draw(); };
  $("section-style-original").onclick = () => { delete project.sectionStyles[selected]; styleDraft = null; draw(); };
  $("section-score-play").onclick = guard(async () => {
    if (scorePlayback) return stopScore();
    if (!score) throw new Error("No hay partitura que escuchar.");
    await context().resume();
    playScore(...sectionTicks(sections.findIndex((section) => section.name === selected)), $("section-score-play"));
  });
  $("score-play").onclick = guard(async () => {
    if (scorePlayback) return stopScore();
    if (!score) throw new Error("No hay partitura que escuchar.");
    await context().resume();
    playScore(Math.min(score.total_ticks, Math.round(position() * score.bpm / 60 * TICKS_PER_QUARTER)), score.total_ticks, $("score-play"));
  });
  for (const id of ["song-lora", "song-lora-strength"]) $(id).addEventListener("change", () => { readForm(); draw(); });
  $("section-styles-from-cues").onclick = guard(stylesFromCues);
  $("apply-classical").onclick = guard(applyClassical);
  $("section-classical").addEventListener("change", (event) => { project.classicalPlan[selected] = event.target.value; });
  $("section-singer").addEventListener("change", guard(async (event) => { await setSectionSinger(selected, event.target.value || null); draw(); }));
  $("voice-singer").addEventListener("change", (event) => {
    const voice = selectedVoiceEntry();
    voice.singer = event.target.value || null;
    // A voice still under its default name takes its singer's name.
    if (voice.singer && /^Voz \d+$/.test(voice.name)) voice.name = catalogStyle(voice.singer).name.slice(0, 40);
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
