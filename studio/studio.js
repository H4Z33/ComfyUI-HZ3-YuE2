"use strict";

// HZ3 Studio: song timeline over ComfyUI. A project starts from a source audio:
// analysis transcribes its ABC, lyrics and style, the piano roll corrects the
// sections, and every render is a "take" of the whole song. Each section plays
// from the take chosen for it (comping), with short crossfades at section
// boundaries. Harmony voices are tracks switched per section. Projects are
// saved as .mixmash packages.

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
  return [...TRACKS, ...project.voices.map((voice) => ({ id: `voice-${voice.id}`, name: voice.name, gain: voice.autoGain ?? 0.8, voice }))];
}
const FADE = 0.3;
const PEAKS_PER_SECOND = 100;
const TICKS_PER_QUARTER = 256;
const SECTION_COLORS = ["#e0b04a", "#4aa3e0", "#7bc96f", "#d9714e", "#b07be0", "#4ec9b0", "#e07ba8", "#c9c24a"];
const RENDER_LABELS = { 1: "Cargando modelo", 2: "Tokens YuE2", 4: "KSampler", 5: "Decodificando audio", 7: "Separando voz", 12: "Armonías" };
const ANALYSIS_LABELS = { 3: "SheetSage2 (ABC)", 4: "Separando voz", 5: "Whisper (letra)", 6: "MixMash (Ollama)" };
const COMPOSE_LABELS = { 1: "Cargando modelo", 2: "YuE2 compone el ABC" };

const $ = (id) => document.getElementById(id);
const clientId = crypto.randomUUID();

let project = newProject();
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
    name: "", kind: "song", license: null, source: null, styleInstructions: "", analysis: null,
    style: "", lyrics: "", abc: "", seed: 60, mode: "full",
    ckpt: "yue2_3b_int8_convrot.safetensors",
    sampling: { temperature: 0.9, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, cfg_scale: 2.0 },
    harmonize: true, sectionSeeds: {}, sectionStyles: {}, takes: [], comp: {}, trackOn: {}, mixer: {}, voices: [],
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

// ---------- form <-> project

const SAMPLING = ["temperature", "top_p", "top_k", "repetition_penalty", "cfg_scale"];

function readForm() {
  project.name = $("project-name").value.trim();
  for (const key of ["style", "lyrics", "abc", "mode", "ckpt"]) project[key] = $(key).value;
  project.styleInstructions = $("style-instructions").value;
  project.seed = Number($("seed").value) || 0;
  project.harmonize = $("harmonize").checked;
  for (const key of SAMPLING) project.sampling[key] = Number($(key).value);
  project.arranger = { model: $("arranger-model").value.trim(), instructions: $("arranger-instructions").value };
}

function writeForm() {
  $("project-name").value = project.name;
  for (const key of ["style", "lyrics", "abc", "mode"]) $(key).value = project[key];
  $("style-instructions").value = project.styleInstructions;
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

function renameKeys(from, to) {
  const maps = [project.comp, project.sectionSeeds, project.sectionStyles, project.arrangement, project.classicalPlan, ...Object.values(project.trackOn)];
  for (const take of project.takes) maps.push(take.sectionLyrics, take.sectionSeeds, take.sectionStyles ?? {}, take.sectionAbc ?? {});
  for (const voice of project.voices) maps.push(voice.on, voice.sectionOffsets, voice.sectionGains ?? {});
  for (const map of maps) {
    if (from in map) { map[to] = map[from]; delete map[from]; }
  }
}

async function applySections(edited, renames = {}) {
  readForm();
  // Pin every section's seed by name so reshaping one section does not reroll the others.
  sections.forEach((section, index) => { project.sectionSeeds[section.name] ??= project.seed + index; });
  for (const [from, to] of Object.entries(renames)) renameKeys(from, to);
  const data = await postJson("/hz3/yue2/abc_viewer/data", { abc: project.abc, lyrics: project.lyrics, sections: edited });
  project.abc = data.edited_abc;
  project.lyrics = data.edited_lyrics;
  $("abc").value = project.abc;
  $("lyrics").value = project.lyrics;
  await refreshSections();
}

async function uniqueSectionNames() {
  // Sections are keyed by name (takes, seeds, harmonies); YuE2 may repeat "% chorus".
  if (!score || new Set(sections.map((section) => section.name)).size === sections.length) return;
  const edited = editableSections();
  const headers = [...project.lyrics.matchAll(/^[ \t]*\[([^\]\n]+)\][ \t]*$/gm)].map((match) => match[1].trim());
  if (headers.length === edited.length && new Set(headers.map((name) => name.toLowerCase())).size === headers.length) {
    edited.forEach((section, index) => { section.name = headers[index]; });
    await applySections(edited);
    return;
  }
  const seen = new Set();
  for (const section of edited) {
    let name = section.name;
    for (let count = 2; seen.has(name.toLowerCase()); count++) name = `${section.name} ${count}`;
    seen.add(name.toLowerCase());
    section.name = name;
  }
  await applySections(edited);
}

function lyricBlocks(text) {
  const pattern = /^[ \t]*\[([^\]\n]+)\][ \t]*$/gm;
  const blocks = [];
  let match;
  while ((match = pattern.exec(text))) {
    if (blocks.length) blocks[blocks.length - 1].end = match.index;
    blocks.push({ bodyStart: match.index + match[0].length, end: text.length });
  }
  return blocks;
}

function setSectionLyrics(index, body) {
  const blocks = lyricBlocks(project.lyrics);
  if (blocks.length !== sections.length) throw new Error("La letra no tiene un encabezado [Sección] por cada sección del ABC.");
  const block = blocks[index];
  const tail = index === blocks.length - 1 ? "\n" : "\n\n";
  project.lyrics = project.lyrics.slice(0, block.bodyStart) + "\n" + body.trim() + tail + project.lyrics.slice(block.end);
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
    || (take.sectionStyles?.[section.name] ?? "") !== (project.sectionStyles[section.name] ?? "")
    || Boolean(take.sectionAbc && take.sectionAbc[section.name] !== section.abc);
}

function stylesFromCues() {
  readForm();
  // MixMash-style cues: a global first line, then "[Section] description" lines.
  const lines = project.style.split("\n");
  const global = lines.find((line) => line.trim() && !line.trim().startsWith("[")) ?? "";
  let count = 0;
  for (const line of lines) {
    const match = line.match(/^\s*\[([^\]]+)\]\s*(.+)$/);
    const section = match && sections.find((item) => item.name.toLowerCase() === match[1].trim().toLowerCase());
    if (!section) continue;
    project.sectionStyles[section.name] = `${global.trim()}\n${match[2].trim()}`;
    count++;
  }
  status(count ? `${count} secciones con estilo propio.` : "No hay líneas «[Sección] …» que coincidan con las secciones.", 1, !count);
  draw();
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
  for (const id of ["render-song", "render-section", "analyze", "compose", "voice-render"]) $(id).disabled = busy;
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
    project.name = file.name.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N} _()-]+/gu, "_").slice(0, 64).trim();
    $("project-name").value = project.name;
  }
  const form = new FormData();
  form.append("file", file);
  project.source = await api(`/hz3/studio/source?project=${encodeURIComponent(project.name)}`, { method: "POST", body: form });
  await loadSource();
  await saveProject();
  status(`Audio «${project.source.original}» listo. Pulsa «Analizar audio».`, 1);
  draw();
}

async function analyze() {
  readForm();
  if (!project.source) throw new Error("Abre primero el audio original.");
  const lyrics = project.lyrics.trim();
  const prompt = {
    1: { class_type: "LoadAudio", inputs: { audio: project.source.filename } },
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
        lora_trigger: "", abc_report: ["3", 4], abc: ["3", 0], instructions: project.styleInstructions,
        lyrics: lyrics || ["5", 0], extend_abc: false, karaoke_mode: false,
      },
    },
    10: { class_type: "PreviewAny", inputs: { source: ["6", 0] } },
    11: { class_type: "PreviewAny", inputs: { source: ["6", 1] } },
    12: { class_type: "PreviewAny", inputs: { source: ["6", 2] } },
    13: { class_type: "PreviewAny", inputs: { source: ["3", 0] } },
    14: { class_type: "PreviewAny", inputs: { source: ["5", 0] } },
  };
  const owner = project.name;
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
      target.abc = text(12);
    });
    if (!current) return;
    writeForm();
    await refreshSections();
    await uniqueSectionNames();
    await saveProject();
    status("Análisis listo: revisa las secciones en la partitura, la letra y el ABC.", 1);
  });
  status("Análisis en cola…", 0);
}

async function compose() {
  readForm();
  if (!project.style.trim() || !project.lyrics.trim()) throw new Error("Escribe estilo y letra con encabezados [Sección] antes de generar el ABC.");
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "YuE2GenerateABC",
      inputs: {
        clip: ["1", 1], style: project.style, lyrics: project.lyrics, seed: project.seed, mode: project.mode,
        max_abc_tokens: 8192, temperature: 0.7, top_p: 0.9, top_k: 30, repetition_penalty: 1.005, penalty_window: 100,
      },
    },
    3: { class_type: "PreviewAny", inputs: { source: ["2", 0] } },
  };
  const owner = project.name;
  await queue(prompt, COMPOSE_LABELS, async (outputs) => {
    const abc = outputs[3]?.text?.[0] ?? "";
    if (!abc.trim()) throw new Error("YuE2 no devolvió ABC.");
    if (!await updateProject(owner, (target) => { target.abc = abc; })) return;
    $("abc").value = abc;
    await refreshSections().catch((error) => status(`ABC generado; revisa las secciones: ${error.message}`, null, true));
    await uniqueSectionNames();
    await saveProject();
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
        clip: ["1", 1], style: project.style, lyrics: project.lyrics, abc: project.abc, seed: project.seed,
        mode: project.mode, ...project.sampling, section_seeds: overrides,
        section_styles: Object.entries(project.sectionStyles)
          .filter(([name, text]) => text.trim() && sections.some((section) => section.name === name))
          .map(([name, text]) => `${name}: ${text.replace(/\s*\n\s*/g, " ").trim()}`).join("\n"),
        section_tokens: Object.entries(kept).map(([name, tokens]) => `${name} = ${tokens}`).join("\n"),
      },
    },
    3: { class_type: "EmptyYuE2LatentAudio", inputs: { seconds: ["2", 1], batch_size: 1 } },
    4: {
      class_type: "KSampler",
      inputs: {
        model: ["1", 0], seed: 42, steps: 79, cfg: 4.2, sampler_name: "dpm_2", scheduler: "sgm_uniform",
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
  if (project.harmonize) {
    prompt[12] = { class_type: "HZ3_YuE2_VocalHarmonizer", inputs: { vocals: ["7", 3], score_abc: project.abc } };
    VOICES.forEach((voice, index) => {
      prompt[13 + index] = { class_type: "SaveAudio", inputs: { audio: ["12", index], filename_prefix: `${prefix}/${voice}` } };
      saves[13 + index] = voice;
    });
  }
  return { prompt, saves };
}

async function render(targets) {
  readForm();
  if (!project.name) throw new Error("Ponle nombre al proyecto antes de generar.");
  await refreshSections();
  if (!sections.length) throw new Error("Hacen falta letra y ABC con secciones.");
  const id = Math.max(0, ...project.takes.map((take) => take.id)) + 1;
  const take = { id, created: Date.now(), harmonized: project.harmonize, files: {}, sectionLyrics: {}, sectionSeeds: {} };
  take.sectionStyles = {};
  take.sectionAbc = Object.fromEntries(sections.map((section) => [section.name, section.abc]));
  sections.forEach((section, index) => {
    take.sectionLyrics[section.name] = section.lyrics;
    take.sectionSeeds[section.name] = sectionSeed(section, index);
    if (project.sectionStyles[section.name]) take.sectionStyles[section.name] = project.sectionStyles[section.name];
  });
  // A section takes the new render when asked for, edited since its take, or never rendered.
  const comped = new Set(sections.filter((section, index) =>
    !targets || targets.includes(section.name) || isEdited(section, index) || !takeById(project.comp[section.name])
  ).map((section) => section.name));
  // Untouched sections replay the tokens of their chosen take, so a new ending is matched to what plays next.
  const kept = Object.fromEntries(sections.filter((section) => !comped.has(section.name))
    .map((section) => [section.name, takeById(project.comp[section.name])?.sectionTokens?.[section.name]])
    .filter(([, tokens]) => tokens));
  const { prompt, saves } = buildPrompt(`HZ3-Studio/${project.name}/take-${id}`, kept);
  const owner = project.name;
  await queue(prompt, RENDER_LABELS, async (outputs) => {
    for (const [node, track] of Object.entries(saves)) {
      const file = outputs[node]?.audio?.[0];
      if (file) take.files[track] = file;
    }
    if (!take.files.vocals) throw new Error("La generación terminó sin audio.");
    take.sectionTokens = outputs[2]?.section_tokens?.[0] ?? {};
    const current = await updateProject(owner, (target) => {
      target.takes.push(take);
      for (const name of comped) target.comp[name] = take.id;
    });
    if (!current) return;
    await loadTake(take);
    status(`Take ${take.id} listo · ${[...comped].join(", ")}`, 1);
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
  await Promise.all(Object.entries(take.files).filter(([track]) => track !== "mix").map(([, file]) => loadUrl(fileUrl(file))));
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

function sectionGains(track, takeId) {
  return sections.map((section) => (project.comp[section.name] === takeId ? 1 : 0)
    * (trackEnabled(track, section) ? 1 : 0));
}

function scheduleGain(param, values, start, from) {
  const at = (time) => start + time - from;
  const current = Math.max(0, sections.findLastIndex((section) => section.start <= from));
  param.setValueAtTime(values[current] ?? 0, start);
  for (let index = 1; index < sections.length; index++) {
    if (values[index] === values[index - 1]) continue;
    const boundary = sections[index].start;
    if (boundary + FADE / 2 <= from) continue;
    const fadeStart = Math.max(at(boundary - FADE / 2), start);
    param.setValueAtTime(values[index - 1], fadeStart);
    param.linearRampToValueAtTime(values[index], Math.max(at(boundary + FADE / 2), fadeStart + 0.001));
  }
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
  out.gain.value = 2;
  limiter.connect(into).connect(ceiling).connect(out).connect(destination);
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
  // A voice plays section by section, each read at its own measured shift.
  const connectVoice = (voice, trackGain) => {
    const entry = voice.file && buffers.get(fileUrl(voice.file));
    if (!entry) return;
    const at = (time) => start + time - from;
    for (const section of sections) {
      if (!voiceEnabled(voice, section) || section.end + FADE / 2 <= from) continue;
      const begin = Math.max(section.start - FADE / 2, from);
      const end = section.end + FADE / 2;
      const offset = begin + voiceShift(voice, section);
      if (offset >= entry.buffer.duration) continue;
      const source = target.createBufferSource();
      source.buffer = entry.buffer;
      const gain = target.createGain();
      const level = voice.sectionGains?.[section.name] ?? 1;
      gain.gain.setValueAtTime(0, at(begin));
      gain.gain.linearRampToValueAtTime(level, at(Math.min(begin + FADE, end)));
      gain.gain.setValueAtTime(level, at(Math.max(end - FADE, begin + FADE)));
      gain.gain.linearRampToValueAtTime(0, at(end));
      source.connect(gain).connect(trackGain);
      source.start(at(begin) + Math.max(0, -offset), Math.max(0, offset), end - begin);
      sources.push(source);
    }
  };
  for (const track of allTracks()) {
    const trackGain = target.createGain();
    trackGain.gain.value = mixerGain(track);
    trackGain.connect(limiter);
    trackGains[track.id] = trackGain;
    if (track.source) connect(sourceEntry(), trackGain, null);
    else if (track.voice) connectVoice(track.voice, trackGain);
    else for (const takeId of takeIds) connect(trackBuffer(track.id, takeId), trackGain, sectionGains(track.id, takeId));
  }
  return { sources, trackGains };
}

function voiceEnabled(voice, section) {
  return voice.on[section.name] ?? !/^\(?instrumental\)?$/i.test(section.lyrics.trim() || "instrumental");
}

function voiceShift(voice, section) {
  return voice.offset + (voice.sectionOffsets[section.name] ?? 0);
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
  const blob = new Blob([wav(rendered)], { type: "audio/wav" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${project.name || "hz3-studio"}.wav`;
  link.click();
  URL.revokeObjectURL(link.href);
  status("Mezcla exportada.", 1);
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
  if (!duration()) { updatePlayhead(); drawInspector(); drawVoiceInspector(); return; }
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
  if (score) drawScore();
  for (const track of allTracks()) drawTrack(track);
  updatePlayhead();
  drawInspector();
  drawVoiceInspector();
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
    block.onclick = () => { selected = section.name; selectedVoice = null; draw(); };
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
  const head = document.createElement("span");
  head.className = "name";
  head.textContent = `Partitura · ${score.key} · ${score.bpm} BPM`;
  const { body } = lane("score", head);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(duration() * pxPerSecond));
  canvas.height = 119;
  body.append(canvas);
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
  head.innerHTML = `<span class="name">${track.name}</span><span class="controls">
    <button data-action="mute" class="${muted(track) ? "active" : ""}" title="Silenciar">M</button>
    <button data-action="solo" class="${project.mixer[track.id]?.solo ? "active" : ""}" title="Solo">S</button>
    <input type="range" min="0" max="1.5" step="0.01" value="${project.mixer[track.id]?.gain ?? track.gain}" title="Volumen"></span>`;
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
  const { body } = lane("track", head);
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
    name.onclick = () => { selectedVoice = voice.id; selected = null; draw(); };
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
      (project.trackOn[track.id] ??= {})[section.name] = !enabled;
      draw();
      restartIfPlaying();
    };
    body.append(toggle);
  });
}

function applyMixer() {
  if (!playback) return;
  for (const track of allTracks()) playback.trackGains[track.id].gain.value = mixerGain(track);
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
  if (document.activeElement !== $("section-style")) $("section-style").value = project.sectionStyles[section.name] ?? "";
  $("section-classical").value = project.classicalPlan[section.name] ?? "";
  if (document.activeElement !== $("section-seed")) $("section-seed").value = sectionSeed(section, index);
  $("section-merge").disabled = index === sections.length - 1;
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
  await applySections(edited, { [from]: name });
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
  const edited = editableSections();
  const [next] = edited.splice(index + 1, 1);
  edited[index].end_bar = next.end_bar;
  edited[index].lyrics = [edited[index].lyrics.trim(), next.lyrics.trim()].filter(Boolean).join("\n");
  await applySections(edited);
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
    file: null, offset: 0, sectionOffsets: {}, on: {},
  });
  selectedVoice = id;
  selected = null;
  draw();
}

async function renderVoice(voice) {
  readForm();
  if (!project.name) throw new Error("Ponle nombre al proyecto antes de generar.");
  await refreshSections();
  const octave = Number(voice.octave) || 0;
  // Harmony lines and octave moves come from Vocal Harmony (melody-only ABC); the plain lead keeps the song ABC.
  const line = VOICE_LINES[voice.source] ?? (octave ? 0 : undefined);
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "HZ3_YuE2_GenerateMusicSections",
      inputs: {
        clip: ["1", 1], style: voice.style, lyrics: project.lyrics, abc: line !== undefined ? ["20", line] : project.abc, seed: voice.seed,
        mode: line !== undefined ? "melody" : project.mode, ...project.sampling, section_seeds: "", section_styles: "",
      },
    },
    3: { class_type: "EmptyYuE2LatentAudio", inputs: { seconds: ["2", 1], batch_size: 1 } },
    4: {
      class_type: "KSampler",
      inputs: {
        model: ["1", 0], seed: 42, steps: 79, cfg: 4.2, sampler_name: "dpm_2", scheduler: "sgm_uniform",
        positive: ["2", 0], negative: ["2", 0], latent_image: ["3", 0], denoise: 1,
      },
    },
    5: { class_type: "VAEDecodeAudioTiled", inputs: { samples: ["4", 0], vae: ["1", 2], tile_size: 256, overlap: 32 } },
    7: { class_type: "AudioSeparation", inputs: { audio: ["5", 0] } },
    8: { class_type: "SaveAudio", inputs: { audio: ["7", 3], filename_prefix: `HZ3-Studio/${project.name}/voice-${voice.id}/vocals` } },
  };
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
      if (stored) stored.file = file;
    });
    if (!current) return;  // aligned when that project is open again ("Alinear con la voz principal")
    await loadUrl(fileUrl(file));
    await alignVoice(project.voices.find((item) => item.id === voice.id));
    await saveProject();
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

async function importReferences(files, license) {
  const { projects } = await api("/hz3/studio/projects");
  const taken = new Set(projects.map(({ name }) => name));
  for (const file of files) {
    let name = file.name.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N} _()-]+/gu, "_").slice(0, 56).trim() || "referencia";
    for (let count = 2; taken.has(name); count++) name = `${name.replace(/ \(\d+\)$/, "")} (${count})`;
    taken.add(name);
    const form = new FormData();
    form.append("file", file);
    const source = await api(`/hz3/studio/source?project=${encodeURIComponent(name)}`, { method: "POST", body: form });
    await postJson("/hz3/studio/project", { ...newProject(), name, kind: "reference", license, source });
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

async function saveProject() {
  readForm();
  if (!project.name) throw new Error("Ponle nombre al proyecto.");
  // A new name on an opened project renames its package instead of copying it.
  if (savedName && savedName !== project.name) await postJson("/hz3/studio/rename", { from: savedName, to: project.name });
  await postJson("/hz3/studio/project", project);
  savedName = project.name;
  await listProjects();
  status(`Proyecto «${project.name}» guardado.`, 1);
}

async function listProjects() {
  const { projects } = await api("/hz3/studio/projects");
  const escape = (text) => String(text ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  $("project-list").innerHTML = `<option value="">— proyectos —</option>`
    + projects.map(({ name }) => `<option${name === project.name ? " selected" : ""}>${escape(name)}</option>`).join("");
  $("catalog-rows").innerHTML = projects.map((entry) => `<tr>
    <td><button data-name="${escape(entry.name)}">${escape(entry.name)}</button></td>
    <td>${new Date(entry.updated * 1000).toLocaleString()}</td>
    <td>${entry.kind === "reference" ? "Referencia" : "Canción"}</td>
    <td>${entry.source ? `Audio: ${escape(entry.source)}` : "Compuesta"}${entry.license ? `<br><small>${escape(entry.license.name)}${entry.license.work ? ` · ${escape(entry.license.work)}` : ""}</small>` : ""}</td>
    <td>${entry.sections}</td><td>${entry.takes}</td><td>${entry.voices}</td>
    <td class="style">${escape(entry.style)}</td></tr>`).join("");
  $("catalog-rows").querySelectorAll("button").forEach((button) => {
    button.onclick = guard(async () => {
      $("catalog").close();
      await openProject(button.dataset.name);
    });
  });
}

async function openProject(name) {
  pause();
  pausedAt = 0;
  project = { ...newProject(), ...(await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`)) };
  savedName = project.name;
  selected = null;
  writeForm();
  await refreshSections();
  status("Cargando audio…");
  selectedVoice = null;
  for (const voice of project.voices) {
    voice.octave ??= 0;
    voice.role ??= "-4";
  }
  await Promise.all([loadSource(), ...project.takes.map(loadTake), ...project.voices.filter((voice) => voice.file).map((voice) => loadUrl(fileUrl(voice.file)))]);
  status(`Proyecto «${project.name}» · ${project.takes.length} takes`, 1);
  draw();
}

// ---------- wiring

function guard(action) {
  return (...args) => Promise.resolve(action(...args)).catch((error) => status(error.message, null, true));
}

async function init() {
  const info = await api("/object_info/CheckpointLoaderSimple");
  const names = info.CheckpointLoaderSimple.input.required.ckpt_name[0];
  $("ckpt").innerHTML = names.map((name) => `<option>${name}</option>`).join("");
  writeForm();
  await listProjects();
  connectSocket();

  for (const id of ["lyrics", "abc"]) $(id).addEventListener("input", scheduleSections);
  $("project-list").onchange = guard((event) => event.target.value && openProject(event.target.value));
  $("save-project").onclick = guard(saveProject);
  $("open-catalog").onclick = guard(async () => { await listProjects(); $("catalog").showModal(); });
  $("close-catalog").onclick = () => $("catalog").close();
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
  $("analyze").onclick = guard(analyze);
  $("compose").onclick = guard(compose);
  $("render-song").onclick = guard(() => render(null));
  $("render-section").onclick = guard(() => render([selected]));
  $("export-mix").onclick = guard(exportMix);
  $("arrange").onclick = guard(arrange);
  $("play").onclick = guard(play);
  $("stop").onclick = () => { pause(); pausedAt = 0; updatePlayhead(); };
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
    await listProjects();
    await openProject(name);
  });
  $("download-package").onclick = guard(async () => {
    await saveProject();
    location.href = `/hz3/studio/package?name=${encodeURIComponent(project.name)}`;
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
  $("section-style").addEventListener("change", (event) => {
    if (event.target.value.trim()) project.sectionStyles[selected] = event.target.value.trim();
    else delete project.sectionStyles[selected];
    draw();
  });
  $("section-styles-from-cues").onclick = guard(stylesFromCues);
  $("apply-classical").onclick = guard(applyClassical);
  $("section-classical").addEventListener("change", (event) => { project.classicalPlan[selected] = event.target.value; });
  $("add-voice").onclick = addVoice;
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
