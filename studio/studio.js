"use strict";

// HZ3 Studio: song timeline over ComfyUI. Every render is a "take" of the whole
// song; each section plays from the take chosen for it (comping), with short
// crossfades at section boundaries. Harmony voices are tracks switched per section.

const TRACKS = [
  { id: "vocals", name: "Voz principal", gain: 1 },
  { id: "instrumental", name: "Instrumental", gain: 1 },
  { id: "tenor", name: "Tenor", gain: 0.4, harmony: true },
  { id: "baritone", name: "Barítono", gain: 0.4, harmony: true },
  { id: "low", name: "Grave", gain: 0.3, harmony: true },
  { id: "bass", name: "Bajo", gain: 0.35, harmony: true },
  { id: "countertenor", name: "Contratenor", gain: 0.22, harmony: true },
];
const VOICES = TRACKS.filter((track) => track.harmony).map((track) => track.id);
const FADE = 0.3;
const PEAKS_PER_SECOND = 100;
const NODE_LABELS = { 1: "Cargando modelo", 2: "Tokens YuE2", 4: "KSampler", 5: "Decodificando audio", 7: "Separando voz", 12: "Armonías" };

const $ = (id) => document.getElementById(id);
const clientId = crypto.randomUUID();

let project = newProject();
let sections = [];
let selected = null;
let pxPerSecond = 6;
const buffers = new Map();
const pending = new Map();
let audioContext = null;
let playback = null;
let pausedAt = 0;

function newProject() {
  return {
    name: "", style: "", lyrics: "", abc: "", seed: 60, mode: "full",
    ckpt: "yue2_3b_int8_convrot.safetensors",
    sampling: { temperature: 0.9, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, cfg_scale: 2.0 },
    harmonize: true, sectionSeeds: {}, takes: [], comp: {}, harmonyOn: {}, mixer: {},
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

// ---------- form <-> project

const SAMPLING = ["temperature", "top_p", "top_k", "repetition_penalty", "cfg_scale"];

function readForm() {
  project.name = $("project-name").value.trim();
  for (const key of ["style", "lyrics", "abc", "mode", "ckpt"]) project[key] = $(key).value;
  project.seed = Number($("seed").value) || 0;
  project.harmonize = $("harmonize").checked;
  for (const key of SAMPLING) project.sampling[key] = Number($(key).value);
}

function writeForm() {
  $("project-name").value = project.name;
  for (const key of ["style", "lyrics", "abc", "mode"]) $(key).value = project[key];
  if ([...$("ckpt").options].some((option) => option.value === project.ckpt)) $("ckpt").value = project.ckpt;
  $("seed").value = project.seed;
  $("harmonize").checked = project.harmonize;
  for (const key of SAMPLING) $(key).value = project.sampling[key];
}

// ---------- sections and lyrics

let sectionsTimer = null;
function scheduleSections() {
  clearTimeout(sectionsTimer);
  sectionsTimer = setTimeout(() => refreshSections().catch((error) => status(error.message, null, true)), 400);
}

async function refreshSections() {
  readForm();
  if (!project.abc.trim() || !project.lyrics.trim()) { sections = []; draw(); return; }
  const data = await api("/hz3/studio/sections", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ abc: project.abc, lyrics: project.lyrics }),
  });
  sections = data.sections;
  if (selected && !sections.some((section) => section.name === selected)) selected = null;
  status(`${sections.length} secciones · ${fmt(duration())}`);
  draw();
}

function duration() {
  return sections.length ? sections[sections.length - 1].end : 0;
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
    || take.sectionSeeds[section.name] !== sectionSeed(section, index);
}

function harmonyEnabled(voice, section) {
  const value = project.harmonyOn[voice]?.[section.name];
  return value ?? /^(chorus|coro|bridge|puente)/i.test(section.name);
}

// ---------- rendering through ComfyUI

function buildPrompt(prefix) {
  const overrides = Object.entries(project.sectionSeeds).map(([name, seed]) => `${name} = ${seed}`).join("\n");
  const prompt = {
    1: { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: project.ckpt } },
    2: {
      class_type: "HZ3_YuE2_GenerateMusicSections",
      inputs: {
        clip: ["1", 1], style: project.style, lyrics: project.lyrics, abc: project.abc, seed: project.seed,
        mode: project.mode, ...project.sampling, section_seeds: overrides,
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
  sections.forEach((section, index) => {
    take.sectionLyrics[section.name] = section.lyrics;
    take.sectionSeeds[section.name] = sectionSeed(section, index);
  });
  // A section takes the new render when asked for, edited since its take, or never rendered.
  const comped = new Set(sections.filter((section, index) =>
    !targets || targets.includes(section.name) || isEdited(section, index) || !takeById(project.comp[section.name])
  ).map((section) => section.name));
  const { prompt, saves } = buildPrompt(`HZ3-Studio/${project.name}/take-${id}`);
  const data = await api("/prompt", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  pending.set(data.prompt_id, { take, saves, comped });
  status(`Take ${id} en cola…`, 0);
  setBusy(true);
}

async function finishRender(promptId) {
  const job = pending.get(promptId);
  pending.delete(promptId);
  setBusy(pending.size > 0);
  const history = await api(`/history/${promptId}`);
  const outputs = history[promptId]?.outputs ?? {};
  for (const [node, track] of Object.entries(job.saves)) {
    const file = outputs[node]?.audio?.[0];
    if (file) job.take.files[track] = file;
  }
  if (!job.take.files.vocals) throw new Error("La generación terminó sin audio.");
  project.takes.push(job.take);
  for (const name of job.comped) project.comp[name] = job.take.id;
  await saveProject();
  await loadTake(job.take);
  status(`Take ${job.take.id} listo · ${[...job.comped].join(", ")}`, 1);
  draw();
}

function setBusy(busy) {
  $("render-song").disabled = busy;
  $("render-section").disabled = busy;
}

function connectSocket() {
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?clientId=${clientId}`);
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    const { type, data } = JSON.parse(event.data);
    if (!data?.prompt_id || !pending.has(data.prompt_id)) return;
    if (type === "progress") {
      status(`${NODE_LABELS[data.node] ?? "Procesando"} · ${data.value}/${data.max}`, data.value / data.max);
    } else if (type === "executing" && data.node) {
      status(`${NODE_LABELS[data.node] ?? "Procesando"}…`);
    } else if (type === "execution_success") {
      finishRender(data.prompt_id).catch((error) => status(error.message, null, true));
    } else if (type === "execution_error") {
      pending.delete(data.prompt_id);
      setBusy(pending.size > 0);
      status(`Error en ${data.node_type}: ${data.exception_message}`, null, true);
    } else if (type === "execution_interrupted") {
      pending.delete(data.prompt_id);
      setBusy(pending.size > 0);
      status("Generación interrumpida.", null, true);
    }
  };
  socket.onclose = () => setTimeout(connectSocket, 2000);
}

// ---------- audio

function fileUrl(file) {
  const query = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder, type: file.type });
  return `/view?${query}`;
}

function context() {
  audioContext ??= new AudioContext();
  return audioContext;
}

async function loadTake(take) {
  await Promise.all(Object.entries(take.files).filter(([track]) => track !== "mix").map(async ([, file]) => {
    const url = fileUrl(file);
    if (buffers.has(url)) return;
    const response = await fetch(url);
    if (!response.ok) return;
    const buffer = await context().decodeAudioData(await response.arrayBuffer());
    buffers.set(url, { buffer, peaks: peaks(buffer) });
  }));
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
    * (VOICES.includes(track) && !harmonyEnabled(track, section) ? 0 : 1));
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

function mixerGain(track) {
  const mixer = project.mixer[track.id] ?? {};
  const soloed = TRACKS.some((other) => project.mixer[other.id]?.solo);
  if (mixer.mute || (soloed && !mixer.solo)) return 0;
  return mixer.gain ?? track.gain;
}

function buildGraph(target, destination, from, start) {
  const sources = [];
  const trackGains = {};
  const takeIds = new Set(Object.values(project.comp));
  for (const track of TRACKS) {
    const trackGain = target.createGain();
    trackGain.gain.value = mixerGain(track);
    trackGain.connect(destination);
    trackGains[track.id] = trackGain;
    for (const takeId of takeIds) {
      const entry = trackBuffer(track.id, takeId);
      if (!entry || from >= entry.buffer.duration) continue;
      const source = target.createBufferSource();
      source.buffer = entry.buffer;
      const gain = target.createGain();
      scheduleGain(gain.gain, sectionGains(track.id, takeId), start, from);
      source.connect(gain).connect(trackGain);
      source.start(start, from);
      sources.push(source);
    }
  }
  return { sources, trackGains };
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

function updatePlayhead() {
  const head = document.querySelector(".lane-head")?.offsetWidth ?? 0;
  $("playhead").style.left = `${head + position() * pxPerSecond}px`;
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
  if (!sections.length) { updatePlayhead(); drawInspector(); return; }
  const ruler = lane("ruler", "m:ss");
  const step = pxPerSecond >= 20 ? 5 : pxPerSecond >= 8 ? 10 : 30;
  for (let time = 0; time < duration(); time += step) {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.left = `${time * pxPerSecond}px`;
    tick.textContent = fmt(time);
    ruler.body.append(tick);
  }
  ruler.body.onclick = (event) => seek(event.offsetX / pxPerSecond);

  const sectionLane = lane("sections", "Secciones");
  sections.forEach((section, index) => {
    const block = document.createElement("div");
    block.className = `section-block${section.name === selected ? " selected" : ""}`;
    block.style.left = `${section.start * pxPerSecond}px`;
    block.style.width = `${(section.end - section.start) * pxPerSecond}px`;
    block.title = `${section.name} · ${fmt(section.start)}–${fmt(section.end)}`;
    block.textContent = section.name;
    const takeId = project.comp[section.name];
    if (takeId) block.insertAdjacentHTML("beforeend", `<span class="take">T${takeId}</span>`);
    if (isEdited(section, index)) block.insertAdjacentHTML("beforeend", `<span class="edited">editado</span>`);
    block.onclick = () => { selected = section.name; draw(); };
    sectionLane.body.append(block);
  });

  for (const track of TRACKS) drawTrack(track);
  updatePlayhead();
  drawInspector();
}

function drawTrack(track) {
  const head = document.createElement("div");
  const mixer = project.mixer[track.id] ?? {};
  head.innerHTML = `<span class="name">${track.name}</span><span class="controls">
    <button data-action="mute" class="${mixer.mute ? "active" : ""}" title="Silenciar">M</button>
    <button data-action="solo" class="${mixer.solo ? "active" : ""}" title="Solo">S</button>
    <input type="range" min="0" max="1.5" step="0.01" value="${mixer.gain ?? track.gain}" title="Volumen"></span>`;
  head.querySelectorAll("button").forEach((button) => {
    button.onclick = () => {
      const entry = project.mixer[track.id] ??= {};
      entry[button.dataset.action] = !entry[button.dataset.action];
      applyMixer();
      draw();
    };
  });
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
  sections.forEach((section) => {
    const entry = trackBuffer(track.id, project.comp[section.name]);
    const enabled = !track.harmony || harmonyEnabled(track.id, section);
    graphics.fillStyle = enabled ? (track.harmony ? "#4aa3e0" : "#c9a14a") : "#3a3d45";
    if (entry) {
      for (let x = Math.floor(section.start * pxPerSecond); x < section.end * pxPerSecond; x++) {
        const from = Math.floor((x / pxPerSecond) * PEAKS_PER_SECOND);
        const to = Math.min(entry.peaks.length, Math.floor(((x + 1) / pxPerSecond) * PEAKS_PER_SECOND));
        let peak = 0;
        for (let index = from; index < to; index++) peak = Math.max(peak, entry.peaks[index]);
        const height = Math.min(1, peak) * (middle - 2);
        graphics.fillRect(x, middle - height, 1, height * 2 || 1);
      }
    }
    if (track.harmony) {
      const toggle = document.createElement("div");
      toggle.className = `toggle${enabled ? "" : " off"}`;
      toggle.style.left = `${section.start * pxPerSecond}px`;
      toggle.style.width = `${(section.end - section.start) * pxPerSecond}px`;
      toggle.title = `${track.name} · ${section.name}: ${enabled ? "encendido" : "apagado"} (clic para cambiar)`;
      toggle.onclick = () => {
        (project.harmonyOn[track.id] ??= {})[section.name] = !enabled;
        draw();
        restartIfPlaying();
      };
      body.append(toggle);
    }
  });
}

function applyMixer() {
  if (!playback) return;
  for (const track of TRACKS) playback.trackGains[track.id].gain.value = mixerGain(track);
}

function drawInspector() {
  const index = sections.findIndex((section) => section.name === selected);
  $("inspector").classList.toggle("hidden", index < 0);
  if (index < 0) return;
  const section = sections[index];
  $("inspector-title").textContent = `${section.name} · ${fmt(section.start)}–${fmt(section.end)}`;
  if (document.activeElement !== $("section-lyrics")) $("section-lyrics").value = section.lyrics;
  if (document.activeElement !== $("section-seed")) $("section-seed").value = sectionSeed(section, index);
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

// ---------- projects

async function saveProject() {
  readForm();
  if (!project.name) throw new Error("Ponle nombre al proyecto.");
  await api("/hz3/studio/project", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(project) });
  await listProjects();
  status(`Proyecto «${project.name}» guardado.`, 1);
}

async function listProjects() {
  const { projects } = await api("/hz3/studio/projects");
  $("project-list").innerHTML = `<option value="">— proyectos —</option>`
    + projects.map((name) => `<option${name === project.name ? " selected" : ""}>${name.replace(/</g, "&lt;")}</option>`).join("");
}

async function openProject(name) {
  pause();
  pausedAt = 0;
  project = { ...newProject(), ...(await api(`/hz3/studio/project?name=${encodeURIComponent(name)}`)) };
  selected = null;
  writeForm();
  await refreshSections();
  status("Cargando audio…");
  await Promise.all(project.takes.map(loadTake));
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
  $("toggle-song").onclick = () => document.querySelector(".song").classList.toggle("hidden");
  $("render-song").onclick = guard(() => render(null));
  $("render-section").onclick = guard(() => render([selected]));
  $("export-mix").onclick = guard(exportMix);
  $("play").onclick = guard(play);
  $("stop").onclick = () => { pause(); pausedAt = 0; updatePlayhead(); };
  $("zoom").oninput = (event) => { pxPerSecond = Number(event.target.value); draw(); };
  $("import-json").onchange = guard(async (event) => {
    const data = JSON.parse(await event.target.files[0].text());
    for (const key of ["style", "lyrics", "abc"]) if (typeof data[key] === "string") $(key).value = data[key];
    event.target.value = "";
    await refreshSections();
  });
  $("section-lyrics").addEventListener("input", guard((event) => {
    setSectionLyrics(sections.findIndex((section) => section.name === selected), event.target.value);
    scheduleSections();
  }));
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
