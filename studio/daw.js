// HZ3 Studio DAW: SoundFont instruments written as notes (piano roll or patterns from the ABC) and per-track
// effects. Everything here is deterministic: the same notes and settings always render the same audio, in playback
// and in the exported mix. Loaded before studio.js and shares its globals (project, score, sections, …).

const GM_PROGRAMS = [
  "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
  "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
  "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
  "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
  "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
  "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
  "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice", "Orchestra Hit",
  "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
  "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet",
  "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
  "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
  "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
  "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
  "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
  "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
  "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause", "Gunshot",
];
const INSTRUMENT_PATTERNS = {
  bass: "Bajo: raíz en cada medio compás", bass_walk: "Bajo caminante (negras)", pad: "Pad: acordes largos",
  comping: "Acordes en los tiempos 2 y 4", arpeggio: "Arpegio en corcheas", drums: "Batería básica (bombo 1-3, caja 2-4, hi-hat)",
  melody: "Melodía de la voz (doblaje)", ins: "Línea instrumental del ABC",
};
// A default sound per pattern, so a new part sounds right without choosing one.
const PATTERN_PROGRAMS = { bass: 33, bass_walk: 32, pad: 89, comping: 0, arpeggio: 46, drums: 0, melody: 73, ins: 48 };
const SYNTH_RATE = 48000;
const GRIDS = { 64: "1/16", 128: "1/8", 256: "1/4", 512: "1/2" };
const ROLL_HEIGHT = 119;

let selectedTrack = null;  // a track id (or "master") whose inspector is open
let fluidSynthLoaded = null;
let soundfontBytes = null;
const instrumentAudio = new Map();  // instrument id -> { key, entry }
let instrumentsTimer = null;

function newInstrument(fields = {}) {
  const id = Math.max(0, ...project.instruments.map((item) => item.id)) + 1;
  return { id, name: `Instrumento ${id}`, program: 0, drums: false, notes: [], grid: 128, low: null, ...fields };
}

function instrumentTracks() {
  return project.instruments.map((instrument) => ({ id: `inst-${instrument.id}`, name: instrument.name, gain: 0.8, instrument }));
}

function ticksToSeconds(ticks) {
  return ticks / TICKS_PER_QUARTER * 60 / score.bpm;
}

function secondsToTicks(seconds) {
  return seconds * score.bpm / 60 * TICKS_PER_QUARTER;
}

// ---------- SoundFont rendering (FluidSynth compiled to WebAssembly)

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = Object.assign(document.createElement("script"), { src, onload: resolve, onerror: () => reject(new Error(`No se pudo cargar ${src}`)) });
    document.head.append(script);
  });
}

async function loadFluidSynth() {
  fluidSynthLoaded ??= (async () => {
    await loadScript("/hz3/studio/static/vendor/libfluidsynth-2.4.6.js");
    await loadScript("/hz3/studio/static/vendor/js-synthesizer.min.js");
    await JSSynth.waitForReady();
    const response = await fetch("/hz3/studio/soundfont");
    if (!response.ok) throw new Error(await response.text());
    soundfontBytes = await response.arrayBuffer();
  })();
  try {
    await fluidSynthLoaded;
  } catch (error) {
    fluidSynthLoaded = null;
    throw error;
  }
}

function instrumentKey(instrument) {
  return JSON.stringify([instrument.program, instrument.drums, score?.bpm, instrument.notes]);
}

// A fresh synthesizer per render, without FluidSynth's own reverb and chorus (the track sends add them), so the
// audio depends only on the notes, the sound and the tempo.
async function renderInstrument(instrument) {
  await loadFluidSynth();
  const synth = new JSSynth.Synthesizer();
  synth.init(SYNTH_RATE, { reverbActive: false, chorusActive: false, initialGain: 1.0 });
  try {
    await synth.loadSFont(soundfontBytes.slice(0));
    const channel = instrument.drums ? 9 : 0;
    if (!instrument.drums) synth.midiProgramChange(channel, instrument.program);
    const events = [];
    for (const note of instrument.notes) {
      events.push({ at: Math.round(ticksToSeconds(note.start) * SYNTH_RATE), on: true, note });
      events.push({ at: Math.round(ticksToSeconds(note.start + note.duration) * SYNTH_RATE), on: false, note });
    }
    // Note-offs first at the same instant, so a repeated pitch restarts cleanly.
    events.sort((a, b) => a.at - b.at || a.on - b.on || a.note.pitch - b.note.pitch);
    const frames = Math.max(1, (events.at(-1)?.at ?? 0) + Math.round(2.5 * SYNTH_RATE));
    const left = new Float32Array(frames);
    const right = new Float32Array(frames);
    let cursor = 0;
    const renderTo = (frame) => {
      while (cursor < frame) {
        const count = Math.min(4096, frame - cursor);
        synth.render([left.subarray(cursor, cursor + count), right.subarray(cursor, cursor + count)]);
        cursor += count;
      }
    };
    for (const event of events) {
      renderTo(event.at);
      if (event.on) synth.midiNoteOn(channel, event.note.pitch, event.note.velocity);
      else synth.midiNoteOff(channel, event.note.pitch);
    }
    renderTo(frames);
    const buffer = context().createBuffer(2, frames, SYNTH_RATE);
    buffer.copyToChannel(left, 0);
    buffer.copyToChannel(right, 1);
    return { buffer, peaks: peaks(buffer) };
  } finally {
    synth.close();
  }
}

function instrumentEntry(instrument) {
  return instrumentAudio.get(instrument.id)?.entry;
}

// Renders every instrument whose notes, sound or tempo changed since its last render.
async function renderInstruments() {
  if (!score) return;
  const stale = project.instruments.filter((instrument) => instrumentAudio.get(instrument.id)?.key !== instrumentKey(instrument));
  if (!stale.length) return;
  status(`Renderizando ${stale.length} instrumento(s)…`, 0);
  for (const instrument of stale) {
    const key = instrumentKey(instrument);
    instrumentAudio.set(instrument.id, { key, entry: instrument.notes.length ? await renderInstrument(instrument) : null });
  }
  status("Instrumentos listos.", 1);
  draw();
  restartIfPlaying();
}

function instrumentsChanged() {
  draw();
  clearTimeout(instrumentsTimer);
  instrumentsTimer = setTimeout(guard(renderInstruments), 250);
}

// ---------- piano roll

function rollRange(instrument) {
  const low = instrument.low ?? (instrument.notes.length
    ? Math.max(0, Math.min(Math.min(...instrument.notes.map((note) => note.pitch)) - 2, 127 - 23))
    : instrument.drums ? 35 : 48);
  return { low, high: Math.min(127, low + 23) };
}

function drawPianoRoll(instrument, body) {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(duration() * pxPerSecond));
  canvas.height = ROLL_HEIGHT;
  body.append(canvas);
  if (!score) return;
  const { low, high } = rollRange(instrument);
  const row = ROLL_HEIGHT / (high - low + 1);
  const graphics = canvas.getContext("2d");
  const x = (ticks) => ticksToSeconds(ticks) * pxPerSecond;
  const y = (pitch) => (high - pitch) * row;
  for (let pitch = low; pitch <= high; pitch++) {
    if ([1, 3, 6, 8, 10].includes(pitch % 12)) {
      graphics.fillStyle = "#1b1d22";
      graphics.fillRect(0, y(pitch), canvas.width, row);
    }
    if (pitch % 12 === 0) {
      graphics.fillStyle = "#3a3d45";
      graphics.fillRect(0, y(pitch) + row - 1, canvas.width, 1);
    }
  }
  for (const bar of score.bars) {
    graphics.fillStyle = "#262930";
    graphics.fillRect(Math.round(x(bar.start)), 0, 1, ROLL_HEIGHT);
  }
  const muted = !instrumentAudio.get(instrument.id) || instrumentAudio.get(instrument.id).key !== instrumentKey(instrument);
  for (const note of instrument.notes) {
    if (note.pitch < low || note.pitch > high) continue;
    graphics.fillStyle = muted ? "#7a6a8a" : `hsl(${280 - note.velocity / 2}, 55%, ${45 + note.velocity / 8}%)`;
    graphics.fillRect(x(note.start), y(note.pitch) + 0.5, Math.max(2, x(note.duration) - 1), Math.max(2, row - 1));
  }
  graphics.font = "10px system-ui, sans-serif";
  graphics.fillStyle = "#6a6d78";
  for (let pitch = low; pitch <= high; pitch++) if (pitch % 12 === 0) graphics.fillText(`C${pitch / 12 - 1}`, 2, y(pitch) + row - 2);
  canvas.onmousedown = (event) => editRoll(event, instrument, canvas, { low, high, row });
  canvas.oncontextmenu = (event) => event.preventDefault();
}

// Click on empty space adds a note (drag to lengthen it); drag a note to move it, its right edge to resize it;
// right-click or double-click a note to delete it. Everything snaps to the instrument's grid.
function editRoll(event, instrument, canvas, { low, high, row }) {
  event.preventDefault();
  event.stopPropagation();
  const box = canvas.getBoundingClientRect();
  const grid = instrument.grid || 128;
  const tickAt = (clientX) => Math.max(0, secondsToTicks((clientX - box.left) / pxPerSecond));
  const pitchAt = (clientY) => Math.max(low, Math.min(high, high - Math.floor((clientY - box.top) / row)));
  const snap = (ticks) => Math.round(ticks / grid) * grid;
  const tick = tickAt(event.clientX);
  const pitch = pitchAt(event.clientY);
  const hit = instrument.notes.findLast((note) => note.pitch === pitch && note.start <= tick && tick < note.start + note.duration);
  if (hit && (event.button === 2 || event.detail >= 2)) {
    instrument.notes = instrument.notes.filter((note) => note !== hit);
    instrumentsChanged();
    return;
  }
  if (event.button !== 0) return;
  let note = hit;
  let mode = "move";
  if (!note) {
    note = { start: Math.floor(tick / grid) * grid, duration: grid, pitch, velocity: 96 };
    instrument.notes.push(note);
    mode = "resize";
  } else if (secondsToTicks((event.clientX - box.left) / pxPerSecond) > note.start + note.duration - secondsToTicks(6 / pxPerSecond)) {
    mode = "resize";
  }
  const origin = { start: note.start, pitch: note.pitch, tick, pitchAt: pitch };
  const move = (moved) => {
    if (mode === "resize") note.duration = Math.max(grid, snap(tickAt(moved.clientX) - note.start));
    else {
      note.start = Math.max(0, snap(origin.start + tickAt(moved.clientX) - origin.tick));
      note.pitch = Math.max(0, Math.min(127, origin.pitch + pitchAt(moved.clientY) - origin.pitchAt));
    }
    draw();
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    instrument.notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
    instrumentsChanged();
  };
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  draw();
}

// ---------- notes from the ABC

function sectionBars(names) {
  const edited = editableSections();
  return edited.filter((section) => !names.length || names.includes(section.name))
    .flatMap((section) => Array.from({ length: section.end_bar - section.start_bar + 1 }, (_, index) => section.start_bar + index));
}

// Writes a pattern into the given sections (all when empty), replacing what the instrument had there.
async function writePattern(instrument, { pattern, sections: names = [], octave = 0, velocity = 96 }) {
  if (!score) throw new Error("Hace falta un ABC con secciones.");
  const bars = sectionBars(names);
  const heard = Object.values(project.comp).map(takeById).find((take) => take?.heardAbc)?.heardAbc ?? "";
  const { notes, chords } = await postJson("/hz3/studio/pattern", { abc: project.abc, pattern, bars, octave, velocity, heard_abc: heard });
  const ranges = bars.map((bar) => [score.bars[bar].start, bar + 1 < score.bars.length ? score.bars[bar + 1].start : score.total_ticks]);
  const inside = (note) => ranges.some(([start, end]) => start <= note.start && note.start < end);
  instrument.notes = [...instrument.notes.filter((note) => !inside(note)), ...notes].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  if (pattern === "drums") instrument.drums = true;
  instrumentsChanged();
  const source = { ABC: "los acordes del ABC", heard: "los acordes que se oyen en el take", guessed: "acordes deducidos de la melodía" }[chords];
  return `${notes.length} notas de «${INSTRUMENT_PATTERNS[pattern]}» en ${names.length ? names.join(", ") : "toda la canción"} (con ${source})`;
}

// ---------- effects

const FX_DEFAULTS = { pan: 0, reverb: 0, delay: 0, fadeIn: 0, fadeOut: 0 };

function trackFx(trackId) {
  return { ...FX_DEFAULTS, ...project.mixer[trackId] };
}

function masterFx() {
  return { reverbSeconds: 2.4, delayBeats: 0.75, feedback: 0.35, ...project.fx };
}

// A room built from a fixed-seed noise burst: the same reverb on every render.
function impulse(target, seconds) {
  let seed = 0x9e3779b9;
  const random = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const length = Math.max(1, Math.round(seconds * target.sampleRate));
  const buffer = target.createBuffer(2, length, target.sampleRate);
  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < length; index++) data[index] = (random() * 2 - 1) * Math.pow(1 - index / length, 3) * 0.6;
  }
  return buffer;
}

// Shared reverb and delay returns for one graph (playback or export).
function effectBuses(target, output) {
  const fx = masterFx();
  const reverb = target.createConvolver();
  reverb.buffer = impulse(target, fx.reverbSeconds);
  reverb.connect(output);
  const delay = target.createDelay(4);
  delay.delayTime.value = Math.min(4, (score ? 60 / score.bpm : 0.5) * fx.delayBeats);
  const feedback = target.createGain();
  feedback.gain.value = Math.min(0.9, fx.feedback);
  const tone = target.createBiquadFilter();
  tone.type = "lowpass";
  tone.frequency.value = 4500;
  delay.connect(tone).connect(feedback).connect(delay);
  tone.connect(output);
  return { reverb, delay };
}

// track gain -> fades -> pan -> output, with sends to the reverb and delay buses.
function trackChain(target, track, trackGain, output, buses, start, from) {
  const fx = trackFx(track.id);
  const fade = target.createGain();
  const at = (time) => start + time - from;
  const level = (time) => Math.min(fx.fadeIn > 0 ? time / fx.fadeIn : 1, fx.fadeOut > 0 ? (duration() - time) / fx.fadeOut : 1, 1);
  fade.gain.setValueAtTime(Math.max(0, level(from)), start);
  for (const time of [fx.fadeIn, duration() - fx.fadeOut, duration()]) {
    if (time > from && (fx.fadeIn > 0 || fx.fadeOut > 0)) fade.gain.linearRampToValueAtTime(Math.max(0, level(time)), at(time));
  }
  const pan = target.createStereoPanner();
  pan.pan.value = fx.pan;
  trackGain.connect(fade).connect(pan).connect(output);
  for (const [bus, amount] of [[buses.reverb, fx.reverb], [buses.delay, fx.delay]]) {
    if (amount <= 0) continue;
    const send = target.createGain();
    send.gain.value = amount;
    pan.connect(send).connect(bus);
  }
}

// ---------- track inspector

function dbOf(gain) {
  return gain > 0 ? Math.round(20 * Math.log10(gain) * 10) / 10 : -60;
}

function drawTrackInspector() {
  const track = selectedTrack === "master" ? { id: "master", name: "Master (efectos)" } : allTracks().find((item) => item.id === selectedTrack);
  $("track-inspector").classList.toggle("hidden", !track || Boolean(selected) || Boolean(selectedVoice));
  if (!track || selected || selectedVoice) return;
  $("track-title").textContent = track.name;
  const master = track.id === "master";
  for (const id of ["track-mix", "track-offsets"]) $(id).classList.toggle("hidden", master);
  $("track-master").classList.toggle("hidden", !master);
  $("track-instrument").classList.toggle("hidden", !track.instrument);
  if (master) {
    const fx = masterFx();
    for (const [id, key] of [["fx-reverb-seconds", "reverbSeconds"], ["fx-delay-beats", "delayBeats"], ["fx-feedback", "feedback"]]) {
      if (document.activeElement !== $(id)) $(id).value = fx[key];
    }
    return;
  }
  const fx = trackFx(track.id);
  const fields = {
    "track-volume": dbOf(mixerLevel(track)), "track-pan": Math.round(fx.pan * 100), "track-reverb": Math.round(fx.reverb * 100),
    "track-delay": Math.round(fx.delay * 100), "track-fade-in": fx.fadeIn, "track-fade-out": fx.fadeOut,
  };
  for (const [id, value] of Object.entries(fields)) if (document.activeElement !== $(id)) $(id).value = value;
  // Take tracks can be nudged per section (the lead's timing against the band, say).
  const offsets = $("track-offsets");
  offsets.replaceChildren();
  if (!track.instrument && !track.source) {
    offsets.append(Object.assign(document.createElement("span"), { textContent: "Ajuste de tiempo por sección (ms; + = más tarde)" }));
    for (const section of sections) {
      const label = document.createElement("label");
      label.innerHTML = `${escapeHtml(section.name)} <input type="number" step="5" value="${Math.round((fx.offsets?.[section.name] ?? 0) * 1000)}">`;
      label.querySelector("input").onchange = (event) => {
        const entry = project.mixer[track.id] ??= {};
        entry.offsets = { ...entry.offsets, [section.name]: Number(event.target.value) / 1000 };
        restartIfPlaying();
      };
      offsets.append(label);
    }
  }
  $("track-tune").classList.toggle("hidden", track.id !== "vocals");
  if (!track.instrument) return;
  const instrument = track.instrument;
  if (document.activeElement !== $("instrument-name")) $("instrument-name").value = instrument.name;
  $("instrument-program").innerHTML = GM_PROGRAMS.map((name, index) => `<option value="${index}"${index === instrument.program ? " selected" : ""}>${index + 1}. ${name}</option>`).join("");
  $("instrument-program").disabled = instrument.drums;
  $("instrument-drums").checked = instrument.drums;
  $("instrument-grid").innerHTML = Object.entries(GRIDS).map(([ticks, label]) => `<option value="${ticks}"${Number(ticks) === instrument.grid ? " selected" : ""}>${label}</option>`).join("");
  if (!$("instrument-pattern").options.length) {
    $("instrument-pattern").innerHTML = Object.entries(INSTRUMENT_PATTERNS).map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
  }
  const chosen = new Set([...$("instrument-sections").querySelectorAll("input:checked")].map((input) => input.value));
  $("instrument-sections").innerHTML = sections.map((section) => `<label><input type="checkbox" value="${escapeHtml(section.name)}"${chosen.has(section.name) ? " checked" : ""}> ${escapeHtml(section.name)}</label>`).join("");
  $("instrument-count").textContent = `${instrument.notes.length} notas`;
}

function mixerLevel(track) {
  return project.mixer[track.id]?.gain ?? track.gain;
}

function selectedInstrument() {
  return allTracks().find((track) => track.id === selectedTrack)?.instrument;
}

function setTrackFx(trackId, changes) {
  Object.assign(project.mixer[trackId] ??= {}, changes);
  applyMixer();
  restartIfPlaying();
}

function wireTrackInspector() {
  $("close-track-inspector").onclick = () => { selectedTrack = null; draw(); };
  $("fx-master").onclick = () => { selectedTrack = "master"; selected = null; selectedVoice = null; draw(); };
  $("track-volume").onchange = (event) => setTrackFx(selectedTrack, { gain: Math.pow(10, Number(event.target.value) / 20) });
  $("track-pan").oninput = (event) => setTrackFx(selectedTrack, { pan: Number(event.target.value) / 100 });
  $("track-reverb").onchange = (event) => setTrackFx(selectedTrack, { reverb: Number(event.target.value) / 100 });
  $("track-delay").onchange = (event) => setTrackFx(selectedTrack, { delay: Number(event.target.value) / 100 });
  $("track-fade-in").onchange = (event) => setTrackFx(selectedTrack, { fadeIn: Math.max(0, Number(event.target.value)) });
  $("track-fade-out").onchange = (event) => setTrackFx(selectedTrack, { fadeOut: Math.max(0, Number(event.target.value)) });
  for (const [id, key] of [["fx-reverb-seconds", "reverbSeconds"], ["fx-delay-beats", "delayBeats"], ["fx-feedback", "feedback"]]) {
    $(id).onchange = (event) => { project.fx = { ...project.fx, [key]: Number(event.target.value) }; restartIfPlaying(); };
  }
  $("add-instrument").onclick = () => {
    const instrument = newInstrument();
    project.instruments.push(instrument);
    selectedTrack = `inst-${instrument.id}`;
    selected = null;
    selectedVoice = null;
    draw();
  };
  $("instrument-name").onchange = (event) => { selectedInstrument().name = event.target.value.trim() || selectedInstrument().name; draw(); };
  $("instrument-program").onchange = (event) => { selectedInstrument().program = Number(event.target.value); instrumentsChanged(); };
  $("instrument-drums").onchange = (event) => { selectedInstrument().drums = event.target.checked; instrumentsChanged(); };
  $("instrument-grid").onchange = (event) => { selectedInstrument().grid = Number(event.target.value); };
  $("instrument-up").onclick = () => { const instrument = selectedInstrument(); instrument.low = Math.min(104, rollRange(instrument).low + 12); draw(); };
  $("instrument-down").onclick = () => { const instrument = selectedInstrument(); instrument.low = Math.max(0, rollRange(instrument).low - 12); draw(); };
  $("instrument-write").onclick = guard(async () => {
    const instrument = selectedInstrument();
    const pattern = $("instrument-pattern").value;
    if (!instrument.notes.length && instrument.program === 0 && !instrument.drums) instrument.program = PATTERN_PROGRAMS[pattern];
    const names = [...$("instrument-sections").querySelectorAll("input:checked")].map((input) => input.value);
    status(await writePattern(instrument, { pattern, sections: names, octave: Number($("instrument-octave").value), velocity: Number($("instrument-velocity").value) }), 1);
  });
  $("instrument-clear").onclick = () => {
    const instrument = selectedInstrument();
    const names = [...$("instrument-sections").querySelectorAll("input:checked")].map((input) => input.value);
    const ranges = sectionBars(names).map((bar) => [score.bars[bar].start, bar + 1 < score.bars.length ? score.bars[bar + 1].start : score.total_ticks]);
    instrument.notes = instrument.notes.filter((note) => !ranges.some(([start, end]) => start <= note.start && note.start < end));
    instrumentsChanged();
  };
  $("tune-apply").onclick = guard(() => tuneVocals(Math.max(0, Math.min(1, Number($("tune-amount").value) / 100))));
  $("tune-undo").onclick = guard(untuneVocals);
  $("instrument-delete").onclick = () => {
    const instrument = selectedInstrument();
    project.instruments = project.instruments.filter((item) => item !== instrument);
    delete project.mixer[`inst-${instrument.id}`];
    instrumentAudio.delete(instrument.id);
    selectedTrack = null;
    draw();
    restartIfPlaying();
  };
}

// ---------- assistant tools

const NOTE_NAMES = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

function midiPitch(value) {
  if (Number.isFinite(Number(value))) return Number(value);
  const match = String(value).trim().match(/^([A-Ga-g])([#b]?)(-?\d)$/);
  if (!match) throw new Error(`altura «${value}» no válida (usa MIDI o nombres como C4, F#3)`);
  return 12 * (Number(match[3]) + 1) + NOTE_NAMES[match[1].toUpperCase()] + (match[2] === "#" ? 1 : match[2] === "b" ? -1 : 0);
}

function gmProgram(value) {
  if (Number.isFinite(Number(value))) return Math.max(0, Math.min(127, Number(value) - 1));
  const wanted = String(value).toLowerCase();
  const index = GM_PROGRAMS.findIndex((name) => name.toLowerCase().includes(wanted));
  if (index < 0) throw new Error(`no hay sonido General MIDI «${value}»`);
  return index;
}

// Called from studio.js, where the tool helpers live.
function dawTools() {
  const sound = str("General MIDI sound: number 1-128 or part of its name (e.g. 'Electric Bass (finger)', 'String Ensemble 1', 'nylon')");
  const names = { type: "array", items: { type: "string" }, description: "section names; omit or empty for the whole song" };
  const pattern = { type: "string", enum: Object.keys(INSTRUMENT_PATTERNS), description: Object.entries(INSTRUMENT_PATTERNS).map(([key, label]) => `${key}: ${label}`).join("; ") };
  return [
    tool("add_instrument", "Add a SoundFont instrument track (deterministic, not YuE2) and optionally write a pattern from the ABC into it.", {
      name: str("track name"), sound, drums: { type: "boolean", description: "General MIDI drum kit (channel 10)" }, pattern, sections: names,
      octave: num("octave shift of the pattern, -3..3"), velocity: num("1-127, default 96"),
    }, ["name"]),
    tool("write_pattern", "Write a pattern from the ABC's bars and chords into an instrument, replacing its notes in those sections.", {
      instrument: str("instrument name"), pattern, sections: names, octave: num("octave shift, -3..3"), velocity: num("1-127"),
    }, ["instrument", "pattern"]),
    tool("write_notes", "Write notes into an instrument by bar and beat (1-based; beats may be fractional, e.g. 2.5).", {
      instrument: str("instrument name"),
      notes: { type: "array", items: { type: "object", properties: {
        bar: num("bar, 1-based"), beat: num("beat in the bar, 1-based"), beats: num("length in beats"),
        pitch: str("MIDI number or name like C4, F#3; drums: 36 kick, 38 snare, 42 closed hi-hat, 49 crash"), velocity: num("1-127"),
      }, required: ["bar", "beat", "beats", "pitch"] } },
      replace: { type: "boolean", description: "first remove the instrument's notes in the bars written to" },
    }, ["instrument", "notes"]),
    tool("set_instrument", "Change an instrument's name or sound.", { instrument: str("instrument name"), name: str("new name"), sound, drums: { type: "boolean" } }, ["instrument"]),
    tool("remove_instrument", "Delete an instrument track.", { instrument: str("instrument name") }, ["instrument"]),
    tool("set_mix", "Set a track's mix: volume, pan, reverb and delay sends, fades. Only the given values change.", {
      track: str("vocals, instrumental, tenor, baritone, low, bass, countertenor, an extra voice or an instrument name"),
      volume_db: num("track volume in dB (0 = unity)"), pan: num("-100 left .. 100 right"), reverb: num("reverb send 0-100"), delay: num("delay send 0-100"),
      fade_in: num("seconds"), fade_out: num("seconds"),
    }, ["track"]),
    tool("set_master_effects", "Set the shared reverb and delay.", {
      reverb_seconds: num("reverb length, 0.2-8"), delay_beats: num("delay time in quarter notes (0.75 = dotted eighth)"), feedback: num("delay feedback 0-0.9"),
    }),
  ];
}

function dawState() {
  return {
    instruments: project.instruments.map((instrument) => ({
      name: instrument.name, sound: instrument.drums ? "drum kit" : GM_PROGRAMS[instrument.program], notes: instrument.notes.length,
    })),
    mix: Object.fromEntries(allTracks().map((track) => {
      const fx = trackFx(track.id);
      return [track.instrument || track.voice ? track.name : track.id, {
        volume_db: dbOf(mixerLevel(track)), muted: muted(track), pan: Math.round(fx.pan * 100), reverb: Math.round(fx.reverb * 100),
        delay: Math.round(fx.delay * 100), fade_in: fx.fadeIn, fade_out: fx.fadeOut,
      }];
    })),
    master_effects: masterFx(),
  };
}

// The DAW's assistant actions; undefined when the action is not one of them.
async function runDawAction(action) {
  const instrumentNamed = (name) => {
    const found = project.instruments.find((item) => item.name.toLowerCase() === String(name ?? "").toLowerCase());
    if (!found) throw new Error(`no hay instrumento «${name}»`);
    return found;
  };
  const sectionList = (list) => (Array.isArray(list) ? list : []).map((name) => {
    const found = sections.find((section) => section.name.toLowerCase() === String(name).toLowerCase());
    if (!found) throw new Error(`no hay sección «${name}»`);
    return found.name;
  });
  const given = (key) => action[key] !== undefined && action[key] !== null && Number.isFinite(Number(action[key]));
  const patternOptions = () => ({
    pattern: action.pattern, sections: sectionList(action.sections), octave: given("octave") ? Number(action.octave) : 0, velocity: Number(action.velocity) || 96,
  });
  switch (action.type) {
    case "add_instrument": {
      const instrument = newInstrument({ name: String(action.name).slice(0, 40), drums: Boolean(action.drums) || action.pattern === "drums" });
      instrument.program = action.sound !== undefined ? gmProgram(action.sound) : PATTERN_PROGRAMS[action.pattern] ?? 0;
      project.instruments.push(instrument);
      const written = action.pattern ? await writePattern(instrument, patternOptions()) : "sin notas todavía";
      instrumentsChanged();
      return `Instrumento «${instrument.name}» (${instrument.drums ? "batería" : GM_PROGRAMS[instrument.program]}): ${written}`;
    }
    case "write_pattern":
      return writePattern(instrumentNamed(action.instrument), patternOptions());
    case "write_notes": {
      const instrument = instrumentNamed(action.instrument);
      if (!score) throw new Error("hace falta un ABC con secciones");
      const barOf = (ticks) => score.bars.findLastIndex((bar) => bar.start <= ticks);
      const notes = (Array.isArray(action.notes) ? action.notes : []).map((note) => {
        const bar = score.bars[Number(note.bar) - 1];
        if (!bar) throw new Error(`no hay compás ${note.bar}`);
        return {
          start: Math.round(bar.start + (Number(note.beat) - 1) * TICKS_PER_QUARTER), duration: Math.max(1, Math.round(Number(note.beats) * TICKS_PER_QUARTER)),
          pitch: midiPitch(note.pitch), velocity: Math.max(1, Math.min(127, Number(note.velocity) || 96)),
        };
      });
      if (action.replace) {
        const bars = new Set(notes.map((note) => barOf(note.start)));
        instrument.notes = instrument.notes.filter((note) => !bars.has(barOf(note.start)));
      }
      instrument.notes = [...instrument.notes, ...notes].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
      instrumentsChanged();
      return `${notes.length} notas escritas en «${instrument.name}»`;
    }
    case "set_instrument": {
      const instrument = instrumentNamed(action.instrument);
      if (action.name) instrument.name = String(action.name).slice(0, 40);
      if (action.sound !== undefined) instrument.program = gmProgram(action.sound);
      if (typeof action.drums === "boolean") instrument.drums = action.drums;
      instrumentsChanged();
      return `«${instrument.name}»: ${instrument.drums ? "batería" : GM_PROGRAMS[instrument.program]}`;
    }
    case "remove_instrument": {
      const instrument = instrumentNamed(action.instrument);
      project.instruments = project.instruments.filter((item) => item !== instrument);
      delete project.mixer[`inst-${instrument.id}`];
      instrumentAudio.delete(instrument.id);
      draw();
      restartIfPlaying();
      return `Instrumento «${instrument.name}» eliminado`;
    }
    case "set_mix": {
      const wanted = String(action.track ?? "").toLowerCase();
      const track = allTracks().find((item) => item.id === wanted || item.name.toLowerCase() === wanted);
      if (!track) throw new Error(`no hay pista «${action.track}»`);
      const changes = {};
      if (given("volume_db")) changes.gain = Math.pow(10, Number(action.volume_db) / 20);
      if (given("pan")) changes.pan = Math.max(-1, Math.min(1, Number(action.pan) / 100));
      if (given("reverb")) changes.reverb = Math.max(0, Math.min(1, Number(action.reverb) / 100));
      if (given("delay")) changes.delay = Math.max(0, Math.min(1, Number(action.delay) / 100));
      if (given("fade_in")) changes.fadeIn = Math.max(0, Number(action.fade_in));
      if (given("fade_out")) changes.fadeOut = Math.max(0, Number(action.fade_out));
      setTrackFx(track.id, changes);
      draw();
      return `Mezcla de «${track.name}» actualizada`;
    }
    case "set_master_effects": {
      const fx = { ...project.fx };
      if (given("reverb_seconds")) fx.reverbSeconds = Math.max(0.2, Math.min(8, Number(action.reverb_seconds)));
      if (given("delay_beats")) fx.delayBeats = Math.max(0.125, Math.min(4, Number(action.delay_beats)));
      if (given("feedback")) fx.feedback = Math.max(0, Math.min(0.9, Number(action.feedback)));
      project.fx = fx;
      restartIfPlaying();
      const now = masterFx();
      return `Efectos master: reverb ${now.reverbSeconds} s, delay ${now.delayBeats} negras, retroalimentación ${now.feedback}`;
    }
    default:
      return undefined;
  }
}

// ---------- lead vocal tuning (HZ3_YuE2_VocalTune: PSOLA toward the ABC melody)

function compTakes() {
  return [...new Set(Object.values(project.comp))].map(takeById).filter((take) => take?.files.vocals);
}

// Tunes the lead of every take in use; the take keeps the voice as YuE2 sang it in files.vocalsOriginal.
async function tuneVocals(amount) {
  const takes = compTakes();
  if (!takes.length) throw new Error("Todavía no hay takes con voz principal.");
  const owner = project.name;
  for (const take of takes) {
    const original = take.files.vocalsOriginal ?? take.files.vocals;
    const { filename } = await postJson("/hz3/studio/stage", { name: `tune_${take.id}_${Date.now().toString(36)}`, files: [original] });
    const prompt = {
      1: { class_type: "LoadAudio", inputs: { audio: filename } },
      2: { class_type: "HZ3_YuE2_VocalTune", inputs: { vocals: ["1", 0], score_abc: project.abc, amount } },
      3: { class_type: "SaveAudio", inputs: { audio: ["2", 0], filename_prefix: `HZ3-Studio/${owner}/take-${take.id}/vocals_tuned` } },
    };
    await queue(prompt, { 2: `Afinando la voz del take ${take.id}` }, async (outputs) => {
      const file = outputs[3]?.audio?.[0];
      if (!file) throw new Error("La afinación terminó sin audio.");
      const current = await updateProject(owner, (target) => {
        const stored = target.takes.find((item) => item.id === take.id);
        if (!stored) return;
        stored.files.vocalsOriginal ??= stored.files.vocals;
        stored.files.vocals = file;
        stored.tuned = amount;
      });
      if (!current) return;
      await loadTake(takeById(take.id));
      remember(`Voz del take ${take.id} afinada al ${Math.round(amount * 100)} % hacia el ABC`);
      draw();
      restartIfPlaying();
    });
  }
  status(`Afinación en cola para ${takes.length} take(s).`, 0);
}

function untuneVocals() {
  const takes = compTakes().filter((take) => take.files.vocalsOriginal);
  if (!takes.length) throw new Error("La voz de los takes en uso no está afinada.");
  for (const take of takes) {
    take.files.vocals = take.files.vocalsOriginal;
    delete take.files.vocalsOriginal;
    delete take.tuned;
  }
  draw();
  restartIfPlaying();
  status(`Voz original restaurada en ${takes.length} take(s).`, 1);
}
