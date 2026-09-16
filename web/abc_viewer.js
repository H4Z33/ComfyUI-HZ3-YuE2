import { app } from "../../scripts/app.js";

const PPQ = 256;
const COLORS = {Vocal: "#d99cff", Ins: "#4ee0bd"};
const pitchName = pitch => ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"][pitch % 12] + (Math.floor(pitch / 12) - 1);
const pitchClass = name => {
    let value = {C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11}[name[0]];
    for (const accidental of name.slice(1)) value += accidental === "#" ? 1 : -1;
    return (value + 24) % 12;
};
const CHORD_INTERVALS = {
    "": [0, 4, 7], m: [0, 3, 7], dim: [0, 3, 6], aug: [0, 4, 8],
    "7": [0, 4, 7, 10], maj7: [0, 4, 7, 11], m7: [0, 3, 7, 10],
    dim7: [0, 3, 6, 9], m7b5: [0, 3, 6, 10], sus4: [0, 5, 7],
    sus2: [0, 2, 7], "6": [0, 4, 7, 9], m6: [0, 3, 7, 9],
    "7sus4": [0, 5, 7, 10], "m(maj7)": [0, 3, 7, 11],
};

function chordPitches(symbol) {
    const match = symbol.match(/^([A-G](?:bb|##|b|#)?)(.*?)(?:\/([A-G](?:bb|##|b|#)?))?$/);
    if (!match || !(match[2] in CHORD_INTERVALS)) return [];
    const root = pitchClass(match[1]), pitches = CHORD_INTERVALS[match[2]].map(interval => 48 + root + interval);
    if (match[3]) pitches.unshift(36 + pitchClass(match[3]));
    return [...new Set(pitches)];
}

function el(tag, text, parent, className = "") {
    const item = document.createElement(tag);
    if (text !== undefined && text !== null) item.textContent = text;
    if (className) item.className = className;
    parent?.append(item);
    return item;
}

function svgEl(tag, attrs, parent) {
    const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attrs)) item.setAttribute(name, value);
    parent.append(item);
    return item;
}

function button(parent, text, action, secondary = false) {
    const item = el("button", text, parent, secondary ? "secondary" : "");
    item.type = "button";
    item.onclick = action;
    return item;
}

function select(parent, label, options, value, action) {
    const wrap = el("label", label, parent);
    const item = el("select", null, wrap);
    for (const [optionValue, text] of options) {
        const option = el("option", text, item);
        option.value = optionValue;
    }
    item.value = value;
    item.onchange = () => action(item.value);
    return item;
}

const style = document.createElement("style");
style.textContent = `
.hz3-abc-viewer{height:100%;padding:12px;box-sizing:border-box;background:#111827;color:#edf7f5;font:13px system-ui;border:1px solid #25a98e;border-radius:8px;overflow:hidden}
.hz3-abc-viewer *{box-sizing:border-box}.hz3-abc-viewer .toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:9px}
.hz3-abc-viewer label{display:flex;align-items:center;gap:5px}.hz3-abc-viewer select,.hz3-abc-viewer button{background:#1e293b;color:#f8fafc;border:1px solid #486174;border-radius:5px;padding:7px 9px}
.hz3-abc-viewer button{background:#16816d;border-color:#28b89c;cursor:pointer}.hz3-abc-viewer button.secondary{background:#263548}.hz3-abc-viewer button:disabled{opacity:.45;cursor:default}
.hz3-abc-viewer .summary{padding:8px 10px;border-left:3px solid #4ee0bd;background:#162234;margin:7px 0;line-height:1.45}.hz3-abc-viewer .summary.error{border-color:#f87171;color:#fecaca}
.hz3-abc-viewer .roll{overflow:auto;max-height:610px;border:1px solid #405368;border-radius:6px;background:#101827;overscroll-behavior:contain}.hz3-abc-viewer svg{display:block;user-select:none}
.hz3-abc-viewer details{margin-top:9px;border:1px solid #35475b;border-radius:6px;padding:7px}.hz3-abc-viewer summary{cursor:pointer;color:#bdeee4}.hz3-abc-viewer textarea{width:100%;height:320px;margin-top:8px;background:#0b1220;color:#d8e4ec;border:1px solid #405368;border-radius:5px;padding:8px;font:12px ui-monospace,monospace;resize:vertical}
.hz3-abc-viewer table{width:100%;border-collapse:collapse;margin-top:7px}.hz3-abc-viewer th,.hz3-abc-viewer td{text-align:left;padding:5px 7px;border-bottom:1px solid #304155}.hz3-abc-viewer th{color:#8ee8d5}
.hz3-abc-viewer .legend{display:flex;gap:14px;align-items:center}.hz3-abc-viewer .dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px}.hz3-abc-expanded{position:fixed;inset:2vh 2vw;width:96vw;height:96vh;overflow:auto;z-index:10000;box-shadow:0 0 0 4vh #000a}
`;
document.head.append(style);

class ABCViewer {
    constructor(node) {
        this.node = node;
        this.track = "Both";
        this.section = "all";
        this.zoom = 30;
        this.chords = true;
        this.position = 0;
        this.root = el("div", null, null, "hz3-abc-viewer");
        this.root.tabIndex = 0;
        this.toolbar = el("div", null, this.root, "toolbar");
        this.summary = el("div", "Queue the upstream ABC node to load the score.", this.root, "summary");
        this.roll = el("div", null, this.root, "roll");
        this.rawDetails = el("details", null, this.root);
        el("summary", "Complete ABC source", this.rawDetails);
        this.raw = el("textarea", "", this.rawDetails);
        this.raw.readOnly = true;
        this.sectionDetails = el("details", null, this.root);
        el("summary", "Sections, meters and note counts", this.sectionDetails);
        this.sectionTable = el("div", null, this.sectionDetails);
        this.observer = new ResizeObserver(() => this.layout());
        this.observer.observe(this.root);
        this.root.addEventListener("keydown", event => {
            event.stopPropagation();
            if (event.code === "Space") { event.preventDefault(); if (!event.repeat) this.toggle(); }
            if (event.key === "Escape" && this.expanded) this.expand(false);
        });
    }

    receive(data) {
        this.stop();
        this.data = typeof data === "string" ? JSON.parse(data) : data;
        this.raw.value = this.data.abc || "";
        this.section = "all";
        const notes = [...(this.data.roll.tracks.Vocal || []), ...(this.data.roll.tracks.Ins || [])];
        this.position = notes.length ? Math.min(...notes.map(note => note.start)) : 0;
        this.render();
    }

    range() {
        if (this.section === "all") return {start: 0, end: this.data.roll.total_ticks, name: "Complete score"};
        const section = this.data.roll.sections[Number(this.section)];
        return {start: section.start_tick, end: section.end_tick, name: section.name};
    }

    visibleTracks() {
        return this.track === "Both" ? ["Vocal", "Ins"] : [this.track];
    }

    render() {
        if (!this.data) return;
        this.toolbar.replaceChildren();
        const tracks = this.data.roll.tracks;
        select(this.toolbar, "Tracks", [
            ["Both", `Both (${tracks.Vocal.length + tracks.Ins.length})`],
            ["Vocal", `Vocal (${tracks.Vocal.length})`],
            ["Ins", `Instrumental (${tracks.Ins.length})`],
        ], this.track, value => { this.stop(); this.track = value; this.draw(); });
        select(this.toolbar, "Section", [["all", "Complete score"], ...this.data.roll.sections.map((section, index) => [String(index), `${index + 1}. ${section.name}`])], this.section,
            value => { this.stop(); this.section = value; this.position = this.range().start; this.draw(); });
        select(this.toolbar, "Zoom", [[18, "Compact"], [30, "Normal"], [48, "Large"], [72, "Detailed"]], String(this.zoom),
            value => { this.zoom = Number(value); this.draw(); });
        this.playButton = button(this.toolbar, "Play MIDI", () => this.toggle());
        button(this.toolbar, "Stop", () => this.stop(), true);
        const metronomeLabel = el("label", "Metronome", this.toolbar);
        this.metronomeBox = el("input", null, metronomeLabel); this.metronomeBox.type = "checkbox";
        const chordLabel = el("label", "Chords", this.toolbar);
        this.chordBox = el("input", null, chordLabel); this.chordBox.type = "checkbox"; this.chordBox.checked = this.chords;
        this.chordBox.onchange = () => { this.chords = this.chordBox.checked; };
        button(this.toolbar, this.expanded ? "Close expanded" : "Expand", () => this.expand(!this.expanded), true);
        const legend = el("span", null, this.toolbar, "legend");
        const vocal = el("span", null, legend); const vd = el("span", null, vocal, "dot"); vd.style.background = COLORS.Vocal; vocal.append("Vocal");
        const ins = el("span", null, legend); const id = el("span", null, ins, "dot"); id.style.background = COLORS.Ins; ins.append("Ins");
        this.summary.textContent = `${this.data.bars} bars · ${this.data.bpm} BPM · ${this.data.key} · ${this.data.meter} header · ${this.data.unit} unit · about ${this.data.seconds.toFixed(1)} s · ${tracks.Vocal.length} vocal / ${tracks.Ins.length} instrumental notes · ${this.data.roll.chords.length} chord changes`;
        this.renderSections();
        this.draw();
        this.layout();
    }

    renderSections() {
        this.sectionTable.replaceChildren();
        const table = el("table", null, this.sectionTable), head = el("tr", null, el("thead", null, table));
        for (const title of ["#", "Section", "Bars", "Meter", "Vocal", "Ins", "Chords"]) el("th", title, head);
        const body = el("tbody", null, table);
        this.data.roll.sections.forEach((section, index) => {
            const row = el("tr", null, body);
            const within = note => note.start >= section.start_tick && note.start < section.end_tick;
            const meters = [...new Set(this.data.roll.bars.slice(section.start_bar - 1, section.end_bar).map(bar => bar.meter))].join(" → ");
            const values = [index + 1, section.name, `${section.start_bar}–${section.end_bar}`, meters,
                this.data.roll.tracks.Vocal.filter(within).length, this.data.roll.tracks.Ins.filter(within).length,
                this.data.roll.chords.filter(within).length];
            for (const value of values) el("td", String(value), row);
        });
    }

    draw() {
        this.roll.replaceChildren();
        if (!this.data) return;
        const {start, end} = this.range();
        const notes = this.visibleTracks().flatMap(track => this.data.roll.tracks[track].filter(note => note.start < end && note.start + note.duration > start));
        const allPitches = notes.map(note => note.pitch);
        let low = allPitches.length ? Math.max(0, Math.min(...allPitches) - 4) : 48;
        let high = allPitches.length ? Math.min(127, Math.max(...allPitches) + 4) : 72;
        if (high - low < 23) { const pad = 23 - (high - low); low = Math.max(0, low - Math.ceil(pad / 2)); high = Math.min(127, low + 23); low = Math.max(0, high - 23); }
        const rowHeight = 15, top = 44, keys = 52, rows = high - low + 1;
        const width = Math.max(this.roll.clientWidth || 820, keys + (end - start) / PPQ * this.zoom);
        const height = top + rows * rowHeight;
        const svg = svgEl("svg", {width, height, viewBox: `0 0 ${width} ${height}`, role: "application", "aria-label": "HZ3 ABC piano roll"}, this.roll);
        this.svg = svg; this.svgStart = start; this.svgEnd = end; this.keysWidth = keys; this.rollWidth = width - keys;

        for (let pitch = low; pitch <= high; pitch++) {
            const y = top + (high - pitch) * rowHeight;
            const black = [1, 3, 6, 8, 10].includes(pitch % 12);
            svgEl("rect", {x: 0, y, width, height: rowHeight, fill: black ? "#111827" : "#182437", stroke: "#26384d", "stroke-width": .5}, svg);
            svgEl("text", {x: 4, y: y + 11, fill: pitch % 12 === 0 ? "#fff" : "#91a4b7", "font-size": 10}, svg).textContent = pitchName(pitch);
        }

        const bars = this.data.roll.bars.filter(bar => bar.start < end && bar.start + bar.duration > start);
        for (const bar of bars) {
            const x = keys + (bar.start - start) / (end - start) * (width - keys);
            svgEl("line", {x1: x, x2: x, y1: 0, y2: height, stroke: "#7d91a6", "stroke-width": 1}, svg);
            svgEl("text", {x: x + 4, y: 15, fill: "#d8e4ec", "font-size": 10}, svg).textContent = `${bar.number} · ${bar.meter}`;
            const [beats, denominator] = bar.meter.split("/").map(Number), beatTicks = PPQ * 4 / denominator;
            for (let beat = 1; beat < beats; beat++) {
                const bx = keys + (bar.start + beat * beatTicks - start) / (end - start) * (width - keys);
                svgEl("line", {x1: bx, x2: bx, y1: top, y2: height, stroke: "#31455b", "stroke-width": .6}, svg);
            }
        }

        for (const section of this.data.roll.sections.filter(section => section.start_tick < end && section.end_tick > start)) {
            const x = keys + (Math.max(start, section.start_tick) - start) / (end - start) * (width - keys);
            svgEl("text", {x: x + 4, y: 32, fill: "#64e7ca", "font-size": 11, "font-weight": 600}, svg).textContent = section.name;
        }

        for (const chord of this.data.roll.chords.filter(chord => chord.start >= start && chord.start < end)) {
            const x = keys + (chord.start - start) / (end - start) * (width - keys);
            svgEl("text", {x: x + 3, y: 42, fill: "#ffd580", "font-size": 10}, svg).textContent = chord.symbol;
        }

        for (const track of this.visibleTracks()) for (const note of this.data.roll.tracks[track]) {
            if (note.start >= end || note.start + note.duration <= start || note.pitch < low || note.pitch > high) continue;
            const a = Math.max(start, note.start), b = Math.min(end, note.start + note.duration);
            const x = keys + (a - start) / (end - start) * (width - keys), w = Math.max(2, (b - a) / (end - start) * (width - keys));
            const y = top + (high - note.pitch) * rowHeight + (track === "Vocal" ? 1 : 4), h = track === "Vocal" ? rowHeight - 3 : rowHeight - 7;
            const rect = svgEl("rect", {x, y, width: w, height: h, rx: 2, fill: COLORS[track], opacity: .88, stroke: track === "Vocal" ? "#f3ddff" : "#c4fff2", "stroke-width": .5, cursor: "pointer"}, svg);
            svgEl("title", {}, rect).textContent = `${track} · ${pitchName(note.pitch)} · ${(note.start / PPQ).toFixed(3)} beat · ${(note.duration / PPQ).toFixed(3)} beats`;
            rect.onclick = () => this.audition(note.pitch, track);
        }
        this.cursor = svgEl("line", {y1: 0, y2: height, stroke: "#fff", "stroke-width": 2, "pointer-events": "none"}, svg);
        this.updateCursor();
    }

    updateCursor() {
        if (!this.cursor || !this.data) return;
        const x = this.keysWidth + (Math.max(this.svgStart, Math.min(this.svgEnd, this.position)) - this.svgStart) / (this.svgEnd - this.svgStart) * this.rollWidth;
        this.cursor.setAttribute("x1", x); this.cursor.setAttribute("x2", x);
    }

    tone(context, destination, pitch, start, end, track, volume = .1) {
        const oscillator = context.createOscillator(), gain = context.createGain();
        oscillator.type = track === "Vocal" ? "sine" : "triangle";
        oscillator.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(volume, start + Math.min(.008, Math.max(.002, (end - start) / 3)));
        gain.gain.setValueAtTime(volume, Math.max(start + .009, end - .025));
        gain.gain.exponentialRampToValueAtTime(.0001, end);
        oscillator.connect(gain); gain.connect(destination); oscillator.start(start); oscillator.stop(end + .02);
    }

    async audition(pitch, track) {
        this.auditionContext?.close();
        const context = new AudioContext(); this.auditionContext = context; await context.resume();
        this.tone(context, context.destination, pitch, context.currentTime + .01, context.currentTime + .3, track, .12);
        setTimeout(() => { if (this.auditionContext === context) { context.close(); this.auditionContext = null; } }, 450);
    }

    toggle() { return this.audio ? this.pause() : this.play(); }

    async play() {
        if (!this.data) return;
        this.pause();
        const range = this.range();
        if (this.position < range.start || this.position >= range.end) this.position = range.start;
        const context = new AudioContext(); this.audio = context; await context.resume();
        if (this.audio !== context) return;
        const master = context.createGain(); master.gain.value = this.track === "Both" ? .55 : .8; master.connect(context.destination);
        const now = context.currentTime + .05, secondsPerTick = 60 / this.data.bpm / PPQ, from = this.position;
        for (const track of this.visibleTracks()) for (const note of this.data.roll.tracks[track]) {
            const a = Math.max(from, note.start), b = Math.min(range.end, note.start + note.duration);
            if (b <= a) continue;
            this.tone(context, master, note.pitch, now + (a - from) * secondsPerTick, now + (b - from) * secondsPerTick, track, track === "Vocal" ? .12 : .085);
        }
        if (this.chords) {
            const chords = [...this.data.roll.chords].sort((a, b) => a.start - b.start);
            for (let index = 0; index < chords.length; index++) {
                const chord = chords[index], next = chords[index + 1]?.start ?? range.end;
                const a = Math.max(from, range.start, chord.start), b = Math.min(range.end, next);
                if (b <= a) continue;
                for (const pitch of chordPitches(chord.symbol)) {
                    this.tone(context, master, pitch, now + (a - from) * secondsPerTick, now + (b - from) * secondsPerTick, "Ins", .022);
                }
            }
        }
        if (this.metronomeBox?.checked) for (const bar of this.data.roll.bars.filter(bar => bar.start < range.end && bar.start + bar.duration > from)) {
            const [beats, denominator] = bar.meter.split("/").map(Number), beatTicks = PPQ * 4 / denominator;
            for (let beat = 0; beat < beats; beat++) {
                const tick = bar.start + beat * beatTicks; if (tick < from || tick >= range.end) continue;
                this.tone(context, master, beat === 0 ? 96 : 88, now + (tick - from) * secondsPerTick, now + (tick - from) * secondsPerTick + .025, "Ins", .035);
            }
        }
        this.playButton.textContent = "Pause";
        this.startedAt = now; this.startedFrom = from; this.secondsPerTick = secondsPerTick;
        const animate = () => {
            if (this.audio !== context) return;
            this.position = Math.min(range.end, from + Math.max(0, context.currentTime - now) / secondsPerTick);
            this.updateCursor();
            if (this.position >= range.end) { this.stop(); return; }
            this.animation = requestAnimationFrame(animate);
        };
        this.animation = requestAnimationFrame(animate);
    }

    pause() {
        if (this.audio) {
            if (this.startedAt !== undefined) this.position = Math.min(this.range().end, this.startedFrom + Math.max(0, this.audio.currentTime - this.startedAt) / this.secondsPerTick);
            this.audio.close(); this.audio = null;
        }
        this.startedAt = undefined; cancelAnimationFrame(this.animation);
        if (this.playButton) this.playButton.textContent = "Play MIDI";
        this.updateCursor();
    }

    stop() {
        this.pause(); this.auditionContext?.close(); this.auditionContext = null;
        this.position = this.data ? this.range().start : 0; this.updateCursor();
    }

    expand(value) {
        if (value === !!this.expanded) return;
        this.expanded = value;
        if (value) { this.home = this.root.parentNode; document.body.append(this.root); this.root.classList.add("hz3-abc-expanded"); }
        else { this.root.classList.remove("hz3-abc-expanded"); this.home?.append(this.root); }
        if (this.data) this.render();
    }

    layout() {
        if (this.expanded) return;
        const height = Math.max(760, Math.min(1050, this.root.scrollHeight + 30));
        if (this.node.size[1] < height + 70) this.node.setSize([Math.max(880, this.node.size[0]), height + 70]);
        this.node.graph?.setDirtyCanvas(true, true);
    }

    destroy() {
        this.stop(); this.expand(false); this.observer.disconnect(); cancelAnimationFrame(this.animation);
    }
}

app.registerExtension({
    name: "HZ3.YuE2.ABCViewer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HZ3_YuE2_ABCViewer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            this.hz3ABCViewer = new ABCViewer(this);
            const widget = this.addDOMWidget("hz3_abc_viewer", "custom", this.hz3ABCViewer.root, {
                serialize: false, getMinHeight: () => 760, getHeight: () => Math.max(760, this.hz3ABCViewer.root.scrollHeight),
            });
            widget.computeSize = () => [840, 760]; this.setSize([880, 850]);
        };
        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            resized?.apply(this, arguments); size[0] = this.size[0] = Math.max(880, size[0]); this.hz3ABCViewer?.layout();
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            const data = message.abc_viewer?.[0];
            if (data) this.hz3ABCViewer?.receive(data);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () { this.hz3ABCViewer?.destroy(); return removed?.apply(this, arguments); };
    },
});
