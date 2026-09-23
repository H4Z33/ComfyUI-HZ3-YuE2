import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function element(tag, text, parent, className = "") {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = text;
    if (className) node.className = className;
    parent?.append(node);
    return node;
}

function svgElement(tag, parent, attrs = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    parent?.append(node);
    return node;
}

function button(parent, label, action, className = "") {
    const node = element("button", label, parent, className);
    node.type = "button";
    node.onclick = action;
    return node;
}

function select(parent, label, options, value, action) {
    const wrapper = element("label", label, parent, "control-label");
    const input = element("select", null, wrapper, "control-select");
    for (const [optionValue, text] of options) {
        const option = element("option", text, input);
        option.value = optionValue;
    }
    input.value = value;
    input.onchange = () => action(input.value);
    return input;
}

function numberInput(parent, label, value, min, max, action) {
    const wrapper = element("label", label, parent, "control-label numeric-control");
    const input = element("input", null, wrapper, "bar-input");
    input.type = "number";
    input.step = "1";
    input.min = String(min);
    input.max = String(max);
    input.value = String(value);
    input.setAttribute("aria-label", label + " BAR");
    input.onchange = () => action(Number(input.value));
    return input;
}

function pitchClass(name) {
    let value = {C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11}[name[0]];
    for (const accidental of name.slice(1)) value += accidental === "#" ? 1 : -1;
    return (value + 24) % 12;
}

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
    const root = pitchClass(match[1]);
    const pitches = CHORD_INTERVALS[match[2]].map(interval => 48 + root + interval);
    if (match[3]) pitches.unshift(36 + pitchClass(match[3]));
    return [...new Set(pitches)];
}

const BAR_WIDTH = 120;
const AXIS_WIDTH = 52;
const NOTE_ROW_HEIGHT = 13;
const SECTION_COLORS = ["#45d6b0", "#f3a64a", "#a78bfa", "#60a5fa", "#fb7185", "#a3e635", "#22d3ee", "#f472b6"];

function sectionColor(index) { return SECTION_COLORS[index % SECTION_COLORS.length]; }

function hexRgb(hex) {
    const value = hex.slice(1);
    return [0, 2, 4].map(offset => parseInt(value.slice(offset, offset + 2), 16));
}

function rgba(hex, alpha) {
    return "rgba(" + hexRgb(hex).join(",") + "," + alpha + ")";
}

const style = document.createElement("style");
style.textContent = [
    ".hz3-section-editor{height:100%;min-height:540px;padding:8px;box-sizing:border-box;background:#111827;color:#edf7f5;font:13px system-ui;border:1px solid #25a98e;border-radius:8px;display:flex;flex-direction:column;overflow:hidden}",
    ".hz3-section-editor *{box-sizing:border-box}.hz3-section-editor .toolbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:5px;flex:0 0 auto}",
    ".hz3-section-editor button{background:#16816d;color:#fff;border:1px solid #28b89c;border-radius:5px;padding:6px 9px;cursor:pointer}.hz3-section-editor button.secondary{background:#263548;border-color:#486174}.hz3-section-editor button.danger{background:#54343b;border-color:#8e5961}.hz3-section-editor button:disabled{opacity:.4;cursor:default}",
    ".hz3-section-editor .control-label{display:flex;align-items:center;gap:5px;color:#b7c9d6;font-size:11px}.hz3-section-editor .control-select{max-width:180px;background:#1e293b;color:#f8fafc;border:1px solid #486174;border-radius:5px;padding:6px 8px}.hz3-section-editor .bar-input{width:62px;background:#1e293b;color:#f8fafc;border:1px solid #486174;border-radius:5px;padding:5px 6px;font:11px ui-monospace,SFMono-Regular,Consolas,monospace}.hz3-section-editor .numeric-control{font-size:10px}",
    ".hz3-section-editor .hint{color:#b6c7d6;font-size:11px;line-height:1.3;padding:5px 7px;border-left:3px solid #4ee0bd;background:#162234;margin-bottom:5px;flex:0 0 auto}",
    ".hz3-section-editor .legend{display:flex;align-items:center;gap:14px;color:#b6c7d6;font-size:11px;margin:0 0 5px 4px;flex:0 0 auto}.hz3-section-editor .legend-item{display:flex;align-items:center;gap:5px}.hz3-section-editor .legend-swatch{width:10px;height:10px;border-radius:2px}",
    ".hz3-section-editor .status{color:#8fe3d0;margin-left:auto;min-width:150px;font-size:11px}",
    ".hz3-section-editor .timeline-scroll{overflow:auto;min-height:0;flex:1;border:1px solid #405368;border-radius:7px;background:#0d1522;overscroll-behavior:contain;scrollbar-gutter:stable}",
    ".hz3-section-editor .timeline-canvas{position:relative;min-height:100%}",
    ".hz3-section-editor .bar-ruler{display:flex;height:44px;position:sticky;top:0;z-index:4;background:#111c2b;border-bottom:1px solid #405368}",
    ".hz3-section-editor .roll-axis{position:sticky;left:0;z-index:3;flex:0 0 " + AXIS_WIDTH + "px;width:" + AXIS_WIDTH + "px;background:#172436;border-right:1px solid #405368}",
    ".hz3-section-editor .ruler-axis{z-index:5;display:flex;align-items:center;justify-content:center;color:#91a7b8;font:10px ui-monospace,SFMono-Regular,Consolas,monospace}",
    ".hz3-section-editor .bar-ruler-cell{position:relative;flex:0 0 " + BAR_WIDTH + "px;width:" + BAR_WIDTH + "px;border-right:1px solid #34475a;text-align:center}",
    ".hz3-section-editor .section-tag{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin:3px 3px 0;padding:2px 3px;height:18px;background:#6b4d1e;color:#fff0bd;border:1px solid #b78b3b;border-radius:3px;font-size:9px;font-weight:700}",
    ".hz3-section-editor .bar-number{display:block;margin:2px auto 0;padding:1px 4px;background:transparent;border:0;color:#9fb3c4;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer}.hz3-section-editor .bar-number:hover{color:#fff;background:#263548;border-radius:3px}",
    ".hz3-section-editor .piano-row{display:flex;align-items:flex-start;border-bottom:1px solid #405368}",
    ".hz3-section-editor .pitch-axis{flex:none;background:#172436;border-right:1px solid #405368}",
    ".hz3-section-editor .piano-svg{display:block;flex:none;background:#101a28}",
    ".hz3-section-editor .lyrics-row{display:flex;align-items:stretch}",
    ".hz3-section-editor .lyrics-axis{display:flex;align-items:flex-start;justify-content:center;padding-top:9px;color:#d2a7ee;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;z-index:3}",
    ".hz3-section-editor .lyrics-track{position:relative;flex:none}",
    ".hz3-section-editor .lyric-card{position:absolute;overflow:hidden;border:1px solid var(--section-color);border-radius:6px;box-shadow:0 3px 10px #0005}",
    ".hz3-section-editor .lyric-card.selected{box-shadow:0 0 0 1px var(--section-color),0 3px 10px #0006}",
    ".hz3-section-editor .lyric-card-head{height:34px;display:flex;align-items:center;gap:4px;padding:3px 5px;background:var(--section-tint);border-bottom:1px solid var(--section-color)}",
    ".hz3-section-editor .lyric-card-title{min-width:0;flex:1;background:#111827;color:#fff;border:1px solid var(--section-color);border-radius:4px;padding:4px 5px;font:600 11px system-ui}",
    ".hz3-section-editor .lyric-card-bar{white-space:nowrap;color:#e7edf4;font:9px ui-monospace,SFMono-Regular,Consolas,monospace}",
    ".hz3-section-editor .remove-button{padding:4px 6px!important}",
    ".hz3-section-editor .lyric-text{display:block;width:100%;height:112px;min-height:88px;resize:vertical;overflow:auto;background:#101827;color:#f3eafa;border:0;padding:7px;font:12px/1.4 system-ui;outline:none}.hz3-section-editor .lyric-text:focus{box-shadow:inset 0 0 0 1px var(--section-color)}",
    ".hz3-section-editor .file-input{display:none!important}",
    ".hz3-section-editor .widget-hidden{display:none!important}",
    ".hz3-section-editor-expanded{position:fixed!important;inset:2vh 2vw!important;width:96vw!important;height:96vh!important;z-index:10000!important;box-shadow:0 0 0 4vh #000a}"
].join("\n");
document.head.append(style);

class ABCSectionEditor {
    constructor(node) {
        this.node = node;
        this.root = element("div", null, null, "hz3-section-editor");
        this.toolbar = element("div", null, this.root, "toolbar");
        this.hint = element("div", "Use the piano roll to see ABC notes. Each lyric box is pinned to its section's starting bar.", this.root, "hint");
        this.legend = element("div", null, this.root, "legend");
        this.timelineScroll = element("div", null, this.root, "timeline-scroll");
        this.timelineCanvas = element("div", null, this.timelineScroll, "timeline-canvas");
        this.selectedSection = "0";
        this.playbackSection = "all";
        this.track = "Both";
        this.chords = true;
        this.position = 0;
        this.zoom = 1;
        this.expanded = false;
        this.status = element("span", "Run this node to load the ABC.", this.toolbar, "status");
        this.fileInput = element("input", null, this.root, "file-input");
        this.fileInput.type = "file";
        this.fileInput.accept = ".json,.abc,.txt,.lrc";
        this.fileInput.multiple = true;
        this.fileInput.onchange = () => this.loadFiles([...this.fileInput.files]);
        this.observer = new ResizeObserver(() => this.layout());
        this.observer.observe(this.root);
    }

    bindStateWidget() {
        this.stateWidget = this.node.widgets?.find(widget => widget.name === "editor_state");
        if (!this.stateWidget) return;
        this.stateWidget.computeSize = () => [0, -4];
        this.stateWidget.options = {...(this.stateWidget.options || {}), serialize: true};
        this.stateWidget.element?.classList.add("widget-hidden");
        this.stateWidget.element?.setAttribute("aria-hidden", "true");
        for (const name of ["loaded_abc", "loaded_lyrics"]) {
            const widget = this.node.widgets?.find(item => item.name === name);
            if (!widget) continue;
            widget.computeSize = () => [0, -4];
            widget.options = {...(widget.options || {}), serialize: true};
            widget.element?.classList.add("widget-hidden");
            widget.element?.setAttribute("aria-hidden", "true");
            this[name + "Widget"] = widget;
        }
        this.node.setSize([Math.max(1000, this.node.size[0]), 640]);
    }

    receive(data) {
        this.stop();
        this.data = typeof data === "string" ? JSON.parse(data) : data;
        this.hasEdits = false;
        this.selectedSection = this.data.sections.length ? "0" : "all";
        this.playbackSection = "all";
        this.position = 0;
        this.state = {
            source_hash: this.data.source_hash,
            editor_version: 3,
            sections: this.data.sections.map(section => ({...section})),
        };
        this.writeState(false);
        this.render();
    }

    writeState(showQueuedHint = true) {
        if (!this.state) return;
        const value = JSON.stringify(this.state);
        if (this.stateWidget) {
            this.stateWidget.value = value;
            this.stateWidget.callback?.(value);
        }
        if (showQueuedHint) this.hasEdits = true;
        if (showQueuedHint && this.status) this.status.textContent = "Edits saved · queue this node to update its outputs.";
        this.node.graph?.setDirtyCanvas(true, true);
    }

    playbackRange() {
        if (!this.data?.bars?.length) return {start: 0, end: 0};
        if (this.playbackSection === "all") return {start: 0, end: this.data.total_ticks};
        const index = Number(this.playbackSection);
        const section = this.state.sections[index];
        if (!section) return {start: 0, end: this.data.total_ticks};
        const start = this.data.bars[section.start_bar]?.start ?? this.data.total_ticks;
        const end = this.data.bars[section.end_bar + 1]?.start ?? this.data.total_ticks;
        return {start, end};
    }

    addSection() {
        if (!this.data || this.state.sections.length >= Math.min(64, this.data.bars.length)) return;
        const used = new Set(this.state.sections.map(section => section.start_bar));
        const selected = this.selectedSection === "all" ? 0 : this.state.sections[Number(this.selectedSection)].start_bar;
        let startBar = -1;
        for (let bar = Math.max(1, selected + 1); bar < this.data.bars.length; bar++) {
            if (!used.has(bar)) { startBar = bar; break; }
        }
        if (startBar < 0) {
            for (let bar = 1; bar < this.data.bars.length; bar++) {
                if (!used.has(bar)) { startBar = bar; break; }
            }
        }
        if (startBar < 0) {
            this.status.textContent = "Every bar already has a section start.";
            return;
        }
        this.stop();
        this.state.sections.push({
            id: "section-" + Date.now(),
            name: "Section " + (this.state.sections.length + 1),
            start_bar: startBar,
            end_bar: this.data.bars.length - 1,
            lyrics: "",
        });
        this.state.sections.sort((a, b) => a.start_bar - b.start_bar);
        const newIndex = this.state.sections.findIndex(section => section.start_bar === startBar);
        const previous = this.state.sections[newIndex - 1];
        const next = this.state.sections[newIndex + 1];
        if (previous) previous.end_bar = Math.min(previous.end_bar, startBar - 1);
        this.state.sections[newIndex].end_bar = (next?.start_bar ?? this.data.bars.length) - 1;
        this.selectedSection = String(newIndex);
        this.playbackSection = this.selectedSection;
        this.position = this.playbackRange().start;
        this.writeState();
        this.render();
        this.focusSection(this.selectedSection);
    }

    removeSection(index) {
        if (this.state.sections.length <= 1) return;
        this.stop();
        const removed = this.state.sections[index];
        this.state.sections.splice(index, 1);
        const previous = this.state.sections[index - 1];
        const next = this.state.sections[index];
        if (previous && removed) previous.end_bar = Math.min(next ? next.start_bar - 1 : removed.end_bar, removed.end_bar);
        this.selectedSection = String(Math.max(0, Math.min(index, this.state.sections.length - 1)));
        this.playbackSection = "all";
        this.position = this.playbackRange().start;
        this.writeState();
        this.render();
        this.focusSection(this.selectedSection);
    }

    setSectionBar(index, field, displayedValue) {
        const section = this.state.sections[index];
        if (!section || !Number.isFinite(displayedValue)) return;
        const previous = this.state.sections[index - 1];
        const next = this.state.sections[index + 1];
        this.stop();
        if (field === "start_bar") {
            const minBar = previous ? previous.start_bar + 1 : 0;
            const maxBar = next ? next.start_bar - 1 : this.data.bars.length - 1;
            section.start_bar = Math.max(minBar, Math.min(maxBar, Math.round(displayedValue) - 1));
            section.end_bar = Math.max(section.start_bar, Math.min(section.end_bar, (next?.start_bar ?? this.data.bars.length) - 1));
            if (previous) previous.end_bar = Math.min(previous.end_bar, section.start_bar - 1);
        } else {
            const maxBar = (next?.start_bar ?? this.data.bars.length) - 1;
            section.end_bar = Math.max(section.start_bar, Math.min(maxBar, Math.round(displayedValue) - 1));
        }
        this.selectedSection = String(index);
        this.playbackSection = this.selectedSection;
        this.position = this.playbackRange().start;
        this.writeState();
        this.render();
        this.focusSection(this.selectedSection);
    }

    focusSection(index) {
        if (!this.data) return;
        const section = index === "all" ? null : this.state.sections[Number(index)];
        const left = section ? section.start_bar * BAR_WIDTH * this.zoom : 0;
        this.timelineScroll.scrollTo({left: Math.max(0, left - AXIS_WIDTH), behavior: "smooth"});
    }

    selectSection(index) {
        this.selectedSection = String(index);
        if (this.editSectionSelect) this.editSectionSelect.value = String(index);
        this.timelineCanvas.querySelectorAll(".lyric-card").forEach((card, cardIndex) => {
            card.classList.toggle("selected", cardIndex === Number(index));
        });
        this.syncRangeInputs();
    }

    syncRangeInputs() {
        const index = this.selectedSection === "all" ? -1 : Number(this.selectedSection);
        const section = this.state?.sections?.[index];
        if (!section || !this.data?.bars?.length) {
            this.startBarInput && (this.startBarInput.disabled = true);
            this.endBarInput && (this.endBarInput.disabled = true);
            return;
        }
        const previous = this.state.sections[index - 1];
        const next = this.state.sections[index + 1];
        if (this.startBarInput) {
            this.startBarInput.value = String(section.start_bar + 1);
            this.startBarInput.min = String((previous?.start_bar ?? -1) + 2);
            this.startBarInput.max = String((next?.start_bar ?? this.data.bars.length) );
            this.startBarInput.disabled = false;
        }
        if (this.endBarInput) {
            this.endBarInput.value = String(section.end_bar + 1);
            this.endBarInput.min = String(section.start_bar + 1);
            this.endBarInput.max = String((next?.start_bar ?? this.data.bars.length));
            this.endBarInput.disabled = false;
        }
    }

    playFromBar(barIndex) {
        if (!this.data?.bars?.[barIndex]) return;
        this.stop();
        this.playbackSection = "all";
        if (this.playRangeSelect) this.playRangeSelect.value = "all";
        this.position = this.data.bars[barIndex].start;
        this.updatePlaybackStatus();
        this.play();
    }

    visibleTracks() {
        return this.track === "Both" ? ["Vocal", "Ins"] : [this.track];
    }

    tone(context, destination, pitch, start, end, track, volume) {
        if (end <= start) return;
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.type = track === "Vocal" ? "sine" : "triangle";
        oscillator.frequency.value = 440 * 2 ** ((pitch - 69) / 12);
        const duration = end - start;
        const attack = Math.min(.008, duration * .3);
        const release = Math.min(.02, duration * .3);
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(volume, start + attack);
        gain.gain.setValueAtTime(volume, Math.max(start + attack, end - release));
        gain.gain.exponentialRampToValueAtTime(.0001, end);
        oscillator.connect(gain);
        gain.connect(destination);
        oscillator.start(start);
        oscillator.stop(end + .02);
    }

    togglePlayback() {
        return this.audio ? this.pause() : this.play();
    }

    async play() {
        if (!this.data) return;
        const range = this.playbackRange();
        if (range.end <= range.start) {
            this.status.textContent = "This section has no ABC bars to play.";
            return;
        }
        if (this.position < range.start || this.position >= range.end) this.position = range.start;
        const AudioContextType = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextType) {
            this.status.textContent = "Audio playback is not supported in this browser.";
            return;
        }
        const context = new AudioContextType();
        this.audio = context;
        await context.resume();
        if (this.audio !== context) return;

        const master = context.createGain();
        master.gain.value = this.track === "Both" ? .55 : .8;
        master.connect(context.destination);
        const secondsPerTick = 60 / this.data.bpm / 256;
        const now = context.currentTime + .05;
        const from = this.position;
        for (const track of this.visibleTracks()) {
            for (const note of this.data.tracks[track] || []) {
                const startTick = Math.max(from, range.start, note.start);
                const endTick = Math.min(range.end, note.start + note.duration);
                if (endTick <= startTick) continue;
                this.tone(context, master, note.pitch,
                    now + (startTick - from) * secondsPerTick,
                    now + (endTick - from) * secondsPerTick,
                    track, track === "Vocal" ? .11 : .075);
            }
        }
        if (this.chords && this.track !== "Vocal") {
            const chords = [...(this.data.chords || [])].sort((a, b) => a.start - b.start);
            for (let index = 0; index < chords.length; index++) {
                const chord = chords[index];
                const next = chords[index + 1]?.start ?? range.end;
                const startTick = Math.max(from, range.start, chord.start);
                const endTick = Math.min(range.end, next);
                if (endTick <= startTick) continue;
                for (const pitch of chordPitches(chord.symbol)) {
                    this.tone(context, master, pitch,
                        now + (startTick - from) * secondsPerTick,
                        now + (endTick - from) * secondsPerTick,
                        "Ins", .018);
                }
            }
        }

        this.playButton.textContent = "❚❚ Pause ABC";
        this.startedAt = now;
        this.startedFrom = from;
        this.secondsPerTick = secondsPerTick;
        const animate = () => {
            if (this.audio !== context) return;
            this.position = Math.min(range.end, from + Math.max(0, context.currentTime - now) / secondsPerTick);
            this.updatePlaybackStatus();
            if (this.position >= range.end) { this.stop(); return; }
            this.animation = requestAnimationFrame(animate);
        };
        this.animation = requestAnimationFrame(animate);
    }

    pause() {
        if (this.audio) {
            if (this.startedAt !== undefined) {
                this.position = Math.min(this.playbackRange().end,
                    this.startedFrom + Math.max(0, this.audio.currentTime - this.startedAt) / this.secondsPerTick);
            }
            this.audio.close();
            this.audio = null;
        }
        this.startedAt = undefined;
        cancelAnimationFrame(this.animation);
        if (this.playButton) this.playButton.textContent = "▶ Play ABC";
        this.updatePlaybackStatus();
    }

    stop() {
        this.pause();
        this.position = this.data ? this.playbackRange().start : 0;
        this.updatePlaybackStatus();
    }

    updatePlaybackStatus() {
        if (!this.status || !this.data) return;
        let barIndex = 0;
        for (let index = 0; index < this.data.bars.length; index++) {
            if (this.data.bars[index].start > this.position) break;
            barIndex = index;
        }
        const barNumber = this.data.bars.length ? String(barIndex + 1).padStart(3, "0") : "000";
        const seconds = this.position / 256 * 60 / this.data.bpm;
        const clock = Math.floor(seconds / 60) + ":" + String(Math.floor(seconds % 60)).padStart(2, "0");
        const state = this.audio
            ? "Playing ABC · bar " + barNumber + " · " + clock
            : this.hasEdits
                ? "Edits saved · queue this node to update its outputs."
                : this.data.bars.length + " bars · " + this.data.bpm + " BPM · " + this.data.key;
        this.status.textContent = state;
        if (this.playhead && this.data.bars.length) {
            const x = this.tickToX(this.position);
            this.playhead.setAttribute("x1", x);
            this.playhead.setAttribute("x2", x);
            this.playhead.setAttribute("visibility", this.audio ? "visible" : "hidden");
        }
    }

    tickToX(tick) {
        const bars = this.data.bars;
        for (let index = 0; index < bars.length; index++) {
            const bar = bars[index];
            const end = bar.start + bar.duration;
            if (tick <= end || index === bars.length - 1) {
                const fraction = Math.max(0, Math.min(1, (tick - bar.start) / bar.duration));
                return (index + fraction) * BAR_WIDTH * this.zoom;
            }
        }
        return bars.length * BAR_WIDTH * this.zoom;
    }

    renderTimeline() {
        const bars = this.data.bars || [];
        const sections = this.state.sections;
        const barWidth = BAR_WIDTH * this.zoom;
        const scoreWidth = Math.max(1, bars.length * barWidth);
        const timelineWidth = scoreWidth;
        const notes = [...(this.data.tracks.Vocal || []), ...(this.data.tracks.Ins || [])];
        let lowPitch = notes.length ? Math.min(...notes.map(note => note.pitch)) - 2 : 48;
        let highPitch = notes.length ? Math.max(...notes.map(note => note.pitch)) + 2 : 72;
        lowPitch = Math.max(0, lowPitch);
        highPitch = Math.min(127, highPitch);
        if (highPitch - lowPitch < 23) {
            const middle = (highPitch + lowPitch) / 2;
            lowPitch = Math.max(0, Math.floor(middle - 12));
            highPitch = Math.min(127, lowPitch + 23);
        }
        const rowHeight = NOTE_ROW_HEIGHT;
        const rollHeight = (highPitch - lowPitch + 1) * rowHeight;
        this.rollHeight = rollHeight;
        this.timelineCanvas.replaceChildren();
        this.timelineCanvas.style.width = (AXIS_WIDTH + timelineWidth) + "px";

        const sectionAtBar = new Map(sections.map((section, index) => [section.start_bar, {section, index}]));
        const ruler = element("div", null, this.timelineCanvas, "bar-ruler");
        element("div", "PITCH", ruler, "roll-axis ruler-axis");
        for (const bar of bars) {
            const cell = element("div", null, ruler, "bar-ruler-cell");
            cell.style.width = barWidth + "px";
            cell.style.flexBasis = barWidth + "px";
            const tagged = sectionAtBar.get(bar.number - 1);
            if (tagged) {
                const tag = element("span", tagged.section.name, cell, "section-tag");
                const color = sectionColor(tagged.index);
                tag.style.background = rgba(color, .22);
                tag.style.borderColor = color;
                tag.style.color = "#f8fafc";
                tag.title = tagged.section.name + " · starts at bar " + String(bar.number).padStart(3, "0");
                tag.onclick = () => {
                    this.selectSection(tagged.index);
                    this.render();
                    this.focusSection(this.selectedSection);
                };
            } else {
                element("span", "", cell, "section-tag");
            }
            const barButton = button(cell, String(bar.number).padStart(3, "0"), event => {
                event.stopPropagation();
                this.playFromBar(bar.number - 1);
            }, "bar-number");
            barButton.title = "Play from BAR " + String(bar.number).padStart(3, "0");
        }
        if (timelineWidth > scoreWidth) {
            const tail = element("div", null, ruler, "bar-ruler-cell");
            tail.style.width = (timelineWidth - scoreWidth) + "px";
            tail.style.flexBasis = (timelineWidth - scoreWidth) + "px";
        }

        const pianoRow = element("div", null, this.timelineCanvas, "piano-row");
        const pitchAxis = svgElement("svg", pianoRow, {class: "roll-axis pitch-axis", width: AXIS_WIDTH, height: rollHeight, viewBox: "0 0 " + AXIS_WIDTH + " " + rollHeight});
        const roll = svgElement("svg", pianoRow, {class: "piano-svg", width: scoreWidth, height: rollHeight, viewBox: "0 0 " + scoreWidth + " " + rollHeight});
        this.playhead = null;

        const blackKeys = new Set([1, 3, 6, 8, 10]);
        for (let pitch = lowPitch; pitch <= highPitch; pitch++) {
            const y = (highPitch - pitch) * rowHeight;
            const black = blackKeys.has(pitch % 12);
            const fill = black ? "#172333" : (pitch % 12 === 0 ? "#243144" : "#1d2939");
            svgElement("rect", roll, {x: 0, y, width: scoreWidth, height: rowHeight, fill});
            svgElement("line", roll, {x1: 0, y1: y + rowHeight, x2: scoreWidth, y2: y + rowHeight, stroke: "#34475a", "stroke-width": .5});
            if (pitch % 12 === 0) {
                const octave = Math.floor(pitch / 12) - 1;
                const label = svgElement("text", pitchAxis, {x: AXIS_WIDTH - 5, y: y + rowHeight - 3, "text-anchor": "end", fill: "#d0dce6", "font-size": 9, "font-family": "monospace"});
                label.textContent = "C" + octave;
            }
        }

        for (let index = 0; index < bars.length; index++) {
            const bar = bars[index];
            const x = index * barWidth;
            const beats = Math.max(1, Math.round(bar.duration / 256));
            for (let beat = 1; beat < beats; beat++) {
                const beatX = x + beat * barWidth / beats;
                svgElement("line", roll, {x1: beatX, y1: 0, x2: beatX, y2: rollHeight, stroke: "#53657a", "stroke-width": .7, "stroke-dasharray": "2 4"});
            }
            svgElement("line", roll, {x1: x, y1: 0, x2: x, y2: rollHeight, stroke: "#71849a", "stroke-width": 1});
        }
        svgElement("line", roll, {x1: scoreWidth - 1, y1: 0, x2: scoreWidth - 1, y2: rollHeight, stroke: "#71849a", "stroke-width": 1});

        for (let index = 0; index < sections.length; index++) {
            const section = sections[index];
            const x = section.start_bar * barWidth;
            const width = (section.end_bar - section.start_bar + 1) * barWidth;
            const color = sectionColor(index);
            svgElement("rect", roll, {x, y: 0, width, height: rollHeight, fill: color, opacity: .09});
            svgElement("line", roll, {x1: x, y1: 0, x2: x, y2: rollHeight, stroke: color, "stroke-width": 2, "stroke-dasharray": "5 3", opacity: .95});
            svgElement("line", roll, {x1: x + width, y1: 0, x2: x + width, y2: rollHeight, stroke: color, "stroke-width": 1.5, opacity: .75});
        }

        const trackColors = {Vocal: "#d99cff", Ins: "#58dfc4"};
        for (const track of ["Vocal", "Ins"]) {
            for (const note of this.data.tracks[track] || []) {
                const x1 = this.tickToX(note.start);
                const x2 = this.tickToX(note.start + note.duration);
                const y = (highPitch - note.pitch) * rowHeight + 2;
                const rect = svgElement("rect", roll, {
                    x: x1 + .7, y, width: Math.max(2, x2 - x1 - 1.4), height: rowHeight - 4,
                    rx: 2, fill: trackColors[track], opacity: track === "Vocal" ? .95 : .78,
                    stroke: "#0b1220", "stroke-width": .6,
                });
                const title = svgElement("title", rect);
                title.textContent = track + " · MIDI " + note.pitch;
            }
        }

        this.playhead = svgElement("line", roll, {
            x1: 0, y1: 0, x2: 0, y2: rollHeight, stroke: "#ffffff", "stroke-width": 2,
            visibility: "hidden", "pointer-events": "none",
        });

        const lyricsRow = element("div", null, this.timelineCanvas, "lyrics-row");
        element("div", "LYRICS", lyricsRow, "roll-axis lyrics-axis");
        const lyricsTrack = element("div", null, lyricsRow, "lyrics-track");
        lyricsTrack.style.width = timelineWidth + "px";
        this.lyricsResizeObserver?.disconnect();
        this.lyricsResizeObserver = new ResizeObserver(() => this.syncLyricsLayout(lyricsTrack));
        for (let index = 0; index < sections.length; index++) {
            const section = sections[index];
            const left = section.start_bar * barWidth;
            const width = Math.max(barWidth, (section.end_bar - section.start_bar + 1) * barWidth);
            const color = sectionColor(index);
            const card = element("article", null, lyricsTrack, "lyric-card" + (String(index) === this.selectedSection ? " selected" : ""));
            card.style.left = left + "px";
            card.style.top = "0px";
            card.style.width = width + "px";
            card.style.setProperty("--section-color", color);
            card.style.setProperty("--section-tint", rgba(color, .14));
            card.style.background = rgba(color, .07);
            card.onclick = () => this.selectSection(index);

            const head = element("div", null, card, "lyric-card-head");
            const title = element("input", null, head, "lyric-card-title");
            title.value = section.name;
            title.setAttribute("aria-label", "Section name");
            title.onclick = event => { event.stopPropagation(); this.selectSection(index); };
            title.onchange = () => {
                section.name = title.value.replace(/[\r\n\[\]]+/g, " ").trim().slice(0, 100) || "Section " + (index + 1);
                this.writeState();
                this.render();
                this.focusSection(String(index));
            };
            element("span", String(section.start_bar + 1).padStart(3, "0") + "–" + String(section.end_bar + 1).padStart(3, "0"), head, "lyric-card-bar");
            const textarea = element("textarea", null, card, "lyric-text");
            textarea.value = section.lyrics;
            textarea.setAttribute("aria-label", "Lyrics for " + section.name);
            textarea.spellcheck = false;
            textarea.onclick = event => { event.stopPropagation(); this.selectSection(index); };
            textarea.oninput = () => {
                section.lyrics = textarea.value;
                textarea.style.height = "auto";
                textarea.style.height = Math.max(88, textarea.scrollHeight) + "px";
                this.syncLyricsLayout(lyricsTrack);
                this.writeState();
            };
            this.lyricsResizeObserver.observe(textarea);
        }
        this.syncLyricsLayout(lyricsTrack);

        this.legend.replaceChildren();
        element("span", "PIANO ROLL", this.legend, "legend-item");
        for (const [name, color] of [["Vocal", trackColors.Vocal], ["Ins", trackColors.Ins]]) {
            const item = element("span", null, this.legend, "legend-item");
            const swatch = element("span", null, item, "legend-swatch");
            swatch.style.background = color;
            element("span", name, item);
        }
    }

    syncLyricsLayout(lyricsTrack) {
        if (!lyricsTrack) return;
        let height = 0;
        for (const card of lyricsTrack.querySelectorAll(".lyric-card")) {
            const textarea = card.querySelector(".lyric-text");
            const head = card.querySelector(".lyric-card-head");
            card.style.height = (head.offsetHeight + textarea.offsetHeight) + "px";
            height = Math.max(height, card.offsetTop + card.offsetHeight);
        }
        height = Math.max(112, height + 8);
        lyricsTrack.style.height = height + "px";
        this.lyricsHeight = height;
        this.renderedTimelineHeight = 44 + this.rollHeight + height;
        this.timelineScroll.style.height = this.renderedTimelineHeight + "px";
        this.layout();
    }

    async requestViewerData(payload) {
        const response = await fetch(api.fileURL("/hz3/yue2/abc_viewer/data"), {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Could not read this ABC and lyrics pair.");
        return result;
    }

    setLoadedWidget(name, value) {
        const widget = this[name + "Widget"] || this.node.widgets?.find(item => item.name === name);
        if (!widget) return;
        widget.value = value;
        widget.callback?.(value);
    }

    async loadFiles(files) {
        if (!files.length) return;
        try {
            let abcText = "";
            let lyricsText = "";
            let sections;
            const jsonFile = files.find(file => file.name.toLowerCase().endsWith(".json"));
            if (jsonFile) {
                const bundle = JSON.parse(await jsonFile.text());
                abcText = bundle.abc || bundle.abc_with_sections || "";
                lyricsText = bundle.lyrics || bundle.lyrics_with_sections || "";
                sections = Array.isArray(bundle.sections) ? bundle.sections : undefined;
            } else {
                const abcFile = files.find(file => file.name.toLowerCase().endsWith(".abc"));
                const lyricFile = files.find(file => /\.(txt|lrc)$/i.test(file.name));
                if (!abcFile) throw new Error("Select an .abc file, or an exported .json pair.");
                abcText = await abcFile.text();
                lyricsText = lyricFile ? await lyricFile.text() : "";
            }
            if (!abcText.trim()) throw new Error("The selected ABC file is empty.");
            const data = await this.requestViewerData({abc: abcText, lyrics: lyricsText, sections});
            this.stop();
            this.data = data;
            this.state = JSON.parse(data.editor_state);
            this.selectedSection = this.state.sections.length ? "0" : "all";
            this.playbackSection = "all";
            this.position = 0;
            this.setLoadedWidget("loaded_abc", abcText);
            this.setLoadedWidget("loaded_lyrics", lyricsText);
            this.writeState(false);
            this.hasEdits = true;
            this.render();
            this.status.textContent = "ABC + lyrics loaded · queue this node to update its outputs.";
        } catch (error) {
            this.status.textContent = "Load failed · " + error.message;
        } finally {
            this.fileInput.value = "";
        }
    }

    async exportPair() {
        if (!this.data || !this.state) return;
        try {
            const rendered = await this.requestViewerData({
                abc: this.data.abc,
                lyrics: this.data.lyrics,
                editor_state: JSON.stringify(this.state),
            });
            const bundle = {
                format: "hz3-abc-lyrics-v1",
                abc: rendered.edited_abc,
                lyrics: rendered.edited_lyrics,
                sections: rendered.sections.map(section => ({...section})),
            };
            const blob = new Blob([JSON.stringify(bundle, null, 2)], {type: "application/json"});
            const url = URL.createObjectURL(blob);
            const anchor = element("a", null, this.root);
            anchor.href = url;
            anchor.download = "abc-lyrics.json";
            anchor.click();
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            this.status.textContent = "Exported ABC + lyrics.";
        } catch (error) {
            this.status.textContent = "Export failed · " + error.message;
        }
    }

    clearImportedPair() {
        this.setLoadedWidget("loaded_abc", "");
        this.setLoadedWidget("loaded_lyrics", "");
        this.hasEdits = true;
        this.status.textContent = "Connected ABC + lyrics will be restored when you queue this node.";
    }

    render() {
        if (!this.data || !this.state) return;
        if (this.selectedSection === "all" || Number(this.selectedSection) >= this.state.sections.length) {
            this.selectedSection = this.state.sections.length ? "0" : "all";
        }
        this.toolbar.replaceChildren();
        const add = button(this.toolbar, "+ Add lyric section", () => this.addSection());
        add.disabled = this.state.sections.length >= Math.min(64, this.data.bars.length);
        button(this.toolbar, "Load ABC + lyrics", () => this.fileInput.click(), "secondary");
        button(this.toolbar, "Export ABC + lyrics", () => this.exportPair(), "secondary");
        const hasLoadedPair = Boolean(this.loaded_abcWidget?.value);
        if (hasLoadedPair) button(this.toolbar, "Use connected inputs", () => this.clearImportedPair(), "secondary");
        this.editSectionSelect = select(this.toolbar, "Edit section", this.state.sections.map((section, index) =>
            [String(index), "BAR " + String(section.start_bar + 1).padStart(3, "0") + " · " + section.name]),
        this.selectedSection, value => {
            this.selectedSection = value;
            this.render();
            this.focusSection(value);
        });
        const selectedIndex = Number(this.selectedSection);
        const selected = this.state.sections[selectedIndex];
        if (selected) {
            const previous = this.state.sections[selectedIndex - 1];
            const next = this.state.sections[selectedIndex + 1];
            this.startBarInput = numberInput(this.toolbar, "Start BAR", selected.start_bar + 1,
                (previous?.start_bar ?? -1) + 2, next?.start_bar ?? this.data.bars.length,
                value => this.setSectionBar(selectedIndex, "start_bar", value));
            this.endBarInput = numberInput(this.toolbar, "End BAR", selected.end_bar + 1,
                selected.start_bar + 1, next?.start_bar ?? this.data.bars.length,
                value => this.setSectionBar(selectedIndex, "end_bar", value));
            if (selectedIndex > 0) {
                button(this.toolbar, "Remove section", () => this.removeSection(selectedIndex), "danger");
            }
        }
        this.playRangeSelect = select(this.toolbar, "Play", [["all", "Full ABC"],
            ...this.state.sections.map((section, index) => [String(index), "BAR " + String(section.start_bar + 1).padStart(3, "0") + " · " + section.name])],
        this.playbackSection, value => {
            this.stop();
            this.playbackSection = value;
            this.position = this.playbackRange().start;
            this.updatePlaybackStatus();
        });
        select(this.toolbar, "Track", [["Both", "Vocal + Ins"], ["Vocal", "Vocal"], ["Ins", "Ins"]], this.track,
            value => { this.stop(); this.track = value; this.render(); });
        const chordLabel = element("label", "Chords", this.toolbar, "control-label");
        const chordToggle = element("input", null, chordLabel);
        chordToggle.type = "checkbox";
        chordToggle.checked = this.chords;
        chordToggle.onchange = () => { this.chords = chordToggle.checked; };
        select(this.toolbar, "Zoom", [["0.7", "70%"], ["0.85", "85%"], ["1", "100%"], ["1.2", "120%"], ["1.5", "150%"]],
            String(this.zoom), value => { this.zoom = Number(value); this.render(); });
        this.playButton = button(this.toolbar, this.audio ? "❚❚ Pause ABC" : "▶ Play ABC", () => this.togglePlayback());
        button(this.toolbar, "■ Stop", () => this.stop(), "secondary");
        this.expandButton = button(this.toolbar, this.expanded ? "Close expanded" : "Expand editor", () => this.expand(!this.expanded), "secondary");
        this.status = element("span", "", this.toolbar, "status");
        this.hint.textContent = this.data.rough_layout
            ? "Section positions are starting estimates. Set each start/end BAR in the number boxes; click any BAR number to play from there."
            : "Set the start/end BAR numbers for each colored lyric section. Click any BAR number to play from there.";
        this.renderTimeline();
        this.syncRangeInputs();
        this.focusSection(this.selectedSection);
        this.updatePlaybackStatus();
        this.layout();
    }

    expand(value) {
        if (value === this.expanded) return;
        this.expanded = value;
        if (value) {
            this.home = this.root.parentNode;
            document.body.append(this.root);
            this.root.classList.add("hz3-section-editor-expanded");
        } else {
            this.root.classList.remove("hz3-section-editor-expanded");
            this.home?.append(this.root);
        }
        if (this.expandButton) this.expandButton.textContent = this.expanded ? "Close expanded" : "Expand editor";
        this.layout();
    }

    layout() {
        if (!this.expanded && this.node.size) {
            const toolbarHeight = this.toolbar.getBoundingClientRect().height || 34;
            const hintHeight = this.hint.getBoundingClientRect().height || 32;
            const legendHeight = this.legend.getBoundingClientRect().height || 18;
            const height = Math.max(540, toolbarHeight + hintHeight + legendHeight + (this.renderedTimelineHeight || 300) + 48);
            this.root.style.height = height + "px";
            if (Math.abs(this.node.size[1] - (height + 55)) > 4) this.node.setSize([Math.max(1000, this.node.size[0]), height + 55]);
        }
        this.node.graph?.setDirtyCanvas(true, true);
    }

    destroy() {
        this.observer.disconnect();
        this.expand(false);
    }
}

app.registerExtension({
    name: "HZ3.YuE2.ABCSectionEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HZ3_YuE2_ABCViewer") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            this.hz3SectionEditor = new ABCSectionEditor(this);
            this.hz3SectionEditor.bindStateWidget();
            requestAnimationFrame(() => this.hz3SectionEditor?.bindStateWidget());
            const widget = this.addDOMWidget("hz3_abc_viewer", "custom", this.hz3SectionEditor.root, {
                serialize: false,
                getMinHeight: () => 540,
                getHeight: () => Math.max(540, this.hz3SectionEditor.root.scrollHeight),
            });
            widget.computeSize = () => [1000, 560];
            this.setSize([Math.max(1000, this.size[0]), 640]);
        };
        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            resized?.apply(this, arguments);
            size[0] = this.size[0] = Math.max(1000, size[0]);
            this.hz3SectionEditor?.layout();
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            const data = message.abc_viewer?.[0];
            if (data) this.hz3SectionEditor?.receive(data);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.hz3SectionEditor?.destroy();
            return removed?.apply(this, arguments);
        };
    },
});
