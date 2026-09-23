import { app } from "../../scripts/app.js";

function element(tag, text, parent, className = "") {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = text;
    if (className) node.className = className;
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

const style = document.createElement("style");
style.textContent = `
.hz3-section-editor{height:100%;min-height:540px;padding:8px;box-sizing:border-box;background:#111827;color:#edf7f5;font:13px system-ui;border:1px solid #25a98e;border-radius:8px;display:flex;flex-direction:column;overflow:hidden}
.hz3-section-editor *{box-sizing:border-box}.hz3-section-editor .toolbar{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;flex:0 0 auto}
.hz3-section-editor button{background:#16816d;color:#fff;border:1px solid #28b89c;border-radius:5px;padding:7px 10px;cursor:pointer}.hz3-section-editor button.secondary{background:#263548;border-color:#486174}.hz3-section-editor button:disabled{opacity:.45;cursor:default}
.hz3-section-editor .control-label{display:flex;align-items:center;gap:5px;color:#b7c9d6;font-size:11px}.hz3-section-editor .control-select{max-width:180px;background:#1e293b;color:#f8fafc;border:1px solid #486174;border-radius:5px;padding:6px 8px}
.hz3-section-editor .hint{color:#b6c7d6;font-size:11px;line-height:1.3;padding:5px 7px;border-left:3px solid #4ee0bd;background:#162234;margin-bottom:6px;flex:0 0 auto}
.hz3-section-editor .status{color:#8fe3d0;margin-left:auto;min-width:180px}.hz3-section-editor .panel-heads{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;margin-bottom:5px;flex:0 0 auto}
.hz3-section-editor .panel-title{background:#192638;color:#d8f5ee;padding:7px 10px;font-weight:700;letter-spacing:.05em;border:1px solid #405368;border-radius:5px}
.hz3-section-editor .workspace{overflow:auto;min-height:0;flex:1;padding:5px 8px;border:1px solid #405368;border-radius:7px;background:#0d1522;overscroll-behavior:contain;scrollbar-gutter:stable}
.hz3-section-editor .section-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;align-items:start;padding-bottom:1px}
.hz3-section-editor .section-column{min-width:0;align-self:start}
.hz3-section-editor .cell{border:1px solid #354c60;border-radius:7px;background:#141f2e;margin:2px 0 10px;overflow:hidden}
.hz3-section-editor .cell-head{display:flex;align-items:center;gap:7px;padding:7px 8px;background:#1b2b3d;border-bottom:1px solid #354c60}
.hz3-section-editor .cell-index{font-size:11px;color:#9fb3c4;white-space:nowrap}.hz3-section-editor .cell-title{min-width:0;flex:1;background:#101827;color:#e7f4f1;border:1px solid #405368;border-radius:4px;padding:5px 7px;font:600 12px system-ui}
.hz3-section-editor .cell-remove{padding:4px 7px!important;background:#54343b!important;border-color:#8e5961!important}.hz3-section-editor .cell-body{padding:3px 7px 7px}
.hz3-section-editor .item{border-left:2px solid #4ee0bd;background:#0f1a28;color:#dce7ef;margin:3px 0;padding:6px 8px;border-radius:3px;line-height:1.45;overflow-wrap:anywhere}
.hz3-section-editor .lyric-item{border-color:#d99cff;white-space:pre-wrap;min-height:32px;outline:none}.hz3-section-editor .lyric-item:focus{box-shadow:0 0 0 1px #d99cff}.hz3-section-editor .abc-item{font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
.hz3-section-editor .abc-item{display:grid;grid-template-columns:52px minmax(0,1fr);gap:5px;padding:3px 5px;margin:2px 0;align-items:start}.hz3-section-editor .abc-bar-index{font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#9fb3c4;text-align:right;white-space:nowrap;padding-top:1px}.hz3-section-editor .abc-bar-content{min-width:0;overflow-x:auto;scrollbar-width:thin}.hz3-section-editor .abc-voice{font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre;overflow-wrap:normal}.hz3-section-editor .abc-vocal{color:#e5bdff}.hz3-section-editor .abc-ins{color:#83ead3}
.hz3-section-editor .drop-slot{height:8px;margin:0 -2px;position:relative;border-radius:4px;transition:background .12s}.hz3-section-editor .drop-slot.drag-over{height:20px;background:#1c6055}
.hz3-section-editor .drop-slot.has-boundary{height:auto;min-height:33px;margin:2px -2px}.hz3-section-editor .drop-slot.has-boundary.drag-over{min-height:35px}
.hz3-section-editor .boundary{display:flex;align-items:center;justify-content:center;width:100%;min-height:33px;margin:0;background:#203c46!important;border:1px dashed #55cbb1!important;color:#c4fff0!important;font-size:10px;line-height:1.25;padding:5px 7px!important;white-space:normal;cursor:grab!important;touch-action:none}
.hz3-section-editor .boundary:active{cursor:grabbing!important}.hz3-section-editor .empty{padding:7px;color:#8499ab;font-style:italic}
.hz3-section-editor .widget-hidden{display:none!important}
.hz3-section-editor-expanded{position:fixed!important;inset:2vh 2vw!important;width:96vw!important;height:96vh!important;z-index:10000!important;box-shadow:0 0 0 4vh #000a}
`;
document.head.append(style);

class ABCSectionEditor {
    constructor(node) {
        this.node = node;
        this.root = element("div", null, null, "hz3-section-editor");
        this.toolbar = element("div", null, this.root, "toolbar");
        this.hint = element("div", "Existing section labels provide a rough starting point. Drag each divider separately in Lyrics and ABC; no syllable-to-note alignment is assumed.", this.root, "hint");
        this.panelHeads = element("div", null, this.root, "panel-heads");
        this.lyricHeading = element("div", "LYRICS", this.panelHeads, "panel-title");
        this.abcHeading = element("div", "ABC · measures", this.panelHeads, "panel-title");
        this.workspace = element("div", null, this.root, "workspace");
        this.lyricPanel = {cells: []};
        this.abcPanel = {cells: []};
        this.track = "Both";
        this.selectedSection = "all";
        this.position = 0;
        this.chords = true;
        this.status = element("span", "Run this node to load the inputs.", this.toolbar, "status");
        this.expanded = false;
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
        this.node.setSize([Math.max(1020, this.node.size[0]), 640]);
    }

    receive(data) {
        this.stop();
        this.data = typeof data === "string" ? JSON.parse(data) : data;
        this.selectedSection = "all";
        this.position = 0;
        this.state = {
            source_hash: this.data.source_hash,
            sections: this.data.sections.map(section => ({...section})),
            lyric_lines: [...this.data.lyric_lines],
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
        if (showQueuedHint) this.status.textContent = "Changes saved here · queue this node to refresh its outputs.";
        this.node.graph?.setDirtyCanvas(true, true);
    }

    setName(index, value) {
        const name = value.replace(/[\r\n\[\]]+/g, " ").trim().slice(0, 100) || `Section ${index + 1}`;
        this.state.sections[index].name = name;
        this.writeState();
        this.render();
    }

    addSection() {
        if (!this.state || this.state.sections.length >= 64) return;
        this.stop();
        const lyricEnd = this.state.lyric_lines.length;
        const abcEnd = this.data.bars.length;
        this.state.sections[this.state.sections.length - 1].lyrics_end = lyricEnd;
        this.state.sections[this.state.sections.length - 1].abc_end = abcEnd;
        const index = this.state.sections.length + 1;
        this.state.sections.push({id: `section-${Date.now()}-${index}`, name: `Section ${index}`, lyrics_end: lyricEnd, abc_end: abcEnd});
        this.selectedSection = String(index - 1);
        this.writeState();
        this.render();
    }

    removeSection(index) {
        const sections = this.state.sections;
        if (sections.length <= 1) return;
        this.stop();
        if (index > 0) {
            sections[index - 1].lyrics_end = sections[index].lyrics_end;
            sections[index - 1].abc_end = sections[index].abc_end;
        }
        sections.splice(index, 1);
        sections.forEach((section, i) => { section.id = section.id || `section-${i + 1}`; });
        sections[sections.length - 1].lyrics_end = this.state.lyric_lines.length;
        sections[sections.length - 1].abc_end = this.data.bars.length;
        if (this.selectedSection !== "all") {
            const selected = Number(this.selectedSection);
            this.selectedSection = selected === index ? String(Math.max(0, index - 1))
                : String(selected > index ? selected - 1 : selected);
        }
        this.writeState();
        this.render();
    }

    moveBoundary(side, sectionIndex, target) {
        const sections = this.state.sections;
        if (sectionIndex < 0 || sectionIndex >= sections.length - 1) return;
        this.stop();
        const field = side === "lyrics" ? "lyrics_end" : "abc_end";
        const lower = sectionIndex === 0 ? 0 : sections[sectionIndex - 1][field];
        const upper = sections[sectionIndex + 1][field];
        sections[sectionIndex][field] = Math.max(lower, Math.min(upper, target));
        this.writeState();
        this.render();
    }

    playbackRange() {
        if (this.selectedSection === "all") return {start: 0, end: this.data.total_ticks};
        const index = Number(this.selectedSection);
        const startBar = index === 0 ? 0 : this.state.sections[index - 1].abc_end;
        const endBar = this.state.sections[index].abc_end;
        const start = startBar < this.data.bars.length ? this.data.bars[startBar].start : this.data.total_ticks;
        const end = endBar < this.data.bars.length ? this.data.bars[endBar].start : this.data.total_ticks;
        return {start, end};
    }

    focusSection(index) {
        if (index === "all") {
            this.workspace.scrollTo({top: 0, behavior: "smooth"});
            return;
        }
        const cell = this.lyricPanel.cells?.[Number(index)] || this.abcPanel.cells?.[Number(index)];
        const row = cell?.closest(".section-row");
        if (!row) return;
        const workspaceTop = this.workspace.getBoundingClientRect().top;
        const rowTop = row.getBoundingClientRect().top;
        this.workspace.scrollTo({top: this.workspace.scrollTop + rowTop - workspaceTop, behavior: "smooth"});
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
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(volume, start + Math.min(.012, Math.max(.002, (end - start) / 3)));
        gain.gain.setValueAtTime(volume, Math.max(start + .014, end - .025));
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
        const context = new AudioContext();
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
            for (const note of this.data.tracks[track]) {
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
            const chords = [...this.data.chords].sort((a, b) => a.start - b.start);
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
        const clock = `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
        const state = this.audio ? `Playing ABC · bar ${barNumber} · ${clock}` : `${this.data.bars.length} bars · ${this.data.bpm} BPM · ${this.data.key}`;
        this.status.textContent = state;
    }

    renderSectionCell(side, parent, sectionIndex, start) {
        const isLyrics = side === "lyrics";
        const rows = isLyrics ? this.state.lyric_lines : this.data.bars;
        const endField = isLyrics ? "lyrics_end" : "abc_end";
        const section = this.state.sections[sectionIndex];
        const end = Math.max(start, Math.min(rows.length, section[endField]));
        const cell = element("article", null, parent, "cell");
        (isLyrics ? this.lyricPanel : this.abcPanel).cells[sectionIndex] = cell;
        const head = element("div", null, cell, "cell-head");
        element("span", `SECTION ${String(sectionIndex + 1).padStart(2, "0")}`, head, "cell-index");
        const title = element("input", null, head, "cell-title");
        title.value = section.name;
        title.setAttribute("aria-label", `Section ${sectionIndex + 1} name`);
        title.onchange = () => this.setName(sectionIndex, title.value);
        if (this.state.sections.length > 1) {
            button(head, "×", () => this.removeSection(sectionIndex), "cell-remove").title = "Remove and merge section";
        }
        const body = element("div", null, cell, "cell-body");

        const makeSlot = rowIndex => {
            const slot = element("div", null, body, "drop-slot");
            slot.dataset.index = String(rowIndex);
            slot.ondragover = event => {
                if (Array.from(event.dataTransfer?.types || []).includes("application/x-hz3-section-boundary")) {
                    event.preventDefault();
                    slot.classList.add("drag-over");
                }
            };
            slot.ondragleave = () => slot.classList.remove("drag-over");
            slot.ondrop = event => {
                event.preventDefault();
                slot.classList.remove("drag-over");
                try {
                    const payload = JSON.parse(event.dataTransfer.getData("application/x-hz3-section-boundary"));
                    if (payload.side === side) this.moveBoundary(side, payload.section, Number(slot.dataset.index));
                } catch (_) { /* Ignore unrelated drag data. */ }
            };
            return slot;
        };

        const addBoundaryIfHere = (slot, position) => {
            if (sectionIndex >= this.state.sections.length - 1 || position !== end) return;
            slot.classList.add("has-boundary");
            const nextName = this.state.sections[sectionIndex + 1].name;
            const handle = button(slot, `↕ Drag boundary · next: ${nextName}`, () => {}, "boundary");
            handle.draggable = true;
            handle.ondragstart = event => {
                event.dataTransfer.setData("application/x-hz3-section-boundary", JSON.stringify({side, section: sectionIndex}));
                event.dataTransfer.effectAllowed = "move";
                this.selectedSection = String(sectionIndex);
            };
        };

        if (start === end) {
            const slot = makeSlot(start);
            addBoundaryIfHere(slot, start);
            element("div", rows.length ? "Empty section · drag its boundary to assign content." : "No lines supplied.", body, "empty");
        } else {
            for (let rowIndex = start; rowIndex < end; rowIndex++) {
                if (rowIndex === start) makeSlot(rowIndex);
                if (isLyrics) {
                    const line = element("div", this.state.lyric_lines[rowIndex], body, "item lyric-item");
                    line.contentEditable = "true";
                    line.spellcheck = false;
                    line.dataset.index = String(rowIndex);
                    line.oninput = () => {
                        this.state.lyric_lines[rowIndex] = line.innerText.replace(/[\r\n]+/g, " ");
                        this.writeState();
                    };
                } else {
                    const bar = this.data.bars[rowIndex];
                    const box = element("div", null, body, "item abc-item");
                    element("span", `${String(bar.number).padStart(3, "0")} →`, box, "abc-bar-index");
                    const content = element("div", null, box, "abc-bar-content");
                    element("div", `V  ${bar.vocal}`, content, "abc-voice abc-vocal");
                    element("div", `I   ${bar.ins}`, content, "abc-voice abc-ins");
                }
                const slot = makeSlot(rowIndex + 1);
                addBoundaryIfHere(slot, rowIndex + 1);
            }
        }
    }

    render() {
        if (!this.data || !this.state) return;
        if (this.selectedSection !== "all" && Number(this.selectedSection) >= this.state.sections.length) {
            this.selectedSection = "all";
        }
        this.toolbar.replaceChildren();
        button(this.toolbar, "+ Add section", () => this.addSection());
        select(this.toolbar, "Align section", [["all", "All sections"],
            ...this.state.sections.map((section, index) => [String(index), `${String(index + 1).padStart(2, "0")} · ${section.name}`])],
        this.selectedSection, value => {
            this.stop();
            this.selectedSection = value;
            this.position = this.playbackRange().start;
            this.focusSection(value);
            this.updatePlaybackStatus();
        });
        select(this.toolbar, "Track", [["Both", "Vocal + Ins"], ["Vocal", "Vocal"], ["Ins", "Ins"]], this.track,
            value => { this.stop(); this.track = value; });
        const chordLabel = element("label", "Chords", this.toolbar, "control-label");
        const chordToggle = element("input", null, chordLabel);
        chordToggle.type = "checkbox";
        chordToggle.checked = this.chords;
        chordToggle.onchange = () => { this.chords = chordToggle.checked; };
        this.playButton = button(this.toolbar, this.audio ? "❚❚ Pause ABC" : "▶ Play ABC", () => this.togglePlayback());
        button(this.toolbar, "■ Stop", () => this.stop(), "secondary");
        this.expandButton = button(this.toolbar, this.expanded ? "Close expanded" : "Expand editor", () => this.expand(!this.expanded), "secondary");
        this.status = element("span", "", this.toolbar, "status");
        this.hint.textContent = this.data.rough_layout
            ? "Los cortes iniciales son aproximados. Arrastra cada límite por separado en Lyrics y ABC."
            : "Arrastra cada límite por separado en Lyrics y ABC para ajustar el inicio y fin de las secciones.";
        this.lyricHeading.textContent = `LYRICS · ${this.state.lyric_lines.length} lines`;
        this.abcHeading.textContent = `ABC · ${this.data.bars.length} bars`;
        this.workspace.replaceChildren();
        this.lyricPanel.cells = [];
        this.abcPanel.cells = [];
        let lyricStart = 0;
        let abcStart = 0;
        this.state.sections.forEach((section, sectionIndex) => {
            const row = element("div", null, this.workspace, "section-row");
            const lyrics = element("div", null, row, "section-column");
            const abc = element("div", null, row, "section-column");
            this.renderSectionCell("lyrics", lyrics, sectionIndex, lyricStart);
            this.renderSectionCell("abc", abc, sectionIndex, abcStart);
            lyricStart = section.lyrics_end;
            abcStart = section.abc_end;
        });
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
            const height = Math.max(540, Math.min(700, this.root.scrollHeight + 20));
            if (this.node.size[1] < height + 55) this.node.setSize([Math.max(1020, this.node.size[0]), height + 55]);
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
            widget.computeSize = () => [1020, 560];
            this.setSize([Math.max(1020, this.size[0]), 640]);
        };
        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            resized?.apply(this, arguments);
            size[0] = this.size[0] = Math.max(1020, size[0]);
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
