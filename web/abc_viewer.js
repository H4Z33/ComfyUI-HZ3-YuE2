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

const style = document.createElement("style");
style.textContent = `
.hz3-section-editor{height:100%;min-height:760px;padding:12px;box-sizing:border-box;background:#111827;color:#edf7f5;font:13px system-ui;border:1px solid #25a98e;border-radius:8px;display:flex;flex-direction:column;overflow:hidden}
.hz3-section-editor *{box-sizing:border-box}.hz3-section-editor .toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;flex:0 0 auto}
.hz3-section-editor button{background:#16816d;color:#fff;border:1px solid #28b89c;border-radius:5px;padding:7px 10px;cursor:pointer}.hz3-section-editor button.secondary{background:#263548;border-color:#486174}.hz3-section-editor button:disabled{opacity:.45;cursor:default}
.hz3-section-editor .hint{color:#b6c7d6;font-size:12px;line-height:1.45;padding:7px 9px;border-left:3px solid #4ee0bd;background:#162234;margin-bottom:9px;flex:0 0 auto}
.hz3-section-editor .status{color:#8fe3d0;margin-left:auto;min-width:180px}.hz3-section-editor .panels{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px;min-height:0;flex:1}
.hz3-section-editor .panel{min-width:0;min-height:0;background:#0d1522;border:1px solid #405368;border-radius:7px;display:flex;flex-direction:column;overflow:hidden}
.hz3-section-editor .panel-title{position:sticky;top:0;z-index:2;background:#192638;color:#d8f5ee;padding:9px 11px;font-weight:700;letter-spacing:.05em;border-bottom:1px solid #405368}
.hz3-section-editor .panel-content{overflow:auto;min-height:0;flex:1;padding:8px;overscroll-behavior:contain;scrollbar-gutter:stable}
.hz3-section-editor .cell{border:1px solid #354c60;border-radius:7px;background:#141f2e;margin:2px 0 10px;overflow:hidden}
.hz3-section-editor .cell-head{display:flex;align-items:center;gap:7px;padding:7px 8px;background:#1b2b3d;border-bottom:1px solid #354c60}
.hz3-section-editor .cell-index{font-size:11px;color:#9fb3c4;white-space:nowrap}.hz3-section-editor .cell-title{min-width:0;flex:1;background:#101827;color:#e7f4f1;border:1px solid #405368;border-radius:4px;padding:5px 7px;font:600 12px system-ui}
.hz3-section-editor .cell-remove{padding:4px 7px!important;background:#54343b!important;border-color:#8e5961!important}.hz3-section-editor .cell-body{padding:3px 7px 7px}
.hz3-section-editor .item{border-left:2px solid #4ee0bd;background:#0f1a28;color:#dce7ef;margin:3px 0;padding:6px 8px;border-radius:3px;line-height:1.45;overflow-wrap:anywhere}
.hz3-section-editor .lyric-item{border-color:#d99cff;white-space:pre-wrap;min-height:32px;outline:none}.hz3-section-editor .lyric-item:focus{box-shadow:0 0 0 1px #d99cff}.hz3-section-editor .abc-item{font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
.hz3-section-editor .abc-bar-head{display:flex;justify-content:space-between;color:#9fb3c4;font:10px system-ui;margin-bottom:3px}.hz3-section-editor .abc-voice{white-space:pre-wrap;overflow-wrap:anywhere}.hz3-section-editor .abc-vocal{color:#e5bdff}.hz3-section-editor .abc-ins{color:#83ead3}
.hz3-section-editor .drop-slot{height:8px;margin:0 -2px;position:relative;border-radius:4px;transition:background .12s}.hz3-section-editor .drop-slot.drag-over{height:20px;background:#1c6055}
.hz3-section-editor .boundary{display:flex;align-items:center;justify-content:center;width:100%;min-height:25px;margin:1px 0;background:#203c46!important;border:1px dashed #55cbb1!important;color:#c4fff0!important;font-size:10px;padding:4px 6px!important;cursor:grab!important;touch-action:none}
.hz3-section-editor .boundary:active{cursor:grabbing!important}.hz3-section-editor .empty{padding:7px;color:#8499ab;font-style:italic}
.hz3-section-editor .panel-foot{font-size:10px;color:#8195a8;padding:4px 8px 7px}.hz3-section-editor .widget-hidden{display:none!important}
.hz3-section-editor-expanded{position:fixed!important;inset:2vh 2vw!important;width:96vw!important;height:96vh!important;z-index:10000!important;box-shadow:0 0 0 4vh #000a}
`;
document.head.append(style);

class ABCSectionEditor {
    constructor(node) {
        this.node = node;
        this.root = element("div", null, null, "hz3-section-editor");
        this.toolbar = element("div", null, this.root, "toolbar");
        this.hint = element("div", "Existing section labels provide a rough starting point. Drag each divider separately in Lyrics and ABC; no syllable-to-note alignment is assumed.", this.root, "hint");
        this.panels = element("div", null, this.root, "panels");
        this.lyricPanel = this.makePanel("LYRICS", this.panels);
        this.abcPanel = this.makePanel("ABC · measures", this.panels);
        this.status = element("span", "Run this node to load the inputs.", this.toolbar, "status");
        this.expanded = false;
        this.observer = new ResizeObserver(() => this.layout());
        this.observer.observe(this.root);
    }

    makePanel(title, parent) {
        const panel = element("section", null, parent, "panel");
        const heading = element("div", title, panel, "panel-title");
        const content = element("div", null, panel, "panel-content");
        const foot = element("div", "Drag a divider onto a line/bar position to change this panel's section range.", panel, "panel-foot");
        return {panel, heading, content, foot};
    }

    bindStateWidget() {
        this.stateWidget = this.node.widgets?.find(widget => widget.name === "editor_state");
        if (!this.stateWidget) return;
        this.stateWidget.computeSize = () => [0, -4];
        this.stateWidget.options = {...(this.stateWidget.options || {}), serialize: true};
        this.stateWidget.element?.classList.add("widget-hidden");
        this.stateWidget.element?.setAttribute("aria-hidden", "true");
        this.node.setSize([Math.max(1120, this.node.size[0]), Math.max(850, this.node.size[1])]);
    }

    receive(data) {
        this.data = typeof data === "string" ? JSON.parse(data) : data;
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
        const lyricEnd = this.state.lyric_lines.length;
        const abcEnd = this.data.bars.length;
        this.state.sections[this.state.sections.length - 1].lyrics_end = lyricEnd;
        this.state.sections[this.state.sections.length - 1].abc_end = abcEnd;
        const index = this.state.sections.length + 1;
        this.state.sections.push({id: `section-${Date.now()}-${index}`, name: `Section ${index}`, lyrics_end: lyricEnd, abc_end: abcEnd});
        this.writeState();
        this.render();
    }

    removeSection(index) {
        const sections = this.state.sections;
        if (sections.length <= 1) return;
        if (index > 0) {
            sections[index - 1].lyrics_end = sections[index].lyrics_end;
            sections[index - 1].abc_end = sections[index].abc_end;
        }
        sections.splice(index, 1);
        sections.forEach((section, i) => { section.id = section.id || `section-${i + 1}`; });
        sections[sections.length - 1].lyrics_end = this.state.lyric_lines.length;
        sections[sections.length - 1].abc_end = this.data.bars.length;
        this.writeState();
        this.render();
    }

    moveBoundary(side, sectionIndex, target) {
        const sections = this.state.sections;
        if (sectionIndex < 0 || sectionIndex >= sections.length - 1) return;
        const field = side === "lyrics" ? "lyrics_end" : "abc_end";
        const lower = sectionIndex === 0 ? 0 : sections[sectionIndex - 1][field];
        const upper = sections[sectionIndex + 1][field];
        sections[sectionIndex][field] = Math.max(lower, Math.min(upper, target));
        this.writeState();
        this.render();
    }

    renderPanel(side, target) {
        target.content.replaceChildren();
        if (!this.state) return;
        const isLyrics = side === "lyrics";
        const rows = isLyrics ? this.state.lyric_lines : this.data.bars;
        const endField = isLyrics ? "lyrics_end" : "abc_end";
        let start = 0;

        this.state.sections.forEach((section, sectionIndex) => {
            const end = Math.max(start, Math.min(rows.length, section[endField]));
            const cell = element("article", null, target.content, "cell");
            const head = element("div", null, cell, "cell-head");
            element("span", `CELL ${String(sectionIndex + 1).padStart(2, "0")}`, head, "cell-index");
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
                const nextName = this.state.sections[sectionIndex + 1].name;
                const handle = button(slot, `↕ Drag boundary · next: ${nextName}`, () => {}, "boundary");
                handle.draggable = true;
                handle.ondragstart = event => {
                    event.dataTransfer.setData("application/x-hz3-section-boundary", JSON.stringify({side, section: sectionIndex}));
                    event.dataTransfer.effectAllowed = "move";
                };
            };

            if (start === end) {
                const slot = makeSlot(start);
                addBoundaryIfHere(slot, start);
                element("div", "Empty section · drag its boundary to assign content.", body, "empty");
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
                        const barHead = element("div", null, box, "abc-bar-head");
                        element("span", `BAR ${String(bar.number).padStart(3, "0")}`, barHead);
                        element("span", bar.meter, barHead);
                        element("div", `Vocal  ${bar.vocal}`, box, "abc-voice abc-vocal");
                        element("div", `Ins      ${bar.ins}`, box, "abc-voice abc-ins");
                    }
                    const slot = makeSlot(rowIndex + 1);
                    addBoundaryIfHere(slot, rowIndex + 1);
                }
            }
            start = end;
        });
        if (!rows.length) element("div", "No lines supplied.", target.content, "empty");
        target.heading.textContent = isLyrics ? `LYRICS · ${rows.length} lines` : `ABC · ${rows.length} bars`;
        target.foot.textContent = isLyrics
            ? "Edit lyric lines inline. Drag section dividers between lines to adjust the range."
            : "Each card shows native Vocal/Ins measures. Drag section dividers between bars to adjust the range.";
    }

    render() {
        if (!this.data || !this.state) return;
        this.toolbar.replaceChildren();
        button(this.toolbar, "+ Add section", () => this.addSection());
        this.expandButton = button(this.toolbar, this.expanded ? "Close expanded" : "Expand editor", () => this.expand(!this.expanded), "secondary");
        this.status = element("span", `${this.data.bpm} BPM · ${this.data.key} · ${this.data.bars.length} bars`, this.toolbar, "status");
        this.hint.textContent = this.data.rough_layout
            ? "Starting cuts use source section labels where possible; missing cuts are rough estimates. Drag each divider independently in Lyrics and ABC. No syllable-to-note alignment is assumed."
            : "Starting cuts use the existing section labels. Drag each divider independently in Lyrics and ABC. No syllable-to-note alignment is assumed.";
        this.renderPanel("lyrics", this.lyricPanel);
        this.renderPanel("abc", this.abcPanel);
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
            const height = Math.max(850, Math.min(1150, this.root.scrollHeight + 50));
            if (this.node.size[1] < height + 70) this.node.setSize([Math.max(1120, this.node.size[0]), height + 70]);
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
                getMinHeight: () => 760,
                getHeight: () => Math.max(760, this.hz3SectionEditor.root.scrollHeight),
            });
            widget.computeSize = () => [1120, 780];
            this.setSize([Math.max(1120, this.size[0]), Math.max(850, this.size[1])]);
        };
        const resized = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            resized?.apply(this, arguments);
            size[0] = this.size[0] = Math.max(1120, size[0]);
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
