import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "HZ3.YuE2.AudioFolder",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HZ3_YuE2_AudioFolder") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const folderWidget = this.widgets?.find(w => w.name === "folder");
            const customFolderWidget = this.widgets?.find(w => w.name === "custom_folder");
            const audioFileWidget = this.widgets?.find(w => w.name === "audio_file");
            const subfoldersWidget = this.widgets?.find(w => w.name === "subfolders");

            // Create audio preview DOM widget container
            const previewContainer = document.createElement("div");
            previewContainer.className = "hz3-audio-preview-container";
            Object.assign(previewContainer.style, {
                width: "100%",
                boxSizing: "border-box",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                padding: "6px 8px",
                background: "#111827",
                border: "1px solid #25a98e",
                borderRadius: "6px",
                marginTop: "4px",
            });

            const infoEl = document.createElement("div");
            infoEl.className = "hz3-audio-preview-info";
            Object.assign(infoEl.style, {
                fontSize: "11px",
                color: "#8ee8d5",
                fontFamily: "system-ui, -apple-system, sans-serif",
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                lineHeight: "1.3",
            });
            infoEl.textContent = "Audio preview: select a file";

            const audioEl = document.createElement("audio");
            audioEl.controls = true;
            audioEl.preload = "metadata";
            Object.assign(audioEl.style, {
                width: "100%",
                height: "32px",
                outline: "none",
                borderRadius: "4px",
            });

            previewContainer.appendChild(infoEl);
            previewContainer.appendChild(audioEl);

            this.hz3AudioEl = audioEl;
            this.hz3AudioInfo = infoEl;

            const previewWidget = this.addDOMWidget("audio_preview", "custom", previewContainer, {
                serialize: false,
                getHeight: () => 64,
                getMinHeight: () => 64,
            });
            previewWidget.computeSize = () => [this.size[0], 64];

            const currentSize = this.size || [300, 200];
            this.setSize([Math.max(currentSize[0], 340), Math.max(currentSize[1], 280)]);

            const formatDuration = (seconds) => {
                if (!seconds || !isFinite(seconds)) return "";
                const m = Math.floor(seconds / 60);
                const s = Math.floor(seconds % 60).toString().padStart(2, "0");
                return `${m}:${s}`;
            };

            const updateAudioPreview = () => {
                if (!audioFileWidget || !audioEl) return;
                const fileVal = (audioFileWidget.value || "").trim();
                if (!fileVal || fileVal === "(no audio files found)") {
                    audioEl.pause();
                    audioEl.removeAttribute("src");
                    audioEl.load();
                    audioEl.dataset.currentSrc = "";
                    infoEl.textContent = "No audio file selected";
                    return;
                }

                const folderVal = folderWidget ? folderWidget.value : "[input]";
                const customVal = customFolderWidget ? customFolderWidget.value : "";
                const subVal = subfoldersWidget ? subfoldersWidget.value : true;

                const params = new URLSearchParams({
                    folder: folderVal || "[input]",
                    custom_folder: customVal || "",
                    audio_file: fileVal,
                    subfolders: subVal ? "true" : "false",
                });

                const newSrc = `/hz3/yue2/audio_folder/audio?${params.toString()}`;
                if (audioEl.dataset.currentSrc !== newSrc) {
                    audioEl.dataset.currentSrc = newSrc;
                    audioEl.src = newSrc;
                    audioEl.load();
                    infoEl.textContent = `🎵 ${fileVal}`;
                }
            };

            audioEl.onloadedmetadata = () => {
                const durStr = formatDuration(audioEl.duration);
                const fileVal = (audioFileWidget?.value || "").trim();
                infoEl.textContent = durStr ? `🎵 ${fileVal} (${durStr})` : `🎵 ${fileVal}`;
            };

            audioEl.onerror = () => {
                if (audioEl.src && audioEl.dataset.currentSrc) {
                    const fileVal = (audioFileWidget?.value || "").trim();
                    infoEl.textContent = `⚠️ Error loading audio: ${fileVal}`;
                }
            };

            const updateAudioFiles = async (preserveCurrent = true) => {
                if (!audioFileWidget) return;
                const folderVal = folderWidget ? folderWidget.value : "[input]";
                const customVal = customFolderWidget ? customFolderWidget.value : "";
                const subVal = subfoldersWidget ? subfoldersWidget.value : true;

                try {
                    const params = new URLSearchParams({
                        folder: folderVal || "[input]",
                        custom_folder: customVal || "",
                        subfolders: subVal ? "true" : "false",
                    });
                    const res = await fetch(`/hz3/yue2/audio_folder/files?${params.toString()}`);
                    if (!res.ok) return;
                    const data = await res.json();
                    if (data && Array.isArray(data.files) && data.files.length > 0) {
                        const currentVal = audioFileWidget.value;
                        audioFileWidget.options = audioFileWidget.options || {};
                        audioFileWidget.options.values = data.files;

                        if (preserveCurrent && currentVal && data.files.includes(currentVal)) {
                            audioFileWidget.value = currentVal;
                        } else {
                            audioFileWidget.value = data.files[0];
                        }
                        this.setDirtyCanvas?.(true, true);
                        app.graph?.setDirtyCanvas(true, true);
                        updateAudioPreview();
                    }
                } catch (err) {
                    console.warn("[HZ3 AudioFolder] Failed to refresh audio file list:", err);
                }
            };

            if (folderWidget) {
                const origCallback = folderWidget.callback;
                folderWidget.callback = function (value) {
                    origCallback?.apply(this, arguments);
                    updateAudioFiles(false);
                };
            }

            if (customFolderWidget) {
                const origCallback = customFolderWidget.callback;
                customFolderWidget.callback = function (value) {
                    origCallback?.apply(this, arguments);
                    updateAudioFiles(false);
                };
            }

            if (subfoldersWidget) {
                const origCallback = subfoldersWidget.callback;
                subfoldersWidget.callback = function (value) {
                    origCallback?.apply(this, arguments);
                    updateAudioFiles(true);
                };
            }

            if (audioFileWidget) {
                const origCallback = audioFileWidget.callback;
                audioFileWidget.callback = function (value) {
                    origCallback?.apply(this, arguments);
                    updateAudioPreview();
                };
            }

            // Add Refresh button widget
            this.addWidget("button", "🔄 Refresh Files", null, () => {
                updateAudioFiles(true);
            });

            // Expose updateAudioPreview for external/configure callers
            this.hz3UpdateAudioPreview = updateAudioPreview;

            // Initial load sync after widgets are configured
            setTimeout(() => {
                updateAudioFiles(true);
                updateAudioPreview();
            }, 200);

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            // Delay slightly to ensure saved widget values are populated
            setTimeout(() => {
                const audioFileWidget = this.widgets?.find(w => w.name === "audio_file");
                const currentVal = audioFileWidget?.value;
                const folderWidget = this.widgets?.find(w => w.name === "folder");
                const customFolderWidget = this.widgets?.find(w => w.name === "custom_folder");
                const subfoldersWidget = this.widgets?.find(w => w.name === "subfolders");

                if (!audioFileWidget) return;

                const params = new URLSearchParams({
                    folder: folderWidget?.value || "[input]",
                    custom_folder: customFolderWidget?.value || "",
                    subfolders: (subfoldersWidget ? subfoldersWidget.value : true) ? "true" : "false",
                });
                fetch(`/hz3/yue2/audio_folder/files?${params.toString()}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data && Array.isArray(data.files)) {
                            audioFileWidget.options = audioFileWidget.options || {};
                            audioFileWidget.options.values = data.files;
                            if (currentVal && data.files.includes(currentVal)) {
                                audioFileWidget.value = currentVal;
                            }
                            this.hz3UpdateAudioPreview?.();
                            app.graph?.setDirtyCanvas(true, true);
                        }
                    })
                    .catch(() => {});
            }, 300);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            this.hz3UpdateAudioPreview?.();
            if (message?.text?.[0] && this.hz3AudioInfo) {
                const firstLine = message.text[0].split("\n")[0];
                if (firstLine) this.hz3AudioInfo.textContent = firstLine;
            }
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.hz3AudioEl) {
                this.hz3AudioEl.pause();
                this.hz3AudioEl.removeAttribute("src");
                this.hz3AudioEl.load();
                this.hz3AudioEl = null;
            }
            return onRemoved?.apply(this, arguments);
        };
    },
});
