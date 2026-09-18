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

            // Add Refresh button widget
            this.addWidget("button", "🔄 Refresh Files", null, () => {
                updateAudioFiles(true);
            });

            // Initial load sync after widgets are configured
            setTimeout(() => {
                updateAudioFiles(true);
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
                            app.graph?.setDirtyCanvas(true, true);
                        }
                    })
                    .catch(() => {});
            }, 300);
            return result;
        };
    },
});
