import { app } from "../../scripts/app.js";

// Opens HZ3 Studio on whatever host and port this ComfyUI is served from.
app.registerExtension({
    name: "HZ3.YuE2.Studio",
    commands: [
        {
            id: "hz3.studio.open",
            label: "Abrir HZ3 Studio",
            icon: "pi pi-music",
            function: () => window.open(new URL("/hz3/studio", window.location.href).href, "_blank"),
        },
    ],
    menuCommands: [{ path: ["HZ3"], commands: ["hz3.studio.open"] }],
});
