# HZ3-YuE2: guía para agentes

Lee esto antes de actuar. Cualquier agente (Claude, Codex, Gemini, otro) debe poder usar la plataforma solo con lo que
está en este repositorio. Manual del usuario y de los nodos: `README.md`. Diseño de la interfaz y su orden: `DISENO_STUDIO.md`. LoRAs: `GUIA_LORA_YUE2.md`. Reglas de código:
`HZ3_CODING_GUIDE.md`. Lo medido sobre YuE2: `EXPERIENCIAS_YUE2.md`.

## Expectativas del dueño

1. **Haz exactamente la acción pedida, sobre el material indicado.** La autonomía está bien; desviarse de la acción no.
   No cambies de canción, no inventes letra o melodía de prueba, no agregues pasos que nadie pidió.
2. **En una prueba cambia solo la variable pedida.** Copia todo lo demás del proyecto tal cual: estilo completo, LoRA y
   fuerza, semilla, checkpoint, muestreo. Antes de correr, compara las condiciones línea por línea; si difieren en algo
   más que la variable, el diseño está mal.
3. **Prompts transparentes.** Nada se agrega al prompt por detrás del usuario. Todo texto que el Studio envía se ve en
   el inspector de sección (`What YuE2 receives`). Las voces se describen en el estilo global y luego cada sección
   nombra a su cantante; nunca se inyecta "duet".
4. **Piloto primero.** Una sola toma, se envía al dueño y se espera su veredicto antes de cualquier lote o de aplicar
   una receta nueva a varias canciones.
5. **El juez es el oído del dueño.** Ningún detector automático distingue hombre solo, mujer sola o dueto (ver
   `EXPERIENCIAS_YUE2.md`). Da posiciones de audio como m:ss.
6. **Nunca escribas en los proyectos del dueño** para experimentar: trabaja en una copia (`Copy as new song` en el Studio).
   En las copias de prueba apaga `Harmonies` antes de renderizar. Los encabezados de la letra llevan solo el nombre de
   la sección (`[Bridge 2]`), nunca voz ni estilo.
7. Responde en el idioma del dueño (español).
8. **Commits** pequeños, uno por cambio coherente, y push a `origin/master`. Sin líneas de atribución. Nunca subas
   `scratch/` ni `.claude/`. Si git no tiene identidad: `git -c user.name=H4Z33 -c user.email=76185651+H4Z33@users.noreply.github.com`.
9. Coordina la GPU: si otra sesión está usando ComfyUI, espera.
10. Documenta al terminar: lo que el dueño corrige o lo que se mide va a `EXPERIENCIAS_YUE2.md` o al README, no solo a
    la memoria privada de un agente.

## Cómo se usa la plataforma

**Usa HZ3 Studio, no scripts propios.** El Studio ya resuelve tomas, congelado y comping. Un script que manda grafos
directo a ComfyUI se salta todo eso y vuelve a muestrear la canción entera: en 2026-11 eso arruinó semanas de pruebas
de voz.

- El Studio corre dentro de ComfyUI: `http://127.0.0.1:8188/hz3/studio` (menú HZ3 → Abrir HZ3 Studio). Comfy Desktop
  usa 8189 si 8188 está ocupado.
- **Tomas y congelado:** cada render es una toma de la canción completa. `Regenerate section` muestrea solo las
  secciones elegidas o editadas; las demás reproducen los tokens exactos de su toma elegida. Esa es la forma de
  conservar una canción y cambiar una parte. El nodo detrás es `HZ3_YuE2_GenerateMusicSections` (`section_tokens`).
- **Reusar tokens sin volver a muestrear:** `Generate Token Stream` → `Music From Token Stream` (README, "Reusing
  semantic music tokens").
- **Automatizar:** los drivers son JavaScript que corre dentro de la página del Studio y llama a sus propias funciones.
  Viven en la carpeta `input/` de ComfyUI y se cargan así:
  `(0, eval)(await (await fetch('/view?filename=<driver>.js&type=input&t=' + Date.now())).text())`.
  El canónico es `razones_driver.js`: `runAlbum(songs)` (proyecto nuevo, compone ABC, renderiza toma 1),
  `runFromAbc(songs)`, `addClassical(name, plan)` y `featureYueRi(name, sections)` (sube una octava la melodía de esas
  secciones y regenera solo esas). Progreso en `window.albumLog`.
- **Historial con ramas** (como git): cada guardado, cada render y cada cambio grande (componer, reimaginar, analizar,
  terminar aquí) es un commit con el estado completo de la canción en la rama actual (`project.history`; los estados
  van en `history/<hash>.json` dentro del `.mixmash`). Abrir un punto viejo crea una rama nueva desde él; cambiar de
  rama o de canción guarda antes. Nada se borra. Desde un driver: `saveProject(nombre)`, `openCommit(id)`,
  `switchBranch(rama)`, `newBranch(rama)`. Las versiones y ABC anteriores de canciones viejas se migran a las ramas
  «versiones guardadas», «versiones borradas» y «ABC anteriores» al abrirlas.
- Las llamadas de JavaScript desde un agente expiran a los ~45 s: arranca el driver sin esperar y espera el archivo de
  salida desde la terminal.
- Nombres de proyecto: letras, dígitos, espacios y `- _ ( )`, máximo 64, sin puntos ni `#`.

**Dónde queda todo** (carpeta `output/` de ComfyUI):
- Proyectos: `HZ3-YuE2/studio/<nombre>.mixmash` (zip; `manifest.json` → `project` con tomas, estilo, letra, ABC, LoRA).
- Tokens por sección: `HZ3-YuE2/section_tokens/<digest>.json`; cada toma guarda sus digests en `sectionTokens`.
- Audio de tomas: `HZ3-Studio/<nombre>/take-N/{mix,vocals,instrumental}_00001.flac`. El `mix` es el decode crudo; para
  entregar usa `Export FLAC` del Studio.
- Álbumes y cantantes: `HZ3-YuE2/studio/albums/<álbum>.json` (`/hz3/studio/album`). Perfiles de artista:
  `HZ3-YuE2/artists/<artista>.json`.

**Canción de prueba:** cada artista tiene una canción de prueba; úsala para experimentos (Nadie te pregunto:
`NTP Prueba - Chamarra (Fa sost menor 192)`). La canción publicada es referencia, no material de prueba.

## Comportamientos que hay que conocer

- Al elegir cantante de sección, el Studio etiqueta la letra de esa sección con el cantante y suma su descripción al
  estilo de esa sección. **Ya no mueve la melodía** (2026-11-09): mover la octava cambia al cantante, no solo el
  género (dueño). Quien lo quiera lo pide con `Move melody to the singer's range` o
  `setSectionSinger(sección, id, true)`.
- `🔒 Keep this take` (pestaña Takes; `keepSections(nombres, true)`): `Generate song` reproduce los tokens de la toma
  elegida de las secciones guardadas y solo canta las demás. `Regenerate section` sí las vuelve a cantar.
- `▶` y `▶ in song` junto a cada toma: oír una sección en esa toma sin elegirla.
- `Voice contrast (all sections)` (`voiceContrast()`): escribe en el negativo de cada sección su estilo más la
  descripción de los otros cantantes de la canción. Es el negativo del CFG de la etapa 1: el mismo prompt (instrucciones,
  letra, ABC) con otro `[Tags]`; no actúa con CFG 1 y un LoRA entrenado sobre una toma pesa más.
- `Concept song` del álbum (Catalog → álbum; `album.concept`): la canción de referencia de voz y banda. Hoy solo sirve
  para `Context from the concept song` (`contextFromConcept(sección)`). El trasplante de K/V desde sus tokens
  congelados no está construido: falta el experimento con 3 canciones del mismo estilo y cantantes.
- Componer ABC no respeta la estructura de la letra ni la tonalidad o BPM del estilo: revisa `K:` y `Q:` y notas
  contra sílabas por sección antes de renderizar.
- Un render lleva un solo LoRA.
- `Section context (another song)`, formato `Proyecto · toma N · Sección`, o `setSectionContext(sección,
  proyecto, toma, sección de origen)` desde un driver: YuE2 ve los tokens de esa sección de otra canción justo antes
  de muestrear esta, como si acabaran de sonar; no quedan en la canción ni en su audio (nodo: `section_contexts`).
  Sirve para llevar la voz y la banda de una canción concepto a otra. Aparece en `What YuE2 receives`.
- La interfaz del Studio está en inglés (2026-11-08): los nombres de botón de este archivo son los que se ven.
  El panel lateral va en pestañas (Song, Score, Source y Engine con Advanced), el inspector de sección a la derecha en
  pestañas (Lyrics, Style, Takes, Melody, What YuE2 receives) y el catálogo en tarjetas.
- `Instrumental (no vocal)` (`sectionToInstrumental`) mueve la melodía de la voz a la línea `Ins`: la voz se vuelve
  instrumento. Para una instrumental de verdad, deja la línea `Vocal` en silencio y la `Ins` como estaba.
