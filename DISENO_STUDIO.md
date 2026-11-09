# Diseño de HZ3 Studio: lo que cambia y en qué orden

Decidido 2026-11-08 con el mismo método que h3vn (`docs/29`): tres revisores de proveedores distintos, Gemini 3.8 Flash
(`agy`), DeepSeek v4.1 Flash (Ollama) y Fable, propusieron cada uno un rediseño por separado. Después cada uno atacó las
propuestas de los otros con una investigación de teoría UX y de herramientas establecidas, y se comprometió con una
especificación. Material local (no versionado): `scratch/ui_audit/r2/` (`brief*.md`, `*_r1.md`, `*_r2.md`, capturas) y
`scratch/ui_audit/investigacion_ux.md`.

Pedido del dueño: "algo más intuitivo y mejor", "controles más intuitivos y ventanas más útiles, ocultando lo advanced
cuando no es necesario", versiones y catálogo menos confusos, ramas como git sin perder datos.

Ya construido antes de este documento: modo simple / Avanzado (un campo oculto con valor se ve igual, marcado); historial
con ramas (cada guardado, render o cambio grande es una versión; abrir una vieja crea rama; nada se borra); arreglos del
catálogo (nombres que pisaban canciones, renombres, mayúsculas, álbum fantasma, LoRA al abrir, dobles espacios).

## Veredicto del dueño sobre el primer prototipo (2026-11-08)

- **La interfaz va en inglés.** Este documento usa los nombres en español; en pantalla se traducen (tabla de §1).
- **El timeline se queda como está**: secciones, letra, partitura (ABC) y las pistas le gustaban más que la versión
  simplificada del prototipo. No se toca.
- Lo que no le gustaba y sí se rediseña: el **inspector de sección**, el **panel de control izquierdo** y los
  **paneles del catálogo** y demás diálogos.
- Consecuencia en las vistas: la Canción conserva su timeline completo; el panel izquierdo se vuelve un cajón
  «Song settings» que se abre solo cuando hace falta; el inspector se vuelve una tarjeta compacta junto a la sección
  (Lyrics · Style · Takes); el catálogo y los diálogos pasan a páginas tipo biblioteca.

## 0. Reglas que el diseño obedece

Las de h3vn (inventario §A): la app abre en lo que necesita al dueño; nunca ocultar estado; la GPU solo con un toque;
mostrar el radio de impacto antes del toque; palabras llanas; nada se destruye. Más las medidas de YuE2: el texto del
estilo no elige al cantante, la voz que ya cantó se queda, un render cambia todo salvo lo fijado. Ningún control promete
lo que el modelo no hace.

## 1. Vocabulario en pantalla

| Interno | En pantalla | Nota |
|---|---|---|
En pantalla, en inglés (veredicto del dueño): Song, Album, Artist, Section, Take, Chosen take, Kept, Version, Branch,
Artist voice, Score, Interpretation, Sing, Job. La tabla da el concepto.

| project | **Canción** | |
| album | **Álbum** | su ficha de **Artista** guarda voz, estilo madre y voz del artista |
| section | **Sección** | se reconoce por su primera línea de letra |
| take | **Toma** | un render completo; `T3` solo como sufijo |
| comp | **Toma elegida** | por sección |
| section_tokens kept | **Fijada** | estados: sin toma · fijada · editada (se volverá a cantar) · en cola · fallida |
| commit | **Versión** | con nombre o automática |
| branch | **Rama** | la palabra del dueño; en modo simple nace sola con «Seguir desde aquí» |
| LoRA | **Voz del artista** + fuerza | «LoRA» solo en Avanzado y en el título del control |
| ABC | **Partitura** | el texto crudo solo en Avanzado |
| seed | **Interpretación** | «Otra interpretación» = semilla nueva |
| render | **Cantar** | «Cantar cambios», «Cantar canción» |
| job | **Trabajo** | en la barra de trabajos |

Nunca en modo simple: commit, hash, token, digest, CFG, top_p, checkpoint, stage.

## 2. Vistas

1. **Inicio** (al abrir): qué te necesita, por prioridad fija: trabajo fallido, toma terminada sin escuchar, secciones
   sin toma, trabajos en cola con su tiempo. Debajo, canciones recientes y álbumes. Vacío honesto: «Nada pendiente. GPU
   libre.» (Gemini lo quería muerto por ser "un gestor de incidencias"; queda porque el dueño trabaja en remoto y lo
   primero que necesita al abrir es saber qué terminó.)
2. **Canción** (la de hoy, ordenada):
   - Cabecera de una fila: álbum, nombre editable, chip **«Rama: principal · versión 7»** que abre el Historial, y una
     franja de color cuando la rama no es la principal.
   - Timeline como hoy (secciones, letra, partitura, carriles de audio). Cada sección muestra su estado y su toma
     elegida; al tocarla se despliega su **tira de tomas** (T1 T2 T3 con play, «Escuchar en contexto», elegir con
     deshacer).
   - Inspector por pestañas: Letra · Estilo (Guardar / Deshacer / Original) · Tomas.
   - Botón primario **«Cantar cambios: 2 secciones (Verse 2, Chorus 2) · ~3 min»** que dice antes qué se vuelve a cantar
     y qué queda fijado; sustituye a «Generar canción» / «Regenerar sección» y al aviso que hoy sale después.
   - Barra de trabajos fija abajo.
   - Avanzado: partitura cruda, semillas, muestreo, checkpoint, negativos, contexto, inicio fijo, armonías con agente,
     importar JSON, reimaginar, líneas clásicas, afinación, desfases, y «Cantante de la sección» con la advertencia
     medida.
3. **Historial** (panel lateral; página completa en el teléfono): versiones con nombre arriba y automáticas plegadas;
   cada una dice qué cambió en palabras y trae **Escuchar** (su comp sin abrirla), **Seguir desde aquí** (rama nueva) y
   **Nombrar**. Selector de rama, renombrar y archivar.
4. **Catálogo** (página, no diálogo): álbumes y canciones con buscador (también por letra), estado (borrador · en cola ·
   con tomas · exportada), número de tomas, fecha y play en línea; acciones en «…» (Abrir, Copiar como canción nueva,
   Mover, Archivar con deshacer, Descargar). Audios y referencias en una pestaña «Importar».
5. **Artista** (ficha del álbum, sustituye «Estilos del álbum»): voz (descripción, registro), **voz del artista** con
   fuerza 0.6 por defecto y **«Hacer la voz desde una toma aprobada»** (la receta style que funciona), estilo madre,
   estilos base, canción de prueba y notas. La receta vieja (joint/voice a 1) pasa a Avanzado.

**Teléfono (390 px)**: barra de 48 px (☰, nombre, ▶, «Cantar»). La canción es una lista vertical de secciones, cada una
con su onda pequeña, toma elegida y estado (Fable quería quitar la onda; DeepSeek, conservar el timeline: la onda por
sección deja ver el audio sin carriles anchos). Tocar una abre una hoja desde abajo con Letra · Estilo · Tomas. Zona
del pulgar: play, sección anterior/siguiente, «Cantar cambios». Historial, Catálogo y Avanzado son páginas completas.

## 3. Orden de construcción ((dolor × frecuencia) / costo)

| # | Qué | Listo cuando | Tamaño |
|---|---|---|---|
| 1 | **Barra de trabajos**: trabajos de ComfyUI con sección y etapa, tiempo, cancelar, «terminó» que persiste | el teléfono reconectado muestra qué terminó | S |
| 2 | **«Cantar cambios» con su radio de impacto** antes del toque | el texto coincide siempre con lo que se vuelve a cantar | S |
| 3 | **Tira de tomas en el timeline** + Escuchar en contexto + deshacer | se elige toma sin abrir el inspector | M |
| 4 | **Historial legible**: versiones con nombre, Escuchar, Seguir desde aquí, franja de rama | el dueño vuelve a una versión confirmándola de oído | M |
| 5 | **Teléfono** (`@media`): lista de secciones, hoja inferior, barra del pulgar, páginas | elegir tomas se hace con una mano | L |
| 6 | **Catálogo biblioteca**: buscador, estado, play, «…» | cabe en 390 px sin scroll lateral | M |
| 7 | **Ficha de Artista** con la receta de voz (style 0.6) en la interfaz | una canción nueva hereda voz y estilo sin scripts | M |
| 8 | **Inicio** con lo pendiente | abrir la app dice qué escuchar | M |
| 9 | **Exportar**: pistas, pico y LUFS medidos en el servidor, copia en `HZ3-Studio/<canción>/export/` | la versión exportada queda marcada | S |
| 10 | **Deshacer en vez de confirmar** al archivar, eliminar o elegir toma | solo borrar un álbum pide confirmación | S |
| 11 | **Cantante de la sección honesto** (a Avanzado con la advertencia) | el control no promete elegir la voz | S |
| 12 | **Aviso de partitura**: `K:`/`Q:` distintos del estilo, notas contra sílabas por sección | se ve antes de cantar | S |

## 4. Ideas muertas (no repropuestas)

- **Tomas en una vista aparte** (Gemini): rompe el ciclo oír → elegir → oír; las tomas viven bajo su sección.
- **A/B en caliente al compás** (Gemini): lo cubre «Escuchar en contexto» sin un reproductor nuevo.
- **Inferir las secciones de la letra** (Gemini): componer no respeta la estructura; los `[Sección]` siguen explícitos.
- **Dado de tres interpretaciones** (DeepSeek): triplica GPU sin mostrar el costo; una a la vez, piloto primero.
- **Diff de tokens o hashes**: información de máquina.
- **Fusionar ramas**: inventa una canción que nadie oyó.
- **Detector de quién canta** o badge hombre/mujer/dueto: medido, ningún detector acierta.
- **Re-cantar automático** hasta que suene: la GPU solo con un toque.
- **Chat como interfaz principal**: actúa sin radio de impacto; queda como asistente.
- **Ocultar la palabra «rama»** (investigación UX): el dueño la pidió; se oculta la gestión, no el nombre.
- **Arrastrar tarjetas entre álbumes**: imposible en el teléfono; «Mover» basta.
- **Comping libre por debajo de la sección** (swipe continuo): el motor fija por sección.
