# Experiencias con YuE2

Lo medido y lo que el dueño corrigió, para no repetir errores. Fecha y fuente en cada punto. Agrega aquí lo nuevo.

## Cómo genera YuE2

- Dos etapas. La etapa 1 (modelo autorregresivo) recibe `[Tags] estilo`, `[Lyrics] letra` y el ABC, y produce tokens
  de códec a 25 por segundo en **un solo flujo: voz y banda juntas**. La etapa 2 convierte esos tokens en audio; cada
  cuadro atiende a todo el prefijo y a todos los tokens.
- Orden del prefijo (`comfy/text_encoders/yue2.py`): `[EOD]` instrucciones, `[Tags]` estilo, `[Lyrics]` letra completa,
  `[ABC_START]` ABC completo `[ABC_END]`, `[MUSIC_START]` y los tokens. Las secciones congeladas llevan el mismo prefijo
  completo; el ABC fija cuántos cuadros dura cada sección, y un token congelado solo se reusa si su duración coincide.
  Como el modelo es causal, instrucciones y estilo dan el mismo K/V siempre que su texto no cambie; letra y ABC vienen
  después y cambian todo lo que sigue.
- La etapa 1 es autorregresiva (decide notas, voz y banda de izquierda a derecha). La etapa 2 (KSampler) es difusión
  sobre toda la canción a la vez: no avanza en el tiempo, los primeros pasos fijan lo grueso y los últimos el detalle.
- Misma semilla y mismas entradas reproducen los tokens exactos. Con la misma semilla y otro estilo, solo 1 de cada
  125 tokens coincide: **el estilo no cambia solo la voz**, cambia toda la canción (2026-11-07).
- Las palabras cantadas viven en los tokens: cambiar la letra del prompt con los mismos tokens no cambia lo que se
  canta. Editar letra exige volver a muestrear esa sección (2026-10-01).
- El contexto decae: 34 % de diferencia 1 s después de una sección editada, 5 % a los 20 s (2026-10-01).
- Las voces extra y armonías son generaciones aparte: se desfasan segundos respecto a la principal (0.8 s al inicio,
  3.5 s después). El Studio las alinea por tono contra el ABC; la precisión todavía falla (2026-10-22).

## Voces

- El ABC `c` suena a 277 Hz (C#4), `C` a 140 Hz, `c'` a 554 Hz.
- Primera sección sin historia: en `C` sale hombre aunque se pida mujer; en `c` decide el texto del estilo.
- **La historia amarra al cantante:** la voz de lo ya generado sigue en las secciones siguientes aunque el estilo pida
  otra. Insistir con palabras casi no cambia nada.
- El texto `duet, ...` hace que YuE alterne mujer y hombre por sección, siempre empezando por mujer. Así salieron las
  dos voces de la toma 24 de "El amor no es lo tuyo". Las voces de Nadie te pregunto nacieron con esa línea en el
  estilo: quitarla cambia la voz (2026-11-07).
- Mover la melodía de octava para conseguir un cantante cambia su timbre: el dueño no lo acepta como solución para
  conservar una voz (2026-10-24). Se usó a propósito para la invitada Yue Ri en los duetos de Amoriv.
- Mismos tokens con otros tokens antes de ellos en la etapa 2: la voz cambia, pero casi siempre a dueto (oído del dueño,
  2026-11-07). Cambiar solo el texto del estilo en la etapa 2 no cambia la voz.
- Ningún detector sirve para decidir la voz (2026-11-07): un clasificador MFCC propio marca duetos como hombre;
  Discogs-EffNet encuentra a los hombres pero da muchos falsos; Gemini Flash adivinó por el orden de los archivos.
- Clasificadores contra las etiquetas del dueño (2026-11-08, `scratch/exp_voz/voice_cls.py`; ventanas de 3 s de la voz
  separada; El y Ella de la toma 24, más los renders "desde cero solo El" y "solo Ella"). Probados: el codificador de
  voz de chatterbox (`ve.safetensors`), el encoder de Whisper small, MFCC más tono, y solo el tono (F0).
  - Dejando fuera una sección de la toma 24, separan bien a El de Ella: Whisper 0.95, chatterbox 0.90, MFCC más tono
    0.83.
  - Entrenados en la toma 24 no reconocen al El de otro render: solo marcan como El el 23-60 % del render solo El.
  - Las ventanas de los renders solo El y solo Ella quedan igual de lejos de El y de Ella de la toma 24: cada render
    forma su propio grupo, y lo que separa un render de otro pesa más que quién canta.
  - El tono solo no sirve (0.18): el registro lo pone el ABC, no el cantante.
- Error de diseño a no repetir (2026-11-07): una prueba con una línea inventada, otra semilla y otro orden en cada
  condición no puede decir qué cambió la voz. Usa la canción de prueba y congela todo menos la voz.

### Toma 24 de "El amor no es lo tuyo" sin LoRA, una variable por prueba (2026-11-08, oído del dueño)
- La toma 24 se hizo sin LoRA. Con los mismos valores y sin caché, los tokens divergen a los 2.3 s, pero suena muy
  parecida: el resultado es casi determinista aunque los tokens no.
- Con los tokens congelados, cambiar el texto de la etapa 2 (solo El, solo Ella) no cambia la voz.
- Desde cero, sin `duet` y con todo el estilo nombrando solo a El: sale solo El. Con solo Ella: sale Ella, salvo
  1:01-1:10 (estribillo), que salió masculino.
- CFG 1.0: voz masculina clara y banda más rockera y sucia. CFG 3.0: voz e instrumental más claros, pero "muy metódico".
- **Negativo del CFG con el mismo estilo y la voz opuesta** (`negative_style`, campo "Estilo negativo" del Studio):
  la voz más clara obtenida hasta ahora. Con negativo distinto, la melodía no sale idéntica entre las versiones de El y
  de Ella.
- Conditioning de la etapa 2 por capa: con el mismo prefijo, llevar las filas de los tokens de otro render a la toma 24
  cambia el audio **solo en la capa 10, parte V, head 1** (128 valores por cuadro). Banda y voz se mueven juntas ahí:
  ningún corte de capa, K/V o head las separó. Las capas 0-9 y 11-27 casi no cambian el audio. Al oído, las capas
  0-13 de El sobre la toma 24 dieron voz masculina con "feeling" femenino donde antes cantaba Ella.
- Ese canal (capa 10, V, head 1) decide quién canta y la banda: Ella + ese canal de la toma 24 suena idéntica a la
  toma 24, dueto incluido. El "feeling" vive repartido en las otras capas (la 16 lo cambia junto con la melodía).
- Toma 24 intacta y solo ese canal cambiado en las secciones de la otra voz (`scratch/exp_voz/kvswap.py`, `spans`):
  sale solo Ella o solo El, **sin cortes**, con la instrumentación apenas cambiada. Pero suena a "ella intentando
  cantar como él" y al revés: el feeling sigue siendo el de la voz original de esa sección.
- El feeling del cantante está sobre todo en la **capa 16**, y algo en la 19 y la 13 (K y V, los 8 heads). Sumar las
  capas 13, 16 y 19 del donante en esos mismos tramos se oye ligeramente mejor y el feeling del cantante es notorio en
  las dos direcciones (dueño, prueba a ciegas). Lo demás queda igual a la toma 24 (0.97-1.00), sin cortes. La banda de
  los tramos cambiados se acerca a la del donante (0.01-0.08 más lejos de la toma 24 que solo con la capa 10). Las
  capas 0-9, 11-12, 14-15, 17-18 y 20-27 no aportan.
- **Positivo sin ninguna descripción de género y un negativo por sección con la voz contraria**
  (`section_negative_styles`, campo "Estilo negativo de la sección"): casi la toma 24, con el feeling correcto y la
  instrumentación ligeramente distinta. Fallos: desde 1:50 la voz queda rara, y el Bridge 2 cantado
  (1:40-1:51, «Con la cortina entreabierta»; 1:31-1:40 es instrumental), que canta El, salió femenina aunque su negativo pedía a Ella (también con el Interlude 2 pidiendo a El).
- Lo mismo con el LoRA de la toma 24 a 0.6: la canción sale prácticamente idéntica a la toma 24, pero todo lo canta
  Ella y los negativos no tienen efecto. Ese LoRA aprendió los tokens de la toma 24 sin ABC ni texto y pesa más que
  el estilo y el negativo.
- El LoRA de estilo entrenado sobre la instrumental separada de la toma 24 metió una reverb muy notoria y tampoco deja
  actuar a los negativos.
- Quitar del LoRA las capas 10, 13, 16 y 19 (archivo filtrado, misma fuerza 0.6): lo más parecido a la toma 24, pero
  otra vez solo Ella.
- LoRA solo en las secciones de Ella (las de El regeneradas sin LoRA y con su negativo): la instrumentación cambió
  demasiado y siguió solo Ella; la historia hecha con LoRA amarra la voz.
- El LoRA solo en el conditioning de la etapa 2 (tokens sin LoRA) casi no cambia el audio (0.997): lo que aporta el
  LoRA entra por los tokens. «Con la cortina entreabierta» (1:40) sigue femenina en el 0-shot sin género con negativos.
- **Ojo, las dos pruebas siguientes no son instrumentales reales** (dueño, 2026-11-08): `sectionToInstrumental` pasa la
  melodía de la voz a la línea `Ins`, o sea convierte la voz en instrumento. Una instrumental real deja la línea `Vocal`
  en silencio y la línea `Ins` como estaba.
- Instrumental desde el ABC (todas las secciones con `sectionToInstrumental`, letra `(instrumental)`, mismo LoRA
  filtrado y mismas condiciones que la anterior): sale otra canción, con un estilo completamente diferente. Sin letra ni
  melodía en la voz, los tokens se generan de nuevo y el LoRA ya no arrastra al sonido de la toma 24.
- Toma 24 instrumental desde el ABC, sin LoRA y con todo lo demás igual a la toma 24 (proyecto «NTP Exp 17 - Toma24
  instrumental sin voces», toma 26): la pista de voz separada queda en silencio desde 0:20 (-75 dB). Copiar a la toma 24
  la capa 10, V, head 1 de esa instrumental en las filas de tokens (`kvswap.py render ... 10 v 1-1 "" 1`, prefijos de
  distinto largo alineados por cuadro) quita la voz desde 0:20, pero la banda queda igual a la de la instrumental (mel
  0.93-0.98) y lejos de la toma 24 (0.38-0.78): ese canal lleva la banda y la voz juntas (2026-11-08, medido).
- **Contexto de otra canción** (`section_contexts`, 2026-11-08, oído del dueño). Canción de prueba Chamarra con el Verse 1
  de la toma 24 (Ella) antes de Verse 1 y Chorus, y el Verse 2 (El) antes de Verse 2 y Chorus 2:
  - Con el LoRA y los estilos por sección de la canción de prueba (Exp 18): el estilo sí pasó, la voz no; la voz da
    brincos raros. La toma 1 sin contexto sale bien.
  - Con la configuración de la toma 24 (sin LoRA, positivo sin género, negativo con la voz contraria por sección;
    Exp 18b): suena muy rara, como si no tuviera letra, aunque la letra sí se envió. La voz se parece un poco más a la de
    la toma 24, sin ser concluyente.
  - Probable causa, sin medir: los tokens del contexto cantan otra letra («Next day to the sun», «Dejaste la puerta
    abierta») y YuE2 pierde en qué parte de la letra va.
- La receta de capas con donantes generados con negativo da voces más claras, pero las divergencias de generación se
  notan demasiado: no sirve.
- Cambiar el encabezado de la letra (`[Verse 2]` frente a `[Verse 2 – Her]`) cambia los tokens tanto como volver a
  generar sin cambiar nada.

## LoRA de estilo

- Receta en uso: `HZ3_YuE2_AudioToLoRA` en modo style sobre los tokens de una toma aprobada, 150 pasos, rank 32,
  alpha 64, lr 5e-4, fuerza 0.6 al usarlo (`GUIA_LORA_YUE2.md` §0). El botón `Congelar voz` del Studio aún usa la
  receta vieja (joint/voice, fuerza 1).
- El Studio antepone el disparador del LoRA al estilo; no escribas "in the style of".
- En Luna Pop el LoRA solo no bastaba; se reusaban los tokens de las tomas (dueño, 2026-11-07).
- Un LoRA con captions por sección no decide quién canta: en melodía nueva todas las voces salieron mujer (2026-10-24).

## Componer y estructura

- YuE2 canta unas 10-14 sílabas por compás de 4/4 (unas 3.6 por segundo cantadas, 4.6 habladas). Si una sección tiene
  más compases de los que su letra necesita, repite líneas de secciones anteriores. Recorta cada sección a lo que
  piden sus sílabas (`reshape` en `input/estratega_v3.js`) (2026-10-22).
- Componer funde bloques repetidos (Intro = Verse 1), une versos contiguos y renombra o borra secciones; nunca
  respetó una letra de 10 secciones en 22 semillas. Recomponer desde una sección se descarrila. Lo que funcionó:
  letra sin bloques repetidos, elegir la versión de ABC con notas más cerca de las sílabas y partir el verso largo
  con marcadores `% Nombre` (2026-10-24).
- Componer ignora la tonalidad y el BPM del estilo: el piloto de NTP pidió F#m 192 y salió D mayor 125 (2026-10-24).

## Gusto del dueño (por artista)

- Cali Moshi: verso con métrica y rima constantes, juegos de palabras, mezcla de líneas cantadas y medio habladas; sin
  secciones enteras de melodía libre; final compuesto, sin fade (2026-10-16).
