# Experiencias con YuE2

Lo medido y lo que el dueño corrigió, para no repetir errores. Fecha y fuente en cada punto. Agrega aquí lo nuevo.

## Cómo genera YuE2

- Dos etapas. La etapa 1 (modelo autorregresivo) recibe `[Tags] estilo`, `[Lyrics] letra` y el ABC, y produce tokens
  de códec a 25 por segundo en **un solo flujo: voz y banda juntas**. La etapa 2 convierte esos tokens en audio; cada
  cuadro atiende a todo el prefijo y a todos los tokens.
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
