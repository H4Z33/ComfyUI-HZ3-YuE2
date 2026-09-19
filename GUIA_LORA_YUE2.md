# Guía Detallada: Entrenamiento e Inferencia de LoRAs en YuE2 con HZ3

Esta guía documenta la arquitectura interna, el procedimiento de entrenamiento, la gestión de prompts y la configuración de nodos en ComfyUI para capturar y reproducir voces, artistas y estilos musicales usando **YuE2** y el ecosistema **HZ3**.

---

## 1. Arquitectura de YuE2: ¿Por qué un LoRA de voz necesita dos modelos?

A diferencia de modelos de imagen como Stable Diffusion (donde un solo UNet o DiT genera todo), **YuE2 es un sistema desacoplado en dos etapas independientes**:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                         ETAPA 1                         │
[Style Prompt + Trigger] ───► │                     AR Language Model                   │ ───► [Tokens Acústicos Discretos]
[Letra estructurada]     ───► │                (cargado como CLIP en ComfyUI)           │      (25 frames por segundo)
                              └─────────────────────────────────────────────────────────┘
                                                              │
                                                KV-Cache de Condicionamiento
                                                              ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │                         ETAPA 2                         │
                              │                  Flow-Matching Diffusion                │ ───► [Audio Waveform / VAE]
                              │                (cargado como MODEL en ComfyUI)          │      (Música final en alta fidelidad)
                              └─────────────────────────────────────────────────────────┘
```

### Etapa 1: El Compositor e Intérprete (AR LLM / `clip`)
- Lee el prompt de estilo, el trigger y las letras estructuradas con secciones (`[Verse]`, `[Chorus]`, etc.).
- Genera una secuencia de **tokens acústicos discretos** (código a 25 Hz).
- **Es quien define la identidad vocal**: tesitura, registro (soprano, mezzosoprano, etc.), notas musicales, melodía, vibrato, inflexión emocional y pronunciación fonética de la letra.

### Etapa 2: El Diseñador de Sonido y Renderizador (Flow-Matching Diffusion / `model`)
- Recibe los tokens acústicos de la Etapa 1 como condicionamiento mediante *cross-attention* (KV-cache).
- Decodifica los tokens en latentes de audio continuos mediante integración de velocidades ODE (flow-matching).
- **Es quien define la textura acústica**: calidad de estudio, brillo tímbrico, presencia estereofónica y respuesta en frecuencia.

---

## 2. El Diagnóstico: ¿Por qué fallaba antes?

| Síntoma | Causa Raíz |
|---|---|
| **A fuerza `<= 2.0` se escucha idéntico** | Se entrenó en modo `voice`, el cual **solo parcheaba la Etapa 2 (`diffusion_model`)** y dejaba la Etapa 1 (`clip`) con 0 cambios. Además, el LLM borró el trigger del prompt. La Etapa 1 corrió virgen y generó tokens de una cantante genérica. El modelo de difusión no puede convertir a una soprano genérica en otra cantante si los tokens acústicos mandan lo contrario. |
| **A fuerza `> 2.5` se escucha distorsionado** | En flow-matching, forzar `strength_model > 2.5` multiplica artificialmente el vector de velocidad fuera de la variedad matemática (*manifold*) entrenada. No produce "más voz de la artista", sino **cancelación de fase, zumbido metálico y ruido digital**. |
| **MixMashStyle borró `que_hablen_de_mi`** | La regla lingüística de Ollama prohibía palabras que no fueran en inglés en el estilo. Ollama vio palabras en español y las eliminó para cumplir con la regla. |

---

## 3. Entrenamiento con el Nodo `HZ3_YuE2_AudioToLoRA`

Para capturar la voz y estilo de una referencia (ej. `que_hablen_de_mi.opus`), utiliza el nodo `HZ3 YuE2 · Audio to LoRA (Style / Voice)`.

### Parámetros del Nodo

| Parámetro | Valor Recomendado | Explicación |
|---|---|---|
| `audio` | Entrada de audio | Tu archivo de referencia limpio (voz o stem vocal). |
| `mode` | `joint (voice + style) [Recommended]` | **Fundamental**. Entrena tanto la Etapa 1 (`clip`) para melodía y fraseo como la Etapa 2 (`model`) para textura y timbre. |
| `trigger` | `que_hablen_de_mi` | El handle o identificador único del artista/canción (sin espacios ni caracteres especiales). |
| `lora_name` | `yue2_que_hablen_de_mi` | Nombre del archivo `.safetensors` resultante. |
| `steps` | `100` a `150` | Suficientes pasos para que AdamW adapte los pesos sin memorizar ni sobreajustar. |
| `rank` | `32` | Capacidad dimensional del adaptador LoRA. |
| `learning_rate` | `0.0005` (`5e-4`) | Tasa de aprendizaje óptima para adaptación auditiva rápida. |
| `alpha` | `64.0` | Factor de escala ($\alpha / r = 64/32 = 2.0\times$). Otorga presencia clara al LoRA a fuerza estándar `1.0`. |
| `model`, `clip`, `vae` *(opcionales)* | Conectados | Conéctalos desde tu cargador de YuE2 para que el nodo use los modelos ya cargados en VRAM en lugar de leerlos de disco. |

> **Resultado**: Se generará un archivo en `models/loras/YuE2/<lora_name>.safetensors` con 112 matrices para `diffusion_model` y 112 matrices para `text_encoders`.

---

## 4. Gestión del Trigger con `HZ3_YuE2_MixMashStyle`

Para que YuE2 active el LoRA durante la generación musical, el trigger debe figurar estrictamente en el prompt de estilo en el formato `..., in the style of <trigger>.`

### Opciones de Conexión en MixMashStyle:

1. **Vía Pin Dedicado (Recomendado)**:
   - Conecta o escribe en el pin `lora_trigger`: `que_hablen_de_mi`.
2. **Vía Instrucciones Escritas**:
   - En el campo `instructions`, puedes escribir:
     ```text
     Spanish female lead vocal in the style of que_hablen_de_mi, entrada sinfónica con coros...
     ```
   - El nodo **extrae automáticamente** `que_hablen_de_mi` mediante expresiones regulares.

### Inyección Procedural Garantizada
Incluso si el modelo de lenguaje (Ollama) olvida, omite o traduce el trigger por las reglas de idioma, el código de Python post-procesa el texto y **fuerza determinísticamente** la presencia de `, in the style of que_hablen_de_mi.` al final de la línea resumen de producción en:
- `style` (Detailed multi-section prompt)
- `style_balanced`
- `style_compact`

---

## 5. Inferencia y Carga en ComfyUI

Carga el LoRA entrenado utilizando el nodo nativo de ComfyUI **`Load LoRA (Model and CLIP)`**.

```
[Load YuE2 Checkpoint]
     ├─► MODEL ──────────┐
     ├─► CLIP ─────┐     │
     └─► VAE       │     │
                   ▼     ▼
         [Load LoRA (Model and CLIP)]
         • lora_name: yue2_que_hablen_de_mi.safetensors
         • strength_model: 1.00  ◄── (Timbre y textura acústica)
         • strength_clip:  0.90  ◄── (Composición, melodía y técnica vocal)
                   │     │
                   ▼     ▼
           [YuE2 Generation Workflow]
```

### Configuración de Fuerzas (*Strengths*):

- **`strength_clip` = 0.80 a 1.00**:
  - Controla la **Etapa 1**. Hace que el modelo autorregresivo escriba secuencias de tokens que imiten los giros vocales, el vibrato, la tesitura y el estilo rítmico del intérprete.
- **`strength_model` = 0.90 a 1.20**:
  - Controla la **Etapa 2**. Aplica las características de ecualización, armónicos vocales y textura sonora aprendidas en la difusión.
- **Evitar `strength_model > 1.8`**:
  - Con la nueva escala `alpha=64.0` (escala `2.0x`), un valor de `1.0` ya ejerce el doble de influencia sin perturbar el campo vectorial de difusión. No uses valores superiores a `1.8` para evitar artefactos de fase.

---

## 6. Estructura Óptima de Letra y Estilo para YuE2

Para que el modelo aproveche al máximo el estilo del intérprete sin desincronizarse:

1. **Línea Resumen de Producción (Inicio de Style)**:
   ```text
   BPM: 72, Meter: 4/4, Key: Bb major, symphonic rock with romantic piano and glam rock finale, Spanish female lead vocal, in the style of que_hablen_de_mi.
   ```
2. **Secciones de Letra con Frases Cortas (4 a 8 sílabas por línea)**:
   ```text
   [Verse 1]
   Lento cae el piano,
   tu recuerdo me alcanza,
   entre cuerdas y sombra,
   mi voz te canta.
   ```
   *Evita líneas continuas de más de 12 sílabas; dividir las líneas con saltos de línea (`\n`) permite al modelo respirar y sostener notas melismáticas.*

---

## 7. Preguntas Frecuentes y Solución de Problemas

### ¿Por qué mi cantante no se parece si uso solo `voice`?
El modo `voice` sólo entrena la difusión (frecuencias/latentes). Si Stage 1 compone notas para una cantante genérica, la difusión no puede cambiar el tono fundamental ni el fraseo. **Utiliza siempre `joint (voice + style)`** para clonar un estilo o intérprete.

### ¿Qué hago si el audio de referencia tiene batería y bajo muy fuertes?
Si buscas aislar únicamente la voz del cantante:
1. Pasa el audio por Demucs o UVR5 para extraer la pista `vocals`.
2. Utiliza la pista acapella limpia en el nodo `HZ3_YuE2_AudioToLoRA`.
3. Así el modelo aprenderá las características puras de la voz sin transferir la reverberación o batería de la canción original.

### ¿Puedo entrenar múltiples canciones del mismo artista?
Sí. Puedes concatenar fragmentos vocales representativos de varias canciones en un solo archivo de audio de 60 a 120 segundos antes de pasarlo al nodo `HZ3_YuE2_AudioToLoRA`. Esto amplía el rango dinámico del LoRA.
