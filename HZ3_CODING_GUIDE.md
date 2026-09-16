# HZ3 · Coding Hygiene Rules (para que yo las siga)

Cuando el usuario da una **instrucción explícita sobre un nodo**, seguirla al pie de la
letra. Estas reglas son obligatorias:

## Reglas

1. **Editar SOLO el nodo que se me indica.** Nunca modificar otro nodo/archivo para
   "caber" la lógica que necesito. Si una función vive en otro módulo, reutilizarla
   (importarla), no copiarla ni tocar ese módulo salvo que el usuario lo pida.

2. **Confiar en el usuario.** Si da una instrucción concreta de un nodo, asumir que sabe
   qué está pidiendo. No reinterpretar, no desviarse, no inventar sub-tareas.

3. **Nunca reutilizar una variable para dos significados** (en especial parámetros de
   función). Cada cosa con su nombre (`whisper_segments` vs `segments`/`segments_abc`,
   etc.). Esto evita pisar/colisiones que rompen otras partes.

4. **Cambios aislados probados por separado.** Cada cambio se comprueba en aislamiento
   (el nodo afectado), mockeando dependencias externas (modelo/Ollama), ANTES de conectar
   el resto.

5. **No encadenar fixes con parches sobre parches.** Atacar la causa raíz de la primera
   vez, no remendar síntomas.

6. **Commits lógicos y descriptivos**, hechos con regularidad, uno por cambio coherente.

7. **Coercionar inputs defensivamente** solo cuando el runtime los envuelve (p. ej.
   listas de 1 elemento), en el propio nodo que los recibe; un `first()`/`scalar()`
   no debe alterar la semántica del resto.

## Checklist antes de commit
- [ ] Toqué únicamente el/los nodo(s) que el usuario indicó.
- [ ] No sobreescribí ninguna variable con otro significado.
- [ ] El nodo compila (`py_compile`) y probé su lógica aislada.
- [ ] No agregué funcionalidad no pedida.
- [ ] Commit con nombre claro.
