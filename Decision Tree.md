# DNH-DT · Decision Tree with Non-Homogeneous Distribution Criterion

> **Árbol de Decisión con Criterio de Distribución No Homogénea**  
> *David Cortés — Investigación independiente, 2026*

---

## ¿De dónde viene la idea?

Todo empezó con una pregunta de química:

> *Si mezclas agua, etanol y ácido acético, ¿puedes predecir la densidad de la mezcla sin medirla experimentalmente?*

La respuesta es sí — y el método que lo hace posible, llamado **Distribución No Homogénea (DNH)**, esconde un principio matemático mucho más general.

### La analogía del pastel con pepitas 🍫

Imagina que quieres repartir un pastel con pepitas de chocolate entre tres personas de forma *exactamente equitativa*. Es imposible satisfacer a la vez:
- Igual número de pepitas
- Igual tamaño de porción
- Igual masa de porción

¿Por qué? Porque las pepitas no están distribuidas de forma homogénea. Esta observación cotidiana encierra la misma matemática que predice la densidad de mezclas químicas — y que ahora usamos para construir árboles de decisión mejores.

---

## ¿Qué es la Distribución No Homogénea (DNH)?

Una distribución es **homogénea** cuando su densidad es idéntica en cualquier punto. La distribución uniforme estadística es su análogo: todos los eventos tienen la misma probabilidad.

La naturaleza rara vez es homogénea:
- La densidad de una estrella decrece hacia los bordes
- La riqueza se concentra desigualmente entre países
- Las moléculas de una mezcla tienen masas muy distintas

El método DNH **cuantifica esta no-homogeneidad** y usa esa medida para hacer predicciones.

### La ley clave del paper

El hallazgo central del trabajo original es que el **error de predicción crece racionalmente** con la asimetría de la distribución:

```
E = S / (1 - S)
```

donde `S` es una medida de la asimetría. Si los componentes tienen propiedades similares, `S ≈ 0` y el error es mínimo. Si hay componentes muy dispares (por ejemplo, mezclar agua con un polímero gigante), `S` es grande y el error se dispara.

---

## ¿Cómo se conecta esto con los árboles de decisión?

Un árbol de decisión divide los datos haciéndose preguntas del tipo *"¿es este valor menor que X?"*. En cada nodo, elige el corte que mejor **separa las clases**.

La medida estándar para evaluar esa separación es el **índice de Gini** — una función cuadrática. **DNH-DT propone reemplazarla con la ley racional del paper.**

### La analogía formal

| Marco DNH (química) | DNH-DT (clasificación) |
|---|---|
| Componente `i` de la mezcla | Clase `k` del dataset |
| Masa molecular `MM_i` | Conteo de clase `n_k` en el nodo |
| Distribución homogénea (MMs iguales) | Nodo mezclado (clases equiproporcionales) |
| Distribución asimétrica (una MM domina) | Nodo puro (una clase domina) |
| Minimizar `σ²(MM)` en el grupo | Maximizar asimetría de clases en la partición |

---

## El criterio de impureza DNH

En lugar del índice de Gini, DNH-DT usa:

```
I_DNH(nodo, γ) = 1 / (1 + γ · S)
```

donde:
- `S = σ²(n_k) / μ²` — varianza normalizada de los conteos de clase (la "asimetría")
- `γ` — parámetro de sensibilidad (cuanto mayor, más agresivo detectando pureza)

**Casos límite:**
- **Nodo puro** (una clase tiene todo): `S` es grande → `I_DNH ≈ 0` ✓
- **Nodo mezclado** (clases iguales): `S = 0` → `I_DNH = 1` ✓
- **Con `γ` pequeño**: se comporta casi como Gini (respuesta suave)
- **Con `γ` grande**: respuesta exponencialmente sensible — detecta incluso pequeñas ventajas de pureza

---

## Las tres versiones del algoritmo

### 1. `DNHDecisionTree` — Cortes axis-aligned

El árbol de decisión más directo. Igual que CART, pero con el criterio DNH en lugar de Gini. Los cortes son paralelos a los ejes (`x₁ ≤ t`, `x₂ ≤ t`, etc.).

```
         ¿x₁ ≤ 3.5?
        /           \
   Clase A      ¿x₂ ≤ 1.2?
               /           \
           Clase B       Clase C
```

**Cuándo usarlo:** punto de partida para comparar con CART. Idéntica complejidad computacional `O(N·D·log N)`.

---

### 2. `DNHObliqueDecisionTree` — Cortes oblicuos

Aquí entra la extensión más importante. En lugar de cortes paralelos a los ejes, el árbol busca **cortes diagonales**:

```
w₁·x₁ + w₂·x₂ ≤ t
```

La pregunta ya no es *"¿es x₁ menor que 3.5?"*, sino *"¿está este punto al lado izquierdo de esta línea?"*. Esto permite separar clases que un árbol normal necesitaría muchos niveles para aproximar.

#### ¿Cómo se encuentra la mejor dirección de corte?

Aquí es donde el **criterio de agrupación óptima del paper DNH** se traduce directamente:

> *Paper DNH (Sec. 5.2):* minimizar `σ²(MM_j)` dentro de cada grupo virtual  
> *DNH-DT oblicuo:* maximizar la asimetría de clases en cada partición

Son el mismo principio. La dirección `w*` que maximiza la ganancia DNH es el análogo del "agrupamiento óptimo" del paper.

**En 2D:** se barren ángulos de 0° a 180° y se elige el mejor.  
**En alta dimensión:** se usa ascenso Riemanniano sobre la hipersfera `S^{d-1}`.

---

### 3. `DNHRandomForest` — Ensemble

Un bosque de `n` árboles `DNHDecisionTree`, cada uno entrenado con:
- Una muestra bootstrap de los datos
- Un subconjunto aleatorio de características por nodo (`sqrt(D)` por defecto)

La predicción final es el promedio de probabilidades de todos los árboles. Reduce la varianza y mejora la generalización.

---

## Optimización Riemanniana en alta dimensión

Cuando los datos tienen muchas características (digamos, 30 variables como en el dataset de cáncer), buscar la mejor dirección de corte es un problema de optimización sobre la **hipersfera** `S^{d-1}`:

```
encontrar  w ∈ ℝᵈ  con  ‖w‖ = 1  que maximice Ganancia_DNH(w)
```

La restricción `‖w‖ = 1` complica el gradiente estándar. La solución es proyectarlo al **espacio tangente** de la esfera en el punto actual — técnica llamada gradiente Riemanniano:

```
∇_S f(w) = ∇f(w) − ⟨w, ∇f(w)⟩ · w
w_{t+1}  = normalizar(w_t + α_t · ∇_S f(w_t))
```

La actualización siempre devuelve `w` a la esfera. Como tasa de aprendizaje se usa decaimiento geométrico: `α_t = α₀ · decay^t`.

**Bonus:** el vector `w*` que resulta da importancias de características gratuitas — las características irrelevantes reciben peso `≈ 0` automáticamente.


## Parámetros principales

| Parámetro | Dónde | Descripción |
|---|---|---|
| `gamma` | Todos | Sensibilidad del criterio DNH. Probar entre 1.0 y 5.0. |
| `max_depth` | Todos | Profundidad máxima del árbol. |
| `strategy` | Oblicuo | `'lda'` (recomendado), `'random'`, `'best_random'` |
| `n_iter` | Oblicuo | Iteraciones de ascenso Riemanniano por nodo. |
| `n_estimators` | Forest | Número de árboles del ensemble. |
| `max_features` | Forest | Características por nodo. `'sqrt'` por defecto. |

### ¿Cómo elegir `gamma`?

- `γ = 0.5 – 1.0` → comportamiento similar a Gini, diferencias mínimas
- `γ = 2.0 – 3.0` → punto de partida recomendado
- `γ = 5.0 – 8.0` → muy sensible, riesgo de sobreajuste en datasets pequeños

Usa `GridSearchCV` para encontrar el valor óptimo en tu problema.

---

## Importancias de características

`DNHObliqueDecisionTree` proporciona importancias de características **direccionales**, más ricas que las de CART:

```python
clf = DNHObliqueDecisionTree(max_depth=4, gamma=3.0, strategy='lda')
clf.fit(X, y)

for feat, imp in zip(feature_names, clf.feature_importances_):
    print(f"{feat:<30} {imp:.4f}  {'█' * int(imp * 40)}")
```

Las características irrelevantes (ruido puro) reciben peso `≈ 0` automáticamente, sin necesidad de selección explícita de características.

---

## Extensiones más allá de la clasificación

El índice DNH es una medida general de no-homogeneidad. Los mismos principios se aplican a:

- **Economía:** medir concentración de riqueza por sectores (el "componente" es el sector, la "masa" es el PIB)
- **Demografía:** cuantificar concentración urbana frente a dispersión rural
- **Biología:** detectar acumulación anómala de biomoléculas en compartimentos celulares

---

## Referencia

Si usas este trabajo, por favor cita este repositorio:

```bibtex
@unpublished{,
  author = {Cortés, David},
  title  = {Distribución No Homogénea (DNH) en Mezclas Líquidas:
             Un Método Predictivo de Densidad mediante
             Descomposición Binaria Virtual},
  year   = {2026},
  note   = {Investigación independiente}
}
```

---

## Licencia

MIT © David Cortés, 2026
