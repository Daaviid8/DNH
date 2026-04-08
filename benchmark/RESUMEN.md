# Resultados del Experimento: DNH vs CART

## 📊 Resumen Estadístico

| Método     | Accuracy Media ± DE | Victorias |
|------------|---------------------|-----------|
| **CART**   | **0.8278 ± 0.0824** | 9         |
| **DNH-axis** | **0.8109 ± 0.0880** | -         |
| **DNH-RF** | **0.8608 ± 0.0621** | **19**    |

## 🔬 Tests Estadísticos (Wilcoxon, α=0.05)

| Comparación     | Δ Accuracy | p-valor   | Resultado                  |
|-----------------|------------|-----------|----------------------------|
| DNH-axis vs CART| **−0.017** | **0.018** | ⚠️ Diferencia significativa |
| **DNH-RF vs CART** | **+0.033** | **0.0002**| ✅ **Mejora significativa** |
| **DNH-RF vs DNH-axis** | **+0.050** | **≈0**  | 🚀 **Ensemble domina**     |

## 📈 Análisis por Régimen

- **DNH-axis (γ=3.0 fijo)**: Equiparable a CART sin tuning
- **CART gana** en: fronteras no lineales (circles/moons), datasets levemente desbalanceados
- **DNH-RF domina** en: alta dimensión, ruido/redundancia, datasets reales

## 🎯 Conclusiones Clave

✅ **DNH-RF > CART**: Mejora estadísticamente significativa (+3.3%) con γ fijo  
⚠️ **Árbol individual**: Requiere GridSearchCV sobre γ para competir  
🚀 **Ensemble amplifica**: +5.0% sobre DNH-axis simple  
