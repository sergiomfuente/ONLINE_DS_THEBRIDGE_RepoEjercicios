# Predicción de la Dirección Diaria de Bitcoin
### 1) Descripción del problema
Este proyecto aborda un problema de **clasificación binaria** aplicado a mercado cripto: estimar la probabilidad de que Bitcoin cierre en positivo al día siguiente.

- Objetivo de negocio: apoyar decisiones direccionales con una probabilidad `P(sube mañana)`.
- Target: `1` si el retorno de `t+1` es positivo, `0` si es negativo o cero.
- Restricción metodológica clave: evitar *data leakage* y respetar la naturaleza temporal de los datos.

### 2) Dataset utilizado
- Archivo: `src/data_sample/Bitcoin_history_data.csv`
- Tipo: histórico diario de Bitcoin (OHLCV)
- Variables principales:
  - `Date`
  - `Open`
  - `High`
  - `Low`
  - `Close`
  - `Volume`
- Acceso: datos financieros históricos de fuente pública (incluidos como muestra ejecutable en el repositorio).

### 3) Solución adoptada
Se implementó una pipeline de ML clásico:

1. Validación y limpieza temporal de datos.
2. Construcción del target sin fuga de información futura.
3. Mini EDA orientado al modelado.
4. Split temporal train/test (sin `shuffle`).
5. Baseline + comparativa de modelos clásicos:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
6. Optimización de hiperparámetros con `GridSearchCV` + `TimeSeriesSplit`.
7. Evaluación final en test (umbral base 0.5).
8. Persistencia del modelo en `joblib`.


### 4) Tecnologías utilizadas
- Python 3
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib

### 5) Principales resultados
- Métrica principal: **ROC-AUC**.
- El enfoque final compara de forma rigurosa contra baseline en test.
- El modelo final queda guardado en `src/models/` para reutilización.

### 6) Autores
- **Sergio Martinez** - [GitHub](https://github.com/sergiomfuente)
---
