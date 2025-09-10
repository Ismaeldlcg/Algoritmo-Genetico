# Algoritmos Genéticos aplicados a Machine Learning

Este repositorio contiene tres ejemplos prácticos del uso de **Algoritmos Genéticos (GA)** en problemas de aprendizaje automático:

1. **Selección de características (Feature Selection)**
2. **Optimización de hiperparámetros (Hyperparameter Optimization)**
3. **Neuroevolución de arquitecturas de redes neuronales (Neuroevolution)**

---

## 1. Selección de Características 🧬

- **Dataset:** [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)  
- **Clasificador:** K-Nearest Neighbors (KNN)  
- **Objetivo:** Seleccionar el subconjunto óptimo de características que maximiza la precisión en un conjunto de validación.  
- **Representación:** Cada cromosoma es un vector binario donde `1` indica que la característica está seleccionada.  

**Estructura GA:**
- **Selección:** Torneo (Tournament Selection)  
- **Cruzamiento:** One-Point Crossover  
- **Mutación:** Inversión de bits con probabilidad `pm`  

---

## 2. Optimización de Hiperparámetros 🌹

- **Dataset:** [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)  
- **Clasificador:** Random Forest  
- **Hiperparámetros optimizados:**
  - `n_estimators` (10–200)
  - `max_depth` (2–10)
  - `min_samples_split` (2–10)

- **Objetivo:** Encontrar la combinación de hiperparámetros que maximiza la precisión en el conjunto de validación.  

**Estructura GA:**
- **Selección:** Torneo  
- **Cruzamiento:** One-Point Crossover  
- **Mutación:** Cambio aleatorio de hiperparámetros con probabilidad `pm`  

---

## 3. Neuroevolución 🐱🐶

- **Dataset:** [Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) (subconjunto reducido)  
- **Modelo:** CNN (red convolucional)  
- **Objetivo:** Evolucionar arquitecturas de CNN y tasas de aprendizaje para maximizar `val_accuracy`.  

**Cromosoma:** `[f1, f2, dense_units, dropout_idx, lr_idx]`  
- `f1, f2`: número de filtros en las capas Conv2D  
- `dense_units`: número de neuronas en la capa densa  
- `dropout_idx`: 0 (sin dropout), 1 (0.25), 2 (0.5)  
- `lr_idx`: índice de tasa de aprendizaje (`1e-2, 1e-3, 1e-4`)  

**Estructura GA:**
- **Selección:** Torneo  
- **Cruzamiento:** Two-Point Crossover  
- **Mutación:** Reemplazo aleatorio de un gen  

---

## Ejecución ⚡

Clonar el repositorio y ejecutar cada script:

```bash
# Selección de características
Feature_Selection.ipynb

# Optimización de hiperparámetros
Hyperparameter_Optimization.ipynb

# Neuroevolución CNN
Neuroevolution.ipynb

```

---

## Integrantes del Grupo 👥

- De La Cruz Gomez Ismael
- Mamani Condemayta Sandra
- Paredes Arcaya Mery Luz
- Yana Mullisaca Sandra Nayely
