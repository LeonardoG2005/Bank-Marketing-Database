# Proyecto 1 - Bank Marketing (Clasificación de clientes)

## Inteligencia Artificial

**Integrantes:**
- Leonardo Guevara Atehortúa
- Ángel David Avirama de Oro

---

## 1. Descripción del proyecto

Este proyecto desarrolla un problema de Machine Learning de clasificación binaria utilizando el dataset Bank Marketing.

El objetivo es predecir si un cliente de una institución bancaria portuguesa suscribirá (yes) o no suscribirá (no) un depósito a plazo después de una campaña de marketing telefónico.

En este contexto:

- Cliente: persona contactada por el banco.
- Llamada/contacto: interacción telefónica realizada durante una campaña de marketing.
- Suscribir: aceptar o contratar el depósito a plazo ofrecido por el banco.
- Variable objetivo (y): indica si el cliente aceptó (yes) o no aceptó (no) el producto.

El modelo busca aprender patrones a partir de clientes históricos para apoyar la identificación de clientes con mayor probabilidad de aceptar el producto.

---

## 2. Dataset

El dataset utilizado es Bank Marketing, publicado por el UCI Machine Learning Repository.

Fuente: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Se utiliza el archivo **bank-full.csv**.

Características principales:

- 45.211 registros
- 17 columnas
- 16 variables predictoras
- 1 variable objetivo (y)
- Variables numéricas y categóricas
- No se presentan valores faltantes en el dataset utilizado

### Variable objetivo

La variable y representa el resultado de la campaña:

| Valor | Significado |
|---|---|
| yes | El cliente suscribió/contrató el depósito a plazo |
| no | El cliente no suscribió/contrató el depósito a plazo |

Por lo tanto, se trata de un problema de clasificación binaria.

---

## 3. Variables utilizadas

El dataset contiene las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| age | Numérica entera | Edad del cliente |
| job | Categórica | Tipo de trabajo u ocupación |
| marital | Categórica | Estado civil |
| education | Categórica | Nivel educativo |
| default | Binaria/categórica | Si tiene crédito en mora |
| balance | Numérica entera | Saldo promedio anual en euros |
| housing | Binaria/categórica | Si tiene préstamo hipotecario |
| loan | Binaria/categórica | Si tiene préstamo personal |
| contact | Categórica | Tipo de contacto utilizado |
| day | Numérica entera | Día del mes del último contacto |
| month | Categórica | Mes del último contacto |
| duration | Numérica entera | Duración de la última llamada, en segundos |
| campaign | Numérica entera | Número de contactos durante la campaña actual |
| pdays | Numérica entera | Días desde el contacto anterior; -1 indica que no fue contactado previamente |
| previous | Numérica entera | Número de contactos realizados antes de la campaña actual |
| poutcome | Categórica | Resultado de la campaña anterior |
| y | Binaria / objetivo | Si el cliente suscribió el depósito a plazo |

### Tratamiento de duration

La variable duration representa la duración de la última llamada. Esta información solamente está disponible después de que la llamada ocurre.

Como el objetivo del proyecto es utilizar información disponible para predecir qué clientes tienen mayor probabilidad de suscribirse, duration se excluye de los modelos predictivos para evitar data leakage.

---

## 4. Análisis exploratorio de datos (EDA)

El notebook incluye un análisis exploratorio de los datos que contempla:

- Dimensiones del dataset.
- Información general y tipos de datos.
- Revisión de valores faltantes.
- Revisión de registros duplicados.
- Estadísticas descriptivas mediante describe().
- Distribución de variables numéricas mediante histogramas.
- Identificación de posibles valores extremos mediante diagramas de caja.
- Frecuencias de las variables categóricas.
- Distribución de la variable objetivo.
- Tasa de suscripción (yes) según diferentes variables.
- Análisis de correlaciones entre variables numéricas.

El análisis permite identificar las características de cada variable, sus rangos, escalas, tipos y posibles relaciones con la variable objetivo.

También se observa un desbalance entre las clases, por lo que no se utiliza únicamente Accuracy para evaluar los modelos. Se da especial importancia a Precision, Recall y F1 de la clase Yes.

---

## 5. Preprocesamiento

Antes de entrenar los modelos se realiza el siguiente procesamiento:

1. La variable objetivo y se transforma a valores binarios: no : 0, yes : 1.
2. Se excluye duration para evitar data leakage.
3. Las variables numéricas se procesan mediante imputación de valores faltantes utilizando la mediana y estandarización mediante StandardScaler.
4. Las variables categóricas se procesan mediante imputación utilizando la categoría más frecuente y codificación One-Hot mediante OneHotEncoder.
5. Los pasos de preprocesamiento y el modelo se integran mediante Pipeline y ColumnTransformer.
6. Los datos se dividen en 80% entrenamiento y 20% prueba, de forma estratificada para conservar la proporción de las clases.

---

## 6. Metodología de validación

Para el entrenamiento se utiliza validación cruzada estratificada de 10 folds (StratifiedKFold).

La métrica utilizada durante la búsqueda de hiperparámetros es F1, porque permite considerar simultáneamente Precision y Recall, lo cual es especialmente importante debido al desbalance de la variable objetivo.

---

## 7. Modelos utilizados

### 7.1 Regresión Logística

Se utiliza LogisticRegression con `class_weight="balanced"` y `max_iter=2000`.

Se realiza una búsqueda de hiperparámetros mediante GridSearchCV.

Valores evaluados:

- C: 0.001, 0.01, 0.1, 1, 10, 100
- solver: liblinear, lbfgs

El mejor conjunto de hiperparámetros encontrado fue C = 100, solver = lbfgs.

El mejor F1 promedio obtenido durante la validación cruzada fue F1 CV = 0.3734.

### 7.2 Máquinas de Vectores de Soporte (SVM)

Se utiliza SVC con `class_weight="balanced"` y `probability=False`.

Para reducir el costo computacional de la búsqueda de hiperparámetros, se utiliza una muestra estratificada de 6.000 registros del conjunto de entrenamiento para la búsqueda del SVM.

Se evaluaron diferentes configuraciones:

- Kernel linear con C = 0.1, 1, 10
- Kernel rbf con C = 1, 10 y gamma = scale
- Kernel poly con C = 1, grados 2 y 3, y gamma = scale

El mejor SVM encontrado fue kernel = rbf, C = 1, gamma = scale.

El mejor F1 promedio obtenido durante la validación cruzada fue F1 CV = 0.3898.

---

## 8. Ajuste del umbral de decisión del SVM

Después de obtener el mejor SVM, se realizó un ajuste del umbral de decisión para buscar un mejor equilibrio entre Precision y Recall de la clase Yes.

No se utilizó `probability=True`, ya que esto añade un proceso de calibración adicional y aumenta considerablemente el costo computacional. En su lugar, se utilizaron los valores de `decision_function()`.

El umbral se seleccionó utilizando scores obtenidos mediante validación cruzada sobre los datos utilizados para entrenamiento, evitando utilizar el conjunto de prueba para elegir el umbral.

El umbral seleccionado fue 0.30.

---

## 9. Resultados

Las métricas principales corresponden a la clase positiva Yes.

### Regresión Logística

| Métrica | Resultado |
|---|---:|
| Precision | 0.2667 |
| Recall | 0.6238 |
| F1 | 0.3736 |
| Accuracy | 0.7553 |

### SVM con umbral por defecto

| Métrica | Resultado |
|---|---:|
| Precision | 0.3424 |
| Recall | 0.5728 |
| F1 | 0.4286 |
| Accuracy | 0.8213 |

### SVM con umbral optimizado

| Métrica | Resultado |
|---|---:|
| Umbral | 0.30 |
| Precision | 0.4146 |
| Recall | 0.4773 |
| F1 | 0.4438 |
| Accuracy | 0.8600 |

El ajuste del umbral permitió aumentar el F1 del SVM de 0.4286 a 0.4438 y la Precision de la clase Yes de 0.3424 a 0.4146. El Recall disminuyó de 0.5728 a 0.4773, por lo que el ajuste favorece un mejor equilibrio entre Precision y Recall según F1.

---

## 10. Comparación final

| Modelo | Precision (Yes) | Recall (Yes) | F1 (Yes) |
|---|---:|---:|---:|
| Regresión Logística | 0.2667 | 0.6238 | 0.3736 |
| SVM + umbral optimizado | 0.4146 | 0.4773 | 0.4438 |

### Mejor modelo

De acuerdo con la métrica F1 de la clase Yes, el mejor modelo del proyecto es el SVM con kernel RBF y umbral optimizado de 0.30.

Este modelo obtuvo un F1 de 0.4438, superior al 0.3736 obtenido por la Regresión Logística. Aunque la Regresión Logística obtuvo un Recall mayor, el SVM presenta un mejor equilibrio entre Precision y Recall, reflejado en su mayor F1.

---

## 11. Conclusiones

El proyecto permitió desarrollar y comparar dos modelos de clasificación para predecir la suscripción de un depósito a plazo.

La validación cruzada estratificada de 10 folds y la búsqueda de hiperparámetros permitieron seleccionar configuraciones adecuadas para cada modelo.

La Regresión Logística obtuvo un F1 de 0.3736 para la clase Yes, mientras que el SVM alcanzó inicialmente 0.4286. Mediante el ajuste del umbral de decisión del SVM se obtuvo un F1 de 0.4438, convirtiéndolo en el mejor modelo según esta métrica.

El modelo final podría usarse como herramienta de apoyo para priorizar clientes con mayor probabilidad de aceptar el producto financiero. Aún existe margen de mejora, particularmente en la identificación de clientes de la clase positiva.

---

## 12. Estructura del repositorio

El repositorio debe contener como mínimo:

```
Proyecto-1-Bank-Marketing/
│
├── Proyecto_1_Bank_Marketing_Proyecto1.ipynb
├── bank-full.csv
├── README.md
```

---

## 13. Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- UCI Machine Learning Repository
