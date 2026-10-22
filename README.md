# Segmentación de Clientes con K-Means

Taller final de **Minería de Datos**. Se segmenta a los clientes de un centro comercial con el algoritmo **K-Means**, se evalúa el número de clusters con el **método del codo** y el **coeficiente de silueta**, y se despliega una aplicación interactiva en **Streamlit**.

## Dataset

[Mall Customers Dataset (Kaggle)](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python): 200 clientes con `CustomerID`, `Gender`, `Age`, `Annual Income (k$)` y `Spending Score (1-100)`.

## Estructura

| Archivo | Descripción |
|---|---|
| `Laboratorio.ipynb` | Cuaderno con las 7 partes del taller y las respuestas a todas las preguntas. |
| `app.py` | Aplicación de Streamlit (codo, silueta, clusters y clasificación de nuevos clientes). |
| `Mall_Customers.csv` | Dataset. |
| `requirements.txt` | Dependencias. |

## Resultados principales

- Variables usadas: **Ingreso Anual** y **Puntaje de Gasto**, escaladas con `StandardScaler`.
- El método del codo y el coeficiente de silueta coinciden en **K = 5** (silueta ≈ 0.555).
- Perfiles obtenidos:

| Perfil | Ingreso (k$) | Gasto | Clientes |
|---|---|---|---|
| Alto valor (VIP) | ≈ 87 | ≈ 82 | 39 |
| Con potencial (alto ingreso, bajo gasto) | ≈ 88 | ≈ 17 | 35 |
| Promedio | ≈ 55 | ≈ 50 | 81 |
| Impulsivos (bajo ingreso, alto gasto) | ≈ 26 | ≈ 79 | 22 |
| Bajo valor | ≈ 26 | ≈ 21 | 23 |

- El segmento con **mayor oportunidad de crecimiento** es **Con potencial**: tiene el ingreso más alto pero gasta poco.

## Ejecución local

```bash
pip install -r requirements.txt
jupyter notebook Laboratorio.ipynb   # cuaderno
streamlit run app.py                 # aplicación
```

## Despliegue

Aplicación desplegada en Streamlit Community Cloud:

**https://laboratoriomineriadatos-g823mjbeecj3xufmuuxfjy.streamlit.app/**
