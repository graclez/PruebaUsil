# ===============================================================
# STARTER EDA - FINANCIAL SAMPLE (Kaggle)
# Objetivo: 10 minutos para "romper el hielo" y ver relaciones básicas
# ===============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# (Opcional) estilo básico
sns.set(style="whitegrid")

# ---------------------------------------------------------------
# 1) CARGA RÁPIDA Y CHEQUEO
# ---------------------------------------------------------------
# Si lo tenés en CSV, usá: df = pd.read_csv("Financial Sample.csv")
df = pd.read_excel("Financial Sample.xlsx")

print("Dimensiones:", df.shape)
print(df.head(3))         # primeras filas
print(df[["Sales","Profit"]].describe())  # numeritos clave

# ---------------------------------------------------------------
# 2) DISPERSIÓN SIMPLE (Sales vs Profit)
#    Meta: ¿a más ventas, más ganancia? (correlación visual)
# ---------------------------------------------------------------
plt.figure(figsize=(7,4))
sns.scatterplot(data=df, x="Sales", y="Profit")
plt.title("Ventas vs Ganancia (dispersión simple)")
plt.xlabel("Sales ($)")
plt.ylabel("Profit ($)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 3) DISPERSIÓN CON TERCERA VARIABLE (hue=Segment)
#    Meta: ver si el segmento cambia el comportamiento
# ---------------------------------------------------------------
plt.figure(figsize=(7,4))
sns.scatterplot(data=df, x="Sales", y="Profit", hue="Segment")
plt.title("Ventas vs Ganancia por Segmento")
plt.xlabel("Sales ($)")
plt.ylabel("Profit ($)")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 4) BOXplot RÁPIDO (Profit por Segment)
#    Meta: comparar distribuciones y detectar outliers
# ---------------------------------------------------------------
plt.figure(figsize=(7,4))
sns.boxplot(data=df, x="Segment", y="Profit")
plt.title("Distribución de Profit por Segmento")
plt.xlabel("Segmento")
plt.ylabel("Profit ($)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 5) HEATMAP BÁSICO DE CORRELACIÓN
#    Meta: ver “todo el bosque” de correlaciones numéricas
# ---------------------------------------------------------------
numeric_corr = df.corr(numeric_only=True)
plt.figure(figsize=(6,4))
sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación (numéricas)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 6) MINI DESAFÍO (2 minutos)
#    a) Cambiá hue="Country" en el scatter de (2) y observá cambios.
#    b) Hacé un barplot de Sales por Segment (estimator=sum).
# ---------------------------------------------------------------
# Ejemplo guía del barplot (podés descomentar para mostrar):
# plt.figure(figsize=(7,4))
# sns.barplot(data=df, x="Segment", y="Sales", estimator=sum, ci=None)
# plt.title("Ventas Totales por Segmento")
# plt.tight_layout()
# plt.show()
