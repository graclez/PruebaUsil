# ===============================================================
# CLASE 2 y 3 - EJERCITARIO COMPLETO DE VISUALIZACIÓN BIVARIADA
# Dataset: Financial Sample (Kaggle)
# Profesora: Graciela Lezcano
# ===============================================================

# ---------------------------------------------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# ---------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración estética de Seaborn
sns.set(style="whitegrid", palette="pastel")

# ---------------------------------------------------------------
# 2. CARGAR EL DATASET
# ---------------------------------------------------------------
df = pd.read_excel("Financial Sample.xlsx")

print("✅ Dataset cargado correctamente.")
print(df.head())

# ---------------------------------------------------------------
# 3. ANÁLISIS PRELIMINAR
# ---------------------------------------------------------------
print("\nInformación general del dataset:")
print(df.info())
print("\nEstadísticas básicas:")
print(df.describe())

# Eliminamos filas con valores nulos si las hay
df = df.dropna()

# ---------------------------------------------------------------
# 4. REPASO DE MATPLOTLIB - GRÁFICOS SIMPLES
# ---------------------------------------------------------------
# Línea: tendencia de ventas por año
ventas_anuales = df.groupby("Year")["Sales"].sum()

plt.figure(figsize=(8,5))
plt.plot(ventas_anuales.index, ventas_anuales.values, marker="o")
plt.title("Tendencia de Ventas por Año")
plt.xlabel("Año")
plt.ylabel("Ventas ($)")
plt.grid(True)
plt.show()

# Barras: comparación de ventas por segmento
ventas_segmento = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(7,5))
ventas_segmento.plot(kind="bar", color="skyblue")
plt.title("Ventas Totales por Segmento")
plt.xlabel("Segmento")
plt.ylabel("Ventas ($)")
plt.show()

# ---------------------------------------------------------------
# 5. SEABORN – SCATTERPLOT BÁSICO
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", data=df)
plt.title("Relación entre Ventas y Ganancias")
plt.xlabel("Ventas ($)")
plt.ylabel("Ganancia ($)")
plt.show()

# 💬 COMENTARIO:
# Si los puntos suben hacia la derecha, hay correlación positiva.
# Si bajan, correlación negativa. Si están dispersos, no hay relación.

# ---------------------------------------------------------------
# 6. SCATTERPLOT CON HUE (TERCERA VARIABLE)
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", hue="Segment", data=df)
plt.title("Relación Ventas-Ganancias por Segmento")
plt.show()

# 💬 COMENTARIO:
# El color diferencia categorías. Vemos si un grupo tiene márgenes más altos.

# ---------------------------------------------------------------
# 7. SCATTERPLOT CON ESTILO Y TAMAÑO
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x="Sales", y="Profit", hue="Segment", style="Country", size="Discount Band", data=df)
plt.title("Gráfico con hue, estilo y tamaño")
plt.show()

# 💬 COMENTARIO:
# Se pueden combinar múltiples variables visualmente para entender patrones más complejos.

# ---------------------------------------------------------------
# 8. BOXPLOT – DISTRIBUCIÓN DE PROFIT POR SEGMENTO
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Segment", y="Profit", data=df)
plt.title("Distribución de Ganancia por Segmento")
plt.show()

# 💬 COMENTARIO:
# El boxplot muestra mediana, cuartiles y outliers (valores atípicos).

# ---------------------------------------------------------------
# 9. VIOLINPLOT – DISTRIBUCIÓN DETALLADA
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.violinplot(x="Segment", y="Profit", data=df, inner="quartile")
plt.title("Distribución de Ganancia (Violín)")
plt.show()

# 💬 COMENTARIO:
# Muestra densidad de los datos. Las partes más anchas indican donde se concentran los valores.

# ---------------------------------------------------------------
# 10. BARPLOT – COMPARACIÓN DE PROMEDIOS
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.barplot(x="Segment", y="Sales", data=df, estimator=sum, ci=None)
plt.title("Ventas Totales por Segmento")
plt.show()

# 💬 COMENTARIO:
# Barplot puede mostrar sumas o promedios, según el parámetro 'estimator'.

# ---------------------------------------------------------------
# 11. MÚLTIPLES GRÁFICOS CON FACETGRID
# ---------------------------------------------------------------
# Queremos comparar la relación entre Sales y Profit, pero por país.
# ---------------------------------------------------------------
g = sns.FacetGrid(df, col="Country", col_wrap=3, height=4)
g.map(sns.scatterplot, "Sales", "Profit")
g.fig.suptitle("Relación Ventas-Ganancia por País", y=1.02)
plt.show()

# 💬 COMENTARIO:
# FacetGrid permite crear una grilla de gráficos por categoría.

# ---------------------------------------------------------------
# 12. HEATMAP DE CORRELACIÓN
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor - Correlaciones")
plt.show()

# 💬 COMENTARIO:
# Colores cálidos = correlación positiva fuerte.
# Colores fríos = correlación negativa.

# ---------------------------------------------------------------
# 13. EJEMPLO DE CORRELACIÓN NUMÉRICA
# ---------------------------------------------------------------
corr_sales_profit = df["Sales"].corr(df["Profit"])
print(f"🔹 Correlación entre Ventas y Ganancias: {corr_sales_profit:.2f}")

# 💬 COMENTARIO:
# Valores cercanos a 1 indican correlación fuerte positiva.
# Cercanos a -1, fuerte negativa. Cercanos a 0, sin relación.

# ---------------------------------------------------------------
# 14. MINI DESAFÍOS PARA LOS ALUMNOS 💡
# ---------------------------------------------------------------
print("\n💡 DESAFÍOS:")
print("1️⃣ Crear un scatterplot de 'Units Sold' vs 'COGS'.")
print("2️⃣ Usar hue='Country' y style='Segment'.")
print("3️⃣ Hacer un boxplot del campo 'Profit' por 'Country'.")
print("4️⃣ Crear un heatmap sólo de columnas numéricas seleccionadas.")
print("5️⃣ Explicar qué relación observan en los gráficos.")

# ---------------------------------------------------------------
# 15. EXTRA: PALETA DE COLORES PERSONALIZADA
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x="Segment", y="Profit", data=df, palette="Set2")
plt.title("Boxplot con paleta personalizada")
plt.show()

# ---------------------------------------------------------------
# 16. EXTRA: RELACIÓN ENTRE VARIABLES CON REGRESIÓN LINEAL
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
sns.lmplot(x="Sales", y="Profit", data=df, height=5, aspect=1.3, line_kws={'color':'red'})
plt.title("Regresión lineal: Ventas vs Ganancias")
plt.show()

# 💬 COMENTARIO:
# lmplot agrega una línea de tendencia automáticamente,
# útil para visualizar la dirección de la correlación.
