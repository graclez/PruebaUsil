# -------------------------------------------------------------
# CLASE 3 - EDA Integrado (EJERCICIOS SÚPER BÁSICOS, MUY GUIADOS)
# Archivo: clase03_ejercicios_super_basicos.py
# Requisitos: pandas, matplotlib, (opcional) seaborn
# Importante: este .py debe estar en la MISMA carpeta que 'train.csv'
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# (Opcional) activar cuadrícula por defecto
plt.rcParams["figure.figsize"] = (8,5)

print("=== CLASE 3 - EDA (Ejercicios súper básicos) ===")

# =============================================================
# 0) CARGA DEL DATASET Y CHEQUEOS RÁPIDOS
# =============================================================
df = pd.read_csv("train.csv")  # Asegurate que esté en la misma carpeta

print("\n[0.1] Primeras filas:")
print(df.head())

print("\n[0.2] Info general:")
print(df.info())

print("\n[0.3] Nulos por columna:")
print(df.isnull().sum())

print("\n[0.4] Estadísticas numéricas:")
print(df.describe())

# =============================================================
# 1) LIMPIEZA BÁSICA (según lo visto): eliminar nulos de columnas clave
#    * Si alguna no existe en tu dataset, comentá esa parte.
# =============================================================
cols_clave = [c for c in ["SalePrice", "GrLivArea", "OverallQual", "Neighborhood"] if c in df.columns]
df1 = df.dropna(subset=cols_clave)
print(f"\n[1] Dropna en columnas clave {cols_clave} → filas: {len(df)} → {len(df1)}")

# =============================================================
# 2) FEATURE ENGINEERING BÁSICO: price_per_sqft
#    Definición: SalePrice / GrLivArea (evitar división por 0)
# =============================================================
if "SalePrice" in df1.columns and "GrLivArea" in df1.columns:
    df1["price_per_sqft"] = df1["SalePrice"] / df1["GrLivArea"].replace(0, np.nan)
    print("\n[2] Columna creada: price_per_sqft")
    print(df1[["SalePrice", "GrLivArea", "price_per_sqft"]].head())
else:
    df1["price_per_sqft"] = np.nan
    print("\n[2] No se pudo crear price_per_sqft (faltan columnas).")

# =============================================================
# 3) HISTOGRAMAS SENCILLOS (sin estilos ni paletas complejas)
# =============================================================
if "SalePrice" in df1.columns:
    plt.figure()
    df1["SalePrice"].dropna().plot(kind="hist", bins=40)
    plt.title("Distribución: SalePrice")
    plt.xlabel("SalePrice")
    plt.ylabel("Frecuencia")
    plt.show()

if "price_per_sqft" in df1.columns:
    plt.figure()
    df1["price_per_sqft"].dropna().plot(kind="hist", bins=40)
    plt.title("Distribución: price_per_sqft")
    plt.xlabel("price_per_sqft")
    plt.ylabel("Frecuencia")
    plt.show()

# =============================================================
# 4) BOXPLOTS SENCILLOS (usando matplotlib puro)
# =============================================================
if "SalePrice" in df1.columns:
    plt.figure()
    plt.boxplot(df1["SalePrice"].dropna(), vert=False)
    plt.title("Boxplot: SalePrice")
    plt.xlabel("SalePrice")
    plt.show()

if "price_per_sqft" in df1.columns:
    plt.figure()
    plt.boxplot(df1["price_per_sqft"].dropna(), vert=False)
    plt.title("Boxplot: price_per_sqft")
    plt.xlabel("price_per_sqft")
    plt.show()

# =============================================================
# 5) DISPERSIÓN BÁSICA: OverallQual vs SalePrice
# =============================================================
if "OverallQual" in df1.columns and "SalePrice" in df1.columns:
    plt.figure()
    plt.scatter(df1["OverallQual"], df1["SalePrice"], alpha=0.7)
    plt.title("OverallQual vs SalePrice")
    plt.xlabel("OverallQual")
    plt.ylabel("SalePrice")
    plt.show()

# =============================================================
# 6) IQR BÁSICO PASO A PASO (CUANTIFICAR OUTLIERS) en SalePrice
#    Def: IQR = Q3 - Q1; outlier si < Q1-1.5*IQR o > Q3+1.5*IQR
# =============================================================
if "SalePrice" in df1.columns:
    y = df1["SalePrice"].dropna()
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask_out = (y < low) | (y > high)
    n_out = mask_out.sum()
    pct_out = 100 * n_out / len(y)
    print("\n[6] IQR en SalePrice")
    print(f" Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}")
    print(f" Límites: [{low:.2f}, {high:.2f}]  |  Outliers={n_out} ({pct_out:.2f}%)")

    # A) Remover outliers
    df2 = df1.loc[~((df1["SalePrice"] < low) | (df1["SalePrice"] > high))].copy()
    print(f" [6A] Remover outliers → filas: {len(df1)} → {len(df2)}")

    # B) Capping (reemplazar por límite)
    df3 = df1.copy()
    df3.loc[df3["SalePrice"] < low, "SalePrice"] = low
    df3.loc[df3["SalePrice"] > high, "SalePrice"] = high
    print(f" [6B] Capping aplicado → filas: {len(df3)} (mismo tamaño)")

# =============================================================
# 7) BARRIOS: mediana de price_per_sqft (Top 10)
# =============================================================
if "Neighborhood" in df1.columns and "price_per_sqft" in df1.columns:
    mediana_top10 = (df1[["Neighborhood", "price_per_sqft"]]
                     .dropna()
                     .groupby("Neighborhood")["price_per_sqft"]
                     .median()
                     .sort_values(ascending=False)
                     .head(10))
    print("\n[7] Top 10 barrios por mediana de price_per_sqft:")
    print(mediana_top10)

    plt.figure(figsize=(10,5))
    mediana_top10.plot(kind="bar")
    plt.title("Top 10 barrios por mediana de price_per_sqft")
    plt.xlabel("Neighborhood")
    plt.ylabel("Mediana price_per_sqft")
    plt.xticks(rotation=45)
    plt.show()

# =============================================================
# 8) MATRIZ DE CORRELACIÓN SIMPLE (subset recomendado)
# =============================================================
subset = [c for c in ["SalePrice", "GrLivArea", "OverallQual", "price_per_sqft"] if c in df1.columns]
if len(subset) >= 2:
    corr = df1[subset].corr(numeric_only=True)
    print("\n[8] Correlaciones (subset):")
    print(corr)

    plt.figure(figsize=(6,4))
    # Heatmap manual con matplotlib: texto + colores simples
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlación")
    plt.xticks(range(len(subset)), subset, rotation=45, ha="right")
    plt.yticks(range(len(subset)), subset)
    for i in range(len(subset)):
        for j in range(len(subset)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Matriz de correlación (subset)")
    plt.tight_layout()
    plt.show()

# =============================================================
# 9) MINI-REPORTE IMPRESO (para leer en voz alta)
# =============================================================
print("\n[9] Mini-Reporte (ejemplo):")
if len(subset) >= 2 and "SalePrice" in subset and "OverallQual" in subset:
    corr_oq = corr.loc["SalePrice","OverallQual"]
    print(f" • OverallQual se relaciona fuertemente con SalePrice (corr ≈ {corr_oq:.2f}).")
else:
    print(" • Correlaciones principales: ver heatmap anterior.")

if "price_per_sqft" in df1.columns and "Neighborhood" in df1.columns:
    print(" • price_per_sqft permite comparar valor relativo por barrio (ver Top 10).")

if "SalePrice" in df1.columns:
    print(f" • IQR detectó ≈ {n_out} outliers de SalePrice ({pct_out:.2f}%).")

# =============================================================
# 10) ARCHIVOS DE SALIDA (para práctica y entrega)
# =============================================================
df2.to_csv("train_sin_outliers.csv", index=False)
df3.to_csv("train_capped.csv", index=False)
print("\n[10] Guardados: 'train_sin_outliers.csv', 'train_capped.csv'")

print("\n=== FIN DE LOS EJERCICIOS BÁSICOS ===")
