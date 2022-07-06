import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def ObtenerColumnasConNan(df):
    nombresColumnas = df.columns
    columnas = []
    for columna in nombresColumnas:
        if(df[columna].isnull().sum() > 0):
            columnas.append(columna)
    return columnas

def ObtenerTiposColumnas(df):
    columnasNumericasDiscretas = []
    columnasNumericasContinuas = []
    columnasCategoricas = []
    columnasFecha = []
    for columna in df.columns:
        if(df[columna].dtype == "object"):
            columnasCategoricas.append(columna)
        elif((df[columna].dtype == "int64") or (df[columna].dtype == "float64")):
            totalValores = len(df[columna].value_counts())
            if(totalValores <= 30):
                columnasNumericasDiscretas.append(columna)
            else:
                columnasNumericasContinuas.append(columna)
        else:
            columnasFecha.append(columna)
            
    return columnasNumericasDiscretas, columnasNumericasContinuas, columnasCategoricas, columnasFecha

def GraficaVariablesCategoricas(df, columnasCategoricas, y):
    for columna in columnasCategoricas:
        plt.figure(figsize=(12,6))
        plot = sns.countplot(x=df[columna], hue=df[y])
        plt.show()

def GraficoBarrasCategoricas(df, columna):
    df[columna].value_counts().sort_values(ascending=False).plot.bar()
    plt.ylabel("Cantidad")
    plt.xlabel(columna)
    plt.show()

def GraficoBarrasNumericas(df, columna):
    df[columna].sort_values(ascending=False).plot.bar()
    plt.ylabel("Cantidad")
    plt.xlabel(columna)
    plt.show()

def GraficacaDensidadVariable(df, columna):
    
    plt.figure(figsize = (15,6))
    plt.subplot(121)
    df[columna].hist(bins=30)
    plt.title(columna)
    
    plt.subplot(122)
    stats.probplot(df[columna], dist="norm", plot=plt)
    plt.show()

def GraficosOutliers(df, columna):
    
    plt.figure(figsize = (15,6))
    
    plt.subplot(131)
    sns.distplot(df[columna], bins=30)
    plt.title("Densisd-Histograma: " + columna)
    
    plt.subplot(132)
    stats.probplot(df[columna], dist="norm", plot=plt)
    plt.title("QQ-Plot: " + columna)
    
    plt.subplot(133)
    sns.boxplot(y=df[columna])
    plt.title("Boxplot: " + columna)
    
    plt.show()

def MatrizConfusion(nombreModelo, yTest, predicciones):
    matrizConfusion = pd.crosstab(yTest, predicciones, rownames=["observación"], colnames=["Predicción"])
    print(f"\nMatriz de Confusión - {nombreModelo}: \n\n", matrizConfusion)
    
    TP = matrizConfusion.iloc[1,1]
    TN = matrizConfusion.iloc[0,0]
    FN = matrizConfusion.iloc[1,0]
    FP = matrizConfusion.iloc[0,1]
    
    print("\nSentitividad: ", TP/(TP+FN))
    print("Especificidad: ", TN/(TN+FP))