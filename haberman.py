import math
import random

import matplotlib as mpl
import matplotlib.animation
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

column_names = [
    "age_of_patient", "year_of_operation", "axillary_nodes_detected", "survival_status"
]
target_name = "survival_status"

df = pd.read_csv("haberman.data", names=column_names)


def separar_dataframe(dataframe):
    return (dataframe.iloc[:, -1], dataframe.iloc[:, :-1])


def boxplot(data, title=''):
    sns.boxplot(data=data, orient='h').set_title(title)


def pairplot(data, target, columns):
    sns.pairplot(data=data, hue=target, markers=["o", "D"], vars=columns)


target_data, fields_data = separar_dataframe(df)

# Explorando la data
sns.set_style('whitegrid')
pairplot(df, target_name, column_names[:-1])

corr = df.corr()
sns.heatmap(corr)

# Identificando Outliers
boxplot(fields_data, "DataFrame Crudo")
boxplot(df['axillary_nodes_detected'], "Axillary Nodes Detected")
sns.distplot(df['axillary_nodes_detected'])

# Procesando Outliers
df['axillary_nodes_detected'].describe()

# Excluyendo Outliers en axillary_nodes_detected por encima de 4
df_ol = df[df.axillary_nodes_detected <= 4]
df_ol.shape

# Visualizando la data luego de procesar
pairplot(df_ol, target_name, column_names[:-1])
boxplot(df_ol['axillary_nodes_detected'],
        "Axillary Nodes Detected post procesado")

# Procesando Outliers con Z Score
z_score = np.abs(stats.zscore(df))
print(z_score)

threshold = 3
print(np.where(z_score > threshold))

df_z = df[(z_score < 3).all(axis=1)]
df_z.shape

# Visualizando la data luego de procesar
pairplot(df_z, target_name, column_names[:-1])
boxplot(df_z['axillary_nodes_detected'],
        "Axillary Nodes Detected post z-score")

# Procesando Outliers para el DataFrame completo
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1
print(iqr)

df_qol = df[~((df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))).any(axis=1)]
df_qol.shape

# Visualizando la data luego de procesar
pairplot(df_qol, target_name, column_names[:-1])
boxplot(df_qol['axillary_nodes_detected'],
        "Axillary Nodes Detected post IQR")

# Iniciando Data

X_train, X_test, y_train, y_test = train_test_split(
    fields_data, target_data, stratify=target_data)


def print_rendimiento(entrenamiento, prueba, total):
    print("Rendimiento en el conjunto de entrenamiento: ", entrenamiento)
    print("Rendimiento en el conjunto de prueba: ", prueba)
    print("Rendimiento en el conjunto total: ", total)


def generar_modelo(solver="lbfgs", f_activation="logistic", capas=(20,), aprendizaje=0.0001, batch="auto", max_iter=200):
    mlp_modelo = MLPClassifier(
        solver=solver, random_state=0, hidden_layer_sizes=capas, activation=f_activation, alpha=aprendizaje, batch_size=batch, max_iter=max_iter)
    mlp_modelo.fit(X_train, y_train)
    print_rendimiento(
        mlp_modelo.score(X_train, y_train),
        mlp_modelo.score(X_test, y_test),
        mlp_modelo.score(fields_data, target_data)
    )
    return mlp_modelo


def neuronas_capa_oculta(neuronas_entrada: int, neuronas_salida: int):
    return math.pow(neuronas_entrada * neuronas_salida, 1/2)


possible_output = 2
neuronas_ocultas = neuronas_capa_oculta(len(column_names) - 1, possible_output)

generar_modelo()
